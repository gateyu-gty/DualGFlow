"""Decoupled Actor-Critic DAgger + Critic Warmup + Full-Batch PPO for DualGFlow-RG v2.

Key fixes over prototype:
1. Decoupled Actor and Critic GNN backbones (DualGFlow ICLR 2025) so high-magnitude
   value regression gradients never corrupt Actor GNN representations.
2. Proper reward normalization (reward_scale=50000.0) + Huber value loss + Critic Warmup.
3. Full-trajectory PPO updates (~3,500 transitions/update instead of 640 subsample).
4. Strict Train / Validation (checkpoint selection) / Held-out Test (N=16 & Zero-Shot N=32) split.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from dualgflow.topologies import make_flow_scenario
from dualgflow.simulator_flow import FlowWorkflowEnv
from dualgflow.models_rg import DualGFlowPolicy
from dualgflow.run_prototype_experiment import evaluate_policy
from dualgflow.train_ppo_dagger import collect_dagger_states, train_supervised_dagger

REWARD_SCALE = 50000.0


def collect_rollouts(
    model: DualGFlowPolicy,
    scenarios: list[dict],
    num_episodes: int,
    rng: np.random.Generator,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
    anchor_model: DualGFlowPolicy | None = None,
) -> tuple[list[tuple], np.ndarray]:
    """Collect full on-policy episodes and compute GAE returns/advantages."""
    model.eval()
    chosen_idx = rng.choice(len(scenarios), size=min(num_episodes, len(scenarios)), replace=False)
    transitions = []
    ep_flowtimes = []
    device = next(model.parameters()).device

    with torch.no_grad():
        for idx in chosen_idx:
            sc = scenarios[int(idx)]
            env = FlowWorkflowEnv(sc, reward_scale=REWARD_SCALE)
            obs = env.reset()
            ep_obs, ep_act, ep_logp, ep_rew, ep_val, ep_anchor_lp = [], [], [], [], [], []
            while not env.done:
                logits, val = model.forward_obs(obs)
                dist = Categorical(logits=logits)
                act = dist.sample()
                logp = dist.log_prob(act)
                a_lp = (
                    F.log_softmax(anchor_model.forward_actor(obs), dim=0).cpu()
                    if anchor_model is not None
                    else F.log_softmax(logits, dim=0).cpu()
                )
                next_obs, rew, _, _ = env.step(int(act.item()))
                ep_obs.append(obs)
                ep_act.append(int(act.item()))
                ep_logp.append(float(logp.item()))
                ep_rew.append(float(rew))
                ep_val.append(float(val.item()))
                ep_anchor_lp.append(a_lp)
                obs = next_obs

            ep_flowtimes.append(env.metrics()['mean_flowtime'])

            T = len(ep_rew)
            adv = np.zeros(T, dtype=np.float32)
            ret = np.zeros(T, dtype=np.float32)
            gae = 0.0
            next_v = 0.0
            for t in reversed(range(T)):
                delta = ep_rew[t] + gamma * next_v - ep_val[t]
                gae = delta + gamma * gae_lambda * gae
                adv[t] = gae
                ret[t] = gae + ep_val[t]
                next_v = ep_val[t]

            for t in range(T):
                transitions.append((ep_obs[t], ep_act[t], ep_logp[t], ret[t], adv[t], ep_anchor_lp[t]))

    return transitions, np.asarray(ep_flowtimes, dtype=np.float64)


def update_critic(
    model: DualGFlowPolicy,
    opt_c: torch.optim.Optimizer,
    transitions: list[tuple],
    epochs: int = 3,
    batch_size: int = 64,
    rng: np.random.Generator | None = None,
) -> float:
    model.train()
    device = next(model.parameters()).device
    rng = rng or np.random.default_rng(0)
    losses = []
    for _ in range(epochs):
        perm = rng.permutation(len(transitions))
        opt_c.zero_grad()
        for b_i, idx in enumerate(perm):
            obs, _, _, ret_v, _, _ = transitions[int(idx)]
            val = model.forward_critic(obs)
            target = torch.tensor(ret_v, dtype=torch.float32, device=device)
            loss = F.smooth_l1_loss(val, target)
            (loss / batch_size).backward()
            losses.append(float(loss.item()))
            if (b_i + 1) % batch_size == 0 or (b_i + 1) == len(perm):
                nn.utils.clip_grad_norm_(model.critic_parameters(), 1.0)
                opt_c.step()
                opt_c.zero_grad()
    return float(np.mean(losses)) if losses else 0.0


def decoupled_ppo_train(
    model: DualGFlowPolicy,
    train_scenarios: list[dict],
    val_scenarios: list[dict],
    warmup_rounds: int = 3,
    ppo_updates: int = 10,
    episodes_per_update: int = 9,
    lr_a: float = 1e-4,
    lr_c: float = 5e-4,
    clip_eps: float = 0.15,
    kl_coef: float = 0.04,
    device: str = 'cuda',
) -> DualGFlowPolicy:
    """Critic Warmup + Decoupled Full-Batch On-Policy PPO."""
    model.to(device)
    anchor_model = deepcopy(model).eval()
    for p in anchor_model.parameters():
        p.requires_grad_(False)

    opt_a = torch.optim.AdamW(model.actor_parameters(), lr=lr_a, weight_decay=1e-4)
    opt_c = torch.optim.AdamW(model.critic_parameters(), lr=lr_c, weight_decay=1e-4)
    rng = np.random.default_rng(2026)

    # 1. Warmup Critic with Actor frozen
    print('  [Critic Warmup] Collecting rollouts & fitting Critic...', flush=True)
    for w_i in range(warmup_rounds):
        trans, _ = collect_rollouts(model, train_scenarios, episodes_per_update, rng, anchor_model=anchor_model)
        c_loss = update_critic(model, opt_c, trans, epochs=3, batch_size=64, rng=rng)
        print(f'    Warmup {w_i+1}/{warmup_rounds} | Transitions: {len(trans)} | Huber Loss: {c_loss:.4f}', flush=True)

    def eval_val() -> float:
        model.eval()
        vals = []
        with torch.no_grad():
            for sc in val_scenarios:
                r = evaluate_policy(sc, lambda e, o: int(torch.argmax(model.forward_actor(o)).item()))
                vals.append(r['mean_flowtime'])
        return float(np.mean(vals))

    best_val = eval_val()
    best_state = deepcopy(model.state_dict())
    print(f'  [PPO Init] Val Mean Flowtime: {best_val:.2f}s', flush=True)

    # 2. Decoupled PPO Updates (Full trajectory batch, separate optimizers)
    for upd in range(ppo_updates):
        trans, ep_flows = collect_rollouts(model, train_scenarios, episodes_per_update, rng, anchor_model=anchor_model)
        all_adv = np.array([x[4] for x in trans], dtype=np.float32)
        adv_mean, adv_std = float(all_adv.mean()), max(float(all_adv.std()), 1e-6)

        model.train()
        # Update Critic (2 epochs over all transitions)
        c_loss = update_critic(model, opt_c, trans, epochs=2, batch_size=64, rng=rng)

        # Update Actor (2 epochs over all transitions)
        for _ in range(2):
            perm = rng.permutation(len(trans))
            opt_a.zero_grad()
            for b_i, idx in enumerate(perm):
                obs, act, old_logp, _, raw_adv, anchor_lp = trans[int(idx)]
                norm_adv = (raw_adv - adv_mean) / adv_std

                logits = model.forward_actor(obs)
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(torch.tensor(act, device=device))
                ratio = torch.exp(new_logp - old_logp)
                surr1 = ratio * norm_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * norm_adv
                policy_loss = -torch.min(surr1, surr2)

                cur_lp = F.log_softmax(logits, dim=0)
                cur_p = cur_lp.exp()
                kl = (cur_p * (cur_lp - anchor_lp.to(device))).sum()
                entropy = dist.entropy()

                loss = policy_loss + kl_coef * kl - 0.005 * entropy
                (loss / 64.0).backward()
                if (b_i + 1) % 64 == 0 or (b_i + 1) == len(perm):
                    nn.utils.clip_grad_norm_(model.actor_parameters(), 0.5)
                    opt_a.step()
                    opt_a.zero_grad()

        val_flow = eval_val()
        tag = ''
        if val_flow < best_val:
            best_val = val_flow
            best_state = deepcopy(model.state_dict())
            tag = ' *BEST*'
        print(
            f'  [PPO {upd+1:2d}/{ppo_updates}] TrainEp: {ep_flows.mean():.1f}s | CriticLoss: {c_loss:.4f} | Val: {val_flow:.2f}s{tag}',
            flush=True,
        )

    model.load_state_dict(best_state)
    model.eval()
    return model


def evaluate_suite(scenarios: list[dict], models: dict[str, DualGFlowPolicy]) -> dict:
    methods = {
        'compute_eft': lambda env, obs: int(np.argmin(obs['compute_eft'])),
        'nominal_ca_eft': lambda env, obs: int(np.argmin(obs['nominal_eft'])),
        'flow_ca_eft': lambda env, obs: int(np.argmin(obs['flow_eft'])),
    }
    for k, m in models.items():
        methods[k] = lambda env, obs, mod=m: int(torch.argmax(mod.forward_actor(obs)).item())

    out = {}
    for name, fn in methods.items():
        by_topo = {'leaf_spine': [], 'cloud_edge': [], 'multi_hop_mesh': []}
        all_flows, all_lats = [], []
        with torch.no_grad():
            for sc in scenarios:
                topo = sc['meta']['topology']['topology']
                r = evaluate_policy(sc, fn, name)
                all_flows.append(r['mean_flowtime'])
                all_lats.append(r['p95_decision_ms'])
                by_topo[topo].append(r['mean_flowtime'])
        out[name] = dict(
            mean_flowtime=float(np.mean(all_flows)),
            std_flowtime=float(np.std(all_flows)),
            per_scene=all_flows,
            by_topology={t: float(np.mean(v)) for t, v in by_topo.items() if v},
            p95_decision_ms=float(np.mean(all_lats)),
        )
    return out


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Running Decoupled DualGFlow-RG v2 Full Training & Held-Out Test on device={device}', flush=True)

    topos = ['leaf_spine', 'cloud_edge', 'multi_hop_mesh']
    train_scenarios = [
        make_flow_scenario(seed=100 + i, split='train', workflows=12, num_servers=16,
                           load=0.75, ccr=2.5, topology=topos[i % 3], oversubscription=6.0)
        for i in range(18)
    ]
    val_scenarios = [
        make_flow_scenario(seed=200 + i, split='validation', workflows=14, num_servers=16,
                           load=0.75, ccr=2.5, topology=topos[i % 3], oversubscription=6.0)
        for i in range(6)
    ]
    test_scenarios_16 = [
        make_flow_scenario(seed=400 + i, split='test', workflows=14, num_servers=16,
                           load=0.75, ccr=2.5, topology=topos[i % 3], oversubscription=6.0)
        for i in range(15)
    ]
    test_scenarios_32 = [
        make_flow_scenario(seed=500 + i, split='test', workflows=20, num_servers=32,
                           load=0.75, ccr=2.5, topology=topos[i % 3], oversubscription=6.0)
        for i in range(9)
    ]

    print('Collecting DAgger Round 0 dataset...', flush=True)
    t0 = time.perf_counter()
    base_dataset = collect_dagger_states(train_scenarios, policy=None, states_per_scenario=35, top_k=6, seed=101)
    print(f'Round 0: {len(base_dataset)} states in {time.perf_counter() - t0:.1f}s', flush=True)

    torch.manual_seed(101)
    pilot_rg = DualGFlowPolicy(variant='dualgflow', mode='residual_eft', hidden_dim=64, gnn_layers=4)
    train_supervised_dagger(pilot_rg, base_dataset, epochs=18, lr=1e-3, device=device)
    print('Collecting DAgger Round 1 dataset...', flush=True)
    dagger_extra = collect_dagger_states(train_scenarios[:12], policy=pilot_rg, states_per_scenario=30, top_k=6, seed=202)
    full_dataset = base_dataset + dagger_extra
    print(f'Total DAgger dataset: {len(full_dataset)} states', flush=True)

    models = {}
    for mode in ('pure_structural', 'residual_eft'):
        for variant in ('dualgflow', 'global_mlp', 'no_mp_keep_links'):
            key = f'{variant}_{mode}'
            torch.manual_seed(101)
            m = DualGFlowPolicy(variant=variant, mode=mode, hidden_dim=64, gnn_layers=4)
            actor_p = sum(p.numel() for p in m.actor_parameters())
            print(f'\n=== Training {key} (actor_params={actor_p:,}) ===', flush=True)
            train_supervised_dagger(m, full_dataset, epochs=22, lr=1e-3, device=device)
            if mode == 'residual_eft':
                decoupled_ppo_train(
                    m, train_scenarios, val_scenarios,
                    warmup_rounds=3, ppo_updates=10, episodes_per_update=9,
                    lr_a=1e-4, lr_c=5e-4, device=device,
                )
            models[key] = m

    print('\nEvaluating on Held-Out Test N=16 (15 unseen scenarios)...', flush=True)
    test_16 = evaluate_suite(test_scenarios_16, models)
    print('Evaluating on Held-Out Zero-Shot Test N=32 (9 unseen scenarios)...', flush=True)
    test_32 = evaluate_suite(test_scenarios_32, models)

    out_path = Path(__file__).resolve().parent / 'full_ppo_results.json'
    out_path.write_text(json.dumps(dict(test_16=test_16, test_32=test_32), indent=2), encoding='utf-8')

    for tag, res in [('HELD-OUT TEST N=16 (15 SCENES)', test_16), ('ZERO-SHOT TEST N=32 (9 SCENES)', test_32)]:
        print(f'\n=== {tag} ===')
        base = res['flow_ca_eft']['mean_flowtime']
        for k, v in res.items():
            gain = (base - v['mean_flowtime']) / base * 100.0
            ls = v['by_topology']['leaf_spine']
            ce = v['by_topology']['cloud_edge']
            mh = v['by_topology']['multi_hop_mesh']
            print(f"{k:32s} | Mean: {v['mean_flowtime']:9.1f}s ({gain:+6.2f}%) | LS: {ls:8.1f} | CE: {ce:8.1f} | Mesh: {mh:8.1f} | P95: {v['p95_decision_ms']:5.2f}ms")


if __name__ == '__main__':
    main()
