"""DualGFlow-RG v2 Training Pipeline v3: Unclipped Residual + Flow-CA-EFT KL Prior + Low-Noise Decoupled PPO.

Fixes from v2 diagnostics:
1. Residual swing increased to [-4, +4] around -2.0 * rel_flow_eft (covers 100% of Lookahead overrides up to rel_eft=1.37+).
2. Stage 1 Lookahead Distillation uses validation gating (never starts PPO worse than Flow-CA-EFT).
3. Stage 2 PPO anchors KL directly to the Flow-CA-EFT prior (-2.0 * rel_flow_eft) instead of a degraded Stage 1 model,
   using T=0.125 low-noise sampling and 12 episodes/update.
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
from dualgflow.train_ppo_dagger import collect_dagger_states
from dualgflow.train_ppo_full import evaluate_suite, update_critic, REWARD_SCALE


def flow_eft_prior_logits(obs: dict, device: torch.device | str) -> torch.Tensor:
    raw_eft = obs['flow_eft'].astype(np.float32)
    eft_min = float(np.min(raw_eft))
    eft_med = float(np.median(raw_eft))
    rel_eft = np.clip((raw_eft - eft_min) / max(eft_med - eft_min, 1e-3), 0.0, 5.0)
    return -2.0 * torch.as_tensor(rel_eft, dtype=torch.float32, device=device)


def train_stage1_gated(
    model: DualGFlowPolicy,
    dataset: list[dict],
    val_scenarios: list[dict],
    epochs: int = 18,
    lr: float = 8e-4,
    device: str = 'cuda',
) -> DualGFlowPolicy:
    """Advantage-weighted distillation with strict validation gating against Flow-CA-EFT."""
    model.to(device)
    opt = torch.optim.AdamW(model.actor_parameters(), lr=lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rng = np.random.default_rng(101)

    def eval_val() -> float:
        model.eval()
        with torch.no_grad():
            return float(np.mean([
                evaluate_policy(sc, lambda e, o: int(torch.argmax(model.forward_actor(o)).item()))['mean_flowtime']
                for sc in val_scenarios
            ]))

    best_val = eval_val()
    best_state = deepcopy(model.state_dict())
    print(f'  [Stage 1 Init (Flow-CA-EFT)] Val: {best_val:.2f}s', flush=True)

    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(dataset))
        opt.zero_grad()
        for idx_pos, i in enumerate(order):
            sample = dataset[int(i)]
            obs = sample['obs']
            costs = torch.as_tensor(sample['costs'], dtype=torch.float32, device=device)

            logits = model.forward_actor(obs)
            eft_a = int(np.argmin(obs['flow_eft']))
            best_a = int(sample['best_a'])
            adv_best = float((costs[eft_a] - costs[best_a]) / costs[eft_a].clamp_min(1.0))

            # Pairwise preference + advantage-weighted ranking over Flow-CA-EFT anchor
            # Only push best_a above eft_a proportionally to real lookahead gain;
            # otherwise gently regularize residual logits toward Flow-CA-EFT prior.
            prior = flow_eft_prior_logits(obs, device)
            residual = logits - prior
            if adv_best > 0.005:
                margin = logits[best_a] - logits[eft_a]
                target_margin = min(2.0, 12.0 * adv_best)
                pref_loss = F.smooth_l1_loss(margin, torch.tensor(target_margin, device=device))
                loss = pref_loss + 0.02 * residual.pow(2).mean()
            else:
                loss = 0.05 * residual.pow(2).mean()

            (loss / 16.0).backward()
            if (idx_pos + 1) % 16 == 0 or (idx_pos + 1) == len(order):
                nn.utils.clip_grad_norm_(model.actor_parameters(), 1.0)
                opt.step()
                opt.zero_grad()
        scheduler.step()

        if (ep + 1) % 3 == 0:
            v = eval_val()
            tag = ''
            if v < best_val:
                best_val = v
                best_state = deepcopy(model.state_dict())
                tag = ' *BEST*'
            print(f'  [Stage 1 Ep {ep+1:2d}/{epochs}] Val: {v:.2f}s{tag}', flush=True)

    model.load_state_dict(best_state)
    model.eval()
    return model


def collect_rollouts_v3(
    model: DualGFlowPolicy,
    scenarios: list[dict],
    num_episodes: int,
    rng: np.random.Generator,
    temperature: float = 0.125,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
) -> tuple[list[tuple], np.ndarray]:
    model.eval()
    device = next(model.parameters()).device
    chosen_idx = rng.choice(len(scenarios), size=min(num_episodes, len(scenarios)), replace=False)
    transitions = []
    ep_flowtimes = []

    with torch.no_grad():
        for idx in chosen_idx:
            sc = scenarios[int(idx)]
            env = FlowWorkflowEnv(sc, reward_scale=REWARD_SCALE)
            obs = env.reset()
            ep_obs, ep_act, ep_logp, ep_rew, ep_val, ep_prior_lp = [], [], [], [], [], []
            while not env.done:
                logits, val = model.forward_obs(obs)
                scaled_logits = logits / temperature
                dist = Categorical(logits=scaled_logits)
                act = dist.sample()
                logp = dist.log_prob(act)
                prior_lp = F.log_softmax(flow_eft_prior_logits(obs, device) / temperature, dim=0).cpu()

                next_obs, rew, _, _ = env.step(int(act.item()))
                ep_obs.append(obs)
                ep_act.append(int(act.item()))
                ep_logp.append(float(logp.item()))
                ep_rew.append(float(rew))
                ep_val.append(float(val.item()))
                ep_prior_lp.append(prior_lp)
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
                transitions.append((ep_obs[t], ep_act[t], ep_logp[t], ret[t], adv[t], ep_prior_lp[t]))

    return transitions, np.asarray(ep_flowtimes, dtype=np.float64)


def ppo_train_v3(
    model: DualGFlowPolicy,
    train_scenarios: list[dict],
    val_scenarios: list[dict],
    temperature: float = 0.125,
    warmup_rounds: int = 3,
    ppo_updates: int = 12,
    episodes_per_update: int = 12,
    lr_a: float = 1e-4,
    lr_c: float = 5e-4,
    clip_eps: float = 0.15,
    kl_coef: float = 0.008,
    device: str = 'cuda',
) -> DualGFlowPolicy:
    model.to(device)
    opt_a = torch.optim.AdamW(model.actor_parameters(), lr=lr_a, weight_decay=1e-4)
    opt_c = torch.optim.AdamW(model.critic_parameters(), lr=lr_c, weight_decay=1e-4)
    rng = np.random.default_rng(2026)

    print('  [Critic Warmup] Fitting Critic...', flush=True)
    for w_i in range(warmup_rounds):
        trans, ep_flows = collect_rollouts_v3(model, train_scenarios, episodes_per_update, rng, temperature=temperature)
        c_loss = update_critic(model, opt_c, trans, epochs=3, batch_size=64, rng=rng)
        print(f'    Warmup {w_i+1}/{warmup_rounds} | TrainEp: {ep_flows.mean():.1f}s | Huber: {c_loss:.4f}', flush=True)

    def eval_val() -> float:
        model.eval()
        with torch.no_grad():
            return float(np.mean([
                evaluate_policy(sc, lambda e, o: int(torch.argmax(model.forward_actor(o)).item()))['mean_flowtime']
                for sc in val_scenarios
            ]))

    best_val = eval_val()
    best_state = deepcopy(model.state_dict())
    print(f'  [PPO Init] Val Mean Flowtime: {best_val:.2f}s', flush=True)

    for upd in range(ppo_updates):
        trans, ep_flows = collect_rollouts_v3(model, train_scenarios, episodes_per_update, rng, temperature=temperature)
        all_adv = np.array([x[4] for x in trans], dtype=np.float32)
        adv_mean, adv_std = float(all_adv.mean()), max(float(all_adv.std()), 1e-6)

        model.train()
        c_loss = update_critic(model, opt_c, trans, epochs=2, batch_size=64, rng=rng)

        for _ in range(2):
            perm = rng.permutation(len(trans))
            opt_a.zero_grad()
            for b_i, idx in enumerate(perm):
                obs, act, old_logp, _, raw_adv, prior_lp = trans[int(idx)]
                norm_adv = (raw_adv - adv_mean) / adv_std

                scaled_logits = model.forward_actor(obs) / temperature
                dist = Categorical(logits=scaled_logits)
                new_logp = dist.log_prob(torch.tensor(act, device=device))
                ratio = torch.exp(new_logp - old_logp)
                surr1 = ratio * norm_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * norm_adv
                policy_loss = -torch.min(surr1, surr2)

                cur_lp = F.log_softmax(scaled_logits, dim=0)
                cur_p = cur_lp.exp()
                kl = (cur_p * (cur_lp - prior_lp.to(device))).sum()

                loss = policy_loss + kl_coef * kl - 0.001 * dist.entropy()
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
            f'  [PPO {upd+1:2d}/{ppo_updates}] TrainEp: {ep_flows.mean():.1f}s | Critic: {c_loss:.4f} | Val: {val_flow:.2f}s{tag}',
            flush=True,
        )

    model.load_state_dict(best_state)
    model.eval()
    return model


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Running DualGFlow-RG v3 Pipeline on device={device}', flush=True)

    topos = ['leaf_spine', 'cloud_edge', 'multi_hop_mesh']
    train_scenarios = [
        make_flow_scenario(seed=100 + i, split='train', workflows=12, num_servers=16,
                           load=0.75, ccr=2.5, topology=topos[i % 3], oversubscription=6.0)
        for i in range(24)
    ]
    val_scenarios = [
        make_flow_scenario(seed=200 + i, split='validation', workflows=14, num_servers=16,
                           load=0.75, ccr=2.5, topology=topos[i % 3], oversubscription=6.0)
        for i in range(9)
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

    print('Collecting DAgger Round 0 dataset (24 train scenarios)...', flush=True)
    t0 = time.perf_counter()
    base_dataset = collect_dagger_states(train_scenarios, policy=None, states_per_scenario=45, top_k=6, seed=101)
    print(f'Round 0: {len(base_dataset)} states in {time.perf_counter() - t0:.1f}s', flush=True)

    torch.manual_seed(101)
    pilot_rg = DualGFlowPolicy(variant='dualgflow', mode='residual_eft', hidden_dim=64, gnn_layers=4)
    train_stage1_gated(pilot_rg, base_dataset, val_scenarios, epochs=15, lr=8e-4, device=device)
    dagger_extra = collect_dagger_states(train_scenarios[:15], policy=pilot_rg, states_per_scenario=35, top_k=6, seed=202)
    full_dataset = base_dataset + dagger_extra
    print(f'Total DAgger dataset: {len(full_dataset)} states', flush=True)

    models = {}
    for variant in ('dualgflow', 'global_mlp', 'no_mp_keep_links'):
        key = f'{variant}_residual_eft'
        torch.manual_seed(101)
        m = DualGFlowPolicy(variant=variant, mode='residual_eft', hidden_dim=64, gnn_layers=4)
        print(f'\n=== Training {key} ===', flush=True)
        train_stage1_gated(m, full_dataset, val_scenarios, epochs=12, lr=3e-4, device=device)
        ppo_train_v3(
            m, train_scenarios, val_scenarios,
            temperature=0.125, warmup_rounds=3, ppo_updates=14, episodes_per_update=12,
            lr_a=1e-4, lr_c=5e-4, device=device,
        )
        models[key] = m

    test_16 = evaluate_suite(test_scenarios_16, models)
    test_32 = evaluate_suite(test_scenarios_32, models)

    out_path = Path(__file__).resolve().parent / 'ppo_v3_results.json'
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
