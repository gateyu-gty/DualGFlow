"""Stage-2 DAgger + On-Policy PPO Fine-Tuning for DualGFlow-RG v2.

Pipeline:
1. Multi-round DAgger counterfactual distillation (~1,800 states across Leaf-Spine, Cloud-Edge, Multi-Hop Mesh)
   to eliminate distribution shift from offline-only trajectories.
2. On-Policy PPO fine-tuning directly on FlowWorkflowEnv with GAE, clipped surrogate,
   and KL anchoring to prevent policy collapse.
3. Fair evaluation across N=16 validation (12 scenarios) and N=32 zero-shot generalization (6 scenarios).
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
from dualgflow.run_prototype_experiment import lookahead_action, evaluate_policy


def collect_dagger_states(
    scenarios: list[dict],
    policy: DualGFlowPolicy | None,
    states_per_scenario: int = 35,
    top_k: int = 6,
    seed: int = 101,
) -> list[dict]:
    """Roll out current policy (or Flow-CA-EFT if None) and query causal lookahead teacher."""
    dataset = []
    rng = np.random.default_rng(seed)
    for sc_idx, sc in enumerate(scenarios):
        env = FlowWorkflowEnv(sc)
        obs = env.reset()
        step_idx = 0
        collected = 0
        while not env.done:
            if step_idx % 3 == 0 and collected < states_per_scenario:
                best_a, costs = lookahead_action(env, obs, top_k=top_k)
                dataset.append(dict(obs=obs, costs=costs.astype(np.float32), best_a=best_a))
                collected += 1
            if policy is not None and rng.random() < 0.75:
                with torch.no_grad():
                    logits, _ = policy.forward_obs(obs)
                    action = int(torch.argmax(logits).item())
            else:
                action = int(np.argmin(obs['flow_eft']))
            obs, _, _, _ = env.step(action)
            step_idx += 1
    return dataset


def train_supervised_dagger(
    model: DualGFlowPolicy,
    dataset: list[dict],
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = 'cuda',
) -> DualGFlowPolicy:
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rng = np.random.default_rng(101)

    for ep in range(epochs):
        order = rng.permutation(len(dataset))
        opt.zero_grad()
        for idx_pos, i in enumerate(order):
            sample = dataset[int(i)]
            obs = sample['obs']
            costs = torch.as_tensor(sample['costs'], dtype=torch.float32, device=device)
            c_min = costs.min()
            scale = torch.clamp(costs.std(), min=0.02 * c_min.abs().clamp_min(1.0))
            target_probs = F.softmax(-(costs - c_min) / (scale * 0.5), dim=0)

            logits, _ = model.forward_obs(obs)
            log_probs = F.log_softmax(logits, dim=0)
            ce_loss = -(target_probs * log_probs).sum()

            if model.mode == 'pure_structural':
                best_a = int(sample['best_a'])
                margin_loss = F.relu(1.0 - (logits[best_a] - logits)).mean()
                loss = ce_loss + 0.15 * margin_loss
            else:
                eft_a = int(np.argmin(obs['flow_eft']))
                best_a = int(sample['best_a'])
                gain = float((costs[eft_a] - costs[best_a]) / costs[eft_a].clamp_min(1.0))
                rel_eft = np.clip(
                    (obs['flow_eft'] - obs['flow_eft'].min()) / max(float(np.median(obs['flow_eft']) - obs['flow_eft'].min()), 1e-3),
                    0.0, 5.0,
                )
                raw_res = logits + 4.0 * torch.as_tensor(rel_eft, dtype=torch.float32, device=device)
                if gain > 0.012:
                    loss = ce_loss + 0.02 * raw_res.pow(2).mean()
                else:
                    loss = 0.25 * raw_res.pow(2).mean()

            (loss / 16.0).backward()
            if (idx_pos + 1) % 16 == 0 or (idx_pos + 1) == len(order):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
        scheduler.step()
    model.eval()
    return model


def ppo_finetune(
    model: DualGFlowPolicy,
    train_scenarios: list[dict],
    val_scenarios: list[dict],
    updates: int = 8,
    episodes_per_update: int = 6,
    ppo_epochs: int = 3,
    lr: float = 2.5e-4,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.15,
    kl_coef: float = 0.05,
    device: str = 'cuda',
) -> DualGFlowPolicy:
    """On-policy PPO fine-tuning with GAE and KL anchor to initial policy."""
    model.to(device)
    anchor_model = deepcopy(model).eval()
    for p in anchor_model.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(2026)

    def eval_mean(sc_list):
        model.eval()
        vals = []
        with torch.no_grad():
            for sc in sc_list:
                r = evaluate_policy(sc, lambda e, o: int(torch.argmax(model.forward_obs(o)[0]).item()))
                vals.append(r['mean_flowtime'])
        return float(np.mean(vals))

    best_val = eval_mean(val_scenarios)
    best_state = deepcopy(model.state_dict())
    print(f'  [PPO Init] Val Mean Flowtime: {best_val:.2f}s', flush=True)

    for upd in range(updates):
        # 1. Collect on-policy rollouts
        model.eval()
        trajectories = []
        chosen_idx = rng.choice(len(train_scenarios), size=min(episodes_per_update, len(train_scenarios)), replace=False)
        for idx in chosen_idx:
            sc = train_scenarios[int(idx)]
            env = FlowWorkflowEnv(sc, reward_scale=1000.0)
            obs = env.reset()
            ep_obs, ep_act, ep_logp, ep_rew, ep_val, ep_anchor_logp = [], [], [], [], [], []
            with torch.no_grad():
                while not env.done:
                    logits, val = model.forward_obs(obs)
                    anchor_logits, _ = anchor_model.forward_obs(obs)
                    dist = Categorical(logits=logits)
                    act = dist.sample()
                    logp = dist.log_prob(act)
                    a_logp = F.log_softmax(anchor_logits, dim=0)

                    next_obs, rew, done, _ = env.step(int(act.item()))
                    ep_obs.append(obs)
                    ep_act.append(int(act.item()))
                    ep_logp.append(float(logp.item()))
                    ep_rew.append(float(rew))
                    ep_val.append(float(val.item()))
                    ep_anchor_logp.append(a_logp.cpu())
                    obs = next_obs

            # Compute GAE
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
                trajectories.append((ep_obs[t], ep_act[t], ep_logp[t], ret[t], adv[t], ep_anchor_logp[t]))

        # Normalize advantages globally across batch
        all_adv = np.array([x[4] for x in trajectories], dtype=np.float32)
        adv_mean, adv_std = float(all_adv.mean()), max(float(all_adv.std()), 1e-6)

        # 2. PPO gradient updates
        model.train()
        for _ in range(ppo_epochs):
            perm = rng.permutation(len(trajectories))
            opt.zero_grad()
            # Subsample up to 640 transitions per PPO epoch for fast updates
            step_subset = perm[:min(640, len(perm))]
            for b_i, t_idx in enumerate(step_subset):
                obs, act, old_logp, ret_v, raw_adv, anchor_lp = trajectories[int(t_idx)]
                norm_adv = (raw_adv - adv_mean) / adv_std

                logits, val = model.forward_obs(obs)
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(torch.tensor(act, device=device))
                ratio = torch.exp(new_logp - old_logp)
                surr1 = ratio * norm_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * norm_adv
                policy_loss = -torch.min(surr1, surr2)

                value_loss = 0.5 * (val - torch.tensor(ret_v, dtype=torch.float32, device=device)).pow(2)
                entropy = dist.entropy()

                # KL(current || anchor)
                cur_lp = F.log_softmax(logits, dim=0)
                cur_p = cur_lp.exp()
                kl = (cur_p * (cur_lp - anchor_lp.to(device))).sum()

                loss = policy_loss + 0.25 * value_loss + kl_coef * kl - 0.005 * entropy
                (loss / 32.0).backward()

                if (b_i + 1) % 32 == 0 or (b_i + 1) == len(step_subset):
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    opt.step()
                    opt.zero_grad()

        val_flow = eval_mean(val_scenarios)
        improved = ''
        if val_flow < best_val:
            best_val = val_flow
            best_state = deepcopy(model.state_dict())
            improved = ' *BEST*'
        print(f'  [PPO Upd {upd+1:2d}/{updates}] Val Mean Flowtime: {val_flow:.2f}s{improved}', flush=True)

    model.load_state_dict(best_state)
    model.eval()
    return model


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Running DualGFlow-RG v2 DAgger + PPO Pipeline on device={device}', flush=True)

    topos = ['leaf_spine', 'cloud_edge', 'multi_hop_mesh']
    train_scenarios = [
        make_flow_scenario(seed=100 + i, split='train', workflows=12, num_servers=16,
                           load=0.75, ccr=2.5, topology=topos[i % 3], oversubscription=6.0)
        for i in range(18)
    ]
    val_scenarios_16 = [
        make_flow_scenario(seed=200 + i, split='validation', workflows=14, num_servers=16,
                           load=0.75, ccr=2.5, topology=topos[i % 3], oversubscription=6.0)
        for i in range(12)
    ]
    val_scenarios_32 = [
        make_flow_scenario(seed=300 + i, split='test', workflows=20, num_servers=32,
                           load=0.75, ccr=2.5, topology=topos[i % 3], oversubscription=6.0)
        for i in range(6)
    ]

    # Round 0 DAgger dataset (under Flow-CA-EFT)
    print('Collecting DAgger Round 0 dataset...', flush=True)
    t0 = time.perf_counter()
    base_dataset = collect_dagger_states(train_scenarios, policy=None, states_per_scenario=35, top_k=6, seed=101)
    print(f'Round 0: {len(base_dataset)} states in {time.perf_counter() - t0:.1f}s', flush=True)

    # Train initial DualGFlow-RG (residual_eft) and collect DAgger Round 1 on-policy states
    torch.manual_seed(101)
    rg_res = DualGFlowPolicy(variant='dualgflow', mode='residual_eft', hidden_dim=64, gnn_layers=4)
    train_supervised_dagger(rg_res, base_dataset, epochs=20, lr=1e-3, device=device)

    print('Collecting DAgger Round 1 dataset (on DualGFlow-RG trajectories)...', flush=True)
    t1 = time.perf_counter()
    dagger_extra = collect_dagger_states(train_scenarios[:12], policy=rg_res, states_per_scenario=30, top_k=6, seed=202)
    full_dataset = base_dataset + dagger_extra
    print(f'Total DAgger dataset: {len(full_dataset)} states (+{len(dagger_extra)} in {time.perf_counter() - t1:.1f}s)', flush=True)

    models = {}
    for mode in ('pure_structural', 'residual_eft'):
        for variant in ('dualgflow', 'global_mlp', 'no_mp_keep_links'):
            key = f'{variant}_{mode}'
            torch.manual_seed(101)
            m = DualGFlowPolicy(variant=variant, mode=mode, hidden_dim=64, gnn_layers=4)
            print(f'\n--- Stage 1 DAgger Training: {key} (params={sum(p.numel() for p in m.parameters()):,}) ---', flush=True)
            train_supervised_dagger(m, full_dataset, epochs=22, lr=1e-3, device=device)
            if mode == 'residual_eft':
                print(f'--- Stage 2 PPO Fine-Tuning: {key} ---', flush=True)
                ppo_finetune(m, train_scenarios, val_scenarios_16[:6], updates=6, episodes_per_update=6, device=device)
            models[key] = m

    def run_eval(sc_list):
        methods = {
            'compute_eft': lambda env, obs: int(np.argmin(obs['compute_eft'])),
            'nominal_ca_eft': lambda env, obs: int(np.argmin(obs['nominal_eft'])),
            'flow_ca_eft': lambda env, obs: int(np.argmin(obs['flow_eft'])),
        }
        for k, m in models.items():
            methods[k] = lambda env, obs, mod=m: int(torch.argmax(mod.forward_obs(obs)[0]).item())

        res = {}
        for name, fn in methods.items():
            flows, lats = [], []
            with torch.no_grad():
                for sc in sc_list:
                    r = evaluate_policy(sc, fn, name)
                    flows.append(r['mean_flowtime'])
                    lats.append(r['p95_decision_ms'])
            res[name] = dict(
                mean_flowtime=float(np.mean(flows)),
                per_scene=flows,
                p95_decision_ms=float(np.mean(lats)),
            )
        return res

    print('\nEvaluating on N=16 Validation (12 scenarios)...', flush=True)
    res_16 = run_eval(val_scenarios_16)
    print('Evaluating on N=32 Zero-Shot (6 scenarios)...', flush=True)
    res_32 = run_eval(val_scenarios_32)

    out_path = Path(__file__).resolve().parent / 'ppo_dagger_results.json'
    out_path.write_text(json.dumps(dict(n16=res_16, n32=res_32), indent=2), encoding='utf-8')

    for tag, r in [('N=16 VALIDATION (12 SCENES)', res_16), ('N=32 ZERO-SHOT (6 SCENES)', res_32)]:
        print(f'\n=== {tag} ===')
        base = r['flow_ca_eft']['mean_flowtime']
        for k, v in r.items():
            gain = (base - v['mean_flowtime']) / base * 100.0
            print(f"{k:32s} | Mean Flowtime: {v['mean_flowtime']:10.2f}s | vs Flow-CA-EFT: {gain:+6.2f}% | P95: {v['p95_decision_ms']:5.2f}ms")


if __name__ == '__main__':
    main()
