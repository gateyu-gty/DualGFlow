"""End-to-end validation of the DualGFlow-RG v2 Flow-Level Network Model & Resource GNN.

Runs:
1. Network Model & Heuristic Headroom Diagnostic across Leaf-Spine, Cloud-Edge, and Multi-Hop Mesh.
2. Counterfactual Rollout Preference Training comparing:
   - DualGFlow-RG (4-layer Edge-Attentive Resource GNN with Switch Nodes & Parent-Data Injection)
   - Global-MLP (Complete X/A/E dense MLP encoder)
   - No-MP-Keep-Links (1-hop link summary without multi-hop message passing)
   under both:
   - Mode A: Pure Structural Reasoning (No pre-computed EFT / route delay features)
   - Mode B: Residual over Flow-CA-EFT (Targeting >3% improvement over CA-EFT & non-GNN baselines)
3. Zero-Shot Scale Generalization from N=16 trained models to N=32 servers.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dualgflow.topologies import make_flow_scenario
from dualgflow.simulator_flow import FlowWorkflowEnv
from dualgflow.models_rg import DualGFlowPolicy


def lookahead_action(env: FlowWorkflowEnv, obs: dict, horizon: int = 200, top_k: int = 16) -> tuple[int, np.ndarray]:
    """Evaluate candidate servers using a 100% causal rollout of currently arrived workflows."""
    n_srv = obs['num_servers']
    flow_eft = obs['flow_eft']
    comp_eft = obs['compute_eft']

    if top_k >= n_srv:
        candidates = list(range(n_srv))
    else:
        cand_set = set(np.argsort(flow_eft)[:top_k].tolist())
        cand_set.update(np.argsort(obs['nominal_eft'])[:3].tolist())
        cand_set.update(np.argsort(comp_eft)[:2].tolist())
        parent_hosts = np.where(obs['resource_x'][:n_srv, 5] > 0)[0].tolist()
        cand_set.update(parent_hosts[:4])
        candidates = sorted(cand_set)

    eval_scores = {}
    for a in candidates:
        sim = env.clone_causal_active()
        sim.step(a, return_observation=False)
        while not sim.done:
            efts = sim.candidate_eft_estimates()
            cont_a = int(np.argmin(efts['flow_eft']))
            sim.step(cont_a, return_observation=False)
        eval_scores[a] = -sim.total_reward_unscaled

    min_eval = min(eval_scores.values())
    max_eval = max(eval_scores.values())
    spread = max(max_eval - min_eval, 1.0)
    costs = np.zeros(n_srv, dtype=np.float64)
    for a in range(n_srv):
        if a in eval_scores:
            costs[a] = eval_scores[a]
        else:
            costs[a] = max_eval + spread * 0.5 + (flow_eft[a] - np.min(flow_eft))

    best_a = int(min(eval_scores, key=eval_scores.get))
    return best_a, costs


def evaluate_policy(scenario: dict, policy_fn, name: str = '') -> dict:
    """Run a single scenario with policy_fn(env, obs) -> action and measure flowtime + decision latency."""
    env = FlowWorkflowEnv(scenario)
    obs = env.reset()
    latencies_ms = []
    while not env.done:
        t0 = time.perf_counter()
        action = int(policy_fn(env, obs))
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
        obs, _, _, _ = env.step(action)
    m = env.metrics()
    m['p95_decision_ms'] = float(np.quantile(latencies_ms, 0.95)) if latencies_ms else 0.0
    m['mean_decision_ms'] = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    m['decisions'] = len(latencies_ms)
    return m


def collect_counterfactual_dataset(scenarios: list[dict], states_per_scenario: int = 45, horizon: int = 10) -> list[dict]:
    """Collect (observation, counterfactual_costs) pairs along mixed rollout trajectories."""
    dataset = []
    rng = np.random.default_rng(20261117)
    for sc_idx, sc in enumerate(scenarios):
        env = FlowWorkflowEnv(sc)
        obs = env.reset()
        step_idx = 0
        while not env.done:
            if step_idx % 2 == 0 and len(dataset) < (sc_idx + 1) * states_per_scenario:
                best_a, costs = lookahead_action(env, obs, horizon=horizon, top_k=8)
                dataset.append(dict(obs=obs, costs=costs.astype(np.float32), best_a=best_a))
                action = best_a if rng.random() < 0.7 else int(np.argmin(obs['flow_eft']))
            else:
                action = int(np.argmin(obs['flow_eft']))
            obs, _, _, _ = env.step(action)
            step_idx += 1
    return dataset


def train_model_on_dataset(
    model: DualGFlowPolicy,
    dataset: list[dict],
    epochs: int = 30,
    lr: float = 1e-3,
    device: str = 'cpu',
) -> DualGFlowPolicy:
    """Train policy via soft counterfactual preference cross-entropy (with safe residual regularization)."""
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
            # Scale cost differences relative to baseline cost magnitude to avoid amplifying noise
            c_min = costs.min()
            scale = torch.clamp(costs.std(), min=0.02 * c_min.abs().clamp_min(1.0))
            norm_c = (costs - c_min) / (scale * 0.5)
            target_probs = F.softmax(-norm_c, dim=0)

            logits, _ = model.forward_obs(obs)
            log_probs = F.log_softmax(logits, dim=0)
            ce_loss = -(target_probs * log_probs).sum()

            if model.mode == 'pure_structural':
                best_a = int(sample['best_a'])
                margin_loss = F.relu(1.0 - (logits[best_a] - logits)).mean()
                loss = ce_loss + 0.15 * margin_loss
            else:
                # Only learn residual corrections where lookahead genuinely beats Flow-CA-EFT;
                # otherwise keep residual close to zero so routine Flow-CA-EFT decisions are preserved!
                eft_a = int(np.argmin(obs['flow_eft']))
                best_a = int(sample['best_a'])
                gain = float((costs[eft_a] - costs[best_a]) / costs[eft_a].clamp_min(1.0))
                raw_res = logits + 4.0 * torch.as_tensor(
                    np.clip((obs['flow_eft'] - obs['flow_eft'].min()) / max(float(np.median(obs['flow_eft']) - obs['flow_eft'].min()), 1e-3), 0.0, 5.0),
                    dtype=torch.float32, device=device
                )
                if gain > 0.015:
                    loss = ce_loss + 0.02 * raw_res.pow(2).mean()
                else:
                    # Routine state: preserve Flow-CA-EFT by regularizing residual toward 0
                    loss = 0.2 * raw_res.pow(2).mean()

            (loss / 16.0).backward()
            if (idx_pos + 1) % 16 == 0 or (idx_pos + 1) == len(order):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
        scheduler.step()

    model.eval()
    return model


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Running DualGFlow-RG v2 Prototype Experiment on device={device}', flush=True)

    # 1. Build training and validation scenarios across all 3 topologies
    topos = ['leaf_spine', 'cloud_edge', 'multi_hop_mesh']
    train_scenarios = []
    for i in range(15):
        topo = topos[i % 3]
        sc = make_flow_scenario(
            seed=100 + i,
            split='train',
            workflows=12,
            num_servers=16,
            load=0.75,
            ccr=2.5,
            topology=topo,
            oversubscription=6.0,
        )
        train_scenarios.append(sc)

    val_scenarios_16 = []
    for i in range(9):
        topo = topos[i % 3]
        sc = make_flow_scenario(
            seed=200 + i,
            split='validation',
            workflows=14,
            num_servers=16,
            load=0.75,
            ccr=2.5,
            topology=topo,
            oversubscription=6.0,
        )
        val_scenarios_16.append(sc)

    val_scenarios_32 = []
    for i in range(6):
        topo = topos[i % 3]
        sc = make_flow_scenario(
            seed=300 + i,
            split='test',
            workflows=20,
            num_servers=32,
            load=0.75,
            ccr=2.5,
            topology=topo,
            oversubscription=6.0,
        )
        val_scenarios_32.append(sc)

    print('Collecting counterfactual rollout dataset...', flush=True)
    t0 = time.perf_counter()
    dataset = collect_counterfactual_dataset(train_scenarios, states_per_scenario=45, horizon=8)
    print(f'Collected {len(dataset)} states in {time.perf_counter() - t0:.1f}s', flush=True)

    models = {}
    for mode in ('pure_structural', 'residual_eft'):
        for variant in ('dualgflow', 'global_mlp', 'no_mp_keep_links'):
            key = f'{variant}_{mode}'
            torch.manual_seed(101)
            m = DualGFlowPolicy(variant=variant, mode=mode, hidden_dim=64, gnn_layers=4)
            print(f'Training {key} (params={sum(p.numel() for p in m.parameters()):,})...', flush=True)
            train_model_on_dataset(m, dataset, epochs=30, lr=1e-3, device=device)
            models[key] = m

    # Evaluate on N=16 Validation Scenarios
    def run_eval_suite(scenarios: list[dict], include_lookahead: bool = True) -> dict:
        methods = {
            'compute_eft': lambda env, obs: int(np.argmin(obs['compute_eft'])),
            'nominal_ca_eft': lambda env, obs: int(np.argmin(obs['nominal_eft'])),
            'flow_ca_eft': lambda env, obs: int(np.argmin(obs['flow_eft'])),
        }
        if include_lookahead:
            methods['lookahead_h8'] = lambda env, obs: lookahead_action(env, obs, horizon=8, top_k=6)[0]
        for k, m in models.items():
            methods[k] = lambda env, obs, mod=m: int(torch.argmax(mod.forward_obs(obs)[0]).item())

        summary = {}
        per_scene = {m_name: [] for m_name in methods}
        latencies = {m_name: [] for m_name in methods}
        throttles = []

        for sc in scenarios:
            for m_name, fn in methods.items():
                res = evaluate_policy(sc, fn, m_name)
                per_scene[m_name].append(res['mean_flowtime'])
                latencies[m_name].append(res['p95_decision_ms'])
                if m_name == 'flow_ca_eft':
                    throttles.append(res['throttled_flow_events'])

        for m_name in methods:
            arr = np.array(per_scene[m_name], dtype=np.float64)
            summary[m_name] = dict(
                mean_flowtime=float(np.mean(arr)),
                per_scene=arr.tolist(),
                p95_decision_ms=float(np.mean(latencies[m_name])),
            )
        summary['_meta'] = dict(mean_throttled_events=float(np.mean(throttles)))
        return summary

    print('\nEvaluating on N=16 Validation Suite (9 scenarios across 3 topologies)...', flush=True)
    res_16 = run_eval_suite(val_scenarios_16, include_lookahead=False)

    print('\nEvaluating Zero-Shot Generalization on N=32 Suite (6 scenarios across 3 topologies)...', flush=True)
    res_32 = run_eval_suite(val_scenarios_32, include_lookahead=False)

    out = dict(n16_validation=res_16, n32_zero_shot=res_32)
    out_path = Path(__file__).resolve().parent / 'prototype_results.json'
    out_path.write_text(json.dumps(out, indent=2), encoding='utf-8')

    print('\n=== N=16 VALIDATION SUMMARY ===')
    base_flow_eft = res_16['flow_ca_eft']['mean_flowtime']
    for k, v in res_16.items():
        if k.startswith('_'):
            continue
        gain_vs_eft = (base_flow_eft - v['mean_flowtime']) / base_flow_eft * 100.0
        print(f"{k:32s} | Mean Flowtime: {v['mean_flowtime']:10.2f}s | vs Flow-CA-EFT: {gain_vs_eft:+6.2f}% | P95 Latency: {v['p95_decision_ms']:6.2f}ms")

    print('\n=== N=32 ZERO-SHOT GENERALIZATION SUMMARY ===')
    base_32_eft = res_32['flow_ca_eft']['mean_flowtime']
    for k, v in res_32.items():
        if k.startswith('_'):
            continue
        gain_vs_eft = (base_32_eft - v['mean_flowtime']) / base_32_eft * 100.0
        print(f"{k:32s} | Mean Flowtime: {v['mean_flowtime']:10.2f}s | vs Flow-CA-EFT: {gain_vs_eft:+6.2f}% | P95 Latency: {v['p95_decision_ms']:6.2f}ms")


if __name__ == '__main__':
    main()
