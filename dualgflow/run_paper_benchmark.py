"""Publication-Grade 3-Seed Benchmark & Figure Generator for DualGFlow-RG v2.

Evaluates:
1. Pure Structural Mode (zero handcrafted EFT/route features) across 3 seeds (101, 202, 303):
   - DualGFlow-RG vs No-MP-Keep-Links vs Global-MLP
2. Residual EFT Mode (Batched Q-Ranking + Decoupled Batched PPO) across 3 seeds:
   - DualGFlow-RG vs No-MP-Keep-Links vs Global-MLP vs Flow-CA-EFT / Nominal-CA-EFT / Compute-EFT
3. Generalization & Sensitivity Axes (all on unseen split='test' scenarios):
   - In-scale N=16 (18 scenarios: 6 Leaf-Spine, 6 Cloud-Edge, 6 Multi-Hop Mesh)
   - Zero-shot Scale Generalization: N=32 (9 scenes) and N=64 (6 scenes)
   - Zero-shot CCR Sweep: CCR in {0.5, 2.5, 5.0} (9 scenes each)
   - Budget-Matched Top-2 Lookahead Pareto comparison (Flowtime vs Decision Latency)
4. Generates 95% paired bootstrap CIs and 3 publication-ready figures.
"""
from __future__ import annotations

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

from copy import deepcopy
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dualgflow.topologies import make_flow_scenario
from dualgflow.simulator_flow import FlowWorkflowEnv
from dualgflow.models_rg import DualGFlowPolicy
from dualgflow.batch_rg import collate_obs_batch, forward_actor_batch
from dualgflow.run_prototype_experiment import evaluate_policy
from dualgflow.train_fast_batched import (
    collect_masked_counterfactuals,
    train_batched_q_ranking,
    train_batched_ppo,
    guided_topk_action,
)


def prepare_full_costs(dataset: list[dict]) -> None:
    """Fill un-evaluated candidate servers smoothly so Pure Structural receives full negative signal."""
    for s in dataset:
        m = s['mask']
        c = s['costs']
        ev_c = c[m]
        c_min, c_max = float(ev_c.min()), float(ev_c.max())
        spread = max(c_max - c_min, 1.0)
        eft = s['obs']['flow_eft']
        full_c = np.where(m, c, c_max + 0.5 * spread + (eft - eft.min()))
        s['full_costs'] = full_c.astype(np.float32)


def train_pure_structural_fullcost(
    variant: str,
    dataset: list[dict],
    seed: int = 101,
    epochs: int = 35,
    batch_size: int = 64,
    lr: float = 1.5e-3,
    device: str = 'cuda',
) -> DualGFlowPolicy:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mod = DualGFlowPolicy(variant=variant, mode='pure_structural', hidden_dim=64, gnn_layers=4).to(device)
    opt = torch.optim.AdamW(mod.actor_parameters(), lr=lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for _ in range(epochs):
        mod.train()
        order = rng.permutation(len(dataset))
        for st in range(0, len(order), batch_size):
            sub = [dataset[int(i)] for i in order[st:st + batch_size]]
            b = collate_obs_batch([s['obs'] for s in sub], device=device)
            logits = forward_actor_batch(mod, b)
            fc = torch.as_tensor(np.stack([s['full_costs'] for s in sub]), device=device)
            c_min = fc.min(dim=1, keepdim=True).values
            temp = torch.clamp(fc.std(dim=1, keepdim=True), min=0.02 * c_min) * 0.5
            tp = F.softmax(-(fc - c_min) / temp, dim=1)
            kl = -(tp * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            best_idx = torch.as_tensor([s['best_a'] for s in sub], dtype=torch.long, device=device)
            ml = F.relu(1.0 - (logits.gather(1, best_idx.unsqueeze(1)) - logits)).mean()
            loss = kl + 0.15 * ml
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(mod.actor_parameters(), 1.0)
            opt.step()
        scheduler.step()

    mod.eval()
    return mod


def eval_policy_suite(scenarios: list[dict], fn_dict: dict) -> dict:
    out = {}
    for name, fn in fn_dict.items():
        by_topo = {'leaf_spine': [], 'cloud_edge': [], 'multi_hop_mesh': []}
        flows, lats = [], []
        with torch.no_grad():
            for sc in scenarios:
                topo = sc['meta']['topology']['topology']
                r = evaluate_policy(sc, fn, name)
                flows.append(r['mean_flowtime'])
                lats.append(r['p95_decision_ms'])
                by_topo[topo].append(r['mean_flowtime'])
        out[name] = dict(
            mean_flowtime=float(np.mean(flows)),
            std_flowtime=float(np.std(flows)),
            per_scene=flows,
            by_topology={t: float(np.mean(v)) for t, v in by_topo.items() if v},
            p95_decision_ms=float(np.mean(lats)),
        )
    return out


def paired_gain_ci(baseline_arr: list[float], target_arr: list[float], n_boot: int = 5000, seed: int = 42) -> dict:
    b = np.asarray(baseline_arr, dtype=np.float64)
    t = np.asarray(target_arr, dtype=np.float64)
    rel = (b - t) / b * 100.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(rel), size=(n_boot, len(rel)))
    boot_means = rel[idx].mean(axis=1)
    lo, hi = np.quantile(boot_means, [0.025, 0.975])
    wins = int(np.sum(t < b))
    return dict(
        gain_pct=float(rel.mean()),
        ci95_low=float(lo),
        ci95_high=float(hi),
        wins=wins,
        total=len(rel),
    )


def generate_figures(results: dict, fig_dir: Path) -> list[str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.25})
    saved = []

    # Fig 1: Pure Structural Topology Breakdown (N=16) & Zero-Shot Scale Generalization (N=16 -> 32 -> 64)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3), dpi=180)
    topos = ['leaf_spine', 'cloud_edge', 'multi_hop_mesh']
    topo_labels = ['Leaf-Spine', 'Cloud-Edge', 'Multi-Hop Mesh']
    x = np.arange(len(topos))
    w = 0.24
    s16 = results['n16_in_scale']['summary']
    for offset, (key, label, color) in enumerate([
        ('dualgflow_pure', 'DualGFlow-RG (Pure)', '#1f77b4'),
        ('no_mp_pure', 'No-MP-Keep-Links (Pure)', '#ff7f0e'),
        ('global_mlp_pure', 'Global-MLP (Pure)', '#d62728'),
    ]):
        vals = [s16[key]['by_topology'][t] / 1000.0 for t in topos]
        ax1.bar(x + (offset - 1) * w, vals, width=w, label=label, color=color, alpha=0.9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(topo_labels)
    ax1.set_ylabel('Mean Workflow Flowtime (×10³ s)')
    ax1.set_title('(a) Pure Structural Reasoning by Topology (N=16)')
    ax1.legend(frameon=True, fontsize=8.5)

    scales = ['N=16', 'N=32', 'N=64']
    scale_keys = ['n16_in_scale', 'n32_zero_shot', 'n64_zero_shot']
    for key, label, marker, color in [
        ('flow_ca_eft', 'Flow-CA-EFT (Analytical)', 's', '#2ca02c'),
        ('dualgflow_pure', 'DualGFlow-RG (Pure)', 'o', '#1f77b4'),
        ('no_mp_pure', 'No-MP-Keep-Links (Pure)', '^', '#ff7f0e'),
        ('global_mlp_pure', 'Global-MLP (Pure)', 'x', '#d62728'),
    ]:
        ys = [results[sk]['summary'][key]['mean_flowtime'] / 1000.0 for sk in scale_keys]
        ax2.plot(scales, ys, marker=marker, linewidth=2.2, markersize=7, label=label, color=color)
    ax2.set_ylabel('Mean Workflow Flowtime (×10³ s)')
    ax2.set_title('(b) Zero-Shot Scale Generalization (Trained on N=16)')
    ax2.legend(frameon=True, fontsize=8.5)
    plt.tight_layout()
    p1 = fig_dir / 'fig1_structural_and_scale.png'
    fig.savefig(p1)
    plt.close(fig)
    saved.append(str(p1))

    # Fig 2: CCR Sensitivity Sweep
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=180)
    ccrs = ['0.5', '2.5', '5.0']
    for key, label, marker, color in [
        ('flow_ca_eft', 'Flow-CA-EFT', 's', '#2ca02c'),
        ('nominal_ca_eft', 'Nominal-CA-EFT', 'd', '#9467bd'),
        ('dualgflow_pure', 'DualGFlow-RG (Pure)', 'o', '#1f77b4'),
        ('no_mp_pure', 'No-MP-Keep-Links (Pure)', '^', '#ff7f0e'),
        ('global_mlp_pure', 'Global-MLP (Pure)', 'x', '#d62728'),
    ]:
        ys = [results['ccr_sweep'][c][key]['mean_flowtime'] / 1000.0 for c in ccrs]
        ax.plot([f'CCR={c}' for c in ccrs], ys, marker=marker, linewidth=2.2, markersize=7, label=label, color=color)
    ax.set_ylabel('Mean Workflow Flowtime (×10³ s)')
    ax.set_title('Communication-to-Computation Ratio (CCR) Sensitivity (N=16)')
    ax.legend(frameon=True, fontsize=8.5)
    plt.tight_layout()
    p2 = fig_dir / 'fig2_ccr_sensitivity.png'
    fig.savefig(p2)
    plt.close(fig)
    saved.append(str(p2))

    # Fig 3: Decision Latency vs Flowtime Pareto Frontier (N=16)
    fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=180)
    pareto_pts = [
        ('nominal_ca_eft', 'Nominal-CA-EFT', '#9467bd', 'd'),
        ('flow_ca_eft', 'Flow-CA-EFT (K=1)', '#2ca02c', 's'),
        ('global_mlp_pure', 'Global-MLP (Pure)', '#d62728', 'x'),
        ('no_mp_pure', 'No-MP-Keep-Links (Pure)', '#ff7f0e', '^'),
        ('dualgflow_pure', 'DualGFlow-RG (Pure, K=1)', '#1f77b4', 'o'),
        ('dualgflow_res', 'DualGFlow-RG (Residual, K=1)', '#17becf', 'P'),
    ]
    for key, label, color, marker in pareto_pts:
        if key not in s16:
            continue
        d = s16[key]
        ax.scatter(max(d['p95_decision_ms'], 0.01), d['mean_flowtime'] / 1000.0, s=85, color=color, marker=marker, label=label, zorder=4)
    # Add Top-2 screening points
    for key, label, color, marker in [
        ('flow_ca_eft_top2', 'Flow-CA-EFT (Top-2)', '#2ca02c', '*'),
        ('dualgflow_top2', 'DualGFlow-RG (Top-2)', '#1f77b4', '*'),
    ]:
        if key in results['pareto_top2']:
            d = results['pareto_top2'][key]
            ax.scatter(d['p95_decision_ms'], d['mean_flowtime'] / 1000.0, s=140, color=color, marker=marker, label=label, zorder=5)

    ax.set_xscale('log')
    ax.set_ylim(20, 60)
    ax.set_xlabel('P95 Decision Latency (ms, log scale)')
    ax.set_ylabel('Mean Workflow Flowtime (×10³ s)')
    ax.set_title('Scheduling Quality vs. Decision Latency Trade-off (N=16 Held-Out)')
    ax.legend(frameon=True, fontsize=8, loc='upper right')
    plt.tight_layout()
    p3 = fig_dir / 'fig3_pareto_frontier.png'
    fig.savefig(p3)
    plt.close(fig)
    saved.append(str(p3))
    return saved


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Running 3-Seed Publication Benchmark on device={device}', flush=True)

    topos = ['leaf_spine', 'cloud_edge', 'multi_hop_mesh']
    seeds = [101, 202, 303]

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
    test_16 = [
        make_flow_scenario(seed=400 + i, split='test', workflows=14, num_servers=16,
                           load=0.75, ccr=2.5, topology=topos[i % 3], oversubscription=6.0)
        for i in range(18)
    ]
    test_32 = [
        make_flow_scenario(seed=500 + i, split='test', workflows=18, num_servers=32,
                           load=0.75, ccr=2.5, topology=topos[i % 3], oversubscription=6.0)
        for i in range(9)
    ]
    test_64 = [
        make_flow_scenario(seed=600 + i, split='test', workflows=24, num_servers=64,
                           load=0.75, ccr=2.5, topology=topos[i % 3], oversubscription=6.0)
        for i in range(6)
    ]
    ccr_suites = {
        str(ccr): [
            make_flow_scenario(seed=700 + int(ccr * 10) * 20 + i, split='test', workflows=14, num_servers=16,
                               load=0.75, ccr=ccr, topology=topos[i % 3], oversubscription=6.0)
            for i in range(9)
        ]
        for ccr in (0.5, 2.5, 5.0)
    }

    ckpt_dir = Path(__file__).resolve().parent / 'checkpoints_paper'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ds_path = ckpt_dir / 'dataset_18scenes.pt'
    if ds_path.exists():
        dataset = torch.load(ds_path, weights_only=False)
        print(f'Loaded cached dataset ({len(dataset)} states)', flush=True)
    else:
        print('Collecting shared counterfactual training dataset (18 train scenes)...', flush=True)
        t0 = time.perf_counter()
        dataset = collect_masked_counterfactuals(train_scenarios, policy=None, states_per_scenario=45, top_k=8, seed=101)
        prepare_full_costs(dataset)
        torch.save(dataset, ds_path)
        print(f'Collected & cached {len(dataset)} states in {time.perf_counter()-t0:.1f}s', flush=True)

    trained = {k: [] for k in (
        'dualgflow_pure', 'no_mp_pure', 'global_mlp_pure',
        'dualgflow_res', 'no_mp_res', 'global_mlp_res',
    )}

    # Phase 1: Train 3 seeds of Pure Structural (~1.5 mins total!)
    for sd in seeds:
        for var, short in [('dualgflow', 'dualgflow'), ('no_mp_keep_links', 'no_mp'), ('global_mlp', 'global_mlp')]:
            pure_ckpt = ckpt_dir / f'seed_{sd}_{short}_pure.pt'
            if pure_ckpt.exists():
                m_pure = DualGFlowPolicy(variant=var, mode='pure_structural', hidden_dim=64, gnn_layers=4).to(device)
                m_pure.load_state_dict(torch.load(pure_ckpt, map_location=device, weights_only=True))
                m_pure.eval()
                print(f'Loaded cached {pure_ckpt.name}', flush=True)
            else:
                print(f'Training {short}_pure seed={sd}...', flush=True)
                m_pure = train_pure_structural_fullcost(var, dataset, seed=sd, epochs=35, batch_size=64, lr=1.5e-3, device=device)
                torch.save(m_pure.state_dict(), pure_ckpt)
            trained[f'{short}_pure'].append(m_pure)

    def aggregate_across_seeds(scenarios: list[dict], active_keys: tuple[str, ...]) -> dict:
        heuristics = {
            'compute_eft': lambda e, o: int(np.argmin(o['compute_eft'])),
            'nominal_ca_eft': lambda e, o: int(np.argmin(o['nominal_eft'])),
            'flow_ca_eft': lambda e, o: int(np.argmin(o['flow_eft'])),
        }
        summary = eval_policy_suite(scenarios, heuristics)
        per_seed = {}
        for mkey in active_keys:
            mod_list = trained[mkey]
            if not mod_list:
                continue
            seed_runs = []
            for mod in mod_list:
                res = eval_policy_suite(scenarios, {mkey: lambda e, o, m=mod: int(torch.argmax(m.forward_actor(o)).item())})[mkey]
                seed_runs.append(res)
            per_seed[mkey] = seed_runs
            pooled_per_scene = np.mean([sr['per_scene'] for sr in seed_runs], axis=0).tolist()
            summary[mkey] = dict(
                mean_flowtime=float(np.mean([sr['mean_flowtime'] for sr in seed_runs])),
                seed_std=float(np.std([sr['mean_flowtime'] for sr in seed_runs])),
                per_scene=pooled_per_scene,
                by_topology={
                    t: float(np.mean([sr['by_topology'][t] for sr in seed_runs]))
                    for t in topos
                },
                p95_decision_ms=float(np.mean([sr['p95_decision_ms'] for sr in seed_runs])),
            )
        return dict(summary=summary, per_seed=per_seed)

    def run_and_save_all(active_keys: tuple[str, ...], include_rg_top2: bool = False):
        print('\nEvaluating Suite 1: Held-Out N=16 (18 scenes)...', flush=True)
        res_n16 = aggregate_across_seeds(test_16, active_keys)
        print('Evaluating Suite 2a: Zero-Shot N=32 (9 scenes)...', flush=True)
        res_n32 = aggregate_across_seeds(test_32, active_keys)
        print('Evaluating Suite 2b: Zero-Shot N=64 (6 scenes)...', flush=True)
        res_n64 = aggregate_across_seeds(test_64, active_keys)
        print('Evaluating Suite 3: CCR Sweep (0.5, 2.5, 5.0)...', flush=True)
        ccr_res = {c_str: aggregate_across_seeds(sc_list, active_keys)['summary'] for c_str, sc_list in ccr_suites.items()}
        print('Evaluating Suite 4: Top-2 Screening Pareto on N=16 (18 scenes)...', flush=True)
        top2_fns = {'flow_ca_eft_top2': lambda e, o: guided_topk_action(e, o, lambda ob: -ob['flow_eft'], k=2)}
        if include_rg_top2 and trained['dualgflow_res']:
            top2_fns['dualgflow_top2'] = lambda e, o: guided_topk_action(e, o, lambda ob: trained['dualgflow_res'][0].forward_actor(ob).detach().cpu().numpy(), k=2)
        elif trained['dualgflow_pure']:
            top2_fns['dualgflow_top2'] = lambda e, o: guided_topk_action(e, o, lambda ob: trained['dualgflow_pure'][0].forward_actor(ob).detach().cpu().numpy(), k=2)
        pareto_top2 = eval_policy_suite(test_16, top2_fns)

        cis = {}
        for tag, r in [('n16', res_n16), ('n32', res_n32), ('n64', res_n64)]:
            s = r['summary']
            cis[tag] = dict(
                rg_pure_vs_nomp=paired_gain_ci(s['no_mp_pure']['per_scene'], s['dualgflow_pure']['per_scene']),
                rg_pure_vs_global=paired_gain_ci(s['global_mlp_pure']['per_scene'], s['dualgflow_pure']['per_scene']),
                floweft_vs_nomeft=paired_gain_ci(s['nominal_ca_eft']['per_scene'], s['flow_ca_eft']['per_scene']),
            )
            if 'dualgflow_res' in s:
                cis[tag]['rg_res_vs_floweft'] = paired_gain_ci(s['flow_ca_eft']['per_scene'], s['dualgflow_res']['per_scene'])

        out = dict(
            seeds=seeds,
            n16_in_scale=res_n16,
            n32_zero_shot=res_n32,
            n64_zero_shot=res_n64,
            ccr_sweep=ccr_res,
            pareto_top2=pareto_top2,
            paired_95ci=cis,
        )
        out_path = Path(__file__).resolve().parent / 'paper_benchmark_results.json'
        out_path.write_text(json.dumps(out, indent=2), encoding='utf-8')
        fig_dir = Path(__file__).resolve().parent / 'figures'
        figs = generate_figures(out, fig_dir)
        out['figures'] = figs
        out_path.write_text(json.dumps(out, indent=2), encoding='utf-8')
        return out

    # Save Phase 1 (Pure Structural 3 seeds + Heuristics + Figures) immediately (~4 mins!)
    out = run_and_save_all(('dualgflow_pure', 'no_mp_pure', 'global_mlp_pure'), include_rg_top2=False)
    print('\n[Phase 1 Complete] Pure Structural 3-Seed Results & Figures saved!', flush=True)

    # Phase 2: Train 3 seeds of Residual EFT (4-update PPO, ~3-4 mins/model)
    for sd in seeds:
        print(f'\n================ RESIDUAL SEED {sd} ================', flush=True)
        for var, short in [('dualgflow', 'dualgflow'), ('no_mp_keep_links', 'no_mp'), ('global_mlp', 'global_mlp')]:
            res_ckpt = ckpt_dir / f'seed_{sd}_{short}_res.pt'
            if res_ckpt.exists():
                m_res = DualGFlowPolicy(variant=var, mode='residual_eft', hidden_dim=64, gnn_layers=4).to(device)
                m_res.load_state_dict(torch.load(res_ckpt, map_location=device, weights_only=True))
                m_res.eval()
                print(f'Loaded cached {res_ckpt.name}', flush=True)
            else:
                torch.manual_seed(sd)
                m_res = DualGFlowPolicy(variant=var, mode='residual_eft', hidden_dim=64, gnn_layers=4)
                train_batched_q_ranking(m_res, dataset, val_scenarios=val_scenarios, epochs=16, batch_size=64, lr=8e-4, device=device)
                train_batched_ppo(
                    m_res, train_scenarios, val_scenarios,
                    temperature=0.125, warmup_rounds=2, ppo_updates=4, episodes_per_update=6,
                    batch_size=128, lr_a=1e-4, lr_c=5e-4, device=device,
                )
                torch.save(m_res.state_dict(), res_ckpt)
            trained[f'{short}_res'].append(m_res)

    out = run_and_save_all(
        ('dualgflow_pure', 'no_mp_pure', 'global_mlp_pure', 'dualgflow_res', 'no_mp_res', 'global_mlp_res'),
        include_rg_top2=True,
    )
    res_n16, res_n32, res_n64, cis = out['n16_in_scale'], out['n32_zero_shot'], out['n64_zero_shot'], out['paired_95ci']

    print('\n================ FINAL 3-SEED BENCHMARK SUMMARY ================')
    for tag, r in [('N=16 HELD-OUT (18 scenes)', res_n16), ('N=32 ZERO-SHOT (9 scenes)', res_n32), ('N=64 ZERO-SHOT (6 scenes)', res_n64)]:
        print(f'\n--- {tag} ---')
        s = r['summary']
        for k in ('compute_eft', 'nominal_ca_eft', 'flow_ca_eft', 'dualgflow_pure', 'no_mp_pure', 'global_mlp_pure', 'dualgflow_res', 'no_mp_res', 'global_mlp_res'):
            v = s[k]
            sd_str = f"±{v['seed_std']:6.1f}" if 'seed_std' in v else '       '
            print(f"{k:18s} | Mean: {v['mean_flowtime']:9.1f}s {sd_str} | LS: {v['by_topology']['leaf_spine']:8.1f} | CE: {v['by_topology']['cloud_edge']:8.1f} | Mesh: {v['by_topology']['multi_hop_mesh']:8.1f} | P95: {v['p95_decision_ms']:5.2f}ms")

    print('\n--- PAIRED 95% BOOTSTRAP CONFIDENCE INTERVALS ---')
    for tag, cdict in cis.items():
        print(f'[{tag}]')
        for cname, c in cdict.items():
            print(f"  {cname:22s}: {c['gain_pct']:+6.2f}%  (95% CI [{c['ci95_low']:+.2f}%, {c['ci95_high']:+.2f}%], wins={c['wins']}/{c['total']})")


if __name__ == '__main__':
    main()
