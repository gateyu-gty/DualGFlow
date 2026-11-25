"""Parallel 48-scene test evaluation for the already-trained checkpoints_tw_clean (3 seeds)."""
from __future__ import annotations

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

from dualgflow.topologies import make_flow_scenario
from dualgflow.models_rg import DualGFlowPolicy
from dualgflow.run_paper_benchmark import paired_gain_ci
from dualgflow.run_fair_nominal_experiment import eval_strict
from dualgflow.run_mesh_topk_diag import make_topk_mask_1d, apply_mask


VARIANTS = ('dualgflow', 'no_mp_keep_links', 'global_mlp', 'no_rg')


def eval_seed_48(seed: int, device: str = 'cuda'):
    topo, ccr, load, ov, top_k = 'multi_hop_mesh', 4.5, 0.90, 12.0, 4
    out_dir = Path(__file__).resolve().parent
    ckpt_dir = out_dir / 'checkpoints_tw_clean'
    test_48 = [make_flow_scenario(seed=4000 + i, split='test', workflows=14, num_servers=16, load=load, ccr=ccr, topology=topo, oversubscription=ov) for i in range(48)]

    res = dict(seed=seed, offline={}, online={}, heuristics={})
    if seed == 101:
        for hname, hfn in [
            ('nominal_ca_eft', lambda e, o: int(np.argmin(o['nominal_eft']))),
            ('flow_oracle_eft', lambda e, o: int(np.argmin(o['flow_eft']))),
        ]:
            flows, lats = [], []
            for sc in test_48:
                r = eval_strict(sc, hfn, hname)
                flows.append(r['mean_flowtime']); lats.append(r['p95_decision_ms'])
            res['heuristics'][hname] = dict(mean_flowtime=float(np.mean(flows)), per_scene=flows, p95_decision_ms=float(np.mean(lats)))
        print(f'[Seed {seed}] Heuristics done.', flush=True)

    for stage in ('offline', 'online'):
        for var in VARIANTS:
            use_tw = (var == 'dualgflow')
            mod = DualGFlowPolicy(
                variant=var, mode='residual_eft', hidden_dim=64, gnn_layers=4,
                use_nominal_anchor=True, residual_scale=2.0, use_tropical_wave=use_tw,
            ).to(device)
            mod.load_state_dict(torch.load(ckpt_dir / f'{var}_{stage}_seed{seed}.pt', map_location=device, weights_only=True))
            mod.eval()
            flows, lats = [], []
            with torch.no_grad():
                for sc in test_48:
                    r = eval_strict(
                        sc,
                        lambda e, o, m=mod: int(torch.argmax(apply_mask(m.forward_actor(o), make_topk_mask_1d(o['nominal_eft'], k=top_k).to(device))).item()),
                        var,
                    )
                    flows.append(r['mean_flowtime']); lats.append(r['p95_decision_ms'])
            res[stage][var] = dict(mean_flowtime=float(np.mean(flows)), per_scene=flows, p95_decision_ms=float(np.mean(lats)))
            print(f'[Seed {seed}] {stage:7s} | {var:18s}: {np.mean(flows):.1f}s', flush=True)

    with open(ckpt_dir / f'eval48_seed{seed}.json', 'w') as f:
        json.dump(res, f, indent=2)
    print(f'[Seed {seed}] Saved eval48_seed{seed}.json', flush=True)


def merge_and_plot(seeds: list[int]):
    out_dir = Path(__file__).resolve().parent
    ckpt_dir = out_dir / 'checkpoints_tw_clean'
    sdata = {sd: json.load(open(ckpt_dir / f'eval48_seed{sd}.json')) for sd in seeds}
    gen_results = json.load(open(out_dir / 'tw_clean_3seed_results.json'))['gen_suites']

    res_off_48, res_on_48 = {}, {}
    for hname in ('nominal_ca_eft', 'flow_oracle_eft'):
        hd = sdata[101]['heuristics'][hname]
        entry = dict(mean_flowtime=hd['mean_flowtime'], seed_std=0.0, per_scene=hd['per_scene'], p95_decision_ms=hd['p95_decision_ms'])
        res_off_48[hname] = entry
        res_on_48[hname] = entry

    for stage, target in (('offline', res_off_48), ('online', res_on_48)):
        for var in VARIANTS:
            runs = [sdata[sd][stage][var] for sd in seeds]
            target[var] = dict(
                mean_flowtime=float(np.mean([r['mean_flowtime'] for r in runs])),
                seed_std=float(np.std([r['mean_flowtime'] for r in runs])),
                per_scene=np.mean([r['per_scene'] for r in runs], axis=0).tolist(),
                p95_decision_ms=float(np.mean([r['p95_decision_ms'] for r in runs])),
            )

    cis_on_48 = {
        f'{v}_vs_nominal': paired_gain_ci(res_on_48['nominal_ca_eft']['per_scene'], res_on_48[v]['per_scene'])
        for v in ('dualgflow', 'no_mp_keep_links', 'global_mlp', 'no_rg', 'flow_oracle_eft')
    }
    cis_on_48['rg_online_vs_offline'] = paired_gain_ci(res_off_48['dualgflow']['per_scene'], res_on_48['dualgflow']['per_scene'])
    cis_on_48['rg_vs_nomp'] = paired_gain_ci(res_on_48['no_mp_keep_links']['per_scene'], res_on_48['dualgflow']['per_scene'])
    cis_on_48['rg_vs_norg'] = paired_gain_ci(res_on_48['no_rg']['per_scene'], res_on_48['dualgflow']['per_scene'])
    cis_on_48['rg_vs_global'] = paired_gain_ci(res_on_48['global_mlp']['per_scene'], res_on_48['dualgflow']['per_scene'])

    with open(out_dir / 'tw_clean_48_results.json', 'w') as f:
        json.dump(dict(seeds=seeds, offline_48=res_off_48, online_48=res_on_48, cis_online_48=cis_on_48, gen_suites=gen_results), f, indent=2)

    fig_dir = out_dir / 'figures'
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.9))
    order = [
        ('no_rg', 'DualGFlow (No-RG)', '#8c564b'),
        ('global_mlp', 'Global-MLP', '#d62728'),
        ('no_mp_keep_links', 'No-MP-Keep-Links', '#ff7f0e'),
        ('nominal_ca_eft', 'Nominal-CA-EFT', '#9467bd'),
        ('dualgflow', 'DualGFlow-RG (Ours)', '#1f77b4'),
    ]
    y = np.arange(len(order))
    h = 0.36
    off_m = [res_off_48[o[0]]['mean_flowtime'] / 1000.0 for o in order]
    off_e = [res_off_48[o[0]]['seed_std'] / 1000.0 for o in order]
    on_m = [res_on_48[o[0]]['mean_flowtime'] / 1000.0 for o in order]
    on_e = [res_on_48[o[0]]['seed_std'] / 1000.0 for o in order]

    axes[0].barh(y - h / 2, off_m, xerr=off_e, height=h, color=[o[2] for o in order], alpha=0.35, edgecolor=[o[2] for o in order], hatch='//', capsize=2.5)
    axes[0].barh(y + h / 2, on_m, xerr=on_e, height=h, color=[o[2] for o in order], alpha=0.92, edgecolor='black', linewidth=0.5, capsize=3)
    axes[0].set_yticks(y); axes[0].set_yticklabels([o[1] for o in order])
    axes[0].set_xlim(min(on_m + off_m) - 1.1, max(on_m + off_m) + 1.55)
    axes[0].set_xlabel('Mean Workflow Flowtime (×10³ s)')
    axes[0].set_title('(a) 3-Seed Offline vs Online Main (N=16, 48 Scenes)')
    axes[0].grid(axis='x', alpha=0.25)
    legend_handles = [
        Patch(facecolor='#cccccc', edgecolor='#444444', hatch='//', label='Offline (Stage-1 WarmStart)'),
        Patch(facecolor='#444444', edgecolor='black', label='Online (Stage-1 + SIL-PPO)'),
    ]
    axes[0].legend(handles=legend_handles, fontsize=8.5, loc='lower right')
    for yi, m, err in zip(y, on_m, on_e):
        axes[0].text(m + err + 0.08, yi + h / 2, f'{m:.2f}k', va='center', fontsize=8.5, fontweight='bold')

    g_labels = ['Scale N=24', 'Scale N=32', 'CCR=2.5', 'CCR=6.0', 'Load=0.75', 'Load=1.05', 'Cloud-Edge', 'Leaf-Spine']
    g_keys = ['scale_n24', 'scale_n32', 'ccr_2.5', 'ccr_6.0', 'load_0.75', 'load_1.05', 'topo_ce', 'topo_ls']
    xg = np.arange(len(g_keys))
    w = 0.25
    for idx, (var, lbl, col) in enumerate([
        ('dualgflow', 'DualGFlow-RG (Ours)', '#1f77b4'),
        ('no_mp_keep_links', 'No-MP-Keep-Links', '#ff7f0e'),
        ('no_rg', 'DualGFlow (No-RG)', '#8c564b'),
    ]):
        gains = [(gen_results[k]['nominal_ca_eft']['mean_flowtime'] - gen_results[k][var]['mean_flowtime']) / gen_results[k]['nominal_ca_eft']['mean_flowtime'] * 100.0 for k in g_keys]
        axes[1].bar(xg + (idx - 1) * w, gains, width=w, color=col, alpha=0.9, label=lbl)
    axes[1].axhline(0.0, color='#9467bd', linestyle='--', linewidth=1.2)
    axes[1].set_xticks(xg); axes[1].set_xticklabels(g_labels, rotation=22, ha='right', fontsize=8.5)
    axes[1].set_ylabel('Zero-Shot Gain vs Nominal-CA-EFT (%)')
    axes[1].set_title('(b) 3-Seed Zero-Shot Generalization (Table 14)')
    axes[1].grid(axis='y', alpha=0.25); axes[1].legend(fontsize=8.3)
    plt.tight_layout()
    fig.savefig(fig_dir / 'fig39_tw_clean_48_main_and_gen.png')
    plt.close(fig)

    print('\n=== 3-SEED CLEAN (kl=0.05) MAIN N=16 (48 SCENES) ===')
    b48 = res_on_48['nominal_ca_eft']['mean_flowtime']
    for k in ('dualgflow', 'no_mp_keep_links', 'global_mlp', 'no_rg', 'nominal_ca_eft', 'flow_oracle_eft'):
        moff, soff = res_off_48[k]['mean_flowtime'], res_off_48[k]['seed_std']
        mon, son = res_on_48[k]['mean_flowtime'], res_on_48[k]['seed_std']
        print(f"  {k:18s} | Off: {moff:8.1f}s ± {soff:5.1f}s ({(b48-moff)/b48*100:+5.2f}%) | On: {mon:8.1f}s ± {son:5.1f}s ({(b48-mon)/b48*100:+5.2f}%) | p95={res_on_48[k]['p95_decision_ms']:.2f}ms")

    print('\n=== PAIRED 95% BOOTSTRAP CIs (ONLINE, N=16, 48 SCENES) ===')
    for ck, cv in cis_on_48.items():
        print(f"  {ck:22s}: {cv['gain_pct']:+.2f}% [95% CI: {cv['ci95_low']:+.2f}%, {cv['ci95_high']:+.2f}%] | wins={cv['wins']}/{cv['total']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--merge', action='store_true')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.seed is not None:
        eval_seed_48(args.seed, device=device)
    if args.merge:
        merge_and_plot([101, 202, 303])


if __name__ == '__main__':
    main()
