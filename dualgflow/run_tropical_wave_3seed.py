"""3-Seed Final Tropical-Wave Paper Suite on Multi-Hop Mesh HC.

Implements the 3-step rectification plan:
1. Gentle Stage-2 PPO fine-tuning (lr_a=1.5e-5, kl_coef=0.015) around the strong
   Stage-1 Tropical-Wave policy so Online preserves/improves Offline gains.
2. Full 3-seed coverage (seeds 101, 202, 303) across all 4 models.
3. Evaluates Offline (Table 1), Online Main (N=16, 24 scenes), Learning Curves,
   and all 8 Table 14 Zero-Shot Generalization suites (N=24, N=32, CCR=2.5/6.0,
   Load=0.75/1.05, Cloud-Edge HC, Leaf-Spine HC).
"""
from __future__ import annotations

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dualgflow.topologies import make_flow_scenario
from dualgflow.models_rg import DualGFlowPolicy
from dualgflow.run_paper_benchmark import paired_gain_ci
from dualgflow.run_fair_nominal_experiment import eval_strict, collect_strict_fifo_dataset
from dualgflow.run_mesh_topk_diag import make_topk_mask_1d, apply_mask
from dualgflow.run_mesh_stage1_ppo_4models import train_stage1_topk
from dualgflow.run_ce_hc_aligned_3seed import train_stage2_aligned


def eval_3seed_suite(
    sc_list: list[dict],
    models_by_var: dict[str, list[DualGFlowPolicy]],
    top_k: int,
    device: str,
) -> dict:
    out = {}
    for hname, hfn in [
        ('nominal_ca_eft', lambda e, o: int(np.argmin(o['nominal_eft']))),
        ('flow_oracle_eft', lambda e, o: int(np.argmin(o['flow_eft']))),
    ]:
        flows, lats = [], []
        for sc in sc_list:
            r = eval_strict(sc, hfn, hname)
            flows.append(r['mean_flowtime']); lats.append(r['p95_decision_ms'])
        out[hname] = dict(mean_flowtime=float(np.mean(flows)), seed_std=0.0, per_scene=flows, p95_decision_ms=float(np.mean(lats)))

    for var, mod_list in models_by_var.items():
        seed_runs = []
        for mod in mod_list:
            flows, lats = [], []
            with torch.no_grad():
                for sc in sc_list:
                    r = eval_strict(
                        sc,
                        lambda e, o, m=mod: int(torch.argmax(apply_mask(m.forward_actor(o), make_topk_mask_1d(o['nominal_eft'], k=top_k).to(device))).item()),
                        var,
                    )
                    flows.append(r['mean_flowtime']); lats.append(r['p95_decision_ms'])
            seed_runs.append(dict(mean_flowtime=float(np.mean(flows)), per_scene=flows, p95_decision_ms=float(np.mean(lats))))
        pooled = np.mean([sr['per_scene'] for sr in seed_runs], axis=0).tolist()
        out[var] = dict(
            mean_flowtime=float(np.mean([sr['mean_flowtime'] for sr in seed_runs])),
            seed_std=float(np.std([sr['mean_flowtime'] for sr in seed_runs])),
            per_scene=pooled,
            p95_decision_ms=float(np.mean([sr['p95_decision_ms'] for sr in seed_runs])),
        )
    return out


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    topo = 'multi_hop_mesh'
    ccr, load, ov = 4.5, 0.90, 12.0
    seeds = [101, 202, 303]
    top_k = 4
    out_dir = Path(__file__).resolve().parent
    ckpt_dir = out_dir / 'checkpoints_tropical_wave_3seed'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f'Running 3-Seed Final Tropical-Wave Suite on {topo} HC (lr_a=1.5e-5, kl=0.015) on {device}', flush=True)

    train_sc = [make_flow_scenario(seed=1000 + i, split='train', workflows=14, num_servers=16, load=load, ccr=ccr, topology=topo, oversubscription=ov) for i in range(36)]
    val_sc = [make_flow_scenario(seed=2000 + i, split='validation', workflows=14, num_servers=16, load=load, ccr=ccr, topology=topo, oversubscription=ov) for i in range(12)]
    test_16 = [make_flow_scenario(seed=4000 + i, split='test', workflows=14, num_servers=16, load=load, ccr=ccr, topology=topo, oversubscription=ov) for i in range(24)]

    gen_suites = {
        'scale_n24': [make_flow_scenario(seed=5100 + i, split='test', workflows=16, num_servers=24, load=load, ccr=ccr, topology=topo, oversubscription=ov) for i in range(8)],
        'scale_n32': [make_flow_scenario(seed=5000 + i, split='test', workflows=18, num_servers=32, load=load, ccr=ccr, topology=topo, oversubscription=ov) for i in range(8)],
        'ccr_2.5':   [make_flow_scenario(seed=6100 + i, split='test', workflows=14, num_servers=16, load=load, ccr=2.5, topology=topo, oversubscription=ov) for i in range(8)],
        'ccr_6.0':   [make_flow_scenario(seed=6200 + i, split='test', workflows=14, num_servers=16, load=load, ccr=6.0, topology=topo, oversubscription=ov) for i in range(8)],
        'load_0.75': [make_flow_scenario(seed=7100 + i, split='test', workflows=14, num_servers=16, load=0.75, ccr=ccr, topology=topo, oversubscription=ov) for i in range(8)],
        'load_1.05': [make_flow_scenario(seed=7200 + i, split='test', workflows=14, num_servers=16, load=1.05, ccr=ccr, topology=topo, oversubscription=ov) for i in range(8)],
        'topo_ce':   [make_flow_scenario(seed=8100 + i, split='test', workflows=14, num_servers=16, load=load, ccr=ccr, topology='cloud_edge', oversubscription=ov) for i in range(8)],
        'topo_ls':   [make_flow_scenario(seed=8200 + i, split='test', workflows=14, num_servers=16, load=load, ccr=ccr, topology='leaf_spine', oversubscription=ov) for i in range(8)],
    }

    ds_path = out_dir / 'checkpoints_mesh_hc' / 'dataset_mesh_hc_36scenes_80st.pt'
    if ds_path.exists():
        dataset = torch.load(ds_path, weights_only=False)
        print(f'Loaded cached dataset ({len(dataset)} states)', flush=True)
    else:
        dataset = collect_strict_fifo_dataset(train_sc, states_per_sc=80, top_k=4, seed=101)
        ds_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dataset, ds_path)

    train_nom = [eval_strict(sc, lambda e, o: int(np.argmin(o['nominal_eft'])), 'nom')['mean_flowtime'] for sc in train_sc]
    val_nom = float(np.mean([eval_strict(sc, lambda e, o: int(np.argmin(o['nominal_eft'])), 'nom')['mean_flowtime'] for sc in val_sc]))

    variants = ('dualgflow', 'no_mp_keep_links', 'global_mlp', 'no_rg')
    offline_models = {v: [] for v in variants}
    online_models = {v: [] for v in variants}
    histories = {v: {} for v in variants}

    for sd in seeds:
        print(f'\n================ SEED {sd} ================', flush=True)
        for var in variants:
            ckpt_off = ckpt_dir / f'{var}_offline_seed{sd}.pt'
            ckpt_on = ckpt_dir / f'{var}_online_seed{sd}.pt'
            hpath = ckpt_dir / f'history_{var}_seed{sd}.json'
            use_tw = (var == 'dualgflow')

            if ckpt_off.exists() and ckpt_on.exists() and hpath.exists():
                m_off = DualGFlowPolicy(variant=var, mode='residual_eft', hidden_dim=64, gnn_layers=4,
                                       use_nominal_anchor=True, residual_scale=2.0, use_tropical_wave=use_tw).to(device)
                m_off.load_state_dict(torch.load(ckpt_off, map_location=device, weights_only=True))
                m_off.eval()

                m_on = DualGFlowPolicy(variant=var, mode='residual_eft', hidden_dim=64, gnn_layers=4,
                                      use_nominal_anchor=True, residual_scale=2.0, use_tropical_wave=use_tw).to(device)
                m_on.load_state_dict(torch.load(ckpt_on, map_location=device, weights_only=True))
                m_on.eval()
                with open(hpath) as hf:
                    hist = json.load(hf)
                print(f'Loaded cached {var} seed {sd}', flush=True)
            else:
                print(f'--- Training {var} (seed={sd}, TropicalWave={use_tw}) ---', flush=True)
                torch.manual_seed(sd)
                m = DualGFlowPolicy(variant=var, mode='residual_eft', hidden_dim=64, gnn_layers=4,
                                   use_nominal_anchor=True, residual_scale=2.0, use_tropical_wave=use_tw).to(device)
                m, h1 = train_stage1_topk(m, dataset, val_sc, val_nom, top_k=top_k, epochs=8, lr=1.5e-4, seed=sd, device=device)
                m_off = deepcopy(m).eval()
                torch.save(m_off.state_dict(), ckpt_off)

                m, h2 = train_stage2_aligned(
                    m, train_sc, val_sc, train_nom, val_nom,
                    seed=sd, top_k=top_k, temperature=0.05, ppo_updates=8, eps_per_upd=12,
                    lr_a=1.5e-5, kl_coef=0.015, ent_coef=0.0003, device=device,
                )
                m_on = m.eval()
                torch.save(m_on.state_dict(), ckpt_on)
                hist = dict(stage1=h1, stage2=h2)
                with open(hpath, 'w') as hf:
                    json.dump(hist, hf, indent=2)

            offline_models[var].append(m_off)
            online_models[var].append(m_on)
            histories[var][str(sd)] = hist

    print('\nEvaluating 3-Seed Offline vs Online Main Performance (N=16, 24 scenes)...', flush=True)
    res_off_16 = eval_3seed_suite(test_16, offline_models, top_k, device)
    res_on_16 = eval_3seed_suite(test_16, online_models, top_k, device)

    print('Evaluating 3-Seed Zero-Shot Generalization Suites (Table 14)...', flush=True)
    gen_results = {gk: eval_3seed_suite(gscs, online_models, top_k, device) for gk, gscs in gen_suites.items()}

    cis_on_16 = {
        f'{v}_vs_nominal': paired_gain_ci(res_on_16['nominal_ca_eft']['per_scene'], res_on_16[v]['per_scene'])
        for v in ('dualgflow', 'no_mp_keep_links', 'global_mlp', 'no_rg', 'flow_oracle_eft')
    }
    cis_on_16['rg_online_vs_offline'] = paired_gain_ci(res_off_16['dualgflow']['per_scene'], res_on_16['dualgflow']['per_scene'])
    cis_on_16['rg_vs_nomp'] = paired_gain_ci(res_on_16['no_mp_keep_links']['per_scene'], res_on_16['dualgflow']['per_scene'])
    cis_on_16['rg_vs_norg'] = paired_gain_ci(res_on_16['no_rg']['per_scene'], res_on_16['dualgflow']['per_scene'])
    cis_on_16['rg_vs_global'] = paired_gain_ci(res_on_16['global_mlp']['per_scene'], res_on_16['dualgflow']['per_scene'])

    with open(out_dir / 'tropical_wave_3seed_results.json', 'w') as f:
        json.dump(dict(
            seeds=seeds, offline_16=res_off_16, online_16=res_on_16,
            cis_online_16=cis_on_16, gen_suites=gen_results,
        ), f, indent=2)

    fig_dir = out_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})

    # Fig 31: 3-Seed Learning Curves
    fig_lc, axes_lc = plt.subplots(1, 2, figsize=(12.2, 4.6))
    styles = [
        ('dualgflow', 'DualGFlow-RG (Ours)', '#1f77b4', '-'),
        ('no_mp_keep_links', 'No-MP-Keep-Links', '#ff7f0e', '--'),
        ('global_mlp', 'Global-MLP', '#d62728', '-.'),
        ('no_rg', 'DualGFlow (No-RG)', '#8c564b', ':'),
    ]
    for var, label, col, ls in styles:
        step_vg: dict[int, list[float]] = {}
        step_ev: dict[int, list[float]] = {}
        for sd_str, hd in histories[var].items():
            for r in hd['stage1']:
                step_vg.setdefault(int(r['step']), []).append(float(r['val_gain_pct']))
            for r in hd['stage2']:
                if r['step'] > 0:
                    step_vg.setdefault(8 + int(r['step']), []).append(float(r['val_gain_pct']))
                if 'explained_variance' in r:
                    step_ev.setdefault(int(r['step']), []).append(float(r['explained_variance']))
        xs = sorted(step_vg.keys())
        ys = np.array([np.mean(step_vg[x]) for x in xs])
        es = np.array([np.std(step_vg[x]) for x in xs])
        axes_lc[0].plot(xs, ys, label=label, color=col, linestyle=ls, linewidth=2.2, marker='o', markersize=3.4)
        axes_lc[0].fill_between(xs, ys - es, ys + es, color=col, alpha=0.14)

        exs = sorted(step_ev.keys())
        eys = np.array([np.mean(step_ev[x]) for x in exs])
        ees = np.array([np.std(step_ev[x]) for x in exs])
        axes_lc[1].plot(exs, eys, label=label, color=col, linestyle=ls, linewidth=2.2, marker='o', markersize=3.4)
        axes_lc[1].fill_between(exs, eys - ees, eys + ees, color=col, alpha=0.14)

    axes_lc[0].axvline(8, color='gray', linestyle=':', linewidth=1.4, label='Offline → Online PPO')
    axes_lc[0].axhline(0.0, color='#9467bd', linestyle='--', linewidth=1.4, label='Nominal-CA-EFT (0%)')
    axes_lc[0].set_title('(a) 3-Seed Offline + Online PPO Val Gain (%)')
    axes_lc[0].set_xlabel('Step (0-8: Offline Stage 1, 9-16: Online PPO)'); axes_lc[0].set_ylabel('Validation Gain (%)')
    axes_lc[0].grid(alpha=0.25); axes_lc[0].legend(fontsize=8.3)
    axes_lc[1].set_title('(b) 3-Seed Online Stage-2 Critic Explained Variance')
    axes_lc[1].set_xlabel('PPO Update'); axes_lc[1].set_ylabel('Explained Variance')
    axes_lc[1].grid(alpha=0.25); axes_lc[1].legend(fontsize=8.5)
    plt.tight_layout()
    fig_lc.savefig(fig_dir / 'fig31_tw_3seed_learning_curves.png')
    plt.close(fig_lc)

    # Fig 32: 3-Seed Main Performance + Table 14 Zero-Shot Generalization
    fig_m, axes_m = plt.subplots(1, 2, figsize=(13.5, 4.8))
    order = [
        ('no_rg', 'DualGFlow (No-RG)', '#8c564b'),
        ('global_mlp', 'Global-MLP', '#d62728'),
        ('no_mp_keep_links', 'No-MP-Keep-Links', '#ff7f0e'),
        ('nominal_ca_eft', 'Nominal-CA-EFT', '#9467bd'),
        ('dualgflow', 'DualGFlow-RG (Ours)', '#1f77b4'),
    ]
    y = np.arange(len(order))
    h = 0.36
    off_m = [res_off_16[o[0]]['mean_flowtime'] / 1000.0 for o in order]
    off_s = [res_off_16[o[0]]['seed_std'] / 1000.0 for o in order]
    on_m = [res_on_16[o[0]]['mean_flowtime'] / 1000.0 for o in order]
    on_s = [res_on_16[o[0]]['seed_std'] / 1000.0 for o in order]
    axes_m[0].barh(y - h / 2, off_m, xerr=off_s, capsize=3, height=h, color='#aec7e8', edgecolor='#1f77b4', label='Offline (Stage-1)')
    axes_m[0].barh(y + h / 2, on_m, xerr=on_s, capsize=3, height=h, color=[o[2] for o in order], alpha=0.92, label='Online (Stage-1+PPO)')
    axes_m[0].set_yticks(y); axes_m[0].set_yticklabels([o[1] for o in order])
    axes_m[0].set_xlim(min(on_m + off_m) - 1.5, max(m + s for m, s in zip(on_m + off_m, on_s + off_s)) + 1.2)
    axes_m[0].set_xlabel('Mean Workflow Flowtime (×10³ s)')
    axes_m[0].set_title('(a) 3-Seed Offline vs Online Main (N=16, 24 Scenes)')
    axes_m[0].grid(axis='x', alpha=0.25); axes_m[0].legend(fontsize=8.5)
    for yi, m, s in zip(y, on_m, on_s):
        axes_m[0].text(m + s + 0.08, yi + h / 2, f'{m:.2f}k', va='center', fontsize=8.5, fontweight='bold')

    gen_keys = ['scale_n24', 'scale_n32', 'ccr_2.5', 'ccr_6.0', 'load_0.75', 'load_1.05', 'topo_ce', 'topo_ls']
    gen_labels = ['N=24', 'N=32', 'CCR=2.5', 'CCR=6.0', 'Load=0.75', 'Load=1.05', 'Cloud-Edge', 'Leaf-Spine']
    x = np.arange(len(gen_keys))
    w = 0.19
    for idx, (var, label, col, _) in enumerate(styles):
        gains = [
            (gen_results[gk]['nominal_ca_eft']['mean_flowtime'] - gen_results[gk][var]['mean_flowtime'])
            / gen_results[gk]['nominal_ca_eft']['mean_flowtime'] * 100.0
            for gk in gen_keys
        ]
        axes_m[1].bar(x + (idx - 1.5) * w, gains, width=w, color=col, label=label, alpha=0.9)
    axes_m[1].axhline(0.0, color='#9467bd', linestyle='--', linewidth=1.2)
    axes_m[1].set_xticks(x); axes_m[1].set_xticklabels(gen_labels, rotation=25, ha='right', fontsize=8.5)
    axes_m[1].set_ylabel('Zero-Shot Gain vs Nominal-CA-EFT (%)')
    axes_m[1].set_title('(b) 3-Seed Zero-Shot Generalization Without Retraining')
    axes_m[1].grid(axis='y', alpha=0.25); axes_m[1].legend(fontsize=8.0)
    plt.tight_layout()
    fig_m.savefig(fig_dir / 'fig32_tw_3seed_main_and_gen.png')
    plt.close(fig_m)

    print('\n=== 3-SEED OFFLINE vs ONLINE N=16 (24 SCENES) ===')
    b16 = res_on_16['nominal_ca_eft']['mean_flowtime']
    for k in ('dualgflow', 'no_mp_keep_links', 'global_mlp', 'no_rg', 'nominal_ca_eft', 'flow_oracle_eft'):
        vo, von = res_off_16[k], res_on_16[k]
        print(
            f"  {k:18s} | Off: {vo['mean_flowtime']:8.1f}s ±{vo['seed_std']:5.1f} ({(b16-vo['mean_flowtime'])/b16*100:+5.2f}%) "
            f"| On: {von['mean_flowtime']:8.1f}s ±{von['seed_std']:5.1f} ({(b16-von['mean_flowtime'])/b16*100:+5.2f}%) | p95={von['p95_decision_ms']:4.2f}ms"
        )

    print('\n=== 95% PAIRED CIs (ONLINE N=16, 24 SCENES) ===')
    for ck, cv in cis_on_16.items():
        print(f"  {ck:24s}: {cv['gain_pct']:+6.2f}% [{cv['ci95_low']:+.2f}%, {cv['ci95_high']:+.2f}%] w={cv['wins']}/{cv['total']}")

    print('\n=== 3-SEED ZERO-SHOT GENERALIZATION WITHOUT RETRAINING (TABLE 14) ===')
    for gk, gl in zip(gen_keys, gen_labels):
        bn = gen_results[gk]['nominal_ca_eft']['mean_flowtime']
        bo = gen_results[gk]['flow_oracle_eft']['mean_flowtime']
        row = f"  {gl:11s} (Ora={(bn-bo)/bn*100:+5.2f}%):"
        for var in variants:
            mv = gen_results[gk][var]['mean_flowtime']
            row += f" {var}={(bn-mv)/bn*100:+5.2f}% |"
        print(row)


if __name__ == '__main__':
    main()
