"""Plan B Experiments:
Part B1: Moderate Oversubscription & CCR Sensitivity (OV=2.0/CCR=2.0, OV=4.0/CCR=2.0, OV=4.0/CCR=4.5)
Part B2: 3-Seed Standalone No-RG+TW Ablation (Task-Only Encoder + 6-Hop Tropical Wave, NO 4L Resource GNN)
"""
from __future__ import annotations

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import json
import time
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch

from dualgflow.topologies import make_flow_scenario
from dualgflow.models_rg import DualGFlowPolicy
from dualgflow.run_mesh_topk_diag import make_topk_mask_1d, apply_mask
from dualgflow.run_fair_nominal_experiment import eval_strict
from dualgflow.run_mesh_stage1_ppo_4models import train_stage1_topk
from dualgflow.run_tw_clean_3seed import train_online_stable_sil
from dualgflow.run_paper_benchmark import paired_gain_ci


def run_b1_ov_sensitivity(device: str, out_dir: Path, ckpt_dir: Path) -> None:
    print('\n=== [Part B1] Moderate Oversubscription & CCR Sensitivity (3 Seeds, Zero-Shot) ===', flush=True)
    t0 = time.perf_counter()
    seeds = (101, 202, 303)
    top_k = 4
    suites = {
        'ov2_ccr2.0': [make_flow_scenario(seed=9100 + i, split='test', workflows=14, num_servers=16, load=0.90, ccr=2.0, topology='multi_hop_mesh', oversubscription=2.0) for i in range(8)],
        'ov4_ccr2.0': [make_flow_scenario(seed=9200 + i, split='test', workflows=14, num_servers=16, load=0.90, ccr=2.0, topology='multi_hop_mesh', oversubscription=4.0) for i in range(8)],
        'ov4_ccr4.5': [make_flow_scenario(seed=9300 + i, split='test', workflows=14, num_servers=16, load=0.90, ccr=4.5, topology='multi_hop_mesh', oversubscription=4.0) for i in range(8)],
    }

    # Load all 5 trained online models (3 seeds each)
    model_specs = [
        ('dualgflow', 'dualgflow', True, 'dualgflow_online_seed'),
        ('no_tw', 'dualgflow', False, 'dualgflow_notw_online_seed'),
        ('no_mp_keep_links', 'no_mp_keep_links', False, 'no_mp_keep_links_online_seed'),
        ('no_rg', 'no_rg', False, 'no_rg_online_seed'),
        ('global_mlp', 'global_mlp', False, 'global_mlp_online_seed'),
    ]
    loaded = {name: [] for name, *_ in model_specs}
    for name, var, use_tw, prefix in model_specs:
        for sd in seeds:
            m = DualGFlowPolicy(
                variant=var, mode='residual_eft', hidden_dim=64, gnn_layers=4,
                use_nominal_anchor=True, residual_scale=2.0, use_tropical_wave=use_tw,
            ).to(device)
            m.load_state_dict(torch.load(ckpt_dir / f'{prefix}{sd}.pt', map_location=device, weights_only=True))
            m.eval()
            loaded[name].append(m)

    results = {}
    with torch.no_grad():
        for sname, scs in suites.items():
            srec = {}
            nom_ps = [eval_strict(sc, lambda e, o: int(np.argmin(o['nominal_eft'])), 'nom')['mean_flowtime'] for sc in scs]
            comp_ps = [eval_strict(sc, lambda e, o: int(np.argmin(o['compute_eft'])), 'comp')['mean_flowtime'] for sc in scs]
            dl_ps = [eval_strict(sc, lambda e, o: int(np.argmin(o['pair_x'][:, 0])), 'dl')['mean_flowtime'] for sc in scs]
            nom_m = float(np.mean(nom_ps))
            srec['nominal_ca_eft'] = dict(mean_flowtime=nom_m, gain_vs_nom_pct=0.0)
            srec['compute_eft'] = dict(mean_flowtime=float(np.mean(comp_ps)), gain_vs_nom_pct=float((nom_m - np.mean(comp_ps)) / nom_m * 100.0))
            srec['data_locality'] = dict(mean_flowtime=float(np.mean(dl_ps)), gain_vs_nom_pct=float((nom_m - np.mean(dl_ps)) / nom_m * 100.0))

            for name, mods in loaded.items():
                sd_means = []
                for m in mods:
                    ps = [
                        eval_strict(
                            sc,
                            lambda e, o, mod=m: int(torch.argmax(apply_mask(mod.forward_actor(o), make_topk_mask_1d(o['nominal_eft'], k=top_k).to(device))).item()),
                            name,
                        )['mean_flowtime']
                        for sc in scs
                    ]
                    sd_means.append(float(np.mean(ps)))
                mm, ms = float(np.mean(sd_means)), float(np.std(sd_means))
                srec[name] = dict(mean_flowtime=mm, seed_std=ms, gain_vs_nom_pct=float((nom_m - mm) / nom_m * 100.0))

            results[sname] = srec
            print(
                f'  {sname:11s} | Nom={nom_m:7.1f}s | Comp={srec["compute_eft"]["gain_vs_nom_pct"]:+6.1f}% | DL={srec["data_locality"]["gain_vs_nom_pct"]:+6.2f}% '
                f'| DualGFlow={srec["dualgflow"]["gain_vs_nom_pct"]:+5.2f}% | NoTW={srec["no_tw"]["gain_vs_nom_pct"]:+5.2f}% '
                f'| NoMP={srec["no_mp_keep_links"]["gain_vs_nom_pct"]:+5.2f}% | NoRG={srec["no_rg"]["gain_vs_nom_pct"]:+5.2f}%',
                flush=True,
            )

    out_json = out_dir / 'tw_ov_sensitivity_results.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f'[Part B1] Completed in {time.perf_counter()-t0:.1f}s -> {out_json}', flush=True)


def run_b2_norg_tw_ablation(device: str, out_dir: Path, ckpt_dir: Path) -> None:
    print('\n=== [Part B2] 3-Seed No-RG+TW Ablation (Task-Only + 6-Hop Tropical Wave, NO 4L Resource GNN) ===', flush=True)
    seeds = (101, 202, 303)
    topo, ccr, load, ov, top_k = 'multi_hop_mesh', 4.5, 0.90, 12.0, 4

    train_sc = [make_flow_scenario(seed=1000 + i, split='train', workflows=14, num_servers=16, load=load, ccr=ccr, topology=topo, oversubscription=ov) for i in range(36)]
    val_sc = [make_flow_scenario(seed=2000 + i, split='validation', workflows=14, num_servers=16, load=load, ccr=ccr, topology=topo, oversubscription=ov) for i in range(16)]
    test_48 = [make_flow_scenario(seed=4000 + i, split='test', workflows=14, num_servers=16, load=load, ccr=ccr, topology=topo, oversubscription=ov) for i in range(48)]
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
    dataset = torch.load(ds_path, weights_only=False)
    train_nom = [eval_strict(sc, lambda e, o: int(np.argmin(o['nominal_eft'])), 'nom')['mean_flowtime'] for sc in train_sc]
    val_nom = float(np.mean([eval_strict(sc, lambda e, o: int(np.argmin(o['nominal_eft'])), 'nom')['mean_flowtime'] for sc in val_sc]))

    per_seed = {}
    for sd in seeds:
        t0 = time.perf_counter()
        ckpt_off = ckpt_dir / f'no_rg_tw_offline_seed{sd}.pt'
        ckpt_on = ckpt_dir / f'no_rg_tw_online_seed{sd}.pt'
        torch.manual_seed(sd)
        mod = DualGFlowPolicy(
            variant='no_rg', mode='residual_eft', hidden_dim=64, gnn_layers=4,
            use_nominal_anchor=True, residual_scale=2.0, use_tropical_wave=True,
        ).to(device)

        if ckpt_off.exists() and ckpt_on.exists():
            print(f'[Seed {sd}] Loaded cached No-RG+TW checkpoints.', flush=True)
            m_off = deepcopy(mod); m_off.load_state_dict(torch.load(ckpt_off, map_location=device, weights_only=True)); m_off.eval()
            m_on = deepcopy(mod); m_on.load_state_dict(torch.load(ckpt_on, map_location=device, weights_only=True)); m_on.eval()
        else:
            print(f'\n[Seed {sd}] === Training No-RG+TW (Task-Only + 6-Hop Tropical Wave) ===', flush=True)
            mod, h1 = train_stage1_topk(mod, dataset, val_sc, val_nom, top_k=top_k, epochs=1, lr=8e-5, seed=sd, device=device)
            m_off = deepcopy(mod).eval()
            torch.save(m_off.state_dict(), ckpt_off)

            mod, h2 = train_online_stable_sil(mod, train_sc, val_sc, train_nom, val_nom, seed=sd, top_k=top_k, device=device)
            m_on = mod.eval()
            torch.save(m_on.state_dict(), ckpt_on)
            with open(ckpt_dir / f'history_no_rg_tw_seed{sd}.json', 'w') as hf:
                json.dump(dict(stage1=h1, stage2=h2), hf, indent=2)

        sd_rec = dict(offline_48={}, online_48={}, gen_suites={})
        with torch.no_grad():
            for st_name, m_eval in (('offline_48', m_off), ('online_48', m_on)):
                flows, lats = [], []
                for sc in test_48:
                    r = eval_strict(
                        sc,
                        lambda e, o, m=m_eval: int(torch.argmax(apply_mask(m.forward_actor(o), make_topk_mask_1d(o['nominal_eft'], k=top_k).to(device))).item()),
                        'norg_tw',
                    )
                    flows.append(r['mean_flowtime'])
                    lats.append(r['p95_decision_ms'])
                sd_rec[st_name] = dict(mean_flowtime=float(np.mean(flows)), per_scene=flows, p95_decision_ms=float(np.mean(lats)))

            for gname, gscs in gen_suites.items():
                gflows = []
                for sc in gscs:
                    r = eval_strict(
                        sc,
                        lambda e, o, m=m_on: int(torch.argmax(apply_mask(m.forward_actor(o), make_topk_mask_1d(o['nominal_eft'], k=top_k).to(device))).item()),
                        'norg_tw',
                    )
                    gflows.append(r['mean_flowtime'])
                sd_rec['gen_suites'][gname] = dict(mean_flowtime=float(np.mean(gflows)), per_scene=gflows)

        per_seed[str(sd)] = sd_rec
        print(
            f'[Seed {sd}] Done in {time.perf_counter()-t0:.1f}s | '
            f'Off48: {sd_rec["offline_48"]["mean_flowtime"]:.1f}s | '
            f'On48: {sd_rec["online_48"]["mean_flowtime"]:.1f}s | '
            f'p95: {sd_rec["online_48"]["p95_decision_ms"]:.2f}ms',
            flush=True,
        )

    d48 = json.load(open(out_dir / 'tw_clean_48_results.json', encoding='utf-8'))
    nom_ps = d48['online_48']['nominal_ca_eft']['per_scene']
    nom_mean = d48['online_48']['nominal_ca_eft']['mean_flowtime']
    rg_ps = d48['online_48']['dualgflow']['per_scene']
    norg_ps = d48['online_48']['no_rg']['per_scene']

    off_runs = [per_seed[str(sd)]['offline_48'] for sd in seeds]
    on_runs = [per_seed[str(sd)]['online_48'] for sd in seeds]
    off_ps = np.mean([r['per_scene'] for r in off_runs], axis=0).tolist()
    on_ps = np.mean([r['per_scene'] for r in on_runs], axis=0).tolist()

    agg = dict(
        seeds=list(seeds),
        offline_48=dict(
            mean_flowtime=float(np.mean([r['mean_flowtime'] for r in off_runs])),
            seed_std=float(np.std([r['mean_flowtime'] for r in off_runs])),
            gain_vs_nom_pct=float((nom_mean - np.mean([r['mean_flowtime'] for r in off_runs])) / nom_mean * 100.0),
            per_scene=off_ps,
            p95_decision_ms=float(np.mean([r['p95_decision_ms'] for r in off_runs])),
        ),
        online_48=dict(
            mean_flowtime=float(np.mean([r['mean_flowtime'] for r in on_runs])),
            seed_std=float(np.std([r['mean_flowtime'] for r in on_runs])),
            gain_vs_nom_pct=float((nom_mean - np.mean([r['mean_flowtime'] for r in on_runs])) / nom_mean * 100.0),
            per_scene=on_ps,
            p95_decision_ms=float(np.mean([r['p95_decision_ms'] for r in on_runs])),
        ),
        cis_online_48=dict(
            norg_tw_vs_norg=paired_gain_ci(norg_ps, on_ps),
            rg_vs_norg_tw=paired_gain_ci(on_ps, rg_ps),
            norg_tw_vs_nominal=paired_gain_ci(nom_ps, on_ps),
        ),
        gen_suites={},
        per_seed=per_seed,
    )
    for gname in gen_suites:
        gruns = [per_seed[str(sd)]['gen_suites'][gname] for sd in seeds]
        gmean = float(np.mean([r['mean_flowtime'] for r in gruns]))
        gstd = float(np.std([r['mean_flowtime'] for r in gruns]))
        gnom = d48['gen_suites'][gname]['nominal_ca_eft']['mean_flowtime']
        agg['gen_suites'][gname] = dict(
            mean_flowtime=gmean,
            seed_std=gstd,
            gain_vs_nom_pct=float((gnom - gmean) / gnom * 100.0),
        )

    out_json = out_dir / 'tw_norg_tw_ablation_results.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(agg, f, indent=2)

    print('\n=== SUMMARY: No-RG+TW (Task-Only + 6-Hop Tropical Wave) ===', flush=True)
    print(f'Offline 48: {agg["offline_48"]["mean_flowtime"]:.1f} +/- {agg["offline_48"]["seed_std"]:.1f}s ({agg["offline_48"]["gain_vs_nom_pct"]:+.2f}%)')
    print(f'Online  48: {agg["online_48"]["mean_flowtime"]:.1f} +/- {agg["online_48"]["seed_std"]:.1f}s ({agg["online_48"]["gain_vs_nom_pct"]:+.2f}%) | p95={agg["online_48"]["p95_decision_ms"]:.2f}ms')
    print(f'No-RG+TW vs No-RG    (Net TW on clean base): {agg["cis_online_48"]["norg_tw_vs_norg"]}')
    print(f'DualGFlow vs No-RG+TW (Net 4L RGNN on TW)  : {agg["cis_online_48"]["rg_vs_norg_tw"]}')
    for gname, rec in agg['gen_suites'].items():
        print(f'  Zero-Shot {gname:10s}: {rec["mean_flowtime"]:.1f}s ({rec["gain_vs_nom_pct"]:+.2f}%)')
    print(f'Saved: {out_json}', flush=True)


def main() -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(__file__).resolve().parent
    ckpt_dir = out_dir / 'checkpoints_tw_clean'
    run_b1_ov_sensitivity(device, out_dir, ckpt_dir)
    run_b2_norg_tw_ablation(device, out_dir, ckpt_dir)


if __name__ == '__main__':
    main()
