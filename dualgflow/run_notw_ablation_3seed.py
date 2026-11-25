"""3-Seed Standalone No-TW Ablation (4-Layer Edge-Attentive Resource GNN WITHOUT Tropical Wave).

Uses 100% identical hyperparameters, training/validation/test scenarios, and seeds (101, 202, 303)
as run_tw_clean_3seed.py and eval_tw_clean_48.py to isolate the standalone contribution of
Six-Hop Tropical Wave Propagation.
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


def main() -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seeds = (101, 202, 303)
    topo, ccr, load, ov, top_k = 'multi_hop_mesh', 4.5, 0.90, 12.0, 4
    out_dir = Path(__file__).resolve().parent
    ckpt_dir = out_dir / 'checkpoints_tw_clean'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

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
        ckpt_off = ckpt_dir / f'dualgflow_notw_offline_seed{sd}.pt'
        ckpt_on = ckpt_dir / f'dualgflow_notw_online_seed{sd}.pt'
        torch.manual_seed(sd)
        mod = DualGFlowPolicy(
            variant='dualgflow', mode='residual_eft', hidden_dim=64, gnn_layers=4,
            use_nominal_anchor=True, residual_scale=2.0, use_tropical_wave=False,
        ).to(device)

        if ckpt_off.exists() and ckpt_on.exists():
            print(f'[Seed {sd}] Loaded cached No-TW checkpoints.', flush=True)
            m_off = deepcopy(mod)
            m_off.load_state_dict(torch.load(ckpt_off, map_location=device, weights_only=True))
            m_off.eval()
            m_on = deepcopy(mod)
            m_on.load_state_dict(torch.load(ckpt_on, map_location=device, weights_only=True))
            m_on.eval()
        else:
            print(f'\n[Seed {sd}] === Training No-TW (4-Layer Resource GNN, use_tropical_wave=False) ===', flush=True)
            mod, h1 = train_stage1_topk(mod, dataset, val_sc, val_nom, top_k=top_k, epochs=1, lr=8e-5, seed=sd, device=device)
            m_off = deepcopy(mod).eval()
            torch.save(m_off.state_dict(), ckpt_off)

            mod, h2 = train_online_stable_sil(mod, train_sc, val_sc, train_nom, val_nom, seed=sd, top_k=top_k, device=device)
            m_on = mod.eval()
            torch.save(m_on.state_dict(), ckpt_on)
            with open(ckpt_dir / f'history_dualgflow_notw_seed{sd}.json', 'w') as hf:
                json.dump(dict(stage1=h1, stage2=h2), hf, indent=2)

        sd_rec = dict(offline_48={}, online_48={}, gen_suites={})
        with torch.no_grad():
            for st_name, m_eval in (('offline_48', m_off), ('online_48', m_on)):
                flows, lats = [], []
                for sc in test_48:
                    r = eval_strict(
                        sc,
                        lambda e, o, m=m_eval: int(torch.argmax(apply_mask(m.forward_actor(o), make_topk_mask_1d(o['nominal_eft'], k=top_k).to(device))).item()),
                        'no_tw',
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
                        'no_tw',
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

    # Aggregate 3 seeds & compute paired CIs against DualGFlow and Nominal-CA-EFT
    d48 = json.load(open(out_dir / 'tw_clean_48_results.json', encoding='utf-8'))
    nom_ps = d48['online_48']['nominal_ca_eft']['per_scene']
    nom_mean = d48['online_48']['nominal_ca_eft']['mean_flowtime']
    rg_ps = d48['online_48']['dualgflow']['per_scene']

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
            rg_vs_notw=paired_gain_ci(on_ps, rg_ps),
            notw_vs_nominal=paired_gain_ci(nom_ps, on_ps),
            notw_online_vs_offline=paired_gain_ci(off_ps, on_ps),
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

    out_json = out_dir / 'tw_notw_ablation_results.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(agg, f, indent=2)

    print('\n=== SUMMARY: No-TW (4-Layer Resource GNN Only) ===', flush=True)
    print(f'Offline 48: {agg["offline_48"]["mean_flowtime"]:.1f} +/- {agg["offline_48"]["seed_std"]:.1f}s ({agg["offline_48"]["gain_vs_nom_pct"]:+.2f}%)')
    print(f'Online  48: {agg["online_48"]["mean_flowtime"]:.1f} +/- {agg["online_48"]["seed_std"]:.1f}s ({agg["online_48"]["gain_vs_nom_pct"]:+.2f}%) | p95={agg["online_48"]["p95_decision_ms"]:.2f}ms')
    print(f'DualGFlow vs No-TW (Online 48): {agg["cis_online_48"]["rg_vs_notw"]}')
    print(f'No-TW vs Nominal   (Online 48): {agg["cis_online_48"]["notw_vs_nominal"]}')
    for gname, rec in agg['gen_suites'].items():
        print(f'  Zero-Shot {gname:10s}: {rec["mean_flowtime"]:.1f}s ({rec["gain_vs_nom_pct"]:+.2f}%)')
    print(f'Saved: {out_json}', flush=True)


if __name__ == '__main__':
    main()
