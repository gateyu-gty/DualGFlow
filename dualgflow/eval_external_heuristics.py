"""Evaluate classic external scheduling heuristics on the exact 48-scene main test set and 8 Table 14 suites."""
from __future__ import annotations

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import json
import time
from pathlib import Path
import numpy as np

from dualgflow.topologies import make_flow_scenario
from dualgflow.run_fair_nominal_experiment import eval_strict
from dualgflow.run_paper_benchmark import paired_gain_ci


def make_heuristics():
    rr_state = {'idx': 0}

    def round_robin(env, obs):
        n = int(obs['num_servers'])
        a = rr_state['idx'] % n
        rr_state['idx'] += 1
        return a

    def reset_rr():
        rr_state['idx'] = 0

    return [
        ('compute_eft', 'Compute-EFT (HEFT-Comp)', lambda e, o: int(np.argmin(o['compute_eft'])), None),
        ('earliest_idle', 'Earliest-Idle (Min-Backlog)', lambda e, o: int(np.argmin(e._estimate_server_cpu_backlog())), None),
        ('data_locality', 'Data-Locality (Min-Transfer)', lambda e, o: int(np.argmin(o['nominal_delay'] + 1e-4 * o['compute_eft'])), None),
        ('round_robin', 'Round-Robin', round_robin, reset_rr),
    ]


def eval_suite(sc_list: list[dict]) -> dict:
    out = {}
    for key, _, fn, reset_fn in make_heuristics():
        flows, lats = [], []
        for sc in sc_list:
            if reset_fn is not None:
                reset_fn()
            r = eval_strict(sc, fn, key)
            flows.append(r['mean_flowtime'])
            lats.append(r['p95_decision_ms'])
        out[key] = dict(
            mean_flowtime=float(np.mean(flows)),
            seed_std=0.0,
            per_scene=flows,
            p95_decision_ms=float(np.mean(lats)),
        )
    return out


def main():
    t0 = time.perf_counter()
    topo, ccr, load, ov = 'multi_hop_mesh', 4.5, 0.90, 12.0
    out_dir = Path(__file__).resolve().parent

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

    res_48 = eval_suite(test_48)
    gen_res = {gk: eval_suite(gscs) for gk, gscs in gen_suites.items()}

    # Load main 48-scene results to compute paired CIs against DualGFlow-RG (Online)
    main48 = json.load(open(out_dir / 'tw_clean_48_results.json'))
    nom_48 = main48['online_48']['nominal_ca_eft']['mean_flowtime']
    rg_on_per_scene = main48['online_48']['dualgflow']['per_scene']

    cis_vs_ext = {
        f'rg_vs_{k}': paired_gain_ci(res_48[k]['per_scene'], rg_on_per_scene)
        for k in res_48
    }

    with open(out_dir / 'tw_external_heuristics_results.json', 'w') as f:
        json.dump(dict(main_48=res_48, gen_suites=gen_res, cis_rg_vs_ext=cis_vs_ext), f, indent=2)

    print(f'Completed in {time.perf_counter()-t0:.1f}s\n=== EXTERNAL HEURISTICS ON 48-SCENE MAIN TEST SET ===')
    for k, v in res_48.items():
        gain = (nom_48 - v['mean_flowtime']) / nom_48 * 100.0
        ci = cis_vs_ext[f'rg_vs_{k}']
        print(f"  {k:16s} | {v['mean_flowtime']:8.1f}s ({gain:+6.2f}% vs Nom) | p95={v['p95_decision_ms']:.2f}ms | DualGFlow-RG vs {k}: {ci['gain_pct']:+.2f}% [{ci['ci95_low']:+.2f}%, {ci['ci95_high']:+.2f}%] ({ci['wins']}/{ci['total']})")

    print('\n=== EXTERNAL HEURISTICS ON TABLE 14 ZERO-SHOT SUITES ===')
    for gk, gr in gen_res.items():
        nom_g = main48['gen_suites'][gk]['nominal_ca_eft']['mean_flowtime']
        rg_g = (nom_g - main48['gen_suites'][gk]['dualgflow']['mean_flowtime']) / nom_g * 100.0
        parts = [f"RG={rg_g:+6.2f}%"]
        for k in ('compute_eft', 'earliest_idle', 'data_locality', 'round_robin'):
            g = (nom_g - gr[k]['mean_flowtime']) / nom_g * 100.0
            parts.append(f"{k}={g:+6.2f}%")
        print(f"  {gk:10s} | " + " | ".join(parts))


if __name__ == '__main__':
    main()
