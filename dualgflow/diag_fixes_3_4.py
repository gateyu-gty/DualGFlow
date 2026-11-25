"""Fast diagnostic for Expert Review Points #3 (M=N collision-free Tropical Wave) and #4 (exact timing & dispatch count)."""
from __future__ import annotations

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import json
import time
from pathlib import Path
import numpy as np
import torch

from dualgflow.topologies import make_flow_scenario
from dualgflow.simulator_flow import FlowWorkflowEnv
from dualgflow.models_rg import DualGFlowPolicy, compute_tropical_wave_delays, transform_obs
from dualgflow.run_mesh_topk_diag import make_topk_mask_1d, apply_mask
from dualgflow.run_fair_nominal_experiment import eval_strict


def eval_with_dynamic_M(mod: DualGFlowPolicy, sc: dict, M_val: int, device: str) -> float:
    """Evaluate DualGFlow with M = M_val wave slots (zero slot collisions when M >= num_servers)."""
    def policy_fn(env, obs):
        t_obs = transform_obs(obs, device=device, use_nominal_anchor=True)
        n_srv = t_obs['num_servers']
        joint = mod._encode_joint(t_obs, mod.task_encoder, mod.task_to_srv, mod.res_encoder)
        cand_h = mod.candidate_proj(joint)
        global_ctx = torch.cat([cand_h.mean(dim=0), cand_h.max(dim=0).values], dim=-1)
        raw_logits = mod.scorer(torch.cat([cand_h, cand_h * global_ctx[:mod.hidden_dim].unsqueeze(0).expand(n_srv, -1)], dim=-1)).squeeze(-1)
        _, d_cong = compute_tropical_wave_delays(
            t_obs['resource_x'], t_obs['resource_edges'], t_obs['resource_edge_attr'], L=6, M=M_val
        )
        exec_t = torch.expm1(t_obs['candidate_features'][:, 0] * 8.0)
        cpu_av = torch.expm1(t_obs['candidate_features'][:, 1] * 8.0)
        nom_d = torch.expm1(t_obs['pair_x'][:, 3] * 8.0)
        wave_eft = torch.maximum(cpu_av, torch.maximum(nom_d, d_cong[:n_srv])) + exec_t
        w_min, w_med = wave_eft.min(), wave_eft.median()
        rel_anchor = torch.clamp((wave_eft - w_min) / (w_med - w_min).clamp_min(1e-3), 0.0, 5.0)
        logits = -2.0 * rel_anchor + mod.residual_scale * torch.tanh(raw_logits)
        return int(torch.argmax(apply_mask(logits, make_topk_mask_1d(obs['nominal_eft'], k=4).to(device))).item())
    return eval_strict(sc, policy_fn, f'M={M_val}')['mean_flowtime']


def main() -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(__file__).resolve().parent
    ckpt_dir = out_dir / 'checkpoints_tw_clean'
    d48 = json.load(open(out_dir / 'tw_clean_48_results.json', encoding='utf-8'))

    # --- 1. Exact Timing & Dispatch Count Breakdown (on 8 unseen 48-scene test cases) ---
    print('=== [Fix #4] Exact Timing & Dispatch Count Breakdown ===', flush=True)
    mod = DualGFlowPolicy(
        variant='dualgflow', mode='residual_eft', hidden_dim=64, gnn_layers=4,
        use_nominal_anchor=True, residual_scale=2.0, use_tropical_wave=True,
    ).to(device)
    mod.load_state_dict(torch.load(ckpt_dir / 'dualgflow_online_seed101.pt', map_location=device, weights_only=True))
    mod.eval()

    test_sub = [make_flow_scenario(seed=4000 + i, split='test', workflows=14, num_servers=16, load=0.90, ccr=4.5, topology='multi_hop_mesh', oversubscription=12.0) for i in range(8)]
    dispatches, fwd_ms_all, obs_ms_all, step_total_ms_all = [], [], [], []
    with torch.no_grad():
        for sc in test_sub:
            env = FlowWorkflowEnv(sc, strict_fifo=True)
            t0 = time.perf_counter()
            obs = env.reset()
            obs_ms_all.append((time.perf_counter() - t0) * 1000.0)
            cnt = 0
            while not env.done:
                tf0 = time.perf_counter()
                logits = mod.forward_actor(obs)
                act = int(torch.argmax(apply_mask(logits, make_topk_mask_1d(obs['nominal_eft'], k=4).to(device))).item())
                if device == 'cuda':
                    torch.cuda.synchronize()
                tf1 = time.perf_counter()
                fwd_ms = (tf1 - tf0) * 1000.0
                fwd_ms_all.append(fwd_ms)

                ts0 = time.perf_counter()
                obs, _, _, _ = env.step(act)
                ts1 = time.perf_counter()
                obs_ms = (ts1 - ts0) * 1000.0
                obs_ms_all.append(obs_ms)
                step_total_ms_all.append(fwd_ms + obs_ms)
                cnt += 1
            dispatches.append(cnt)

    mean_disp = float(np.mean(dispatches))
    fwd_mean, fwd_p95 = float(np.mean(fwd_ms_all)), float(np.percentile(fwd_ms_all, 95))
    obs_mean, obs_p95 = float(np.mean(obs_ms_all)), float(np.percentile(obs_ms_all, 95))
    tot_mean, tot_p95 = float(np.mean(step_total_ms_all)), float(np.percentile(step_total_ms_all, 95))
    cum_fwd_s = mean_disp * fwd_mean / 1000.0
    cum_tot_s = mean_disp * tot_mean / 1000.0
    print(f'  Mean dispatches per 14-workflow episode: {mean_disp:.1f} (min={min(dispatches)}, max={max(dispatches)})')
    print(f'  Policy forward pass : mean={fwd_mean:.2f}ms | p95={fwd_p95:.2f}ms | cumulative/episode={cum_fwd_s:.2f}s')
    print(f'  Sim step + obs build: mean={obs_mean:.2f}ms | p95={obs_p95:.2f}ms')
    print(f'  End-to-end step     : mean={tot_mean:.2f}ms | p95={tot_p95:.2f}ms | cumulative/episode={cum_tot_s:.2f}s')

    # --- 2. Collision-Free Tropical Wave (M = num_servers) on Scale N=24 and N=32 ---
    print('\n=== [Fix #3] Collision-Free Tropical Wave (M=N) on Scale N=24 & N=32 (3 Seeds) ===', flush=True)
    mods = []
    for sd in (101, 202, 303):
        m = DualGFlowPolicy(
            variant='dualgflow', mode='residual_eft', hidden_dim=64, gnn_layers=4,
            use_nominal_anchor=True, residual_scale=2.0, use_tropical_wave=True,
        ).to(device)
        m.load_state_dict(torch.load(ckpt_dir / f'dualgflow_online_seed{sd}.pt', map_location=device, weights_only=True))
        m.eval()
        mods.append(m)

    scale_results = {}
    with torch.no_grad():
        for sname, n_srv, n_wf, base_sd in [('scale_n24', 24, 16, 5100), ('scale_n32', 32, 18, 5000)]:
            scs = [make_flow_scenario(seed=base_sd + i, split='test', workflows=n_wf, num_servers=n_srv, load=0.90, ccr=4.5, topology='multi_hop_mesh', oversubscription=12.0) for i in range(8)]
            nom_m = d48['gen_suites'][sname]['nominal_ca_eft']['mean_flowtime']
            m16_m = d48['gen_suites'][sname]['dualgflow']['mean_flowtime']
            mN_vals = []
            for m in mods:
                mN_vals.append(float(np.mean([eval_with_dynamic_M(m, sc, n_srv, device) for sc in scs])))
            mN_mean, mN_std = float(np.mean(mN_vals)), float(np.std(mN_vals))
            g16 = (nom_m - m16_m) / nom_m * 100.0
            gN = (nom_m - mN_mean) / nom_m * 100.0
            scale_results[sname] = dict(nom=nom_m, m16_mean=m16_m, m16_gain=g16, mN_mean=mN_mean, mN_std=mN_std, mN_gain=gN)
            print(f'  {sname}: Nom={nom_m:.1f}s | M=16: {m16_m:.1f}s ({g16:+.2f}%) | M={n_srv} (Zero-Collision): {mN_mean:.1f} +/- {mN_std:.1f}s ({gN:+.2f}%)', flush=True)

    diag_out = out_dir / 'tw_diag_fixes_3_4.json'
    with open(diag_out, 'w', encoding='utf-8') as f:
        json.dump(dict(
            timing=dict(mean_dispatches=mean_disp, fwd_mean_ms=fwd_mean, fwd_p95_ms=fwd_p95, obs_mean_ms=obs_mean, obs_p95_ms=obs_p95, tot_mean_ms=tot_mean, tot_p95_ms=tot_p95, cum_fwd_s=cum_fwd_s, cum_tot_s=cum_tot_s),
            scale_dynamic_M=scale_results,
        ), f, indent=2)
    print(f'Saved: {diag_out}', flush=True)


if __name__ == '__main__':
    main()
