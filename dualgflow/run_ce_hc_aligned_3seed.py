"""Aligned 3-Seed Stage-1 Counterfactual + Stage-2 Top-K=4 PPO on Cloud-Edge HC.

Unifies all lessons learned:
1. Distribution alignment: train, val, and test_16 ALL use `workflows=14` under
   high-congestion Cloud-Edge (ccr=4.5, load=0.90, oversubscription=12.0),
   eliminating the 12-vs-14 workflow headroom mismatch.
2. Consistent Top-K=4 action masking across Stage-1 counterfactual distillation,
   Stage-1 validation, Stage-2 PPO rollout/update, and final test evaluation.
3. Two-stage Offline-to-Online DualGFlow paradigm (8 epochs Stage-1 ListNet +
   8 updates stabilized PPO anchored to Stage-1, temp=0.06, kl=0.003).
4. Multi-seed statistical rigor (seeds 101, 202, 303) on 24 unseen N=16 scenes
   and 8 unseen N=32 zero-shot scenes.
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
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dualgflow.topologies import make_flow_scenario
from dualgflow.simulator_flow import FlowWorkflowEnv
from dualgflow.models_rg import DualGFlowPolicy
from dualgflow.batch_rg import collate_obs_batch, forward_actor_batch, forward_critic_batch
from dualgflow.run_paper_benchmark import paired_gain_ci
from dualgflow.run_fair_nominal_experiment import eval_strict, collect_strict_fifo_dataset
from dualgflow.run_mesh_topk_diag import make_topk_mask_1d, apply_mask
from dualgflow.run_mesh_stage1_ppo_4models import train_stage1_topk


def train_stage2_aligned(
    model: DualGFlowPolicy,
    train_sc: list[dict],
    val_sc: list[dict],
    train_nom: list[float],
    val_nom: float,
    seed: int = 101,
    top_k: int = 4,
    temperature: float = 0.06,
    ppo_updates: int = 8,
    eps_per_upd: int = 12,
    batch_size: int = 128,
    lr_a: float = 5e-5,
    lr_c: float = 5e-4,
    kl_coef: float = 0.003,
    ent_coef: float = 0.0005,
    device: str = 'cuda',
) -> tuple[DualGFlowPolicy, list[dict]]:
    model.to(device)
    s1_anchor = deepcopy(model).eval()
    for p in s1_anchor.parameters():
        p.requires_grad_(False)

    opt_a = torch.optim.AdamW(model.actor_parameters(), lr=lr_a, weight_decay=1e-4)
    opt_c = torch.optim.AdamW(model.critic_parameters(), lr=lr_c, weight_decay=1e-4)
    rng = np.random.default_rng(seed)

    def collect_batch(upd_idx: int):
        model.eval()
        start = (upd_idx * eps_per_upd) % len(train_sc)
        idxs = [(start + j) % len(train_sc) for j in range(eps_per_upd)]
        trans, ep_gains = [], []
        all_rets, all_vals = [], []
        with torch.no_grad():
            for idx in idxs:
                sc = train_sc[idx]
                nom_f = train_nom[idx]
                n_wf = max(1, len(sc['workflows']))
                env = FlowWorkflowEnv(sc, reward_scale=(nom_f * n_wf) / 10.0, strict_fifo=True)
                obs = env.reset()
                ep_obs, ep_act, ep_logp, ep_rew, ep_val, ep_prior, ep_mask = [], [], [], [], [], [], []
                while not env.done:
                    logits, val = model.forward_obs(obs)
                    m_cpu = make_topk_mask_1d(obs['nominal_eft'], k=top_k)
                    m_dev = m_cpu.to(device)
                    masked = apply_mask(logits, m_dev)
                    dist = Categorical(logits=masked / temperature)
                    act = dist.sample()
                    anc_logits = apply_mask(s1_anchor.forward_actor(obs), m_dev)
                    prior_lp = F.log_softmax(anc_logits / temperature, dim=0).cpu()
                    next_obs, rew, _, _ = env.step(int(act.item()))
                    ep_obs.append(obs); ep_act.append(int(act.item()))
                    ep_logp.append(float(dist.log_prob(act).item()))
                    ep_rew.append(float(rew)); ep_val.append(float(val.item()))
                    ep_prior.append(prior_lp); ep_mask.append(m_cpu)
                    obs = next_obs
                mf = env.metrics()['mean_flowtime']
                ep_gains.append((nom_f - mf) / nom_f * 100.0)

                T = len(ep_rew)
                adv = np.zeros(T, dtype=np.float32); ret = np.zeros(T, dtype=np.float32)
                gae, nv = 0.0, 0.0
                for t in reversed(range(T)):
                    delta = ep_rew[t] + 0.995 * nv - ep_val[t]
                    gae = delta + 0.995 * 0.95 * gae
                    adv[t] = gae; ret[t] = gae + ep_val[t]; nv = ep_val[t]
                all_rets.extend(ret.tolist()); all_vals.extend(ep_val)
                for t in range(T):
                    trans.append(dict(
                        obs=ep_obs[t], act=ep_act[t], old_logp=ep_logp[t],
                        ret=float(ret[t]), adv=float(adv[t]), prior_lp=ep_prior[t], mask=ep_mask[t],
                    ))
        y_true = np.asarray(all_rets, dtype=np.float64)
        y_pred = np.asarray(all_vals, dtype=np.float64)
        ev = float(1.0 - np.var(y_true - y_pred) / max(np.var(y_true), 1e-8))
        return trans, np.asarray(ep_gains), ev

    def update_critic_b(trans, epochs=4):
        model.train()
        losses = []
        for _ in range(epochs):
            perm = rng.permutation(len(trans))
            for st in range(0, len(perm), batch_size):
                sub = [trans[int(i)] for i in perm[st:st + batch_size]]
                b = collate_obs_batch([x['obs'] for x in sub], device=device, use_nominal_anchor=True)
                vals = forward_critic_batch(model, b)
                tgt = torch.tensor([x['ret'] for x in sub], dtype=torch.float32, device=device)
                loss = F.smooth_l1_loss(vals, tgt)
                opt_c.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.critic_parameters(), 1.0); opt_c.step()
                losses.append(float(loss.item()))
        return float(np.mean(losses))

    for w_i in range(2):
        tr, eg, ev = collect_batch(w_i)
        cl = update_critic_b(tr, epochs=5)
        print(f'  [Warmup {w_i+1}/2] TrainGain: {eg.mean():+.2f}% | Huber: {cl:.4f} | EV: {ev:+.3f}', flush=True)

    def eval_val() -> float:
        model.eval()
        with torch.no_grad():
            return float(np.mean([
                eval_strict(
                    sc,
                    lambda e, o: int(torch.argmax(apply_mask(model.forward_actor(o), make_topk_mask_1d(o['nominal_eft'], k=top_k).to(device))).item()),
                )['mean_flowtime']
                for sc in val_sc
            ]))

    best_val = eval_val()
    best_state = deepcopy(model.state_dict())
    vg0 = (val_nom - best_val) / val_nom * 100.0
    print(f'  [PPO Init] Val: {best_val:.1f}s ({vg0:+.2f}%)', flush=True)
    history = [dict(phase='ppo', step=0, val_flowtime=float(best_val), val_gain_pct=float(vg0))]

    for upd in range(ppo_updates):
        t0 = time.perf_counter()
        tr, eg, ev = collect_batch(upd)
        all_adv = np.array([x['adv'] for x in tr], dtype=np.float32)
        am, astd = float(all_adv.mean()), max(float(all_adv.std()), 1e-6)
        update_critic_b(tr, epochs=4)

        model.train()
        clips, ents = [], []
        for _ in range(3):
            perm = rng.permutation(len(tr))
            for st in range(0, len(perm), batch_size):
                sub = [tr[int(i)] for i in perm[st:st + batch_size]]
                b = collate_obs_batch([x['obs'] for x in sub], device=device, use_nominal_anchor=True)
                raw_logits = forward_actor_batch(model, b)
                b_mask = torch.stack([x['mask'] for x in sub]).to(device)
                scaled = apply_mask(raw_logits, b_mask) / temperature
                dist = Categorical(logits=scaled)
                acts = torch.tensor([x['act'] for x in sub], dtype=torch.long, device=device)
                old_lp = torch.tensor([x['old_logp'] for x in sub], dtype=torch.float32, device=device)
                n_adv = torch.tensor([(x['adv'] - am) / astd for x in sub], dtype=torch.float32, device=device)
                pr_lp = torch.stack([x['prior_lp'] for x in sub]).to(device)

                new_lp = dist.log_prob(acts)
                ratio = torch.exp(new_lp - old_lp)
                ploss = -torch.min(ratio * n_adv, torch.clamp(ratio, 0.85, 1.15) * n_adv).mean()
                cur_lp = F.log_softmax(scaled, dim=1)
                kl = (cur_lp.exp() * (cur_lp - pr_lp) * b_mask).sum(dim=1).mean()
                ent = dist.entropy().mean()
                loss = ploss + kl_coef * kl - ent_coef * ent

                opt_a.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.actor_parameters(), 0.5)
                opt_a.step()

                clips.append(float(((ratio - 1.0).abs() > 0.15).float().mean().item()))
                ents.append(float(ent.item()))

        vf = eval_val()
        vg = (val_nom - vf) / val_nom * 100.0
        tag = ''
        if vf < best_val:
            best_val = vf
            best_state = deepcopy(model.state_dict())
            tag = ' *BEST*'
        print(
            f'  [PPO {upd+1:2d}/{ppo_updates}] ({time.perf_counter()-t0:.1f}s) '
            f'| TrainGain: {eg.mean():+.2f}% | ValGain: {vg:+.2f}% '
            f'| EV: {ev:+.3f} | Clip: {np.mean(clips):.3f} | Ent: {np.mean(ents):.3f}{tag}',
            flush=True,
        )
        history.append(dict(
            phase='ppo', step=upd + 1,
            train_gain_pct=float(eg.mean()), val_flowtime=float(vf), val_gain_pct=float(vg),
            explained_variance=float(ev), clip_fraction=float(np.mean(clips)), entropy=float(np.mean(ents)),
        ))

    model.load_state_dict(best_state)
    model.eval()
    return model, history


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    topo = 'cloud_edge'
    ccr, load, ov = 4.5, 0.90, 12.0
    seeds = [101, 202, 303]
    top_k = 4
    out_dir = Path(__file__).resolve().parent
    ckpt_dir = out_dir / 'checkpoints_ce_hc_aligned_3seed'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f'Running Aligned 3-Seed Stage1+PPO on Cloud-Edge HC (workflows=14) on {device}', flush=True)

    train_sc = [make_flow_scenario(seed=1000 + i, split='train', workflows=14, num_servers=16, load=load, ccr=ccr, topology=topo, oversubscription=ov) for i in range(24)]
    val_sc = [make_flow_scenario(seed=2000 + i, split='validation', workflows=14, num_servers=16, load=load, ccr=ccr, topology=topo, oversubscription=ov) for i in range(12)]
    test_16 = [make_flow_scenario(seed=4000 + i, split='test', workflows=14, num_servers=16, load=load, ccr=ccr, topology=topo, oversubscription=ov) for i in range(24)]
    test_32 = [make_flow_scenario(seed=5000 + i, split='test', workflows=18, num_servers=32, load=load, ccr=ccr, topology=topo, oversubscription=ov) for i in range(8)]

    ds_path = ckpt_dir / 'dataset_ce_hc_24scenes_wf14.pt'
    if ds_path.exists():
        dataset = torch.load(ds_path, weights_only=False)
        print(f'Loaded cached dataset ({len(dataset)} states)', flush=True)
    else:
        print('Collecting 24-scene counterfactual dataset (workflows=14)...', flush=True)
        dataset = collect_strict_fifo_dataset(train_sc, states_per_sc=40, top_k=4, seed=101)
        torch.save(dataset, ds_path)
        print(f'Saved {len(dataset)} counterfactual states', flush=True)

    train_nom = [eval_strict(sc, lambda e, o: int(np.argmin(o['nominal_eft'])), 'nom')['mean_flowtime'] for sc in train_sc]
    val_nom = float(np.mean([eval_strict(sc, lambda e, o: int(np.argmin(o['nominal_eft'])), 'nom')['mean_flowtime'] for sc in val_sc]))

    variants = ('dualgflow', 'no_mp_keep_links', 'global_mlp', 'no_rg')
    trained = {v: [] for v in variants}
    histories = {v: {} for v in variants}

    for sd in seeds:
        print(f'\n================ SEED {sd} ================', flush=True)
        for var in variants:
            ckpt = ckpt_dir / f'{var}_seed{sd}.pt'
            hpath = ckpt_dir / f'history_{var}_seed{sd}.json'
            if ckpt.exists() and hpath.exists():
                m = DualGFlowPolicy(variant=var, mode='residual_eft', hidden_dim=64, gnn_layers=4, use_nominal_anchor=True).to(device)
                m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
                m.eval()
                with open(hpath) as hf:
                    hist = json.load(hf)
                print(f'Loaded cached {ckpt.name}', flush=True)
            else:
                print(f'--- Training {var} (seed={sd}) ---', flush=True)
                torch.manual_seed(sd)
                m = DualGFlowPolicy(variant=var, mode='residual_eft', hidden_dim=64, gnn_layers=4, use_nominal_anchor=True).to(device)
                m, h1 = train_stage1_topk(m, dataset, val_sc, val_nom, top_k=top_k, epochs=8, seed=sd, device=device)
                m, h2 = train_stage2_aligned(m, train_sc, val_sc, train_nom, val_nom, seed=sd, top_k=top_k, ppo_updates=8, device=device)
                hist = dict(stage1=h1, stage2=h2)
                torch.save(m.state_dict(), ckpt)
                with open(hpath, 'w') as hf:
                    json.dump(hist, hf, indent=2)
            trained[var].append(m)
            histories[var][str(sd)] = hist

    def eval_suite_3seed(sc_list: list[dict]) -> dict:
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

        for var in variants:
            seed_runs = []
            for mod in trained[var]:
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

    r16 = eval_suite_3seed(test_16)
    r32 = eval_suite_3seed(test_32)
    cis_16 = {f'{v}_vs_nominal': paired_gain_ci(r16['nominal_ca_eft']['per_scene'], r16[v]['per_scene']) for v in ('dualgflow', 'no_mp_keep_links', 'global_mlp', 'no_rg', 'flow_oracle_eft')}
    cis_32 = {f'{v}_vs_nominal': paired_gain_ci(r32['nominal_ca_eft']['per_scene'], r32[v]['per_scene']) for v in ('dualgflow', 'no_mp_keep_links', 'global_mlp', 'no_rg', 'flow_oracle_eft')}

    with open(out_dir / 'ce_hc_aligned_3seed_results.json', 'w') as f:
        json.dump(dict(seeds=seeds, test_16=r16, test_32=r32, cis_16=cis_16, cis_32=cis_32), f, indent=2)

    fig_dir = out_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})

    # Fig 23: Bars
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    order = [
        ('no_rg', 'DualGFlow (No-RG)', '#8c564b'),
        ('global_mlp', 'Global-MLP', '#d62728'),
        ('no_mp_keep_links', 'No-MP-Keep-Links', '#ff7f0e'),
        ('nominal_ca_eft', 'Nominal-CA-EFT', '#9467bd'),
        ('dualgflow', 'DualGFlow-RG (Ours)', '#1f77b4'),
    ]
    for ax, r_data, title in [
        (axes[0], r16, '(a) Cloud-Edge HC Aligned N=16 (24 Scenes, 3 Seeds)'),
        (axes[1], r32, '(b) Zero-Shot Scale N=32 (8 Scenes, 3 Seeds)'),
    ]:
        labels = [o[1] for o in order]
        means = [r_data[o[0]]['mean_flowtime'] / 1000.0 for o in order]
        stds = [r_data[o[0]]['seed_std'] / 1000.0 for o in order]
        colors = [o[2] for o in order]
        y = np.arange(len(labels))
        ax.barh(y, means, xerr=stds, capsize=4, height=0.58, color=colors, alpha=0.9)
        ax.set_yticks(y); ax.set_yticklabels(labels)
        ax.set_xlim(max(0.0, min(means) - 1.5), max(m + s for m, s in zip(means, stds)) + 1.2)
        ax.set_xlabel('Mean Workflow Flowtime (×10³ s)'); ax.set_title(title); ax.grid(axis='x', alpha=0.25)
        for yi, m, s, (k, _, _) in zip(y, means, stds, order):
            ax.text(m + s + 0.1, yi, f'{m:.2f}k', va='center', fontsize=9,
                    fontweight='bold' if k == 'dualgflow' else 'normal')
    plt.tight_layout()
    fig.savefig(fig_dir / 'fig23_ce_hc_aligned_3seed_bars.png')
    plt.close(fig)

    # Fig 24: 3-seed Learning Curves
    fig_lc, axes_lc = plt.subplots(1, 2, figsize=(12.0, 4.5))
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
        axes_lc[0].plot(xs, ys, label=label, color=col, linestyle=ls, linewidth=2.0, marker='o', markersize=3.2)
        axes_lc[0].fill_between(xs, ys - es, ys + es, color=col, alpha=0.14)

        exs = sorted(step_ev.keys())
        eys = np.array([np.mean(step_ev[x]) for x in exs])
        ees = np.array([np.std(step_ev[x]) for x in exs])
        axes_lc[1].plot(exs, eys, label=label, color=col, linestyle=ls, linewidth=2.0, marker='o', markersize=3.2)
        axes_lc[1].fill_between(exs, eys - ees, eys + ees, color=col, alpha=0.14)

    axes_lc[0].axvline(8, color='gray', linestyle=':', linewidth=1.4, label='Stage 1 → PPO')
    axes_lc[0].axhline(0.0, color='#9467bd', linestyle='--', linewidth=1.4, label='Nominal-CA-EFT (0%)')
    axes_lc[0].set_title('(a) 3-Seed Stage-1 + PPO Val Gain (%)')
    axes_lc[0].set_xlabel('Step (0-8: Stage 1, 9-16: PPO)'); axes_lc[0].set_ylabel('Val Gain vs Nominal (%)')
    axes_lc[0].grid(alpha=0.25); axes_lc[0].legend(fontsize=8.2)
    axes_lc[1].set_title('(b) 3-Seed Stage-2 Critic Explained Variance')
    axes_lc[1].set_xlabel('PPO Update'); axes_lc[1].set_ylabel('Explained Variance')
    axes_lc[1].grid(alpha=0.25); axes_lc[1].legend(fontsize=8.5)
    plt.tight_layout()
    fig_lc.savefig(fig_dir / 'fig24_ce_hc_aligned_3seed_curves.png')
    plt.close(fig_lc)

    print('\n=== ALIGNED 3-SEED CLOUD-EDGE HC N=16 (24 SCENES) ===')
    b16 = r16['nominal_ca_eft']['mean_flowtime']
    for k, _, _ in order + [('flow_oracle_eft', 'Flow-Oracle-EFT', '#2ca02c')]:
        v = r16[k]
        print(f"  {k:18s}: {v['mean_flowtime']:8.1f}s ±{v['seed_std']:5.1f} ({(b16 - v['mean_flowtime']) / b16 * 100:+5.2f}%) | {v['p95_decision_ms']:4.2f}ms")

    print('\n=== ALIGNED 3-SEED CLOUD-EDGE HC ZERO-SHOT N=32 (8 SCENES) ===')
    b32 = r32['nominal_ca_eft']['mean_flowtime']
    for k, _, _ in order + [('flow_oracle_eft', 'Flow-Oracle-EFT', '#2ca02c')]:
        v = r32[k]
        print(f"  {k:18s}: {v['mean_flowtime']:8.1f}s ±{v['seed_std']:5.1f} ({(b32 - v['mean_flowtime']) / b32 * 100:+5.2f}%) | {v['p95_decision_ms']:4.2f}ms")

    print('\n--- 95% PAIRED CIs (N=16, 24 SCENES) ---')
    for ck, cv in cis_16.items():
        print(f"  {ck:24s}: {cv['gain_pct']:+6.2f}% [{cv['ci95_low']:+.2f}%, {cv['ci95_high']:+.2f}%] w={cv['wins']}/{cv['total']}")


if __name__ == '__main__':
    main()
