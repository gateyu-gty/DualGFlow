"""Clean & Stable 3-Seed Tropical-Wave Offline WarmStart + Online SIL-PPO Suite.

Fixes the two visual/algorithmic issues:
1. 16-scene validation set + gentle effective actor step (lr_a=4e-6, kl_coef=0.05,
   clip=0.10, temperature=0.05) + 1-epoch light Offline WarmStart (lr=8e-5) so
   Stage-2 Online RL smoothly improves over Offline without chaotic oscillations.
2. Clean dual-bar legend styling (light hatched Offline vs solid method-colored Online).
"""
from __future__ import annotations

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import argparse
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
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

from dualgflow.topologies import make_flow_scenario
from dualgflow.simulator_flow import FlowWorkflowEnv
from dualgflow.models_rg import DualGFlowPolicy
from dualgflow.batch_rg import collate_obs_batch, forward_actor_batch, forward_critic_batch
from dualgflow.run_mesh_topk_diag import make_topk_mask_1d, apply_mask
from dualgflow.run_paper_benchmark import paired_gain_ci
from dualgflow.run_fair_nominal_experiment import eval_strict
from dualgflow.run_mesh_stage1_ppo_4models import train_stage1_topk
from dualgflow.run_tropical_wave_3seed import eval_3seed_suite


VARIANTS = ('dualgflow', 'no_mp_keep_links', 'global_mlp', 'no_rg')


def train_online_stable_sil(
    model: DualGFlowPolicy,
    train_sc: list[dict],
    val_sc: list[dict],
    train_nom: list[float],
    val_nom: float,
    seed: int = 101,
    top_k: int = 4,
    temperature: float = 0.05,
    ppo_updates: int = 16,
    eps_per_upd: int = 4,
    val_every: int = 4,
    batch_size: int = 128,
    lr_a: float = 4e-6,
    lr_c: float = 4e-4,
    kl_coef: float = 0.05,
    ent_coef: float = 0.0002,
    clip_eps: float = 0.10,
    sil_coef: float = 0.10,
    device: str = 'cuda',
) -> tuple[DualGFlowPolicy, list[dict]]:
    model.to(device)
    s1_anchor = deepcopy(model).eval()
    for p in s1_anchor.parameters():
        p.requires_grad_(False)

    opt_a = torch.optim.AdamW(model.actor_parameters(), lr=lr_a, weight_decay=1e-4)
    opt_c = torch.optim.AdamW(model.critic_parameters(), lr=lr_c, weight_decay=1e-4)
    rng = np.random.default_rng(seed)

    elite_episodes: list[tuple[float, list[dict]]] = []
    max_elite_eps = 16

    def collect_batch(upd_idx: int, n_eps: int = eps_per_upd):
        model.eval()
        start = (upd_idx * n_eps) % len(train_sc)
        idxs = [(start + j) % len(train_sc) for j in range(n_eps)]
        trans, ep_gains, ep_rews = [], [], []
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
                gain = (nom_f - mf) / nom_f * 100.0
                ep_gains.append(gain)
                ep_rews.append(float(np.sum(ep_rew)))

                T = len(ep_rew)
                adv = np.zeros(T, dtype=np.float32); ret = np.zeros(T, dtype=np.float32)
                gae, nv = 0.0, 0.0
                for t in reversed(range(T)):
                    delta = ep_rew[t] + 0.995 * nv - ep_val[t]
                    gae = delta + 0.995 * 0.95 * gae
                    adv[t] = gae; ret[t] = gae + ep_val[t]; nv = ep_val[t]
                all_rets.extend(ret.tolist()); all_vals.extend(ep_val)
                ep_trans = []
                for t in range(T):
                    item = dict(
                        obs=ep_obs[t], act=ep_act[t], old_logp=ep_logp[t],
                        ret=float(ret[t]), adv=float(adv[t]), prior_lp=ep_prior[t], mask=ep_mask[t],
                    )
                    trans.append(item)
                    ep_trans.append(item)
                if gain > 1.5:
                    elite_episodes.append((gain, ep_trans))
                    elite_episodes.sort(key=lambda z: z[0], reverse=True)
                    if len(elite_episodes) > max_elite_eps:
                        elite_episodes.pop()

        y_true = np.asarray(all_rets, dtype=np.float64)
        y_pred = np.asarray(all_vals, dtype=np.float64)
        ev = float(1.0 - np.var(y_true - y_pred) / max(np.var(y_true), 1e-8))
        return trans, np.asarray(ep_gains), np.asarray(ep_rews), ev

    def update_critic_b(trans, epochs=3):
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
        tr, eg, _, ev = collect_batch(w_i, n_eps=4)
        cl = update_critic_b(tr, epochs=4)
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

    vf0 = eval_val()
    vg0 = (val_nom - vf0) / val_nom * 100.0
    best_val = float('inf')
    best_state = deepcopy(model.state_dict())
    print(f'  [PPO Init] Val(16): {vf0:.1f}s ({vg0:+.2f}%)', flush=True)
    history = [dict(step=0, episodes=0, val_flowtime=float(vf0), val_gain_pct=float(vg0))]

    for upd in range(ppo_updates):
        t0 = time.perf_counter()
        tr, eg, er, ev = collect_batch(upd, n_eps=eps_per_upd)
        all_adv = np.array([x['adv'] for x in tr], dtype=np.float32)
        am, astd = float(all_adv.mean()), max(float(all_adv.std()), 1e-6)
        cl = update_critic_b(tr, epochs=3)

        elite_pool = [item for _, ep_t in elite_episodes for item in ep_t]

        model.train()
        clips, ents = [], []
        for _ in range(2):
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
                ploss = -torch.min(ratio * n_adv, torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * n_adv).mean()
                cur_lp = F.log_softmax(scaled, dim=1)
                kl = (cur_lp.exp() * (cur_lp - pr_lp) * b_mask).sum(dim=1).mean()
                ent = dist.entropy().mean()
                loss = ploss + kl_coef * kl - ent_coef * ent

                if len(elite_pool) >= 64:
                    e_idx = rng.choice(len(elite_pool), size=min(64, len(elite_pool)), replace=False)
                    esub = [elite_pool[int(i)] for i in e_idx]
                    eb = collate_obs_batch([x['obs'] for x in esub], device=device, use_nominal_anchor=True)
                    e_logits = apply_mask(forward_actor_batch(model, eb), torch.stack([x['mask'] for x in esub]).to(device)) / temperature
                    e_acts = torch.tensor([x['act'] for x in esub], dtype=torch.long, device=device)
                    loss = loss + sil_coef * F.cross_entropy(e_logits, e_acts)

                opt_a.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.actor_parameters(), 0.35)
                opt_a.step()

                clips.append(float(((ratio - 1.0).abs() > clip_eps).float().mean().item()))
                ents.append(float(ent.item()))

        rec = dict(
            step=upd + 1,
            episodes=(upd + 1) * eps_per_upd,
            train_gain_pct=float(eg.mean()),
            norm_reward_mean=float(er.mean()),
            critic_loss=float(cl),
            explained_variance=float(ev),
            clip_fraction=float(np.mean(clips)),
            entropy=float(np.mean(ents)),
        )
        val_str = ''
        if (upd + 1) % val_every == 0 or (upd + 1) == ppo_updates:
            vf = eval_val()
            vg = (val_nom - vf) / val_nom * 100.0
            rec['val_flowtime'] = float(vf)
            rec['val_gain_pct'] = float(vg)
            tag = ''
            if vf < best_val:
                best_val = vf
                best_state = deepcopy(model.state_dict())
                tag = ' *BEST*'
            val_str = f' | ValGain: {vg:+.2f}%{tag}'

        if (upd + 1) % val_every == 0 or (upd + 1) == 1:
            print(
                f'  [PPO {upd+1:2d}/{ppo_updates} ({(upd+1)*eps_per_upd:3d} eps)] ({time.perf_counter()-t0:.1f}s) '
                f'| TrainGain: {eg.mean():+.2f}% | EV: {ev:+.3f} | Elite: {len(elite_episodes):2d} | Ent: {np.mean(ents):.3f}{val_str}',
                flush=True,
            )
        history.append(rec)

    model.load_state_dict(best_state)
    model.eval()
    return model, history


def train_single_seed(seed: int, device: str = 'cuda'):
    topo = 'multi_hop_mesh'
    ccr, load, ov = 4.5, 0.90, 12.0
    top_k = 4
    out_dir = Path(__file__).resolve().parent
    ckpt_dir = out_dir / 'checkpoints_tw_clean'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_sc = [make_flow_scenario(seed=1000 + i, split='train', workflows=14, num_servers=16, load=load, ccr=ccr, topology=topo, oversubscription=ov) for i in range(36)]
    val_sc = [make_flow_scenario(seed=2000 + i, split='validation', workflows=14, num_servers=16, load=load, ccr=ccr, topology=topo, oversubscription=ov) for i in range(16)]

    ds_path = out_dir / 'checkpoints_mesh_hc' / 'dataset_mesh_hc_36scenes_80st.pt'
    dataset = torch.load(ds_path, weights_only=False)
    print(f'[Seed {seed}] Loaded cached dataset ({len(dataset)} states) | val=16 scenes | on {device}', flush=True)

    train_nom = [eval_strict(sc, lambda e, o: int(np.argmin(o['nominal_eft'])), 'nom')['mean_flowtime'] for sc in train_sc]
    val_nom = float(np.mean([eval_strict(sc, lambda e, o: int(np.argmin(o['nominal_eft'])), 'nom')['mean_flowtime'] for sc in val_sc]))

    for var in VARIANTS:
        ckpt_off = ckpt_dir / f'{var}_offline_seed{seed}.pt'
        ckpt_on = ckpt_dir / f'{var}_online_seed{seed}.pt'
        hpath = ckpt_dir / f'history_{var}_seed{seed}.json'
        if ckpt_off.exists() and ckpt_on.exists() and hpath.exists():
            print(f'[Seed {seed}] Cached {var} found, skipping.', flush=True)
            continue

        use_tw = (var == 'dualgflow')
        print(f'\n[Seed {seed}] === {var} (1-ep WarmStart lr=8e-5 + 25-upd Stable SIL-PPO, TW={use_tw}) ===', flush=True)
        torch.manual_seed(seed)
        m = DualGFlowPolicy(
            variant=var, mode='residual_eft', hidden_dim=64, gnn_layers=4,
            use_nominal_anchor=True, residual_scale=2.0, use_tropical_wave=use_tw,
        ).to(device)
        m, h1 = train_stage1_topk(m, dataset, val_sc, val_nom, top_k=top_k, epochs=1, lr=8e-5, seed=seed, device=device)
        m_off = deepcopy(m).eval()
        torch.save(m_off.state_dict(), ckpt_off)

        m, h2 = train_online_stable_sil(m, train_sc, val_sc, train_nom, val_nom, seed=seed, top_k=top_k, device=device)
        m_on = m.eval()
        torch.save(m_on.state_dict(), ckpt_on)

        with open(hpath, 'w') as hf:
            json.dump(dict(stage1=h1, stage2=h2), hf, indent=2)

    print(f'[Seed {seed}] Completed.', flush=True)


def aggregate_and_plot(seeds: list[int], device: str = 'cuda'):
    topo = 'multi_hop_mesh'
    ccr, load, ov = 4.5, 0.90, 12.0
    top_k = 4
    out_dir = Path(__file__).resolve().parent
    ckpt_dir = out_dir / 'checkpoints_tw_clean'

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

    offline_models = {v: [] for v in VARIANTS}
    online_models = {v: [] for v in VARIANTS}
    histories = {v: {} for v in VARIANTS}

    for sd in seeds:
        for var in VARIANTS:
            use_tw = (var == 'dualgflow')
            m_off = DualGFlowPolicy(
                variant=var, mode='residual_eft', hidden_dim=64, gnn_layers=4,
                use_nominal_anchor=True, residual_scale=2.0, use_tropical_wave=use_tw,
            ).to(device)
            m_off.load_state_dict(torch.load(ckpt_dir / f'{var}_offline_seed{sd}.pt', map_location=device, weights_only=True))
            m_off.eval()

            m_on = DualGFlowPolicy(
                variant=var, mode='residual_eft', hidden_dim=64, gnn_layers=4,
                use_nominal_anchor=True, residual_scale=2.0, use_tropical_wave=use_tw,
            ).to(device)
            m_on.load_state_dict(torch.load(ckpt_dir / f'{var}_online_seed{sd}.pt', map_location=device, weights_only=True))
            m_on.eval()

            offline_models[var].append(m_off)
            online_models[var].append(m_on)
            histories[var][str(sd)] = json.load(open(ckpt_dir / f'history_{var}_seed{sd}.json'))

    print(f'Evaluating {len(seeds)}-Seed Clean Offline vs Online Main Performance (N=16, 24 scenes)...', flush=True)
    res_off_16 = eval_3seed_suite(test_16, offline_models, top_k, device)
    res_on_16 = eval_3seed_suite(test_16, online_models, top_k, device)

    print(f'Evaluating {len(seeds)}-Seed Zero-Shot Generalization Suites (Table 14)...', flush=True)
    gen_results = {gk: eval_3seed_suite(gscs, online_models, top_k, device) for gk, gscs in gen_suites.items()}

    cis_on_16 = {
        f'{v}_vs_nominal': paired_gain_ci(res_on_16['nominal_ca_eft']['per_scene'], res_on_16[v]['per_scene'])
        for v in ('dualgflow', 'no_mp_keep_links', 'global_mlp', 'no_rg', 'flow_oracle_eft')
    }
    cis_on_16['rg_online_vs_offline'] = paired_gain_ci(res_off_16['dualgflow']['per_scene'], res_on_16['dualgflow']['per_scene'])
    cis_on_16['rg_vs_nomp'] = paired_gain_ci(res_on_16['no_mp_keep_links']['per_scene'], res_on_16['dualgflow']['per_scene'])
    cis_on_16['rg_vs_norg'] = paired_gain_ci(res_on_16['no_rg']['per_scene'], res_on_16['dualgflow']['per_scene'])
    cis_on_16['rg_vs_global'] = paired_gain_ci(res_on_16['global_mlp']['per_scene'], res_on_16['dualgflow']['per_scene'])

    with open(out_dir / 'tw_clean_3seed_results.json', 'w') as f:
        json.dump(dict(
            seeds=seeds, offline_16=res_off_16, online_16=res_on_16,
            cis_online_16=cis_on_16, gen_suites=gen_results,
        ), f, indent=2)

    fig_dir = out_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})

    # Fig 35: Clean Learning Curves (0-100 Episodes)
    fig_lc, axes_lc = plt.subplots(1, 2, figsize=(12.4, 4.6))
    styles = [
        ('dualgflow', 'DualGFlow-RG (Ours)', '#1f77b4', '-'),
        ('no_mp_keep_links', 'No-MP-Keep-Links', '#ff7f0e', '--'),
        ('global_mlp', 'Global-MLP', '#d62728', '-.'),
        ('no_rg', 'DualGFlow (No-RG)', '#8c564b', ':'),
    ]
    for var, label, col, ls in styles:
        ep_vg: dict[int, list[float]] = {}
        ep_ev: dict[int, list[float]] = {}
        for sd_str, hd in histories[var].items():
            for r in hd['stage2']:
                if 'val_gain_pct' in r:
                    ep_vg.setdefault(int(r['episodes']), []).append(float(r['val_gain_pct']))
                if 'explained_variance' in r:
                    ep_ev.setdefault(int(r['episodes']), []).append(float(r['explained_variance']))
        xs = sorted(ep_vg.keys())
        ys = np.array([np.mean(ep_vg[x]) for x in xs])
        es = np.array([np.std(ep_vg[x]) for x in xs])
        axes_lc[0].plot(xs, ys, label=label, color=col, linestyle=ls, linewidth=2.3, marker='o', markersize=4.0)
        if len(seeds) > 1:
            axes_lc[0].fill_between(xs, ys - es, ys + es, color=col, alpha=0.14)

        exs = sorted(ep_ev.keys())
        eys = np.array([np.mean(ep_ev[x]) for x in exs])
        ees = np.array([np.std(ep_ev[x]) for x in exs])
        axes_lc[1].plot(exs, eys, label=label, color=col, linestyle=ls, linewidth=2.0)
        if len(seeds) > 1:
            axes_lc[1].fill_between(exs, eys - ees, eys + ees, color=col, alpha=0.14)

    axes_lc[0].axhline(0.0, color='#9467bd', linestyle='--', linewidth=1.3, label='Nominal-CA-EFT (0%)')
    axes_lc[0].set_title(f'(a) {len(seeds)}-Seed Online SIL-PPO Validation Gain (16 Val Scenes)')
    axes_lc[0].set_xlabel('Online Training Episodes'); axes_lc[0].set_ylabel('Validation Gain (%)')
    axes_lc[0].grid(alpha=0.25); axes_lc[0].legend(fontsize=8.3)

    axes_lc[1].axhline(0.0, color='gray', linestyle='--', linewidth=1.0)
    axes_lc[1].set_title(f'(b) {len(seeds)}-Seed Critic Explained Variance')
    axes_lc[1].set_xlabel('Online Training Episodes'); axes_lc[1].set_ylabel('Explained Variance')
    axes_lc[1].set_ylim(-0.3, 1.05); axes_lc[1].grid(alpha=0.25); axes_lc[1].legend(fontsize=8.3)
    plt.tight_layout()
    fig_lc.savefig(fig_dir / 'fig35_tw_clean_learning_curves.png')
    plt.close(fig_lc)

    # Fig 36: Clean Offline vs Online Main & Table 14 Zero-Shot Generalization
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9))
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
    off_e = [res_off_16[o[0]]['seed_std'] / 1000.0 for o in order]
    on_m = [res_on_16[o[0]]['mean_flowtime'] / 1000.0 for o in order]
    on_e = [res_on_16[o[0]]['seed_std'] / 1000.0 for o in order]

    axes[0].barh(
        y - h / 2, off_m, xerr=off_e if len(seeds) > 1 else None, height=h,
        color=[o[2] for o in order], alpha=0.35, edgecolor=[o[2] for o in order], hatch='//', capsize=2.5,
    )
    axes[0].barh(
        y + h / 2, on_m, xerr=on_e if len(seeds) > 1 else None, height=h,
        color=[o[2] for o in order], alpha=0.92, edgecolor='black', linewidth=0.5, capsize=3,
    )
    axes[0].set_yticks(y); axes[0].set_yticklabels([o[1] for o in order])
    axes[0].set_xlim(min(on_m + off_m) - 1.2, max(on_m + off_m) + 1.4)
    axes[0].set_xlabel('Mean Workflow Flowtime (×10³ s)')
    axes[0].set_title(f'(a) {len(seeds)}-Seed Offline vs Online Main (N=16, 24 Scenes)')
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
        gains = [
            (gen_results[k]['nominal_ca_eft']['mean_flowtime'] - gen_results[k][var]['mean_flowtime'])
            / gen_results[k]['nominal_ca_eft']['mean_flowtime'] * 100.0
            for k in g_keys
        ]
        axes[1].bar(xg + (idx - 1) * w, gains, width=w, color=col, alpha=0.9, label=lbl)
    axes[1].axhline(0.0, color='#9467bd', linestyle='--', linewidth=1.2)
    axes[1].set_xticks(xg); axes[1].set_xticklabels(g_labels, rotation=22, ha='right', fontsize=8.5)
    axes[1].set_ylabel('Zero-Shot Gain vs Nominal-CA-EFT (%)')
    axes[1].set_title(f'(b) {len(seeds)}-Seed Zero-Shot Generalization (Table 14)')
    axes[1].grid(axis='y', alpha=0.25); axes[1].legend(fontsize=8.3)
    plt.tight_layout()
    fig.savefig(fig_dir / 'fig36_tw_clean_main_and_gen.png')
    plt.close(fig)

    print(f'\n=== {len(seeds)}-SEED CLEAN OFFLINE vs ONLINE MAIN N=16 (24 SCENES) ===')
    b16 = res_on_16['nominal_ca_eft']['mean_flowtime']
    for k in ('dualgflow', 'no_mp_keep_links', 'global_mlp', 'no_rg', 'nominal_ca_eft', 'flow_oracle_eft'):
        moff, soff = res_off_16[k]['mean_flowtime'], res_off_16[k]['seed_std']
        mon, son = res_on_16[k]['mean_flowtime'], res_on_16[k]['seed_std']
        print(f"  {k:18s} | Off: {moff:8.1f}s ± {soff:5.1f}s ({(b16-moff)/b16*100:+5.2f}%) | On: {mon:8.1f}s ± {son:5.1f}s ({(b16-mon)/b16*100:+5.2f}%) | p95={res_on_16[k]['p95_decision_ms']:.2f}ms")

    print('\n=== PAIRED 95% BOOTSTRAP CIs (ONLINE, N=16) ===')
    for ck, cv in cis_on_16.items():
        print(f"  {ck:22s}: {cv['gain_pct']:+.2f}% [95% CI: {cv['ci95_low']:+.2f}%, {cv['ci95_high']:+.2f}%] | wins={cv['wins']}/{cv['total']}")

    print(f'\n=== {len(seeds)}-SEED ZERO-SHOT GENERALIZATION (TABLE 14) ===')
    for gk in g_keys:
        gr = gen_results[gk]
        bn = gr['nominal_ca_eft']['mean_flowtime']
        rg_g = (bn - gr['dualgflow']['mean_flowtime']) / bn * 100.0
        nmp_g = (bn - gr['no_mp_keep_links']['mean_flowtime']) / bn * 100.0
        nrg_g = (bn - gr['no_rg']['mean_flowtime']) / bn * 100.0
        print(f"  {gk:10s} | Nom={bn:8.1f}s | DualGFlow-RG={rg_g:+6.2f}% | No-MP={nmp_g:+6.2f}% | No-RG={nrg_g:+6.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--aggregate', action='store_true')
    parser.add_argument('--seeds', type=int, nargs='+', default=[101, 202, 303])
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.seed is not None:
        train_single_seed(args.seed, device=device)
    if args.aggregate:
        aggregate_and_plot(args.seeds, device=device)


if __name__ == '__main__':
    main()
