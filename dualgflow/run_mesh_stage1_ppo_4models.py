"""Stage-1 Counterfactual Ranking Warm-Start + Stage-2 Top-K=4 PPO on Multi-Hop Mesh (Seed 101).

Fixes the Stage-1 train/eval mask mismatch (restricts both Stage-1 and Stage-2
consistently to Top-K=4 around Nominal-CA-EFT, so unmasked servers never leak into
argmax), distills multi-hop congestion patterns from the cached 1200-state
counterfactual rollout dataset (dataset_mesh_24scenes.pt), and then fine-tunes
with gentle Top-K=4 Masked PPO.
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
from dualgflow.run_fair_nominal_experiment import eval_strict, nominal_prior_logits
from dualgflow.run_mesh_topk_diag import make_topk_mask_1d, apply_mask


def train_stage1_topk(
    model: DualGFlowPolicy,
    dataset: list[dict],
    val_sc: list[dict],
    val_nom: float,
    top_k: int = 4,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 2e-4,
    seed: int = 101,
    device: str = 'cuda',
) -> tuple[DualGFlowPolicy, list[dict]]:
    model.to(device)
    opt = torch.optim.AdamW(model.actor_parameters(), lr=lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rng = np.random.default_rng(seed)

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
    print(f'  [Stage 1 Init] Val: {best_val:.1f}s ({vg0:+.2f}%)', flush=True)
    s1_hist = [dict(phase='stage1', step=0, val_flowtime=float(best_val), val_gain_pct=float(vg0))]

    # Precompute Top-K=4 mask aligned with counterfactual dataset masks
    ds_topk_masks = []
    for s in dataset:
        m_topk = make_topk_mask_1d(s['obs']['nominal_eft'], k=top_k)
        # Intersect with dataset counterfactual mask (always >=1 since nominal_eft[:top_k] <= top_k=6 in dataset)
        m_valid = m_topk & torch.as_tensor(s['mask'], dtype=torch.bool)
        if not m_valid.any():
            m_valid = torch.as_tensor(s['mask'], dtype=torch.bool)
        ds_topk_masks.append(m_valid)

    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(dataset))
        losses = []
        for start in range(0, len(order), batch_size):
            idxs = order[start:start + batch_size]
            sub = [dataset[int(i)] for i in idxs]
            batch = collate_obs_batch([s['obs'] for s in sub], device=device, use_nominal_anchor=True)
            logits = forward_actor_batch(model, batch)

            costs = torch.as_tensor(np.stack([s['costs'] for s in sub]), dtype=torch.float32, device=device)
            masks = torch.stack([ds_topk_masks[int(i)] for i in idxs]).to(device)
            masked_costs = costs.masked_fill(~masks, 1e12)
            c_min = masked_costs.min(dim=1, keepdim=True).values
            temp = 0.015 * c_min.clamp_min(1.0)
            target_probs = F.softmax(-(masked_costs - c_min) / temp, dim=1)

            pred_masked = apply_mask(logits, masks)
            kl_loss = -(target_probs * F.log_softmax(pred_masked, dim=1) * masks).sum(dim=1).mean()
            residual = (logits + 2.0 * batch.rel_flow_eft) * masks
            loss = kl_loss + 0.05 * residual.pow(2).mean()

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.actor_parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        scheduler.step()

        v = eval_val()
        vg = (val_nom - v) / val_nom * 100.0
        tag = ''
        if v < best_val:
            best_val = v
            best_state = deepcopy(model.state_dict())
            tag = ' *BEST*'
        print(f'  [Stage 1 Ep {ep+1:2d}/{epochs}] Loss: {np.mean(losses):.4f} | Val: {v:.1f}s ({vg:+.2f}%){tag}', flush=True)
        s1_hist.append(dict(phase='stage1', step=ep + 1, val_flowtime=float(v), val_gain_pct=float(vg), s1_loss=float(np.mean(losses))))

    model.load_state_dict(best_state)
    model.eval()
    return model, s1_hist


def train_stage2_ppo(
    model: DualGFlowPolicy,
    train_sc: list[dict],
    val_sc: list[dict],
    train_nom: list[float],
    val_nom: float,
    top_k: int = 4,
    temperature: float = 0.08,
    ppo_updates: int = 12,
    batch_size: int = 128,
    lr_a: float = 5e-5,
    lr_c: float = 5e-4,
    kl_coef: float = 0.002,
    ent_coef: float = 0.001,
    seed: int = 101,
    device: str = 'cuda',
) -> tuple[DualGFlowPolicy, list[dict]]:
    model.to(device)
    # Anchor KL to the Stage-1 distilled actor so PPO refines around Stage-1 rather than forgetting it
    s1_anchor = deepcopy(model).eval()
    for p in s1_anchor.parameters():
        p.requires_grad_(False)

    opt_a = torch.optim.AdamW(model.actor_parameters(), lr=lr_a, weight_decay=1e-4)
    opt_c = torch.optim.AdamW(model.critic_parameters(), lr=lr_c, weight_decay=1e-4)
    rng = np.random.default_rng(seed)

    def collect_all():
        model.eval()
        trans, ep_flows, ep_rews, ep_gains = [], [], [], []
        all_rets, all_vals = [], []
        with torch.no_grad():
            for idx, sc in enumerate(train_sc):
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
                    ep_obs.append(obs)
                    ep_act.append(int(act.item()))
                    ep_logp.append(float(dist.log_prob(act).item()))
                    ep_rew.append(float(rew))
                    ep_val.append(float(val.item()))
                    ep_prior.append(prior_lp)
                    ep_mask.append(m_cpu)
                    obs = next_obs
                mf = env.metrics()['mean_flowtime']
                ep_flows.append(mf)
                ep_rews.append(float(np.sum(ep_rew)))
                ep_gains.append((nom_f - mf) / nom_f * 100.0)

                T = len(ep_rew)
                adv = np.zeros(T, dtype=np.float32)
                ret = np.zeros(T, dtype=np.float32)
                gae, nv = 0.0, 0.0
                for t in reversed(range(T)):
                    delta = ep_rew[t] + 0.995 * nv - ep_val[t]
                    gae = delta + 0.995 * 0.95 * gae
                    adv[t] = gae
                    ret[t] = gae + ep_val[t]
                    nv = ep_val[t]
                all_rets.extend(ret.tolist())
                all_vals.extend(ep_val)
                for t in range(T):
                    trans.append(dict(
                        obs=ep_obs[t], act=ep_act[t], old_logp=ep_logp[t],
                        ret=float(ret[t]), adv=float(adv[t]), prior_lp=ep_prior[t],
                        mask=ep_mask[t],
                    ))
        y_true = np.asarray(all_rets, dtype=np.float64)
        y_pred = np.asarray(all_vals, dtype=np.float64)
        ev = float(1.0 - np.var(y_true - y_pred) / max(np.var(y_true), 1e-8))
        return trans, np.asarray(ep_flows), np.asarray(ep_rews), np.asarray(ep_gains), ev

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
                opt_c.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.critic_parameters(), 1.0)
                opt_c.step()
                losses.append(float(loss.item()))
        return float(np.mean(losses))

    for w_i in range(2):
        tr, _, _, eg, ev = collect_all()
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
    s2_hist = [dict(phase='ppo', step=0, val_flowtime=float(best_val), val_gain_pct=float(vg0))]

    for upd in range(ppo_updates):
        t0 = time.perf_counter()
        tr, _, er, eg, ev = collect_all()
        all_adv = np.array([x['adv'] for x in tr], dtype=np.float32)
        am, astd = float(all_adv.mean()), max(float(all_adv.std()), 1e-6)
        cl = update_critic_b(tr, epochs=4)

        model.train()
        clips, ents, gnorms = [], [], []
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

                opt_a.zero_grad()
                loss.backward()
                gn = float(nn.utils.clip_grad_norm_(model.actor_parameters(), 0.5).item())
                opt_a.step()

                clips.append(float(((ratio - 1.0).abs() > 0.15).float().mean().item()))
                ents.append(float(ent.item()))
                gnorms.append(gn)

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
            f'| EV: {ev:+.3f} | Clip: {np.mean(clips):.3f} | Ent: {np.mean(ents):.3f} | GNorm: {np.mean(gnorms):.2f}{tag}',
            flush=True,
        )
        s2_hist.append(dict(
            phase='ppo', step=upd + 1,
            train_gain_pct=float(eg.mean()), val_flowtime=float(vf), val_gain_pct=float(vg),
            explained_variance=float(ev), clip_fraction=float(np.mean(clips)),
            entropy=float(np.mean(ents)), actor_grad_norm=float(np.mean(gnorms)),
        ))

    model.load_state_dict(best_state)
    model.eval()
    return model, s2_hist


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    topo = 'multi_hop_mesh'
    seed = 101
    top_k = 4
    print(f'Running Stage-1 Ranking + Stage-2 Top-K={top_k} PPO on Multi-Hop Mesh (seed={seed}) on {device}', flush=True)

    ds_path = Path(__file__).resolve().parent / 'checkpoints_mesh_3seed' / 'dataset_mesh_24scenes.pt'
    dataset = torch.load(ds_path, weights_only=False)
    print(f'Loaded cached counterfactual dataset ({len(dataset)} states)', flush=True)

    train_sc = [
        make_flow_scenario(seed=1000 + i, split='train', workflows=12, num_servers=16,
                           load=0.75, ccr=2.5, topology=topo, oversubscription=6.0)
        for i in range(8)
    ]
    val_sc = [
        make_flow_scenario(seed=2000 + i, split='validation', workflows=14, num_servers=16,
                           load=0.75, ccr=2.5, topology=topo, oversubscription=6.0)
        for i in range(8)
    ]
    test_16 = [
        make_flow_scenario(seed=4000 + i, split='test', workflows=14, num_servers=16,
                           load=0.75, ccr=2.5, topology=topo, oversubscription=6.0)
        for i in range(12)
    ]
    test_32 = [
        make_flow_scenario(seed=5000 + i, split='test', workflows=18, num_servers=32,
                           load=0.75, ccr=2.5, topology=topo, oversubscription=6.0)
        for i in range(6)
    ]

    train_nom = [eval_strict(sc, lambda e, o: int(np.argmin(o['nominal_eft'])), 'nom')['mean_flowtime'] for sc in train_sc]
    val_nom = float(np.mean([eval_strict(sc, lambda e, o: int(np.argmin(o['nominal_eft'])), 'nom')['mean_flowtime'] for sc in val_sc]))

    variants = ('dualgflow', 'no_mp_keep_links', 'global_mlp', 'no_rg')
    ckpt_dir = Path(__file__).resolve().parent / 'checkpoints_mesh_s1_ppo'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    trained = {}
    histories = {}
    for var in variants:
        print(f'\n================ {var} (seed={seed}) ================', flush=True)
        torch.manual_seed(seed)
        m = DualGFlowPolicy(variant=var, mode='residual_eft', hidden_dim=64, gnn_layers=4, use_nominal_anchor=True).to(device)
        m, h1 = train_stage1_topk(m, dataset, val_sc, val_nom, top_k=top_k, epochs=10, seed=seed, device=device)
        m, h2 = train_stage2_ppo(m, train_sc, val_sc, train_nom, val_nom, top_k=top_k, ppo_updates=12, seed=seed, device=device)
        torch.save(m.state_dict(), ckpt_dir / f'{var}_seed{seed}.pt')
        trained[var] = m
        histories[var] = dict(stage1=h1, stage2=h2)

    def eval_suite(sc_list: list[dict]) -> dict:
        out = {}
        for hname, hfn in [
            ('nominal_ca_eft', lambda e, o: int(np.argmin(o['nominal_eft']))),
            ('flow_oracle_eft', lambda e, o: int(np.argmin(o['flow_eft']))),
        ]:
            flows = [eval_strict(sc, hfn, hname)['mean_flowtime'] for sc in sc_list]
            out[hname] = dict(mean_flowtime=float(np.mean(flows)), per_scene=flows)
        for var in variants:
            mod = trained[var]
            with torch.no_grad():
                flows = [
                    eval_strict(
                        sc,
                        lambda e, o, m=mod: int(torch.argmax(apply_mask(m.forward_actor(o), make_topk_mask_1d(o['nominal_eft'], k=top_k).to(device))).item()),
                        var,
                    )['mean_flowtime']
                    for sc in sc_list
                ]
            out[var] = dict(mean_flowtime=float(np.mean(flows)), per_scene=flows)
        return out

    r16 = eval_suite(test_16)
    r32 = eval_suite(test_32)

    out_dir = Path(__file__).resolve().parent
    with open(out_dir / 'mesh_s1_ppo_results.json', 'w') as f:
        json.dump(dict(seed=seed, top_k=top_k, test_16=r16, test_32=r32, histories=histories), f, indent=2)

    fig_dir = out_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})

    # Fig 15: Comparison bars
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    order = [
        ('no_rg', 'DualGFlow (No-RG)', '#8c564b'),
        ('global_mlp', 'Global-MLP', '#d62728'),
        ('no_mp_keep_links', 'No-MP-Keep-Links', '#ff7f0e'),
        ('nominal_ca_eft', 'Nominal-CA-EFT', '#9467bd'),
        ('dualgflow', 'DualGFlow-RG (Ours)', '#1f77b4'),
    ]
    for ax, r_data, title in [
        (axes[0], r16, '(a) Multi-Hop Mesh N=16 (Stage1+PPO, 12 Scenes)'),
        (axes[1], r32, '(b) Zero-Shot Scale N=32 (Stage1+PPO, 6 Scenes)'),
    ]:
        labels = [o[1] for o in order]
        means = [r_data[o[0]]['mean_flowtime'] / 1000.0 for o in order]
        colors = [o[2] for o in order]
        y = np.arange(len(labels))
        ax.barh(y, means, height=0.58, color=colors, alpha=0.9)
        ax.set_yticks(y); ax.set_yticklabels(labels)
        ax.set_xlim(max(0.0, min(means) - 1.2), max(means) + 1.0)
        ax.set_xlabel('Mean Workflow Flowtime (×10³ s)')
        ax.set_title(title); ax.grid(axis='x', alpha=0.25)
        for yi, m, (k, _, _) in zip(y, means, order):
            ax.text(m + 0.08, yi, f'{m:.2f}k', va='center', fontsize=9,
                    fontweight='bold' if k == 'dualgflow' else 'normal')
    plt.tight_layout()
    fig.savefig(fig_dir / 'fig15_mesh_stage1_ppo_bars.png')
    plt.close(fig)

    # Fig 16: Combined Stage-1 + Stage-2 Learning Curves
    fig_lc, axes_lc = plt.subplots(1, 2, figsize=(12.0, 4.6))
    styles = [
        ('dualgflow', 'DualGFlow-RG (Ours)', '#1f77b4', '-'),
        ('no_mp_keep_links', 'No-MP-Keep-Links', '#ff7f0e', '--'),
        ('global_mlp', 'Global-MLP', '#d62728', '-.'),
        ('no_rg', 'DualGFlow (No-RG)', '#8c564b', ':'),
    ]
    for var, label, col, ls in styles:
        s1 = histories[var]['stage1']
        s2 = histories[var]['stage2']
        xs1 = [r['step'] for r in s1]
        ys1 = [r['val_gain_pct'] for r in s1]
        xs2 = [10 + r['step'] for r in s2]
        ys2 = [r['val_gain_pct'] for r in s2]
        axes_lc[0].plot(xs1 + xs2[1:], ys1 + ys2[1:], label=label, color=col, linestyle=ls, linewidth=2.0, marker='o', markersize=3.2)
        ev_xs = [r['step'] for r in s2 if 'explained_variance' in r]
        ev_ys = [r['explained_variance'] for r in s2 if 'explained_variance' in r]
        axes_lc[1].plot(ev_xs, ev_ys, label=label, color=col, linestyle=ls, linewidth=2.0, marker='o', markersize=3.2)

    axes_lc[0].axvline(10, color='gray', linestyle=':', linewidth=1.4, label='Stage 1 → PPO')
    axes_lc[0].axhline(0.0, color='#9467bd', linestyle='--', linewidth=1.4, label='Nominal-CA-EFT (0%)')
    axes_lc[0].set_title('(a) Stage-1 Ranking + Stage-2 PPO Val Gain (%)')
    axes_lc[0].set_xlabel('Training Step (0-10: Stage 1 Epoch, 10-22: PPO Update)')
    axes_lc[0].set_ylabel('Validation Gain vs Nominal (%)')
    axes_lc[0].grid(alpha=0.25); axes_lc[0].legend(fontsize=8.2)

    axes_lc[1].set_title('(b) Stage-2 Critic Explained Variance')
    axes_lc[1].set_xlabel('PPO Update')
    axes_lc[1].set_ylabel('Explained Variance')
    axes_lc[1].grid(alpha=0.25); axes_lc[1].legend(fontsize=8.5)

    plt.tight_layout()
    fig_lc.savefig(fig_dir / 'fig16_mesh_stage1_ppo_curves.png')
    plt.close(fig_lc)

    print('\n=== STAGE1+PPO MULTI-HOP MESH N=16 (12 SCENES, SEED 101) ===')
    b16 = r16['nominal_ca_eft']['mean_flowtime']
    for k, _, _ in order + [('flow_oracle_eft', 'Flow-Oracle-EFT', '#2ca02c')]:
        mf = r16[k]['mean_flowtime']
        print(f"  {k:18s}: {mf:8.1f}s ({(b16 - mf) / b16 * 100:+5.2f}%)")

    print('\n=== STAGE1+PPO MULTI-HOP MESH ZERO-SHOT N=32 (6 SCENES, SEED 101) ===')
    b32 = r32['nominal_ca_eft']['mean_flowtime']
    for k, _, _ in order + [('flow_oracle_eft', 'Flow-Oracle-EFT', '#2ca02c')]:
        mf = r32[k]['mean_flowtime']
        print(f"  {k:18s}: {mf:8.1f}s ({(b32 - mf) / b32 * 100:+5.2f}%)")


if __name__ == '__main__':
    main()
