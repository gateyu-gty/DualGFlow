"""Diagnostic Top-K=4 Masked PPO Verification for DualGFlow-RG on Multi-Hop Mesh (Seed 101).

Implements the two diagnostic fixes from the RL learning-curve checklist:
1. Top-K=4 action masking (around Nominal-CA-EFT's top 4 candidate servers) applied
   identically during rollout, prior KL, PPO update, and deterministic evaluation,
   eliminating -4% random exploration waste on clearly suboptimal servers.
2. Full PPO health diagnostics logged every update:
   - TrainGain (%) & Scene-Normalized Episode Return (Nominal = -10.0)
   - Validation Gain (%) vs Nominal-CA-EFT
   - Critic Huber Loss & Explained Variance (EV)
   - Policy Loss, Clip Fraction, Approx KL, Actor Grad Norm, Policy Entropy
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


def make_topk_mask_1d(nominal_eft: np.ndarray, k: int = 4) -> torch.Tensor:
    n = len(nominal_eft)
    mask = torch.zeros(n, dtype=torch.bool)
    if k >= n:
        mask[:] = True
        return mask
    order = np.argsort(nominal_eft, kind='stable')[:k]
    mask[order] = True
    return mask


def apply_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~mask, -1e4)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    topo = 'multi_hop_mesh'
    seed = 101
    top_k = 4
    temperature = 0.10
    ppo_updates = 20
    batch_size = 128
    lr_a = 2e-4
    lr_c = 5e-4
    kl_coef = 0.001
    ent_coef = 0.001

    print(f'Running Top-K={top_k} Diagnostic PPO for DualGFlow-RG (seed={seed}) on {device}', flush=True)

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

    torch.manual_seed(seed)
    model = DualGFlowPolicy(
        variant='dualgflow',
        mode='residual_eft',
        hidden_dim=64,
        gnn_layers=4,
        use_nominal_anchor=True,
    ).to(device)

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
                    pr_logits = apply_mask(nominal_prior_logits(obs, device), m_dev)
                    prior_lp = F.log_softmax(pr_logits / temperature, dim=0).cpu()
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
        tr, _, er, eg, ev = collect_all()
        cl = update_critic_b(tr, epochs=5)
        print(f'  [Warmup {w_i+1}/2] TrainGain: {eg.mean():+.2f}% | NormRew: {er.mean():.3f} | Huber: {cl:.4f} | EV: {ev:+.3f}', flush=True)

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
    history = [dict(step=0, val_flowtime=float(best_val), val_gain_pct=float(vg0))]

    for upd in range(ppo_updates):
        t0 = time.perf_counter()
        tr, ef, er, eg, ev = collect_all()
        all_adv = np.array([x['adv'] for x in tr], dtype=np.float32)
        am, astd = float(all_adv.mean()), max(float(all_adv.std()), 1e-6)
        cl = update_critic_b(tr, epochs=4)

        model.train()
        p_losses, kls, ents, clips, approx_kls, gnorms = [], [], [], [], [], []
        for _ in range(3):
            perm = rng.permutation(len(tr))
            for st in range(0, len(perm), batch_size):
                sub = [tr[int(i)] for i in perm[st:st + batch_size]]
                b = collate_obs_batch([x['obs'] for x in sub], device=device, use_nominal_anchor=True)
                raw_logits = forward_actor_batch(model, b)
                b_mask = torch.stack([x['mask'] for x in sub]).to(device)
                masked = apply_mask(raw_logits, b_mask)
                scaled = masked / temperature
                dist = Categorical(logits=scaled)
                acts = torch.tensor([x['act'] for x in sub], dtype=torch.long, device=device)
                old_lp = torch.tensor([x['old_logp'] for x in sub], dtype=torch.float32, device=device)
                n_adv = torch.tensor([(x['adv'] - am) / astd for x in sub], dtype=torch.float32, device=device)
                pr_lp = torch.stack([x['prior_lp'] for x in sub]).to(device)

                new_lp = dist.log_prob(acts)
                ratio = torch.exp(new_lp - old_lp)
                s1 = ratio * n_adv
                s2 = torch.clamp(ratio, 0.85, 1.15) * n_adv
                ploss = -torch.min(s1, s2).mean()
                cur_lp = F.log_softmax(scaled, dim=1)
                kl = (cur_lp.exp() * (cur_lp - pr_lp) * b_mask).sum(dim=1).mean()
                ent = dist.entropy().mean()
                loss = ploss + kl_coef * kl - ent_coef * ent

                opt_a.zero_grad()
                loss.backward()
                gn = float(nn.utils.clip_grad_norm_(model.actor_parameters(), 0.5).item())
                opt_a.step()

                p_losses.append(float(ploss.item()))
                kls.append(float(kl.item()))
                ents.append(float(ent.item()))
                clips.append(float(((ratio - 1.0).abs() > 0.15).float().mean().item()))
                approx_kls.append(float((old_lp - new_lp).mean().item()))
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
        history.append(dict(
            step=upd + 1,
            train_gain_pct=float(eg.mean()),
            norm_reward_mean=float(er.mean()),
            val_flowtime=float(vf),
            val_gain_pct=float(vg),
            critic_loss=float(cl),
            explained_variance=float(ev),
            policy_loss=float(np.mean(p_losses)),
            clip_fraction=float(np.mean(clips)),
            approx_kl=float(np.mean(approx_kls)),
            actor_grad_norm=float(np.mean(gnorms)),
            entropy=float(np.mean(ents)),
        ))

    model.load_state_dict(best_state)
    model.eval()

    def eval_test(sc_list: list[dict]) -> dict:
        nom = [eval_strict(sc, lambda e, o: int(np.argmin(o['nominal_eft'])), 'nom')['mean_flowtime'] for sc in sc_list]
        ora = [eval_strict(sc, lambda e, o: int(np.argmin(o['flow_eft'])), 'ora')['mean_flowtime'] for sc in sc_list]
        with torch.no_grad():
            rg = [
                eval_strict(
                    sc,
                    lambda e, o: int(torch.argmax(apply_mask(model.forward_actor(o), make_topk_mask_1d(o['nominal_eft'], k=top_k).to(device))).item()),
                    'dualgflow',
                )['mean_flowtime']
                for sc in sc_list
            ]
        return dict(
            nominal=float(np.mean(nom)),
            oracle=float(np.mean(ora)),
            dualgflow=float(np.mean(rg)),
        )

    r16 = eval_test(test_16)
    r32 = eval_test(test_32)
    out_dir = Path(__file__).resolve().parent
    with open(out_dir / 'mesh_topk_diag_results.json', 'w') as f:
        json.dump(dict(seed=seed, top_k=top_k, test_16=r16, test_32=r32, history=history), f, indent=2)

    fig_dir = out_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    xs = [r['step'] for r in history if 'train_gain_pct' in r]
    axes[0, 0].plot(xs, [r['train_gain_pct'] for r in history if 'train_gain_pct' in r], '-o', label='Train Rollout Gain (%)', color='#1f77b4')
    axes[0, 0].plot([r['step'] for r in history], [r['val_gain_pct'] for r in history], '-s', label='Val Deterministic Gain (%)', color='#2ca02c')
    axes[0, 0].axhline(0.0, color='#9467bd', linestyle='--', label='Nominal-CA-EFT (0%)')
    axes[0, 0].set_title('(a) Top-K=4 Train vs Val Gain (%)')
    axes[0, 0].set_xlabel('PPO Update'); axes[0, 0].set_ylabel('Gain vs Nominal (%)'); axes[0, 0].grid(alpha=0.25); axes[0, 0].legend(fontsize=8.5)

    axes[0, 1].plot(xs, [r['explained_variance'] for r in history if 'explained_variance' in r], '-o', color='#ff7f0e', label='Explained Variance')
    axes[0, 1].set_title('(b) Critic Explained Variance')
    axes[0, 1].set_xlabel('PPO Update'); axes[0, 1].set_ylabel('EV'); axes[0, 1].grid(alpha=0.25); axes[0, 1].legend(fontsize=8.5)

    axes[1, 0].plot(xs, [r['clip_fraction'] for r in history if 'clip_fraction' in r], '-o', color='#d62728', label='PPO Clip Fraction')
    axes[1, 0].set_title('(c) PPO Clip Fraction')
    axes[1, 0].set_xlabel('PPO Update'); axes[1, 0].set_ylabel('Clip Fraction'); axes[1, 0].grid(alpha=0.25); axes[1, 0].legend(fontsize=8.5)

    axes[1, 1].plot(xs, [r['entropy'] for r in history if 'entropy' in r], '-o', color='#8c564b', label='Policy Entropy (Top-4)')
    axes[1, 1].set_title('(d) Policy Entropy')
    axes[1, 1].set_xlabel('PPO Update'); axes[1, 1].set_ylabel('Entropy'); axes[1, 1].grid(alpha=0.25); axes[1, 1].legend(fontsize=8.5)

    plt.tight_layout()
    fig.savefig(fig_dir / 'fig12_mesh_topk_diag.png')
    plt.close(fig)

    print(f'\n=== TEST N=16 (12 SCENES): Nominal={r16["nominal"]:.1f}s | DualGFlow-RG={r16["dualgflow"]:.1f}s ({(r16["nominal"]-r16["dualgflow"])/r16["nominal"]*100:+.2f}%) | Oracle={r16["oracle"]:.1f}s ===')
    print(f'=== TEST N=32 (6 SCENES):  Nominal={r32["nominal"]:.1f}s | DualGFlow-RG={r32["dualgflow"]:.1f}s ({(r32["nominal"]-r32["dualgflow"])/r32["nominal"]*100:+.2f}%) | Oracle={r32["oracle"]:.1f}s ===')


if __name__ == '__main__':
    main()
