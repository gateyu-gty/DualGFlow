"""Fair Nominal-CA-EFT Anchor Experiment under Strict FIFO Server Queues.

Removes the handcrafted multi-hop `flow_eft` / `flow_delay` shortcut so that
NO neural model receives pre-integrated multi-hop congestion. All models
(`DualGFlow-RG`, `No-MP-Keep-Links`, `Global-MLP`, and external `DualGFlow (No-RG)`)
receive identical static `Nominal-CA-EFT` features + raw physical resource graph.
Only multi-hop Resource GNN message passing can infer 2-3 hop core/WAN congestion!
"""
from __future__ import annotations

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

from copy import deepcopy
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from dualgflow.topologies import make_flow_scenario
from dualgflow.simulator_flow import FlowWorkflowEnv
from dualgflow.models_rg import DualGFlowPolicy
from dualgflow.batch_rg import collate_obs_batch, forward_actor_batch, forward_critic_batch
from dualgflow.run_paper_benchmark import paired_gain_ci

REWARD_SCALE = 50000.0


def eval_strict(sc: dict, fn, name: str = '') -> dict:
    env = FlowWorkflowEnv(sc, strict_fifo=True)
    obs = env.reset()
    lats = []
    while not env.done:
        t0 = time.perf_counter()
        a = int(fn(env, obs))
        lats.append((time.perf_counter() - t0) * 1000.0)
        obs, _, _, _ = env.step(a)
    m = env.metrics()
    m['p95_decision_ms'] = float(np.quantile(lats, 0.95)) if lats else 0.0
    return m


def collect_strict_fifo_dataset(scenarios: list[dict], states_per_sc: int = 45, top_k: int = 6, seed: int = 101) -> list[dict]:
    dataset = []
    rng = np.random.default_rng(seed)
    for sc in scenarios:
        env = FlowWorkflowEnv(sc, strict_fifo=True)
        obs = env.reset()
        step_idx = 0
        collected = 0
        while not env.done:
            if step_idx % 3 == 0 and collected < states_per_sc:
                n_srv = obs['num_servers']
                cand_set = set(np.argsort(obs['nominal_eft'])[:top_k].tolist())
                cand_set.update(np.argsort(obs['flow_eft'])[:4].tolist())
                cands = sorted(cand_set)

                costs = np.zeros(n_srv, dtype=np.float32)
                mask = np.zeros(n_srv, dtype=bool)
                for a in cands:
                    sim = env.clone_causal_active()
                    sim.step(a, return_observation=False)
                    while not sim.done:
                        efts = sim.candidate_eft_estimates()
                        sim.step(int(np.argmin(efts['flow_eft'])), return_observation=False)
                    costs[a] = float(-sim.total_reward_unscaled)
                    mask[a] = True
                best_a = int(cands[int(np.argmin(costs[cands]))])
                dataset.append(dict(obs=obs, costs=costs, mask=mask, best_a=best_a))
                collected += 1
            a = int(np.argmin(obs['flow_eft'])) if rng.random() < 0.5 else int(np.argmin(obs['nominal_eft']))
            obs, _, _, _ = env.step(a)
            step_idx += 1
    return dataset


def nominal_prior_logits(obs: dict, device: torch.device | str) -> torch.Tensor:
    raw = obs['nominal_eft'].astype(np.float32)
    r_min, r_med = float(np.min(raw)), float(np.median(raw))
    rel = np.clip((raw - r_min) / max(r_med - r_min, 1e-3), 0.0, 5.0)
    return -2.0 * torch.as_tensor(rel, dtype=torch.float32, device=device)


def train_stage1_nominal(
    model: DualGFlowPolicy,
    dataset: list[dict],
    val_scenarios: list[dict],
    epochs: int = 16,
    batch_size: int = 64,
    lr: float = 4e-4,
    device: str = 'cuda',
) -> DualGFlowPolicy:
    model.to(device)
    opt = torch.optim.AdamW(model.actor_parameters(), lr=lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rng = np.random.default_rng(101)

    def eval_val() -> float:
        model.eval()
        with torch.no_grad():
            return float(np.mean([
                eval_strict(sc, lambda e, o: int(torch.argmax(model.forward_actor(o)).item()))['mean_flowtime']
                for sc in val_scenarios
            ]))

    best_val = eval_val()
    best_state = deepcopy(model.state_dict())
    print(f'  [Stage 1 Init (Nominal-CA-EFT)] Val: {best_val:.2f}s', flush=True)

    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(dataset))
        for start in range(0, len(order), batch_size):
            sub = [dataset[int(i)] for i in order[start:start + batch_size]]
            batch = collate_obs_batch([s['obs'] for s in sub], device=device, use_nominal_anchor=True)
            logits = forward_actor_batch(model, batch)

            costs = torch.as_tensor(np.stack([s['costs'] for s in sub]), dtype=torch.float32, device=device)
            masks = torch.as_tensor(np.stack([s['mask'] for s in sub]), dtype=torch.bool, device=device)
            masked_costs = costs.masked_fill(~masks, 1e12)
            c_min = masked_costs.min(dim=1, keepdim=True).values
            temp = 0.025 * c_min.clamp_min(1.0)
            target_probs = F.softmax(-(masked_costs - c_min) / temp, dim=1)

            pred_masked = logits.masked_fill(~masks, -1e9)
            kl_loss = -(target_probs * F.log_softmax(pred_masked, dim=1)).sum(dim=1).mean()
            residual = logits + 2.0 * batch.rel_flow_eft
            loss = kl_loss + 0.01 * residual.pow(2).mean()

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.actor_parameters(), 1.0)
            opt.step()
        scheduler.step()

        if (ep + 1) % 2 == 0:
            v = eval_val()
            tag = ''
            if v < best_val:
                best_val = v
                best_state = deepcopy(model.state_dict())
                tag = ' *BEST*'
            print(f'  [Stage 1 Ep {ep+1:2d}/{epochs}] Val: {v:.2f}s{tag}', flush=True)

    model.load_state_dict(best_state)
    model.eval()
    return model


def train_ppo_nominal(
    model: DualGFlowPolicy,
    train_scenarios: list[dict],
    val_scenarios: list[dict],
    temperature: float = 0.125,
    warmup_rounds: int = 2,
    ppo_updates: int = 6,
    episodes_per_update: int = 9,
    batch_size: int = 128,
    lr_a: float = 1e-4,
    lr_c: float = 5e-4,
    device: str = 'cuda',
    history_out: list[dict] | None = None,
) -> DualGFlowPolicy:
    model.to(device)
    opt_a = torch.optim.AdamW(model.actor_parameters(), lr=lr_a, weight_decay=1e-4)
    opt_c = torch.optim.AdamW(model.critic_parameters(), lr=lr_c, weight_decay=1e-4)
    rng = np.random.default_rng(2026)

    def collect_eps():
        model.eval()
        chosen = rng.choice(len(train_scenarios), size=min(episodes_per_update, len(train_scenarios)), replace=False)
        trans, ep_flows, ep_rewards = [], [], []
        with torch.no_grad():
            for idx in chosen:
                env = FlowWorkflowEnv(train_scenarios[int(idx)], reward_scale=REWARD_SCALE, strict_fifo=True)
                obs = env.reset()
                ep_obs, ep_act, ep_logp, ep_rew, ep_val, ep_prior = [], [], [], [], [], []
                while not env.done:
                    logits, val = model.forward_obs(obs)
                    scaled = logits / temperature
                    dist = Categorical(logits=scaled)
                    act = dist.sample()
                    prior_lp = F.log_softmax(nominal_prior_logits(obs, device) / temperature, dim=0).cpu()
                    next_obs, rew, _, _ = env.step(int(act.item()))
                    ep_obs.append(obs); ep_act.append(int(act.item()))
                    ep_logp.append(float(dist.log_prob(act).item()))
                    ep_rew.append(float(rew)); ep_val.append(float(val.item()))
                    ep_prior.append(prior_lp)
                    obs = next_obs
                ep_flows.append(env.metrics()['mean_flowtime'])
                ep_rewards.append(float(np.sum(ep_rew)))
                T = len(ep_rew)
                adv = np.zeros(T, dtype=np.float32); ret = np.zeros(T, dtype=np.float32)
                gae = 0.0; nv = 0.0
                for t in reversed(range(T)):
                    delta = ep_rew[t] + 0.995 * nv - ep_val[t]
                    gae = delta + 0.995 * 0.95 * gae
                    adv[t] = gae; ret[t] = gae + ep_val[t]; nv = ep_val[t]
                for t in range(T):
                    trans.append(dict(obs=ep_obs[t], act=ep_act[t], old_logp=ep_logp[t],
                                      ret=float(ret[t]), adv=float(adv[t]), prior_lp=ep_prior[t]))
        return trans, np.asarray(ep_flows, dtype=np.float64), np.asarray(ep_rewards, dtype=np.float64)

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

    for w_i in range(warmup_rounds):
        tr, ef, er = collect_eps()
        cl = update_critic_b(tr, epochs=4)
        print(f'  [Warmup {w_i+1}/{warmup_rounds}] TrainEp: {ef.mean():.1f}s | Rew: {er.mean():.3f} | Huber: {cl:.4f}', flush=True)
        if history_out is not None:
            history_out.append(dict(
                phase='warmup',
                step=w_i + 1 - warmup_rounds,
                train_flowtime=float(ef.mean()),
                ep_reward_mean=float(er.mean()),
                ep_reward_std=float(er.std()),
                critic_loss=float(cl),
            ))

    def eval_val() -> float:
        model.eval()
        with torch.no_grad():
            return float(np.mean([
                eval_strict(sc, lambda e, o: int(torch.argmax(model.forward_actor(o)).item()))['mean_flowtime']
                for sc in val_scenarios
            ]))

    best_val = eval_val()
    best_state = deepcopy(model.state_dict())
    print(f'  [PPO Init] Val: {best_val:.2f}s', flush=True)
    if history_out is not None:
        history_out.append(dict(
            phase='ppo_init',
            step=0,
            val_flowtime=float(best_val),
        ))

    for upd in range(ppo_updates):
        t0 = time.perf_counter()
        tr, ef, er = collect_eps()
        all_adv = np.array([x['adv'] for x in tr], dtype=np.float32)
        am, astd = float(all_adv.mean()), max(float(all_adv.std()), 1e-6)
        cl = update_critic_b(tr, epochs=3)

        model.train()
        p_losses, kls, ents = [], [], []
        for _ in range(3):
            perm = rng.permutation(len(tr))
            for st in range(0, len(perm), batch_size):
                sub = [tr[int(i)] for i in perm[st:st + batch_size]]
                b = collate_obs_batch([x['obs'] for x in sub], device=device, use_nominal_anchor=True)
                scaled = forward_actor_batch(model, b) / temperature
                dist = Categorical(logits=scaled)
                acts = torch.tensor([x['act'] for x in sub], dtype=torch.long, device=device)
                old_lp = torch.tensor([x['old_logp'] for x in sub], dtype=torch.float32, device=device)
                n_adv = torch.tensor([(x['adv'] - am) / astd for x in sub], dtype=torch.float32, device=device)
                pr_lp = torch.stack([x['prior_lp'] for x in sub]).to(device)

                ratio = torch.exp(dist.log_prob(acts) - old_lp)
                s1 = ratio * n_adv
                s2 = torch.clamp(ratio, 0.85, 1.15) * n_adv
                ploss = -torch.min(s1, s2).mean()
                cur_lp = F.log_softmax(scaled, dim=1)
                kl = (cur_lp.exp() * (cur_lp - pr_lp)).sum(dim=1).mean()
                ent = dist.entropy().mean()
                loss = ploss + 0.008 * kl - 0.001 * ent
                opt_a.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.actor_parameters(), 0.5); opt_a.step()
                p_losses.append(float(ploss.item()))
                kls.append(float(kl.item()))
                ents.append(float(ent.item()))

        vf = eval_val()
        tag = ''
        if vf < best_val:
            best_val = vf
            best_state = deepcopy(model.state_dict())
            tag = ' *BEST*'
        print(f'  [PPO {upd+1:2d}/{ppo_updates}] ({time.perf_counter()-t0:.1f}s) | TrainEp: {ef.mean():.1f}s | Rew: {er.mean():.3f} | Val: {vf:.2f}s{tag}', flush=True)
        if history_out is not None:
            history_out.append(dict(
                phase='ppo',
                step=upd + 1,
                train_flowtime=float(ef.mean()),
                ep_reward_mean=float(er.mean()),
                ep_reward_std=float(er.std()),
                val_flowtime=float(vf),
                critic_loss=float(cl),
                policy_loss=float(np.mean(p_losses)),
                kl=float(np.mean(kls)),
                entropy=float(np.mean(ents)),
            ))

    model.load_state_dict(best_state)
    model.eval()
    return model


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Running Fair Nominal-CA-EFT Anchor Experiment (Strict FIFO) on {device}', flush=True)

    topos = ['leaf_spine', 'cloud_edge', 'multi_hop_mesh']
    train_sc = [make_flow_scenario(seed=100 + i, split='train', workflows=12, num_servers=16,
                                   load=0.75, ccr=2.5, topology=topos[i % 3], oversubscription=6.0) for i in range(36)]
    val_sc = [make_flow_scenario(seed=200 + i, split='validation', workflows=14, num_servers=16,
                                 load=0.75, ccr=2.5, topology=topos[i % 3], oversubscription=6.0) for i in range(12)]
    test_16 = [make_flow_scenario(seed=400 + i, split='test', workflows=14, num_servers=16,
                                  load=0.75, ccr=2.5, topology=topos[i % 3], oversubscription=6.0) for i in range(18)]
    test_32 = [make_flow_scenario(seed=500 + i, split='test', workflows=18, num_servers=32,
                                  load=0.75, ccr=2.5, topology=topos[i % 3], oversubscription=6.0) for i in range(9)]

    ckpt_dir = Path(__file__).resolve().parent / 'checkpoints_nominal_v2'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ds_path = ckpt_dir / 'dataset_36scenes.pt'
    if ds_path.exists():
        dataset = torch.load(ds_path, weights_only=False)
        print(f'Loaded cached dataset ({len(dataset)} states)', flush=True)
    else:
        t0 = time.perf_counter()
        dataset = collect_strict_fifo_dataset(train_sc, states_per_sc=45, top_k=6, seed=101)
        torch.save(dataset, ds_path)
        print(f'Collected {len(dataset)} states in {time.perf_counter()-t0:.1f}s', flush=True)

    models = {}
    for var in ('dualgflow', 'no_mp_keep_links', 'global_mlp', 'no_rg'):
        ckpt = ckpt_dir / f'{var}_seed101.pt'
        torch.manual_seed(101)
        m = DualGFlowPolicy(variant=var, mode='residual_eft', hidden_dim=64, gnn_layers=4, use_nominal_anchor=True)
        if ckpt.exists():
            m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            m.to(device).eval()
            print(f'Loaded cached {ckpt.name}', flush=True)
        else:
            print(f'\n=== Training {var} (Nominal Anchor, No Multi-Hop Shortcut) ===', flush=True)
            train_stage1_nominal(m, dataset, val_sc, epochs=16, batch_size=64, lr=4e-4, device=device)
            train_ppo_nominal(m, train_sc, val_sc, ppo_updates=6, episodes_per_update=9, device=device)
            torch.save(m.state_dict(), ckpt)
        models[var] = m

    def run_eval(sc_list):
        fns = {
            'compute_eft': lambda e, o: int(np.argmin(o['compute_eft'])),
            'nominal_ca_eft': lambda e, o: int(np.argmin(o['nominal_eft'])),
            'flow_oracle_eft': lambda e, o: int(np.argmin(o['flow_eft'])),
        }
        for k, mod in models.items():
            fns[k] = lambda e, o, m=mod: int(torch.argmax(m.forward_actor(o)).item())
        res = {}
        for name, fn in fns.items():
            by_topo = {t: [] for t in topos}
            flows, lats = [], []
            with torch.no_grad():
                for sc in sc_list:
                    r = eval_strict(sc, fn, name)
                    flows.append(r['mean_flowtime'])
                    lats.append(r['p95_decision_ms'])
                    by_topo[sc['meta']['topology']['topology']].append(r['mean_flowtime'])
            res[name] = dict(
                mean_flowtime=float(np.mean(flows)),
                per_scene=flows,
                by_topology={t: float(np.mean(v)) for t, v in by_topo.items() if v},
                p95_decision_ms=float(np.mean(lats)),
            )
        return res

    r16 = run_eval(test_16)
    r32 = run_eval(test_32)
    out = dict(n16=r16, n32=r32)
    out_path = Path(__file__).resolve().parent / 'fair_nominal_results.json'
    out_path.write_text(json.dumps(out, indent=2), encoding='utf-8')

    for tag, r in [('N=16 HELD-OUT (18 scenes)', r16), ('N=32 ZERO-SHOT (9 scenes)', r32)]:
        print(f'\n=== {tag} ===')
        base = r['nominal_ca_eft']['mean_flowtime']
        for k, v in r.items():
            gain = (base - v['mean_flowtime']) / base * 100.0
            ci = paired_gain_ci(r['nominal_ca_eft']['per_scene'], v['per_scene']) if k != 'nominal_ca_eft' else None
            ci_s = f"[{ci['ci95_low']:+.2f}%, {ci['ci95_high']:+.2f}%] w={ci['wins']}/{ci['total']}" if ci else ''
            print(f"{k:18s} | Mean: {v['mean_flowtime']:9.1f}s ({gain:+6.2f}%) {ci_s:26s} | LS={v['by_topology']['leaf_spine']:7.0f} CE={v['by_topology']['cloud_edge']:7.0f} Mesh={v['by_topology']['multi_hop_mesh']:7.0f} | {v['p95_decision_ms']:4.2f}ms")


if __name__ == '__main__':
    main()
