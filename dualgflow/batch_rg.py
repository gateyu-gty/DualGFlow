"""Vectorized disjoint-union graph batching for DualGFlow-RG v2 (30x training speedup).

Produces outputs mathematically identical to single-observation forward passes
while running entire minibatches (e.g. 64 graphs) in a single GPU call.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import torch
from torch.nn import functional as F

from dualgflow.models_rg import DualGFlowPolicy, transform_obs, compute_tropical_wave_delays


@dataclass
class BatchedObs:
    batch_size: int
    num_servers: int
    total_res_nodes: int
    task_x: torch.Tensor
    task_edges: torch.Tensor
    task_batch: torch.Tensor
    task_counts: torch.Tensor
    focus_idx: torch.Tensor
    task_srv_edges: torch.Tensor
    task_srv_attr: torch.Tensor
    resource_x: torch.Tensor
    resource_edges: torch.Tensor
    resource_edge_attr: torch.Tensor
    srv_indices: torch.Tensor
    res_batch_ids: torch.Tensor
    res_local_ids: torch.Tensor
    res_edge_batch: torch.Tensor
    res_edge_local_src: torch.Tensor
    res_edge_local_dst: torch.Tensor
    pair_x: torch.Tensor
    candidate_features: torch.Tensor
    rel_flow_eft: torch.Tensor


def collate_obs_batch(
    obs_list: list[dict],
    device: torch.device | str = 'cuda',
    use_nominal_anchor: bool = False,
) -> BatchedObs:
    """Collate a list of raw simulator observations into a single disjoint-union GPU batch."""
    B = len(obs_list)
    n_srv = int(obs_list[0]['num_servers'])

    tx_list, te_list, tb_list, t_counts, focus_list = [], [], [], [], []
    tse_list, tsa_list = [], []
    rx_list, re_list, ea_list = [], [], []
    srv_idx_list, rb_list, rl_list = [], [], []
    reb_list, rels_list, reld_list = [], [], []
    px_list, cf_list, eft_list = [], [], []

    t_off = 0
    r_off = 0
    for b, obs in enumerate(obs_list):
        if int(obs['num_servers']) != n_srv:
            raise ValueError('All observations in a fast batch must have equal num_servers')
        t_obs = transform_obs(obs, device='cpu', use_nominal_anchor=use_nominal_anchor)
        n_t = int(t_obs['task_x'].shape[0])
        n_r = int(t_obs['total_nodes'])

        tx_list.append(t_obs['task_x'])
        te_list.append(t_obs['task_edges'] + t_off)
        tb_list.append(torch.full((n_t,), b, dtype=torch.long))
        t_counts.append(n_t)
        focus_list.append(t_off + int(t_obs['focus']))

        if t_obs['task_srv_edges'].numel():
            tse = t_obs['task_srv_edges'].clone()
            tse[0] += t_off
            tse[1] += r_off
            tse_list.append(tse)
            tsa_list.append(t_obs['task_srv_attr'])

        rx_list.append(t_obs['resource_x'])
        re_list.append(t_obs['resource_edges'] + r_off)
        ea_list.append(t_obs['resource_edge_attr'])

        srv_idx_list.append(torch.arange(r_off, r_off + n_srv, dtype=torch.long))
        rb_list.append(torch.full((n_r,), b, dtype=torch.long))
        rl_list.append(torch.arange(n_r, dtype=torch.long))

        n_e = int(t_obs['resource_edges'].shape[1])
        reb_list.append(torch.full((n_e,), b, dtype=torch.long))
        rels_list.append(t_obs['resource_edges'][0])
        reld_list.append(t_obs['resource_edges'][1])

        px_list.append(t_obs['pair_x'])
        cf_list.append(t_obs['candidate_features'])
        eft_list.append(t_obs['rel_flow_eft'])

        t_off += n_t
        r_off += n_r

    return BatchedObs(
        batch_size=B,
        num_servers=n_srv,
        total_res_nodes=r_off,
        task_x=torch.cat(tx_list, dim=0).to(device),
        task_edges=torch.cat(te_list, dim=1).to(device),
        task_batch=torch.cat(tb_list, dim=0).to(device),
        task_counts=torch.tensor(t_counts, dtype=torch.float32, device=device).unsqueeze(-1),
        focus_idx=torch.tensor(focus_list, dtype=torch.long, device=device),
        task_srv_edges=torch.cat(tse_list, dim=1).to(device) if tse_list else torch.zeros((2, 0), dtype=torch.long, device=device),
        task_srv_attr=torch.cat(tsa_list, dim=0).to(device) if tsa_list else torch.zeros((0, 2), dtype=torch.float32, device=device),
        resource_x=torch.cat(rx_list, dim=0).to(device),
        resource_edges=torch.cat(re_list, dim=1).to(device),
        resource_edge_attr=torch.cat(ea_list, dim=0).to(device),
        srv_indices=torch.cat(srv_idx_list, dim=0).to(device),
        res_batch_ids=torch.cat(rb_list, dim=0).to(device),
        res_local_ids=torch.cat(rl_list, dim=0).to(device),
        res_edge_batch=torch.cat(reb_list, dim=0).to(device),
        res_edge_local_src=torch.cat(rels_list, dim=0).to(device),
        res_edge_local_dst=torch.cat(reld_list, dim=0).to(device),
        pair_x=torch.cat(px_list, dim=0).to(device),
        candidate_features=torch.cat(cf_list, dim=0).to(device),
        rel_flow_eft=torch.cat(eft_list, dim=0).reshape(B, n_srv).to(device),
    )


def _encode_res_batch(policy: DualGFlowPolicy, res_enc, rx_enriched: torch.Tensor, batch: BatchedObs) -> torch.Tensor:
    if policy.variant == 'dualgflow':
        h = F.gelu(res_enc.in_proj(rx_enriched))
        for layer in res_enc.layers:
            h = layer(h, batch.resource_edges, batch.resource_edge_attr)
        return h[batch.srv_indices]

    if policy.variant == 'no_mp_keep_links':
        n = batch.total_res_nodes
        inc = rx_enriched.new_zeros((n, 6)).index_add(0, batch.resource_edges[1], batch.resource_edge_attr)
        out = rx_enriched.new_zeros((n, 6)).index_add(0, batch.resource_edges[0], batch.resource_edge_attr)
        deg_in = torch.bincount(batch.resource_edges[1], minlength=n).clamp_min(1).to(rx_enriched.dtype).unsqueeze(-1)
        deg_out = torch.bincount(batch.resource_edges[0], minlength=n).clamp_min(1).to(rx_enriched.dtype).unsqueeze(-1)
        feat = torch.cat([rx_enriched, inc / deg_in, out / deg_out], dim=-1)
        return res_enc.net(feat[batch.srv_indices])

    if policy.variant == 'no_rg':
        srv_rx = rx_enriched[batch.srv_indices]
        srv_only = torch.cat([srv_rx[:, :5], srv_rx[:, 9:]], dim=-1)
        return res_enc.net(srv_only)

    # global_mlp
    m = res_enc.max_nodes
    B = batch.batch_size
    padded_x = rx_enriched.new_zeros((B, m, res_enc.rx_dim))
    n_mask = batch.res_local_ids < m
    padded_x[batch.res_batch_ids[n_mask], batch.res_local_ids[n_mask]] = rx_enriched[n_mask]

    dense_e = rx_enriched.new_zeros((B, m, m, 7))
    e_mask = (batch.res_edge_local_src < m) & (batch.res_edge_local_dst < m)
    eb = batch.res_edge_batch[e_mask]
    eu = batch.res_edge_local_src[e_mask]
    ev = batch.res_edge_local_dst[e_mask]
    dense_e[eb, eu, ev, 0] = 1.0
    dense_e[eb, eu, ev, 1:] = batch.resource_edge_attr[e_mask]

    flat = torch.cat([padded_x.reshape(B, -1), dense_e.reshape(B, -1)], dim=-1)
    out = res_enc.net(flat).reshape(B, res_enc.max_servers, res_enc.hidden_dim)
    if batch.num_servers <= res_enc.max_servers:
        global_h = out[:, :batch.num_servers, :].reshape(B * batch.num_servers, res_enc.hidden_dim)
    else:
        repeats = math.ceil(batch.num_servers / res_enc.max_servers)
        global_h = out.repeat(1, repeats, 1)[:, :batch.num_servers, :].reshape(B * batch.num_servers, res_enc.hidden_dim)
    return res_enc.norm(global_h + res_enc.local_proj(rx_enriched[batch.srv_indices]))


def _encode_joint_batch(policy: DualGFlowPolicy, batch: BatchedObs, task_enc, ts_enc, res_enc) -> torch.Tensor:
    B = batch.batch_size
    n_srv = batch.num_servers
    h_tasks = task_enc(batch.task_x, batch.task_edges)
    h_focus = h_tasks[batch.focus_idx].unsqueeze(1).expand(B, n_srv, -1).reshape(B * n_srv, -1)

    wf_sum = h_tasks.new_zeros((B, h_tasks.shape[1])).index_add(0, batch.task_batch, h_tasks)
    h_wf_ctx = (wf_sum / batch.task_counts).unsqueeze(1).expand(B, n_srv, -1).reshape(B * n_srv, -1)

    srv_task_ctx = ts_enc(h_tasks, batch.task_srv_edges, batch.task_srv_attr, batch.total_res_nodes)
    rx_enriched = torch.cat([batch.resource_x, srv_task_ctx], dim=-1)
    h_srv = _encode_res_batch(policy, res_enc, rx_enriched, batch)

    if policy.mode == 'pure_structural':
        local_cpu = batch.candidate_features[:, [0, 1, 3]]
        return torch.cat([h_focus, h_wf_ctx, h_srv, local_cpu], dim=-1)
    return torch.cat([h_focus, h_wf_ctx, h_srv, batch.pair_x, batch.candidate_features], dim=-1)


def forward_actor_batch(policy: DualGFlowPolicy, batch: BatchedObs) -> torch.Tensor:
    """Returns (B, num_servers) actor logits."""
    B, n_srv = batch.batch_size, batch.num_servers
    joint = _encode_joint_batch(policy, batch, policy.task_encoder, policy.task_to_srv, policy.res_encoder)
    cand_h = policy.candidate_proj(joint).reshape(B, n_srv, policy.hidden_dim)
    global_mean = cand_h.mean(dim=1, keepdim=True).expand(B, n_srv, policy.hidden_dim)
    raw_logits = policy.scorer(torch.cat([cand_h, cand_h * global_mean], dim=-1)).squeeze(-1)
    if policy.mode == 'residual_eft':
        if getattr(policy, 'use_tropical_wave', False):
            with torch.no_grad():
                _, d_cong = compute_tropical_wave_delays(
                    batch.resource_x, batch.resource_edges, batch.resource_edge_attr,
                    local_ids=batch.res_local_ids, L=6, M=max(16, n_srv),
                )
                dc_srv = d_cong[batch.srv_indices]
                exec_t = torch.expm1(batch.candidate_features[:, 0] * 8.0)
                cpu_av = torch.expm1(batch.candidate_features[:, 1] * 8.0)
                nom_d = torch.expm1(batch.pair_x[:, 3] * 8.0)
                wave_eft = torch.maximum(cpu_av, torch.maximum(nom_d, dc_srv)).reshape(B, n_srv) + exec_t.reshape(B, n_srv)
                w_min = wave_eft.min(dim=1, keepdim=True).values
                w_med = wave_eft.median(dim=1, keepdim=True).values
                rel_anchor = torch.clamp((wave_eft - w_min) / (w_med - w_min).clamp_min(1e-3), 0.0, 5.0)
        else:
            rel_anchor = batch.rel_flow_eft
        return -2.0 * rel_anchor + getattr(policy, 'residual_scale', 4.0) * torch.tanh(raw_logits)
    return raw_logits


def forward_critic_batch(policy: DualGFlowPolicy, batch: BatchedObs) -> torch.Tensor:
    """Returns (B,) scalar state values."""
    B, n_srv = batch.batch_size, batch.num_servers
    joint = _encode_joint_batch(policy, batch, policy.critic_task_encoder, policy.critic_task_to_srv, policy.critic_res_encoder)
    cand_h = policy.critic_proj(joint).reshape(B, n_srv, policy.hidden_dim)
    global_pool = torch.cat([cand_h.mean(dim=1), cand_h.max(dim=1).values], dim=-1)
    return policy.critic_head(global_pool).squeeze(-1)
