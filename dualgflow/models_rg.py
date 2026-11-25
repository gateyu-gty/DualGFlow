"""Scale-Free Heterogeneous Task-Resource GNN (DualGFlow-RG v2) and Fair Ablations.

Variants:
- 'dualgflow': L-layer Edge-Attentive Message Passing over Server+Switch Resource Graph
               with bidirectional flow propagation and Parent-Data Injection.
- 'no_mp_keep_links': 1-hop incident link aggregation only (no multi-hop message passing).
- 'global_mlp': Flattened global X/A/E MLP encoder (fixed-slot, non-equivariant).

Modes:
- 'pure_structural': No handcrafted EFT or pre-summed route delays given to the network;
                     tests pure structural reasoning over the physical resource graph.
- 'residual_eft': Combines learned structural correction with Flow-CA-EFT prior anchor.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def make_mlp(in_dim: int, hidden_dim: int, out_dim: int, layers: int = 3) -> nn.Sequential:
    mods = []
    d = in_dim
    for _ in range(layers - 1):
        mods.append(nn.Linear(d, hidden_dim))
        mods.append(nn.LayerNorm(hidden_dim))
        mods.append(nn.GELU())
        d = hidden_dim
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


def transform_obs(obs: dict, device: torch.device | str = 'cpu', use_nominal_anchor: bool = False) -> dict[str, torch.Tensor]:
    """Scale-invariant log-normalization of raw physical observations."""
    tx = obs['task_x'].copy()
    # [status, work, wf_rem_work, up_rank, out_deg, max_child_bytes, assigned_speed]
    tx[:, 1] = np.log1p(np.maximum(0.0, tx[:, 1])) / 10.0
    tx[:, 2] = np.log1p(np.maximum(0.0, tx[:, 2])) / 12.0
    tx[:, 3] = np.log1p(np.maximum(0.0, tx[:, 3])) / 10.0
    tx[:, 4] = np.log1p(np.maximum(0.0, tx[:, 4])) / 3.0
    tx[:, 5] = np.log1p(np.maximum(0.0, tx[:, 5])) / 20.0

    rx = obs['resource_x'].copy()
    # [is_server, speed, run_rem, q_work, q_len, has_parent, parent_bytes, max_parent_bytes, active_flows]
    rx[:, 2] = np.log1p(np.maximum(0.0, rx[:, 2])) / 10.0
    rx[:, 3] = np.log1p(np.maximum(0.0, rx[:, 3])) / 10.0
    rx[:, 4] = np.log1p(np.maximum(0.0, rx[:, 4])) / 3.0
    rx[:, 6] = np.log1p(np.maximum(0.0, rx[:, 6])) / 20.0
    rx[:, 7] = np.log1p(np.maximum(0.0, rx[:, 7])) / 20.0
    rx[:, 8] = np.log1p(np.maximum(0.0, rx[:, 8])) / 3.0

    ea = obs['resource_edge_attr'].copy()
    # [bandwidth, latency, n_flows, fair_share_bw, rem_bytes, drain_time]
    ea[:, 0] = np.log1p(np.maximum(0.0, ea[:, 0])) / 20.0
    ea[:, 1] = ea[:, 1] * 100.0
    ea[:, 2] = np.log1p(np.maximum(0.0, ea[:, 2])) / 3.0
    ea[:, 3] = np.log1p(np.maximum(0.0, ea[:, 3])) / 20.0
    ea[:, 4] = np.log1p(np.maximum(0.0, ea[:, 4])) / 20.0
    ea[:, 5] = np.log1p(np.maximum(0.0, ea[:, 5])) / 8.0

    px = obs['pair_x'].copy()
    if use_nominal_anchor and 'nominal_delay' in obs:
        px[:, 3] = obs['nominal_delay']
    # [remote_bytes, local_frac, mean_hops, delay]
    px[:, 0] = np.log1p(np.maximum(0.0, px[:, 0])) / 20.0
    px[:, 2] = px[:, 2] / 6.0
    px[:, 3] = np.log1p(np.maximum(0.0, px[:, 3])) / 8.0

    cf = obs['candidate_features'].copy()
    if use_nominal_anchor:
        cf[:, 2] = obs['nominal_eft'].astype(np.float32)
    # [exec_time, cpu_avail, eft, speed]
    cf[:, 0] = np.log1p(np.maximum(0.0, cf[:, 0])) / 8.0
    cf[:, 1] = np.log1p(np.maximum(0.0, cf[:, 1])) / 8.0
    cf[:, 2] = np.log1p(np.maximum(0.0, cf[:, 2])) / 8.0

    # Relative EFT normalized by (median - min)
    raw_eft = (obs['nominal_eft'] if use_nominal_anchor else obs['flow_eft']).astype(np.float32)
    eft_min = float(np.min(raw_eft))
    eft_med = float(np.median(raw_eft))
    rel_eft = (raw_eft - eft_min) / max(eft_med - eft_min, 1e-3)
    rel_eft = np.clip(rel_eft, 0.0, 5.0)

    raw_comp_eft = obs['compute_eft'].astype(np.float32)
    comp_std = float(np.std(raw_comp_eft))
    rel_comp_eft = (raw_comp_eft - float(np.mean(raw_comp_eft))) / max(comp_std, 1e-3)

    ts_attr = obs['task_srv_attr'].copy()
    if len(ts_attr):
        ts_attr[:, 1] = np.log1p(np.maximum(0.0, ts_attr[:, 1])) / 20.0

    return dict(
        num_servers=int(obs['num_servers']),
        total_nodes=int(obs['total_nodes']),
        task_x=torch.as_tensor(tx, dtype=torch.float32, device=device),
        task_edges=torch.as_tensor(obs['task_edges'], dtype=torch.long, device=device),
        task_srv_edges=torch.as_tensor(obs['task_srv_edges'], dtype=torch.long, device=device),
        task_srv_attr=torch.as_tensor(ts_attr, dtype=torch.float32, device=device),
        focus=int(obs['focus']),
        resource_x=torch.as_tensor(rx, dtype=torch.float32, device=device),
        resource_edges=torch.as_tensor(obs['resource_edges'], dtype=torch.long, device=device),
        resource_edge_attr=torch.as_tensor(ea, dtype=torch.float32, device=device),
        pair_x=torch.as_tensor(px, dtype=torch.float32, device=device),
        candidate_features=torch.as_tensor(cf, dtype=torch.float32, device=device),
        rel_flow_eft=torch.as_tensor(rel_eft, dtype=torch.float32, device=device),
        rel_compute_eft=torch.as_tensor(rel_comp_eft, dtype=torch.float32, device=device),
    )


class TaskDAGEncoder(nn.Module):
    """2-layer bidirectional message-passing encoder over active workflow DAGs.

    Passes messages both forward (parent -> child) and backward (child -> parent)
    so that a ready focus task (whose parents are already done and pruned) receives
    rich downstream subtree, fan-out, and critical-path signals from its children.
    """
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.in_proj = nn.Linear(7, hidden_dim)
        self.fwd1 = make_mlp(hidden_dim * 2, hidden_dim, hidden_dim, layers=2)
        self.bwd1 = make_mlp(hidden_dim * 2, hidden_dim, hidden_dim, layers=2)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fwd2 = make_mlp(hidden_dim * 2, hidden_dim, hidden_dim, layers=2)
        self.bwd2 = make_mlp(hidden_dim * 2, hidden_dim, hidden_dim, layers=2)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def _mp_step(
        self,
        h: torch.Tensor,
        edges: torch.Tensor,
        fwd_net: nn.Module,
        bwd_net: nn.Module,
        norm: nn.Module,
    ) -> torch.Tensor:
        if edges.numel() == 0:
            return h
        src, dst = edges[0], edges[1]
        # Forward: parent -> child
        m_f = fwd_net(torch.cat([h[src], h[dst]], dim=-1))
        agg_f = torch.zeros_like(h).index_add(0, dst, m_f)
        deg_f = torch.bincount(dst, minlength=len(h)).clamp_min(1).to(h.dtype).unsqueeze(-1)
        # Backward: child -> parent (crucial so ready tasks see downstream children!)
        m_b = bwd_net(torch.cat([h[dst], h[src]], dim=-1))
        agg_b = torch.zeros_like(h).index_add(0, src, m_b)
        deg_b = torch.bincount(src, minlength=len(h)).clamp_min(1).to(h.dtype).unsqueeze(-1)
        return norm(h + F.gelu(agg_f / deg_f + agg_b / deg_b))

    def forward(self, x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.in_proj(x))
        h = self._mp_step(h, edges, self.fwd1, self.bwd1, self.norm1)
        h = self._mp_step(h, edges, self.fwd2, self.bwd2, self.norm2)
        return h


class EdgeAttentiveResourceLayer(nn.Module):
    """Edge-aware attention message passing layer with Mean+Max aggregation (captures bottlenecks & slowest parent)."""
    def __init__(self, hidden_dim: int, edge_dim: int = 6):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.msg_net = make_mlp(hidden_dim * 2 + edge_dim, hidden_dim, hidden_dim, layers=2)
        self.update_net = nn.Linear(hidden_dim * 3, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, edges: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        if edges.numel() == 0:
            return h
        src, dst = edges[0], edges[1]
        cat_in = torch.cat([h[src], h[dst], edge_attr], dim=-1)
        gate = torch.sigmoid(self.attn_net(cat_in))
        msg = gate * self.msg_net(cat_in)

        agg_sum = torch.zeros_like(h).index_add(0, dst, msg)
        deg = torch.bincount(dst, minlength=len(h)).clamp_min(1).to(h.dtype).unsqueeze(-1)
        agg_mean = agg_sum / deg

        # Max aggregation captures bottleneck link / slowest parent signal without dilution
        agg_max = torch.full_like(h, -1e4)
        agg_max.index_reduce_(0, dst, msg, reduce='amax', include_self=False)
        has_in = (torch.bincount(dst, minlength=len(h)) > 0).unsqueeze(-1)
        agg_max = torch.where(has_in, agg_max, torch.zeros_like(agg_max))

        out = self.update_net(torch.cat([h, agg_mean, agg_max], dim=-1))
        return self.norm(h + F.gelu(out))


class TaskToServerBipartite(nn.Module):
    """Projects active task embeddings (queued/running + pending data deps) onto physical servers."""
    def __init__(self, hidden_dim: int = 64, out_dim: int = 16):
        super().__init__()
        self.out_dim = out_dim
        self.msg = make_mlp(hidden_dim + 2, hidden_dim, out_dim, layers=2)

    def forward(
        self,
        h_tasks: torch.Tensor,
        ts_edges: torch.Tensor,
        ts_attr: torch.Tensor,
        total_nodes: int,
    ) -> torch.Tensor:
        out = h_tasks.new_zeros((total_nodes, self.out_dim))
        if ts_edges.numel() == 0:
            return out
        t_idx, s_idx = ts_edges[0], ts_edges[1]
        m = self.msg(torch.cat([h_tasks[t_idx], ts_attr], dim=-1))
        agg_sum = h_tasks.new_zeros((total_nodes, self.out_dim)).index_add(0, s_idx, m)
        deg = torch.bincount(s_idx, minlength=total_nodes).clamp_min(1).to(h_tasks.dtype).unsqueeze(-1)
        return agg_sum / deg


def compute_tropical_wave_delays(
    rx: torch.Tensor,
    edges: torch.Tensor,
    edge_attr: torch.Tensor,
    local_ids: torch.Tensor | None = None,
    L: int = 6,
    M: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """L-layer Tropical Wave (min-plus / max-plus) message passing over the physical resource graph.

    Propagates parent-data transfer waves across switches without degree-dilution,
    inferring multi-hop shared-link congestion purely from raw node & edge attributes.
    """
    device = rx.device
    V = rx.shape[0]
    if edges.numel() == 0:
        z = rx.new_zeros((V,))
        return z, z
    if local_ids is None:
        local_ids = torch.arange(V, device=device)
    src, dst = edges[0], edges[1]
    bw = torch.expm1(edge_attr[:, 0] * 20.0).clamp_min(1e-3)
    lat = edge_attr[:, 1] / 100.0
    fair_bw = torch.expm1(edge_attr[:, 3] * 20.0).clamp_min(1e-3)
    drain_t = torch.expm1(edge_attr[:, 5] * 8.0).clamp_min(0.0)

    has_p = rx[:, 5] > 0.5
    p_bytes = torch.expm1(rx[:, 7] * 20.0).clamp_min(0.0)
    INF = 1e9
    d_nom = torch.full((V, M), INF, device=device)
    d_cong = torch.full((V, M), INF, device=device)
    prop = torch.zeros((V, M), device=device)
    t_nom = torch.zeros((V, M), device=device)
    t_cong = torch.zeros((V, M), device=device)
    s_val = torch.zeros((V, M), device=device)

    slots = (local_ids % M).long()
    p_nodes = torch.where(has_p)[0]
    if p_nodes.numel() > 0:
        p_slots = slots[p_nodes]
        d_nom[p_nodes, p_slots] = 0.0
        d_cong[p_nodes, p_slots] = 0.0
        s_val[p_nodes, p_slots] = p_bytes[p_nodes]

    for _ in range(L):
        s_e = s_val[src]
        valid = d_nom[src] < 1e8
        prop_e = prop[src] + lat.unsqueeze(1)
        tn_e = torch.maximum(t_nom[src], s_e / bw.unsqueeze(1))
        tc_link = torch.minimum(drain_t.unsqueeze(1) + s_e / bw.unsqueeze(1), s_e / fair_bw.unsqueeze(1))
        tc_e = torch.maximum(t_cong[src], tc_link)
        dn_e = torch.where(valid, prop_e + tn_e, torch.full_like(prop_e, INF))
        dc_e = torch.where(valid, prop_e + tc_e, torch.full_like(prop_e, INF))

        min_dn = d_nom.clone()
        min_dn.index_reduce_(0, dst, dn_e, reduce='amin', include_self=True)
        improved = (min_dn < d_nom - 1e-5)
        is_best_e = (dn_e <= min_dn[dst] + 1e-5) & (dn_e < 1e8)

        new_dc = d_cong.clone(); new_dc.index_reduce_(0, dst, torch.where(is_best_e, dc_e, torch.full_like(dc_e, INF)), reduce='amin', include_self=False)
        new_prop = prop.clone(); new_prop.index_reduce_(0, dst, torch.where(is_best_e, prop_e, torch.full_like(prop_e, INF)), reduce='amin', include_self=False)
        new_tn = t_nom.clone(); new_tn.index_reduce_(0, dst, torch.where(is_best_e, tn_e, torch.full_like(tn_e, INF)), reduce='amin', include_self=False)
        new_tc = t_cong.clone(); new_tc.index_reduce_(0, dst, torch.where(is_best_e, tc_e, torch.full_like(tc_e, INF)), reduce='amin', include_self=False)
        new_s = s_val.clone(); new_s.index_reduce_(0, dst, torch.where(is_best_e, s_e, torch.full_like(s_e, 0.0)), reduce='amax', include_self=False)

        d_nom = torch.where(improved, min_dn, d_nom)
        d_cong = torch.where(improved, new_dc, d_cong)
        prop = torch.where(improved, new_prop, prop)
        t_nom = torch.where(improved, new_tn, t_nom)
        t_cong = torch.where(improved, new_tc, t_cong)
        s_val = torch.where(improved, new_s, s_val)

    reach = d_nom < 1e8
    max_dn = torch.where(reach, d_nom, torch.zeros_like(d_nom)).max(dim=1).values
    max_dc = torch.where(reach, d_cong, torch.zeros_like(d_cong)).max(dim=1).values
    return max_dn, max_dc


class ResourceGraphGNN(nn.Module):
    """Multi-hop Resource GNN over servers and switches with Parent-Data & Task-Binding Injection."""
    def __init__(self, hidden_dim: int = 128, num_layers: int = 4, rx_dim: int = 25):
        super().__init__()
        self.in_proj = nn.Linear(rx_dim, hidden_dim)
        self.layers = nn.ModuleList([
            EdgeAttentiveResourceLayer(hidden_dim, edge_dim=6) for _ in range(num_layers)
        ])

    def forward(self, rx: torch.Tensor, edges: torch.Tensor, edge_attr: torch.Tensor, num_servers: int) -> torch.Tensor:
        h = F.gelu(self.in_proj(rx))
        for layer in self.layers:
            h = layer(h, edges, edge_attr)
        return h[:num_servers]


class NoMPKeepLinksEncoder(nn.Module):
    """Ablation: Keeps 1-hop incident link attributes, disables multi-hop node message passing."""
    def __init__(self, hidden_dim: int = 128, rx_dim: int = 25):
        super().__init__()
        # rx_dim node features + 6 incoming link mean + 6 outgoing link mean
        self.net = make_mlp(rx_dim + 12, hidden_dim * 2, hidden_dim, layers=3)

    def forward(self, rx: torch.Tensor, edges: torch.Tensor, edge_attr: torch.Tensor, num_servers: int) -> torch.Tensor:
        n = len(rx)
        inc = rx.new_zeros((n, 6)).index_add(0, edges[1], edge_attr)
        out = rx.new_zeros((n, 6)).index_add(0, edges[0], edge_attr)
        deg_in = torch.bincount(edges[1], minlength=n).clamp_min(1).to(rx.dtype).unsqueeze(-1)
        deg_out = torch.bincount(edges[0], minlength=n).clamp_min(1).to(rx.dtype).unsqueeze(-1)
        feat = torch.cat([rx, inc / deg_in, out / deg_out], dim=-1)
        return self.net(feat[:num_servers])


class GlobalMLPEncoder(nn.Module):
    """Fair Global-MLP baseline: complete X/A/E (all 6 edge channels + adjacency) + local node skip."""
    def __init__(self, hidden_dim: int = 64, max_nodes: int = 24, max_servers: int = 16, rx_dim: int = 25):
        super().__init__()
        self.max_nodes = max_nodes
        self.max_servers = max_servers
        self.rx_dim = rx_dim
        # Complete X (rx_dim) + A (1) + E (6) = 7 edge channels
        in_dim = max_nodes * rx_dim + max_nodes * max_nodes * 7
        self.net = make_mlp(in_dim, 512, max_servers * hidden_dim, layers=3)
        self.local_proj = make_mlp(rx_dim, hidden_dim, hidden_dim, layers=2)
        self.norm = nn.LayerNorm(hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, rx: torch.Tensor, edges: torch.Tensor, edge_attr: torch.Tensor, num_servers: int) -> torch.Tensor:
        m = self.max_nodes
        padded_x = rx.new_zeros((m, self.rx_dim))
        use_n = min(len(rx), m)
        padded_x[:use_n] = rx[:use_n]

        dense_e = rx.new_zeros((m, m, 7))
        mask = (edges[0] < m) & (edges[1] < m)
        u, v = edges[0][mask], edges[1][mask]
        dense_e[u, v, 0] = 1.0  # adjacency indicator
        dense_e[u, v, 1:] = edge_attr[mask]

        flat = torch.cat([padded_x.reshape(-1), dense_e.reshape(-1)], dim=0)
        out = self.net(flat).reshape(self.max_servers, self.hidden_dim)
        if num_servers <= self.max_servers:
            global_h = out[:num_servers]
        else:
            repeats = math.ceil(num_servers / self.max_servers)
            global_h = out.repeat(repeats, 1)[:num_servers]
        # Skip-connect each server's own local features so global MLP doesn't waste capacity reconstructing rx[i]
        return self.norm(global_h + self.local_proj(rx[:num_servers]))


class DualGFlowNoRGEncoder(nn.Module):
    """External Baseline (Original DualGFlow ICLR 2025): Server CPU & task-queue state only, NO resource graph edges."""
    def __init__(self, hidden_dim: int = 128, rx_dim: int = 25):
        super().__init__()
        # Uses only server compute/queue features [0:5] + bipartite task context [9:rx_dim]
        # (ignores parent-data injection [5:9] and all physical network link attributes)
        self.net = make_mlp(5 + (rx_dim - 9), hidden_dim * 2, hidden_dim, layers=3)

    def forward(self, rx: torch.Tensor, edges: torch.Tensor, edge_attr: torch.Tensor, num_servers: int) -> torch.Tensor:
        srv_only = torch.cat([rx[:num_servers, :5], rx[:num_servers, 9:]], dim=-1)
        return self.net(srv_only)


class DualGFlowPolicy(nn.Module):
    """Joint Task-DAG and Physical-Resource Actor-Critic Policy."""
    def __init__(
        self,
        variant: str = 'dualgflow',
        mode: str = 'pure_structural',
        hidden_dim: int = 128,
        gnn_layers: int = 4,
        use_nominal_anchor: bool = False,
        residual_scale: float = 4.0,
        use_tropical_wave: bool = False,
    ):
        super().__init__()
        if variant not in ('dualgflow', 'no_mp_keep_links', 'global_mlp', 'no_rg'):
            raise ValueError(f'Unknown variant {variant}')
        if mode not in ('pure_structural', 'residual_eft'):
            raise ValueError(f'Unknown mode {mode}')

        self.variant = variant
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.use_nominal_anchor = bool(use_nominal_anchor)
        self.residual_scale = float(residual_scale)
        self.use_tropical_wave = bool(use_tropical_wave)

        self.task_encoder = TaskDAGEncoder(hidden_dim)
        self.task_to_srv = TaskToServerBipartite(hidden_dim, out_dim=16)
        if variant == 'dualgflow':
            self.res_encoder = ResourceGraphGNN(hidden_dim, num_layers=gnn_layers, rx_dim=25)
        elif variant == 'no_mp_keep_links':
            self.res_encoder = NoMPKeepLinksEncoder(hidden_dim, rx_dim=25)
        elif variant == 'no_rg':
            self.res_encoder = DualGFlowNoRGEncoder(hidden_dim, rx_dim=25)
        else:
            self.res_encoder = GlobalMLPEncoder(hidden_dim, rx_dim=25)

        # In pure_structural mode, only server local compute [exec_time, cpu_avail, speed] is appended (no flow_eft/pair_x)
        extra_dim = 3 if mode == 'pure_structural' else (4 + 4)
        joint_dim = hidden_dim * 3 + extra_dim

        self.candidate_proj = make_mlp(joint_dim, hidden_dim, hidden_dim, layers=2)
        self.scorer = make_mlp(hidden_dim * 2, hidden_dim, 1, layers=2)

        if mode == 'residual_eft':
            # Initialize last scorer layer small so policy starts close to CA-EFT
            nn.init.zeros_(self.scorer[-1].weight)
            nn.init.zeros_(self.scorer[-1].bias)

        # Decoupled Critic encoder (DualGFlow ICLR 2025 principle: prevent value gradients
        # from destabilizing Actor GNN representations)
        self.critic_task_encoder = TaskDAGEncoder(hidden_dim)
        self.critic_task_to_srv = TaskToServerBipartite(hidden_dim, out_dim=16)
        if variant == 'dualgflow':
            self.critic_res_encoder = ResourceGraphGNN(hidden_dim, num_layers=gnn_layers, rx_dim=25)
        elif variant == 'no_mp_keep_links':
            self.critic_res_encoder = NoMPKeepLinksEncoder(hidden_dim, rx_dim=25)
        elif variant == 'no_rg':
            self.critic_res_encoder = DualGFlowNoRGEncoder(hidden_dim, rx_dim=25)
        else:
            self.critic_res_encoder = GlobalMLPEncoder(hidden_dim, rx_dim=25)
        self.critic_proj = make_mlp(joint_dim, hidden_dim, hidden_dim, layers=2)
        self.critic_head = make_mlp(hidden_dim * 2, hidden_dim, 1, layers=2)

    def actor_parameters(self):
        return (
            list(self.task_encoder.parameters())
            + list(self.task_to_srv.parameters())
            + list(self.res_encoder.parameters())
            + list(self.candidate_proj.parameters())
            + list(self.scorer.parameters())
        )

    def critic_parameters(self):
        return (
            list(self.critic_task_encoder.parameters())
            + list(self.critic_task_to_srv.parameters())
            + list(self.critic_res_encoder.parameters())
            + list(self.critic_proj.parameters())
            + list(self.critic_head.parameters())
        )

    def _encode_joint(self, t_obs: dict, task_enc: nn.Module, ts_enc: nn.Module, res_enc: nn.Module) -> torch.Tensor:
        n_srv = t_obs['num_servers']
        h_tasks = task_enc(t_obs['task_x'], t_obs['task_edges'])
        h_focus = h_tasks[t_obs['focus']].unsqueeze(0).expand(n_srv, -1)
        h_wf_ctx = h_tasks.mean(dim=0, keepdim=True).expand(n_srv, -1)
        srv_task_ctx = ts_enc(
            h_tasks,
            t_obs['task_srv_edges'],
            t_obs['task_srv_attr'],
            t_obs['total_nodes'],
        )
        rx_enriched = torch.cat([t_obs['resource_x'], srv_task_ctx], dim=-1)
        h_srv = res_enc(
            rx_enriched,
            t_obs['resource_edges'],
            t_obs['resource_edge_attr'],
            n_srv,
        )
        if self.mode == 'pure_structural':
            local_cpu = t_obs['candidate_features'][:, [0, 1, 3]]
            return torch.cat([h_focus, h_wf_ctx, h_srv, local_cpu], dim=-1)
        return torch.cat([h_focus, h_wf_ctx, h_srv, t_obs['pair_x'], t_obs['candidate_features']], dim=-1)

    def forward_actor(self, obs: dict) -> torch.Tensor:
        device = next(self.parameters()).device
        t_obs = transform_obs(obs, device=device, use_nominal_anchor=self.use_nominal_anchor)
        n_srv = t_obs['num_servers']
        joint = self._encode_joint(t_obs, self.task_encoder, self.task_to_srv, self.res_encoder)
        cand_h = self.candidate_proj(joint)
        global_ctx = torch.cat([cand_h.mean(dim=0), cand_h.max(dim=0).values], dim=-1)
        raw_logits = self.scorer(torch.cat([cand_h, cand_h * global_ctx[:self.hidden_dim].unsqueeze(0).expand(n_srv, -1)], dim=-1)).squeeze(-1)
        if self.mode == 'residual_eft':
            if self.use_tropical_wave:
                with torch.no_grad():
                    _, d_cong = compute_tropical_wave_delays(
                        t_obs['resource_x'], t_obs['resource_edges'], t_obs['resource_edge_attr'], L=6, M=max(16, n_srv)
                    )
                    exec_t = torch.expm1(t_obs['candidate_features'][:, 0] * 8.0)
                    cpu_av = torch.expm1(t_obs['candidate_features'][:, 1] * 8.0)
                    nom_d = torch.expm1(t_obs['pair_x'][:, 3] * 8.0)
                    wave_eft = torch.maximum(cpu_av, torch.maximum(nom_d, d_cong[:n_srv])) + exec_t
                    w_min = wave_eft.min()
                    w_med = wave_eft.median()
                    rel_anchor = torch.clamp((wave_eft - w_min) / (w_med - w_min).clamp_min(1e-3), 0.0, 5.0)
            else:
                rel_anchor = t_obs['rel_flow_eft']
            return -2.0 * rel_anchor + self.residual_scale * torch.tanh(raw_logits)
        return raw_logits

    def forward_critic(self, obs: dict) -> torch.Tensor:
        device = next(self.parameters()).device
        t_obs = transform_obs(obs, device=device, use_nominal_anchor=self.use_nominal_anchor)
        joint = self._encode_joint(t_obs, self.critic_task_encoder, self.critic_task_to_srv, self.critic_res_encoder)
        cand_h = self.critic_proj(joint)
        global_pool = torch.cat([cand_h.mean(dim=0), cand_h.max(dim=0).values], dim=-1)
        return self.critic_head(global_pool).squeeze(-1)

    def forward_obs(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(obs), self.forward_critic(obs)
