"""Standard scalable network topologies (with switch nodes) and DAX workload generator.

Supports:
1. Leaf-Spine (Data Center with configurable oversubscription ratio, e.g. 4:1)
2. Cloud-Edge-Mist (3-tier hierarchy with constrained WAN uplinks and fast cloud core)
3. Multi-Hop Mesh (Irregular multi-hop topology with high routing diameter)
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import networkx as nx
import numpy as np

DAX_DIR = Path(__file__).resolve().parents[1] / 'env' / 'workflow_scheduling_v3' / 'dax'

FILES = {
    'S': ('CyberShake_30', 'Montage_25', 'Inspiral_30', 'Sipht_30'),
    'M': ('CyberShake_50', 'Montage_50', 'Inspiral_50', 'Sipht_60'),
    'L': ('CyberShake_100', 'Montage_100', 'Inspiral_100', 'Sipht_100'),
}
SPLITS = {'train': 1103, 'validation': 2207, 'test': 3301}
STREAMS = {'arrival': 11, 'topology': 23, 'templates': 37, 'parameters': 53}


def _rng(seed: int, split: str, stream: str) -> np.random.Generator:
    if split not in SPLITS:
        raise ValueError(f'Unknown split {split!r}; expected one of {tuple(SPLITS)}')
    return np.random.default_rng(np.random.SeedSequence([int(seed), SPLITS[split], STREAMS[stream]]))


def parse_dax(path: Path, family: str | None = None, size: str | None = None) -> dict:
    """Parse Pegasus DAX XML into task DAG with true file-matched inter-task byte volumes."""
    path = Path(path)
    root = ET.parse(path).getroot()
    ns = {'d': root.tag.split('}')[0].lstrip('{')}
    jobs = root.findall('d:job', ns)
    raw = {}
    for job in jobs:
        key = job.attrib['id']
        runtime = max(1e-3, float(job.attrib['runtime']))
        declared = {'input': {}, 'output': {}}
        for f in job.findall('d:uses', ns):
            side, name = f.attrib.get('link'), f.attrib.get('file')
            if side in declared and name and 'size' in f.attrib:
                amount = max(0.0, float(f.attrib['size']))
                declared[side].setdefault(name, []).append(amount)
        raw[key] = dict(runtime=runtime, files=declared)

    graph = nx.DiGraph()
    graph.add_nodes_from(raw)
    for child in root.findall('d:child', ns):
        target = child.attrib['ref']
        for parent in child.findall('d:parent', ns):
            source = parent.attrib['ref']
            if source in raw and target in raw:
                graph.add_edge(source, target)

    order = list(nx.lexicographical_topological_sort(graph))
    ids = {key: i for i, key in enumerate(order)}
    tasks = []
    for key in order:
        # Root stage-in bytes: input files not produced by any parent
        tasks.append(dict(
            id=ids[key],
            original_id=key,
            work=raw[key]['runtime'] * 16.0,
            xml_runtime=raw[key]['runtime'],
            parents={},
        ))

    edges = []
    for source, target in sorted(graph.edges()):
        out_files = raw[source]['files']['output']
        in_files = raw[target]['files']['input']
        shared = sorted(out_files.keys() & in_files.keys())
        byte_sum = float(sum(max(out_files[name] + in_files[name]) for name in shared))
        tasks[ids[target]]['parents'][ids[source]] = byte_sum
        edges.append(dict(source=ids[source], target=ids[target], bytes=byte_sum))

    return dict(
        name=path.stem,
        family=family or path.stem.split('_')[0],
        size=size,
        tasks=tasks,
        edges=edges,
        total_work=float(sum(t['work'] for t in tasks)),
    )


@lru_cache(maxsize=16)
def load_templates(sizes: tuple[str, ...] = ('S', 'M')) -> tuple[dict, ...]:
    templates = []
    for sz in sizes:
        for fname in FILES[sz]:
            templates.append(parse_dax(DAX_DIR / f'{fname}.xml', family=fname.split('_')[0], size=sz))
    return tuple(templates)


@lru_cache(maxsize=4)
def calibration_reference(sizes: tuple[str, ...] = ('S', 'M')) -> dict:
    templates = load_templates(sizes)
    work = np.array([t['work'] for tpl in templates for t in tpl['tasks']], dtype=np.float64)
    pos_bytes = np.array([e['bytes'] for tpl in templates for e in tpl['edges'] if e['bytes'] > 0], dtype=np.float64)
    return dict(
        reference_bytes=float(np.median(pos_bytes)),
        mean_positive_bytes=float(np.mean(pos_bytes)),
        mean_task_work=float(np.mean(work)),
        mean_workflow_work=float(np.mean([tpl['total_work'] for tpl in templates])),
    )


def make_physical_topology(
    num_servers: int,
    rng: np.random.Generator,
    reference_bytes: float,
    topology: str = 'leaf_spine',
    oversubscription: float = 4.0,
    speed_ratio: float = 4.0,
    link_jitter: float = 0.1,
) -> tuple[list[float], list[str], list[dict], dict, dict]:
    """Build a physical resource graph with compute servers and switch/router nodes.

    Returns:
        server_speeds: list of length num_servers (positive float CPU speeds)
        node_types: list of length total_nodes ('server' for 0..num_servers-1, 'switch' for switches)
        links: list of directed link dicts {src, dst, bandwidth, latency, tier}
        routes: dict mapping (src_node, dst_node) -> list of directed link keys (u, v)
        meta: topology metadata dict
    """
    if num_servers < 4:
        raise ValueError('num_servers must be at least 4')

    # Server speeds normalized to mean 1.0
    speeds = np.resize(np.linspace(1.0, speed_ratio, 4), num_servers).astype(np.float64)
    speeds /= speeds.mean()
    rng.shuffle(speeds)

    graph = nx.DiGraph()
    base_access_bw = 1e8  # 100 MB/s (800 Mbps) server access link

    if topology == 'leaf_spine':
        # Data center Leaf-Spine: racks of ~4 servers each, connected to Leaf (ToR) switches,
        # which connect to Spine switches with configurable oversubscription.
        num_racks = max(2, num_servers // 4)
        num_spines = max(2, num_racks // 2)
        total_nodes = num_servers + num_racks + num_spines
        node_types = ['server'] * num_servers + ['switch'] * (num_racks + num_spines)
        graph.add_nodes_from(range(total_nodes))

        racks = [list(map(int, g)) for g in np.array_split(rng.permutation(num_servers), num_racks)]
        leaf_ids = [num_servers + r for r in range(num_racks)]
        spine_ids = [num_servers + num_racks + s for s in range(num_spines)]

        # Server <-> Leaf (ToR) links: high bandwidth, low latency (0.2 ms)
        for r_idx, srv_group in enumerate(racks):
            tor = leaf_ids[r_idx]
            for s in srv_group:
                graph.add_edge(s, tor, bandwidth=base_access_bw, latency=0.0002, tier='access')
                graph.add_edge(tor, s, bandwidth=base_access_bw, latency=0.0002, tier='access')

            # Total uplink from this ToR to all spines is (len(srv_group) * base_access_bw) / oversubscription
            total_uplink_bw = (len(srv_group) * base_access_bw) / max(1.0, oversubscription)
            per_spine_bw = total_uplink_bw / len(spine_ids)
            for sp in spine_ids:
                graph.add_edge(tor, sp, bandwidth=per_spine_bw, latency=0.001, tier='core')
                graph.add_edge(sp, tor, bandwidth=per_spine_bw, latency=0.001, tier='core')

    elif topology == 'cloud_edge':
        # 3-tier Cloud-Edge: 75% Edge servers (slower CPU, grouped into edge sites with local switches),
        # 25% Cloud servers (fast CPU, connected to a cloud switch), bridged by constrained WAN links.
        n_cloud = max(2, num_servers // 4)
        n_edge = num_servers - n_cloud
        n_edge_sites = max(2, n_edge // 4)
        # Assign speeds and randomly permute server IDs so slot indices do not leak edge vs cloud role
        srv_perm = rng.permutation(num_servers)
        edge_nodes = srv_perm[:n_edge]
        cloud_nodes = srv_perm[n_edge:]
        sorted_speeds = np.sort(speeds)
        speeds[edge_nodes] = sorted_speeds[:n_edge][rng.permutation(n_edge)]
        speeds[cloud_nodes] = sorted_speeds[n_edge:][rng.permutation(n_cloud)]

        edge_sw_ids = [num_servers + i for i in range(n_edge_sites)]
        cloud_sw_id = num_servers + n_edge_sites
        wan_router_id = cloud_sw_id + 1
        total_nodes = wan_router_id + 1
        node_types = ['server'] * num_servers + ['switch'] * (n_edge_sites + 2)
        graph.add_nodes_from(range(total_nodes))

        edge_groups = [list(map(int, g)) for g in np.array_split(edge_nodes, n_edge_sites)]
        for site_idx, group in enumerate(edge_groups):
            esw = edge_sw_ids[site_idx]
            for s in group:
                graph.add_edge(s, esw, bandwidth=base_access_bw * 0.8, latency=0.0005, tier='edge_lan')
                graph.add_edge(esw, s, bandwidth=base_access_bw * 0.8, latency=0.0005, tier='edge_lan')
            # Constrained WAN backhaul from edge site to WAN router
            wan_bw = base_access_bw / max(1.0, oversubscription)
            graph.add_edge(esw, wan_router_id, bandwidth=wan_bw, latency=0.008, tier='wan_backhaul')
            graph.add_edge(wan_router_id, esw, bandwidth=wan_bw, latency=0.008, tier='wan_backhaul')

        # Inter-edge peer links (ring between adjacent edge switches)
        for i in range(n_edge_sites):
            u, v = edge_sw_ids[i], edge_sw_ids[(i + 1) % n_edge_sites]
            peer_bw = base_access_bw / max(1.0, oversubscription * 0.8)
            graph.add_edge(u, v, bandwidth=peer_bw, latency=0.003, tier='edge_peer')
            graph.add_edge(v, u, bandwidth=peer_bw, latency=0.003, tier='edge_peer')

        # WAN router <-> Cloud switch <-> Cloud servers
        cloud_trunk_bw = base_access_bw * 1.5 / max(1.0, oversubscription * 0.5)
        graph.add_edge(wan_router_id, cloud_sw_id, bandwidth=cloud_trunk_bw, latency=0.008, tier='cloud_wan')
        graph.add_edge(cloud_sw_id, wan_router_id, bandwidth=cloud_trunk_bw, latency=0.008, tier='cloud_wan')
        for s in cloud_nodes:
            graph.add_edge(int(s), cloud_sw_id, bandwidth=base_access_bw * 1.5, latency=0.0002, tier='cloud_lan')
            graph.add_edge(cloud_sw_id, int(s), bandwidth=base_access_bw * 1.5, latency=0.0002, tier='cloud_lan')

    elif topology == 'multi_hop_mesh':
        # Multi-hop router mesh: M=num_servers//2 mesh routers connected in a multi-hop graph,
        # each hosting ~2 servers, with heterogeneous link capacities and 3~6 hop paths.
        n_routers = max(4, num_servers // 2)
        total_nodes = num_servers + n_routers
        node_types = ['server'] * num_servers + ['switch'] * n_routers
        graph.add_nodes_from(range(total_nodes))

        router_ids = [num_servers + r for r in range(n_routers)]
        srv_groups = [list(map(int, g)) for g in np.array_split(rng.permutation(num_servers), n_routers)]
        for r_idx, group in enumerate(srv_groups):
            rtr = router_ids[r_idx]
            for s in group:
                graph.add_edge(s, rtr, bandwidth=base_access_bw, latency=0.0003, tier='access')
                graph.add_edge(rtr, s, bandwidth=base_access_bw, latency=0.0003, tier='access')

        # Build a connected sparse multi-hop backbone across routers (ring + random chords)
        backbone_bw = base_access_bw / max(1.0, oversubscription * 0.6)
        for i in range(n_routers):
            u, v = router_ids[i], router_ids[(i + 1) % n_routers]
            lat = float(rng.uniform(0.002, 0.006))
            bw = backbone_bw * float(rng.uniform(0.5, 1.5))
            graph.add_edge(u, v, bandwidth=bw, latency=lat, tier='mesh_backbone')
            graph.add_edge(v, u, bandwidth=bw, latency=lat, tier='mesh_backbone')

        # Add a few chords so diameter is ~3-5 hops
        chords = max(1, n_routers // 3)
        for _ in range(chords):
            i, j = rng.choice(n_routers, size=2, replace=False)
            u, v = router_ids[int(i)], router_ids[int(j)]
            if not graph.has_edge(u, v):
                lat = float(rng.uniform(0.003, 0.008))
                bw = backbone_bw * float(rng.uniform(0.4, 1.2))
                graph.add_edge(u, v, bandwidth=bw, latency=lat, tier='mesh_chord')
                graph.add_edge(v, u, bandwidth=bw, latency=lat, tier='mesh_chord')
    else:
        raise ValueError(f'Unknown topology {topology!r}')

    links = []
    for src, dst, data in sorted(graph.edges(data=True)):
        bw = float(data['bandwidth'] * rng.uniform(1.0 - link_jitter, 1.0 + link_jitter))
        lat = float(data['latency'] * rng.uniform(1.0 - link_jitter, 1.0 + link_jitter))
        tier = data.get('tier', 'generic')
        links.append(dict(src=int(src), dst=int(dst), bandwidth=bw, latency=lat, tier=tier))
        graph[src][dst]['bandwidth'] = bw
        graph[src][dst]['latency'] = lat
        graph[src][dst]['weight'] = lat + reference_bytes / bw

    # Compute shortest routes between all server pairs (and all node pairs)
    paths = dict(nx.all_pairs_dijkstra_path(graph, weight='weight'))
    routes = {}
    for i in range(num_servers):
        for j in range(num_servers):
            if i == j:
                routes[(i, j)] = []
            else:
                p = paths[i][j]
                routes[(i, j)] = list(zip(p[:-1], p[1:]))

    srv_hops = [len(v) for (a, b), v in routes.items() if a != b]
    meta = dict(
        topology=topology,
        num_servers=num_servers,
        num_switches=len(node_types) - num_servers,
        total_nodes=len(node_types),
        directed_links=len(links),
        oversubscription=oversubscription,
        mean_server_hops=float(np.mean(srv_hops)) if srv_hops else 0.0,
        max_server_hops=int(np.max(srv_hops)) if srv_hops else 0,
    )
    return speeds.tolist(), node_types, links, routes, meta


def make_flow_scenario(
    seed: int = 101,
    split: str = 'train',
    workflows: int = 32,
    num_servers: int = 16,
    load: float = 0.70,
    ccr: float = 2.0,
    topology: str = 'leaf_spine',
    oversubscription: float = 4.0,
    sizes: tuple[str, ...] = ('S', 'M'),
    speed_ratio: float = 4.0,
    workload_jitter: float = 0.1,
    data_jitter: float = 0.1,
    link_jitter: float = 0.1,
    warmup: int = 0,
    score: int | None = None,
    tail: int = 0,
) -> dict:
    """Create a complete exogenous multi-workflow scenario for FlowWorkflowEnv."""
    score = workflows - warmup - tail if score is None else score
    if min(warmup, score, tail) < 0 or warmup + score + tail != workflows:
        raise ValueError('warmup + score + tail must equal workflows')

    ref = calibration_reference(sizes)
    streams = {name: _rng(seed, split, name) for name in STREAMS}
    speeds, node_types, links, routes, topo_meta = make_physical_topology(
        num_servers=num_servers,
        rng=streams['topology'],
        reference_bytes=ref['reference_bytes'],
        topology=topology,
        oversubscription=oversubscription,
        speed_ratio=speed_ratio,
        link_jitter=link_jitter,
    )

    lookup = {(x['src'], x['dst']): x for x in links}
    nonlocal_routes = [path for (u, v), path in routes.items() if u != v]
    # Bottleneck inverse bandwidth on each route (since pipelined flow rate is determined by bottleneck link)
    inv_bottleneck_bw = float(np.mean([
        max(1.0 / lookup[e]['bandwidth'] for e in path) for path in nonlocal_routes
    ]))
    mean_prop = float(np.mean([
        sum(lookup[e]['latency'] for e in path) for e in nonlocal_routes for path in [e]
    ]))

    target_comm_time = ccr * ref['mean_task_work']
    if ccr <= 0:
        data_scale = 0.0
    else:
        data_scale = max(0.0, target_comm_time - mean_prop) / (ref['mean_positive_bytes'] * inv_bottleneck_bw)

    templates = load_templates(sizes)
    selection = []
    while len(selection) < workflows:
        selection.extend(streams['templates'].permutation(len(templates)).tolist())
    selection = selection[:workflows]

    expected_wf_work = float(np.mean([tpl['total_work'] for tpl in templates]))
    arrival_rate = float(load * sum(speeds) / expected_wf_work)
    arrivals = np.cumsum(streams['arrival'].exponential(1.0 / arrival_rate, size=workflows))

    episodes = []
    for i, tpl_idx in enumerate(selection):
        tpl = templates[tpl_idx]
        tasks = []
        for t in tpl['tasks']:
            w = t['work'] * float(streams['parameters'].uniform(1.0 - workload_jitter, 1.0 + workload_jitter))
            parents = {
                int(p): float(b * data_scale * streams['parameters'].uniform(1.0 - data_jitter, 1.0 + data_jitter))
                if b > 0 else 0.0
                for p, b in t['parents'].items()
            }
            tasks.append(dict(id=t['id'], work=w, parents=parents))
        episodes.append(dict(
            id=i,
            arrival=float(arrivals[i]),
            score=(warmup <= i < warmup + score),
            tasks=tasks,
            template=tpl['name'],
            family=tpl['family'],
        ))

    return dict(
        nodes=speeds,
        node_types=node_types,
        links=links,
        routes=routes,
        reference_bytes=ref['reference_bytes'],
        workflows=episodes,
        meta=dict(
            seed=int(seed),
            split=split,
            workflows=workflows,
            num_servers=num_servers,
            load=load,
            ccr=ccr,
            data_scale=data_scale,
            topology=topo_meta,
        ),
    )
