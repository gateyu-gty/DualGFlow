"""Event-driven Flow-Level Network Simulator with Max-Min Fair Bandwidth Sharing.

Models:
- Compute servers (0 .. num_servers-1) and Network Switches (num_servers .. total_nodes-1)
- Pipelined multi-hop flows with progressive-filling Max-Min Fair bandwidth allocation (SimGrid LMM style)
- Server-local output data caching and in-flight multicast flow joining
- Parent-Data Injection into physical resource graph node features for GNN message passing
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import heapq
import math
import numpy as np

EPS = 1e-9
PRIORITY = {
    'bandwidth': 0,
    'arrival': 1,
    'flow_serialize_done': 2,
    'propagation_done': 3,
    'compute_done': 4,
}


class FlowWorkflowEnv:
    def __init__(self, scenario: dict, reward_scale: float = 1.0, record: bool = False, strict_fifo: bool = False):
        self._scenario = deepcopy(scenario)
        self.reward_scale = float(reward_scale)
        self.record = bool(record)
        self.strict_fifo = bool(strict_fifo)
        self._initialized = False

    def clone(self) -> 'FlowWorkflowEnv':
        """Fast deep copy for lookahead search (H12) and counterfactual rollouts."""
        c = object.__new__(FlowWorkflowEnv)
        c._scenario = self._scenario
        c.reward_scale = self.reward_scale
        c.record = False
        c.strict_fifo = self.strict_fifo
        c._initialized = self._initialized
        c.now = self.now
        c.total_reward_unscaled = self.total_reward_unscaled
        c._events = list(self._events)
        c._serial = self._serial
        c._flow_gen = self._flow_gen
        c._last_flow_time = self._last_flow_time
        c._next_flow_id = self._next_flow_id
        c._num_servers = self._num_servers
        c._total_nodes = self._total_nodes
        c._node_types = self._node_types
        c._speeds = self._speeds.copy()
        c._queues = [deque(q) for q in self._queues]
        c._running = list(self._running)
        c._busy_time = self._busy_time.copy()
        c._server_cache = [set(s) for s in self._server_cache]
        c._inflight_to_dest = dict(self._inflight_to_dest)
        c._links = {k: dict(v, flows=set(v['flows'])) for k, v in self._links.items()}
        c._routes = self._routes
        c._flows = {
            fid: dict(f, waiters=list(f['waiters']))
            for fid, f in self._flows.items()
        }
        c._active_sending_flows = set(self._active_sending_flows)
        c._workflows = {wid: dict(w, tasks=list(w['tasks'])) for wid, w in self._workflows.items()}
        c._tasks = {
            k: dict(t, transfers=list(t['transfers']))
            for k, t in self._tasks.items()
        }
        c._ready = list(self._ready)
        c._active_workflows = self._active_workflows
        c._focus = self._focus
        c._total_bytes_transferred = self._total_bytes_transferred
        c._throttled_flow_events = self._throttled_flow_events
        return c

    def clone_causal_active(self) -> 'FlowWorkflowEnv':
        """Clone environment keeping ONLY currently arrived workflows (100% causal, zero future leakage)."""
        c = self.clone()
        # Remove future workflow arrival events from the event heap
        c._events = [ev for ev in c._events if ev[3] != 'arrival']
        heapq.heapify(c._events)
        # Keep only arrived workflows in c._workflows so c.done triggers as soon as active workflows finish
        c._workflows = {wid: w for wid, w in c._workflows.items() if w['arrived']}
        return c

    def _push_event(self, when: float, kind: str, payload):
        self._serial += 1
        heapq.heappush(self._events, (float(when), PRIORITY[kind], self._serial, kind, payload))

    def reset(self, return_observation: bool = True):
        self.now = 0.0
        self.total_reward_unscaled = 0.0
        self._events = []
        self._serial = 0
        self._flow_gen = 0
        self._last_flow_time = 0.0
        self._next_flow_id = 0
        self._total_bytes_transferred = 0.0
        self._throttled_flow_events = 0

        self._speeds = np.asarray(self._scenario['nodes'], dtype=np.float64)
        self._num_servers = len(self._speeds)
        self._node_types = list(self._scenario.get('node_types', ['server'] * self._num_servers))
        self._total_nodes = len(self._node_types)

        self._queues = [deque() for _ in range(self._num_servers)]
        self._running = [None for _ in range(self._num_servers)]
        self._busy_time = np.zeros(self._num_servers, dtype=np.float64)
        self._server_cache = [set() for _ in range(self._num_servers)]
        self._inflight_to_dest = {}  # (wid, parent_tid, dst_node) -> flow_id

        self._links = {}
        for spec in self._scenario['links']:
            key = (int(spec['src']), int(spec['dst']))
            self._links[key] = dict(
                src=key[0],
                dst=key[1],
                bandwidth=float(spec['bandwidth']),
                latency=float(spec['latency']),
                flows=set(),
                fair_share_bw=float(spec['bandwidth']),
            )
        self._routes = {
            (int(k[0]), int(k[1])): [tuple(map(int, e)) for e in path]
            for k, path in self._scenario['routes'].items()
        }

        self._flows = {}
        self._active_sending_flows = set()
        self._workflows = {}
        self._tasks = {}
        self._ready = []
        self._active_workflows = 0

        specs = sorted(self._scenario['workflows'], key=lambda w: (w['arrival'], w['id']))
        for order, spec in enumerate(specs):
            wid, arrival = int(spec['id']), float(spec['arrival'])
            raw = {int(t['id']): t for t in spec['tasks']}
            parents = {tid: {int(k): float(v) for k, v in t['parents'].items()} for tid, t in raw.items()}
            children = {tid: [] for tid in raw}
            degree = {tid: len(ps) for tid, ps in parents.items()}
            for tid, ps in parents.items():
                for p in ps:
                    children[p].append(tid)
            heap = [tid for tid in raw if degree[tid] == 0]
            heapq.heapify(heap)
            topo = []
            while heap:
                tid = heapq.heappop(heap)
                topo.append(tid)
                for c in sorted(children[tid]):
                    degree[c] -= 1
                    if degree[c] == 0:
                        heapq.heappush(heap, c)

            # Compute upward rank for priority ordering inside workflow
            up_rank = {}
            mean_speed = float(self._speeds.mean())
            for tid in reversed(topo):
                w_cost = float(raw[tid]['work']) / mean_speed
                c_costs = [up_rank[c] for c in children[tid]]
                up_rank[tid] = w_cost + (max(c_costs) if c_costs else 0.0)

            wf = dict(
                id=wid,
                arrival=arrival,
                score=bool(spec.get('score', True)),
                order=order,
                arrived=False,
                completed=False,
                finish=None,
                tasks=[],
                remaining=len(raw),
            )
            self._workflows[wid] = wf
            for rank, tid in enumerate(topo):
                key = (wid, tid)
                wf['tasks'].append(key)
                self._tasks[key] = dict(
                    key=key,
                    workflow_id=wid,
                    id=tid,
                    work=float(raw[tid]['work']),
                    parents=parents[tid],
                    children=sorted(children[tid]),
                    rank=rank,
                    up_rank=up_rank[tid],
                    status='unassigned',
                    node=None,
                    ready=None,
                    assign=None,
                    data_ready=None,
                    start=None,
                    finish=None,
                    pending_inputs=0,
                    transfers=[],
                )
            self._push_event(arrival, 'arrival', wid)

        self._focus = None
        self._initialized = True
        self._advance_until_decision()
        return None if self.done or not return_observation else self.observe()

    @property
    def done(self) -> bool:
        return all(w['completed'] for w in self._workflows.values())

    def _tick(self, when: float):
        dt = max(0.0, float(when) - self.now)
        if dt > 0:
            self.total_reward_unscaled -= self._active_workflows * dt
            for i in range(self._num_servers):
                if self._running[i] is not None:
                    self._busy_time[i] += dt
            self.now = float(when)

    def _settle_flow_progress(self):
        """Advance byte progress of all active sending flows up to self.now."""
        dt = max(0.0, self.now - self._last_flow_time)
        if dt > 0 and self._active_sending_flows:
            for fid in self._active_sending_flows:
                f = self._flows[fid]
                sent = f['rate'] * dt
                f['remaining'] = max(0.0, f['remaining'] - sent)
                self._total_bytes_transferred += min(sent, f['bytes'])
        self._last_flow_time = self.now

    def _recompute_flow_rates(self):
        """Max-Min Fair Bandwidth Allocation via Progressive Filling (SimGrid LMM)."""
        self._settle_flow_progress()
        self._flow_gen += 1

        if not self._active_sending_flows:
            for link in self._links.values():
                link['fair_share_bw'] = link['bandwidth']
            return

        # Progressive filling
        unfrozen = set(self._active_sending_flows)
        rem_bw = {e: self._links[e]['bandwidth'] for e in self._links}
        rates = {fid: 0.0 for fid in unfrozen}

        # Map active links to unfrozen flows
        while unfrozen:
            link_counts = {}
            for fid in unfrozen:
                for e in self._flows[fid]['path']:
                    link_counts[e] = link_counts.get(e, 0) + 1
            if not link_counts:
                break

            # Find bottleneck increment
            min_inc = math.inf
            bottleneck_links = []
            for e, cnt in link_counts.items():
                inc = rem_bw[e] / cnt
                if inc < min_inc - 1e-12:
                    min_inc = inc
                    bottleneck_links = [e]
                elif abs(inc - min_inc) <= 1e-12:
                    bottleneck_links.append(e)

            for fid in unfrozen:
                rates[fid] += min_inc
            for e, cnt in link_counts.items():
                rem_bw[e] = max(0.0, rem_bw[e] - min_inc * cnt)

            # Freeze flows passing through saturated bottleneck links
            to_freeze = set()
            for e in bottleneck_links:
                for fid in unfrozen:
                    if e in self._flows[fid]['path_set']:
                        to_freeze.add(fid)
            unfrozen -= to_freeze

        # Record rates and find earliest finishing flow
        min_dt = math.inf
        earliest_fid = None
        for fid in self._active_sending_flows:
            f = self._flows[fid]
            old_rate = f['rate']
            new_rate = max(1e-6, rates[fid])
            if old_rate > 0 and new_rate < old_rate * 0.95:
                self._throttled_flow_events += 1
            f['rate'] = new_rate
            dt = f['remaining'] / new_rate
            if dt < min_dt:
                min_dt = dt
                earliest_fid = fid

        # Update each link's fair-share estimate for a hypothetical new flow
        for e, link in self._links.items():
            n_active = len(link['flows'])
            link['fair_share_bw'] = link['bandwidth'] / (n_active + 1)

        if earliest_fid is not None:
            self._push_event(
                self.now + max(0.0, min_dt),
                'flow_serialize_done',
                (earliest_fid, self._flow_gen),
            )

    def _make_ready(self, key: tuple[int, int]):
        task = self._tasks[key]
        task['ready'] = self.now
        wf = self._workflows[key[0]]
        # Prioritize by ready time, workflow arrival order, and negative upward rank (HEFT order)
        heapq.heappush(self._ready, (self.now, wf['order'], -task['up_rank'], task['rank'], key))

    def _start_cpu_services(self):
        """Start queued tasks on idle servers (strict FIFO if strict_fifo=True, else work-conserving)."""
        for node in range(self._num_servers):
            if self._running[node] is None and self._queues[node]:
                ready_idx = None
                if self.strict_fifo:
                    if self._tasks[self._queues[node][0]]['pending_inputs'] == 0:
                        ready_idx = 0
                else:
                    for idx, key in enumerate(self._queues[node]):
                        if self._tasks[key]['pending_inputs'] == 0:
                            ready_idx = idx
                            break
                if ready_idx is not None:
                    # Extract ready_idx from deque
                    if ready_idx == 0:
                        key = self._queues[node].popleft()
                    else:
                        key = self._queues[node][ready_idx]
                        del self._queues[node][ready_idx]
                    task = self._tasks[key]
                    self._running[node] = key
                    task['status'] = 'running'
                    task['start'] = self.now
                    task['finish'] = self.now + task['work'] / self._speeds[node]
                    self._push_event(task['finish'], 'compute_done', (node, key))

    def _handle_event(self, kind: str, payload):
        if kind == 'arrival':
            wid = payload
            wf = self._workflows[wid]
            wf['arrived'] = True
            self._active_workflows += 1
            for key in wf['tasks']:
                if not self._tasks[key]['parents']:
                    self._make_ready(key)

        elif kind == 'flow_serialize_done':
            fid, gen = payload
            if gen != self._flow_gen or fid not in self._active_sending_flows:
                return
            self._settle_flow_progress()
            # Finish all flows whose remaining bytes reached ~0
            finished_fids = [
                f_id for f_id in list(self._active_sending_flows)
                if self._flows[f_id]['remaining'] <= 1e-5
            ]
            if not finished_fids and fid in self._active_sending_flows:
                finished_fids = [fid]

            for f_id in finished_fids:
                f = self._flows[f_id]
                f['remaining'] = 0.0
                f['rate'] = 0.0
                f['status'] = 'propagating'
                self._active_sending_flows.remove(f_id)
                for e in f['path']:
                    self._links[e]['flows'].discard(f_id)
                self._push_event(self.now + f['prop_latency'], 'propagation_done', f_id)

            self._recompute_flow_rates()

        elif kind == 'propagation_done':
            fid = payload
            f = self._flows[fid]
            f['status'] = 'done'
            f['finish'] = self.now
            # Cache parent output at destination server
            self._server_cache[f['dst']].add((f['workflow_id'], f['parent_id']))
            self._inflight_to_dest.pop((f['workflow_id'], f['parent_id'], f['dst']), None)
            for waiter_key in f['waiters']:
                wt = self._tasks[waiter_key]
                wt['pending_inputs'] -= 1
                if wt['pending_inputs'] == 0:
                    wt['data_ready'] = self.now

        elif kind == 'compute_done':
            node, key = payload
            task = self._tasks[key]
            task['status'] = 'done'
            task['finish'] = self.now
            self._running[node] = None
            # Cache output locally on this server
            self._server_cache[node].add(key)

            wf = self._workflows[key[0]]
            wf['remaining'] -= 1
            if wf['remaining'] == 0:
                wf['completed'] = True
                wf['finish'] = self.now
                self._active_workflows -= 1

            for cid in task['children']:
                ckey = (key[0], cid)
                child = self._tasks[ckey]
                if child['status'] == 'unassigned' and child['ready'] is None:
                    if all(self._tasks[(key[0], p)]['status'] == 'done' for p in child['parents']):
                        self._make_ready(ckey)

    def _settle(self):
        while True:
            while self._events and self._events[0][0] <= self.now + 1e-12:
                when, _, _, kind, payload = heapq.heappop(self._events)
                if when > self.now:
                    self._tick(when)
                self._handle_event(kind, payload)
            self._start_cpu_services()
            if not self._events or self._events[0][0] > self.now + 1e-12:
                break

    def _advance_until_decision(self):
        while True:
            self._settle()
            if self._ready:
                self._focus = self._ready[0][4]
                return
            if self.done:
                self._focus = None
                return
            if not self._events:
                raise RuntimeError('Simulator deadlock: unfinished workflows with no events')
            self._tick(self._events[0][0])

    def step(self, action: int, return_observation: bool = True):
        node = int(action)
        if not (0 <= node < self._num_servers):
            raise ValueError(f'Invalid server action {node}')
        before = self.total_reward_unscaled
        _, _, _, _, key = heapq.heappop(self._ready)
        task = self._tasks[key]
        task['node'] = node
        task['assign'] = self.now
        task['status'] = 'queued'
        self._queues[node].append(key)

        new_flows_started = False
        wid = key[0]
        for pid, size in sorted(task['parents'].items()):
            if size <= 0:
                continue
            # Check if parent output is already cached on destination server `node`
            if (wid, pid) in self._server_cache[node]:
                continue
            # Check if an in-flight flow is already delivering (wid, pid) to `node`
            inflight_fid = self._inflight_to_dest.get((wid, pid, node))
            if inflight_fid is not None and self._flows[inflight_fid]['status'] != 'done':
                self._flows[inflight_fid]['waiters'].append(key)
                task['transfers'].append(inflight_fid)
                task['pending_inputs'] += 1
                continue

            src = self._tasks[(wid, pid)]['node']
            if src == node:
                self._server_cache[node].add((wid, pid))
                continue

            path = self._routes[(src, node)]
            prop_lat = sum(self._links[e]['latency'] for e in path)
            fid = self._next_flow_id
            self._next_flow_id += 1
            flow = dict(
                id=fid,
                workflow_id=wid,
                parent_id=pid,
                src=src,
                dst=node,
                bytes=float(size),
                remaining=float(size),
                rate=0.0,
                path=path,
                path_set=set(path),
                prop_latency=prop_lat,
                status='sending',
                start=self.now,
                finish=None,
                waiters=[key],
            )
            self._flows[fid] = flow
            self._active_sending_flows.add(fid)
            self._inflight_to_dest[(wid, pid, node)] = fid
            for e in path:
                self._links[e]['flows'].add(fid)
            task['transfers'].append(fid)
            task['pending_inputs'] += 1
            new_flows_started = True

        if task['pending_inputs'] == 0:
            task['data_ready'] = self.now
        if new_flows_started:
            self._recompute_flow_rates()

        self._advance_until_decision()
        reward = (self.total_reward_unscaled - before) / self.reward_scale
        info = dict(
            time=self.now,
            total_reward_unscaled=self.total_reward_unscaled,
            completed=sum(w['completed'] for w in self._workflows.values()),
        )
        return (None if self.done or not return_observation else self.observe()), float(reward), self.done, info

    def _estimate_server_cpu_backlog(self) -> np.ndarray:
        """Compute remaining CPU work backlog (in seconds) on each server."""
        self._settle_flow_progress()
        avail = np.zeros(self._num_servers, dtype=np.float64)
        for u in range(self._num_servers):
            rem = 0.0
            if self._running[u] is not None:
                rem += max(0.0, self._tasks[self._running[u]]['finish'] - self.now)
            for k in self._queues[u]:
                rem += self._tasks[k]['work'] / self._speeds[u]
            avail[u] = rem
        return avail

    def candidate_eft_estimates(self) -> dict[str, np.ndarray]:
        """Compute Compute-Only EFT, Nominal CA-EFT, and Flow-Contention-Aware CA-EFT."""
        self._settle_flow_progress()
        focus = self._tasks[self._focus]
        wid = focus['workflow_id']
        cpu_avail = self._estimate_server_cpu_backlog()

        compute_eft = np.zeros(self._num_servers, dtype=np.float64)
        nominal_eft = np.zeros(self._num_servers, dtype=np.float64)
        flow_eft = np.zeros(self._num_servers, dtype=np.float64)
        nominal_delay = np.zeros(self._num_servers, dtype=np.float64)
        flow_delay = np.zeros(self._num_servers, dtype=np.float64)
        remote_bytes = np.zeros(self._num_servers, dtype=np.float64)
        mean_hops = np.zeros(self._num_servers, dtype=np.float64)

        for dst in range(self._num_servers):
            exec_time = focus['work'] / self._speeds[dst]
            compute_eft[dst] = cpu_avail[dst] + exec_time

            nom_d = 0.0
            flw_d = 0.0
            rem_b = 0.0
            weighted_hops = 0.0
            for pid, size in focus['parents'].items():
                if size <= 0 or (wid, pid) in self._server_cache[dst]:
                    continue
                src = self._tasks[(wid, pid)]['node']
                if src == dst:
                    continue
                path = self._routes[(src, dst)]
                rem_b += size
                weighted_hops += size * len(path)
                prop = sum(self._links[e]['latency'] for e in path)
                # Nominal bottleneck bandwidth (ignores active flows)
                nom_bw = min(self._links[e]['bandwidth'] for e in path)
                nom_d = max(nom_d, prop + size / nom_bw)
                # Exact Processor-Sharing / Max-Min Fair bottleneck delay on path:
                # An existing flow f with remaining bytes R_f shares bandwidth with this
                # transfer of size S for min(R_f, S) bytes.
                ps_d = prop + max(
                    (size + sum(min(self._flows[fid]['remaining'], size) for fid in self._links[e]['flows']))
                    / self._links[e]['bandwidth']
                    for e in path
                )
                flw_d = max(flw_d, ps_d)

            nominal_delay[dst] = nom_d
            flow_delay[dst] = flw_d
            nominal_eft[dst] = max(cpu_avail[dst], nom_d) + exec_time
            flow_eft[dst] = max(cpu_avail[dst], flw_d) + exec_time
            remote_bytes[dst] = rem_b
            mean_hops[dst] = weighted_hops / rem_b if rem_b > 0 else 0.0

        return dict(
            cpu_avail=cpu_avail,
            compute_eft=compute_eft,
            nominal_eft=nominal_eft,
            flow_eft=flow_eft,
            nominal_delay=nominal_delay,
            flow_delay=flow_delay,
            remote_bytes=remote_bytes,
            mean_hops=mean_hops,
        )

    def observe(self) -> dict | None:
        if self.done or self._focus is None:
            return None
        self._settle_flow_progress()
        focus = self._tasks[self._focus]
        wid = focus['workflow_id']

        # 1. Task Graph (unfinished tasks of arrived workflows)
        arrived = sorted(
            (w for w in self._workflows.values() if w['arrived'] and not w['completed']),
            key=lambda w: w['order'],
        )
        keys = [k for w in arrived for k in w['tasks'] if self._tasks[k]['status'] != 'done']
        idx_map = {k: i for i, k in enumerate(keys)}

        wf_rem_work = {}
        for w in arrived:
            rem = 0.0
            for k in w['tasks']:
                t = self._tasks[k]
                if t['status'] == 'done':
                    continue
                elif t['status'] == 'running':
                    rem += max(0.0, t['finish'] - self.now) * self._speeds[t['node']]
                else:
                    rem += t['work']
            wf_rem_work[w['id']] = rem

        task_x = []
        task_edges = []
        task_srv_edges = []
        task_srv_attr = []
        for k in keys:
            t = self._tasks[k]
            assigned = t['node'] is not None
            out_deg = len(t['children'])
            max_child_bytes = max(
                (self._tasks[(k[0], c)]['parents'].get(t['id'], 0.0) for c in t['children']),
                default=0.0,
            )
            task_x.append([
                float({'unassigned': 0, 'queued': 1, 'running': 2}[t['status']]),
                t['work'],
                wf_rem_work[k[0]],
                t['up_rank'],
                float(out_deg),
                max_child_bytes,
                self._speeds[t['node']] if assigned else 0.0,
            ])
            t_idx = idx_map[k]
            for p in t['parents']:
                if (k[0], p) in idx_map:
                    task_edges.append((idx_map[(k[0], p)], t_idx))
                else:
                    # Parent is already done/assigned: record data dependency from parent's server to this task
                    p_task = self._tasks[(k[0], p)]
                    p_bytes = t['parents'][p]
                    if p_task['node'] is not None and p_bytes > 0:
                        task_srv_edges.append((t_idx, p_task['node']))
                        task_srv_attr.append([0.0, p_bytes])  # type=0: data dependency from server
            if assigned:
                task_srv_edges.append((t_idx, t['node']))
                task_srv_attr.append([1.0, t['work']])  # type=1: task is queued/running on server
            # Self-loop for GAT stability
            task_edges.append((t_idx, t_idx))

        # Add server execution queue chain edges (running -> queued[0] -> queued[1] ...)
        for node in range(self._num_servers):
            chain = ([self._running[node]] if self._running[node] is not None else []) + list(self._queues[node])
            for a_k, b_k in zip(chain, chain[1:]):
                if a_k in idx_map and b_k in idx_map:
                    task_edges.append((idx_map[a_k], idx_map[b_k]))

        # 2. Physical Resource Graph with Parent-Data Injection
        parent_bytes_at_node = np.zeros(self._total_nodes, dtype=np.float64)
        max_parent_at_node = np.zeros(self._total_nodes, dtype=np.float64)
        has_parent_at_node = np.zeros(self._total_nodes, dtype=np.float64)

        for pid, size in focus['parents'].items():
            if size <= 0:
                continue
            src = self._tasks[(wid, pid)]['node']
            if src is not None:
                has_parent_at_node[src] = 1.0
                parent_bytes_at_node[src] += size
                max_parent_at_node[src] = max(max_parent_at_node[src], size)
            # Also mark any server that already cached (wid, pid)
            for u in range(self._num_servers):
                if (wid, pid) in self._server_cache[u]:
                    has_parent_at_node[u] = 1.0

        node_active_flows = np.zeros(self._total_nodes, dtype=np.float64)
        for fid in self._active_sending_flows:
            f = self._flows[fid]
            node_active_flows[f['src']] += 1.0
            for e in f['path']:
                node_active_flows[e[1]] += 1.0

        resource_x = []
        for u in range(self._total_nodes):
            if u < self._num_servers:
                run_key = self._running[u]
                run_rem = max(0.0, self._tasks[run_key]['finish'] - self.now) * self._speeds[u] if run_key else 0.0
                q_work = sum(self._tasks[k]['work'] for k in self._queues[u])
                q_len = float(len(self._queues[u]))
                resource_x.append([
                    1.0,  # is_server
                    self._speeds[u],
                    run_rem,
                    q_work,
                    q_len,
                    has_parent_at_node[u],
                    parent_bytes_at_node[u],
                    max_parent_at_node[u],
                    node_active_flows[u],
                ])
            else:
                # Switch node: aggregates and relays congestion + parent data reachability
                resource_x.append([
                    0.0,  # is_server
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    node_active_flows[u],
                ])

        link_keys = sorted(self._links.keys())
        resource_edge_attr = []
        for e in link_keys:
            lnk = self._links[e]
            n_flw = float(len(lnk['flows']))
            rem_b = sum(self._flows[fid]['remaining'] for fid in lnk['flows'])
            drain_t = rem_b / lnk['bandwidth']
            resource_edge_attr.append([
                lnk['bandwidth'],
                lnk['latency'],
                n_flw,
                lnk['fair_share_bw'],
                rem_b,
                drain_t,
            ])

        efts = self.candidate_eft_estimates()
        total_parent_bytes = max(1.0, sum(focus['parents'].values()))
        pair_x = np.column_stack([
            efts['remote_bytes'],
            1.0 - efts['remote_bytes'] / total_parent_bytes,
            efts['mean_hops'],
            efts['flow_delay'],
        ]).astype(np.float32)

        candidate_features = np.column_stack([
            focus['work'] / self._speeds,
            efts['cpu_avail'],
            efts['flow_eft'],
            self._speeds,
        ]).astype(np.float32)

        return dict(
            num_servers=self._num_servers,
            total_nodes=self._total_nodes,
            task_x=np.asarray(task_x, dtype=np.float32),
            task_edges=np.asarray(task_edges, dtype=np.int64).reshape(-1, 2).T,
            task_srv_edges=np.asarray(task_srv_edges, dtype=np.int64).reshape(-1, 2).T if task_srv_edges else np.zeros((2, 0), dtype=np.int64),
            task_srv_attr=np.asarray(task_srv_attr, dtype=np.float32).reshape(-1, 2) if task_srv_attr else np.zeros((0, 2), dtype=np.float32),
            focus=idx_map[self._focus],
            resource_x=np.asarray(resource_x, dtype=np.float32),
            resource_edges=np.asarray(link_keys, dtype=np.int64).reshape(-1, 2).T,
            resource_edge_attr=np.asarray(resource_edge_attr, dtype=np.float32),
            pair_x=pair_x,
            nominal_delay=efts['nominal_delay'].astype(np.float32),
            candidate_features=candidate_features,
            compute_eft=efts['compute_eft'],
            nominal_eft=efts['nominal_eft'],
            flow_eft=efts['flow_eft'],
            now=self.now,
            workflow_id=wid,
            task_id=focus['id'],
        )

    def metrics(self) -> dict:
        rows = []
        for wf in sorted(self._workflows.values(), key=lambda w: w['order']):
            rows.append(dict(
                workflow_id=wf['id'],
                arrival=wf['arrival'],
                finish=wf['finish'],
                score=wf['score'],
                completed=wf['completed'],
                flowtime=(wf['finish'] - wf['arrival']) if wf['completed'] else None,
            ))
        scoring = [r['flowtime'] for r in rows if r['score'] and r['completed']]
        return dict(
            mean_flowtime=float(np.mean(scoring)) if scoring else None,
            p95_flowtime=float(np.quantile(scoring, 0.95)) if scoring else None,
            makespan=self.now,
            completed=sum(w['completed'] for w in self._workflows.values()),
            total_workflows=len(rows),
            total_bytes_transferred=self._total_bytes_transferred,
            throttled_flow_events=self._throttled_flow_events,
        )
