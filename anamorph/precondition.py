# Based on AnaMorph (https://github.com/NeuroBox3D/AnaMorph).
# Original code copyright 2013-2017 G-CSC, Goethe University Frankfurt - Queisser group.
# Recommended citations retained from the original source:
# Mörschel K, Breit M, Queisser G. Generating neuron geometries for detailed
# three-dimensional simulations using AnaMorph. Neuroinformatics (2017).
# Grein S, Stepniewski M, Reiter S, Knodel MM, Queisser G. 1D-3D hybrid
# modelling - from multi-compartment models to full resolution models in space
# and time. Frontiers in Neuroinformatics 8, 68 (2014).
# Breit M, Stepniewski M, Grein S, Gottmann P, Reinhardt L, Queisser G.
# Anatomically detailed and large-scale simulations studying synapse loss
# and synchrony using NeuroBox. Frontiers in Neuroanatomy 10 (2016).
# See LICENSE and NOTICE.md for the license and additional attribution terms.

"""Morphology preconditioning inspired by AnaMorph's greedy algorithm.

Based on AnaMorph (https://github.com/NeuroBox3D/AnaMorph).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from collections import deque
from heapq import heapify, heappop, heappush
from math import ceil, dist, isfinite, sqrt

from .swc import Morphology, Node
from .collapse import _SourcePriorityQueue
from .source_order import SourceEdgeState
from .source_float import x87_sqrt


@dataclass(frozen=True, slots=True)
class PreconditioningResult:
    morphology: Morphology
    split_segments: int
    collapsed_segments: int
    edge_order: tuple[tuple[int, int], ...] = ()
    source_edge_state: SourceEdgeState | None = None


def _precondition_length(a, b, norm_mode):
    """Measure an edge with robust or source-order norm arithmetic."""
    if norm_mode == "robust":
        return dist(a, b)
    delta = tuple(y-x for x, y in zip(a, b))
    squared = (delta[0]*delta[0]+delta[1]*delta[1])+delta[2]*delta[2]
    if squared > 0 and isfinite(squared):
        return x87_sqrt(squared) if norm_mode == "source-x87" else sqrt(squared)
    # Preserve finite geometric scales when the source's squares overflow or
    # underflow. A truly zero edge still returns zero via dist().
    return dist(a, b)


def precondition(tree: Morphology, alpha: float = 3.0, beta: float = 1.5,
                 gamma: float = 10.0, *, max_splits_per_segment: int = 10000,
                 precondition_norm_mode: str = "robust",
                 _initial_edge_state=None) -> PreconditioningResult:
    """Split long neurite edges and greedily collapse short ones.

    The distance criteria, queue updates, fixed-point iteration and endpoint
    choices follow the C++ preliminary preconditioner. This operates on the
    supplied nodes directly; call prepare_morphology for source-style soma and
    root preparation and, where recoverable, its initial graph-edge ordering.
    Edge IDs and equal-key ordering can still differ.

    ``precondition_norm_mode`` selects scaled Python distances (``robust``),
    source-order unscaled arithmetic (``source``), or that arithmetic with
    Windows-reference x87 square-root rounding (``source-x87``). Source modes
    use scaled distances if the squared length becomes zero or nonfinite.
    """
    if any(not isfinite(x) or x <= 0 for x in (alpha, beta, gamma)):
        raise ValueError("preconditioning parameters must be positive and finite")
    if precondition_norm_mode not in ("robust", "source", "source-x87"):
        raise ValueError("precondition_norm_mode must be robust, source, or source-x87")
    if max_splits_per_segment < 2:
        raise ValueError("max_splits_per_segment must be at least 2")
    nodes = dict(tree.nodes)
    next_id = max(nodes) + 1
    split_count = 0
    neurites = (2, 3, 4)
    # The source initially constructs graph edges breadth-first, then retains
    # an edge's ID when redirecting it during collapse. Outgoing edge lists are
    # sorted by that ID, which decides equal-depth path selection ties.
    free_edge_ids = []
    if _initial_edge_state is not None:
        actual = {(node.parent, node.id) for node in nodes.values() if node.parent != -1}
        entries = _initial_edge_state.entries
        ranks = {rank for rank, _, _ in entries}
        free_ids = set(_initial_edge_state.free_ids)
        if (len(entries) != len(actual) or {(parent, child) for _, parent, child in entries} != actual or
                len(ranks) != len(entries) or len(free_ids) != len(_initial_edge_state.free_ids) or
                ranks & free_ids or len(ranks)+len(free_ids) != _initial_edge_state.next_id or
                any(rank < 0 or rank >= _initial_edge_state.next_id for rank in ranks | free_ids)):
            raise ValueError("initial edge allocator state does not match the morphology")
        edge_rank = {child: rank for rank, _, child in entries}
        free_edge_ids = list(_initial_edge_state.free_ids)
        heapify(free_edge_ids)
        next_edge_rank = _initial_edge_state.next_id
    else:
        edge_rank = {}
        next_edge_rank = 0
        for root in tree.roots:
            queue = deque([root])
            while queue:
                parent_id = queue.popleft()
                for child_id in tree.children[parent_id]:
                    edge_rank[child_id] = next_edge_rank
                    next_edge_rank += 1
                    queue.append(child_id)
    def allocate_edge_id():
        nonlocal next_edge_rank
        if free_edge_ids:
            return heappop(free_edge_ids)
        rank = next_edge_rank
        next_edge_rank += 1
        return rank
    initial_segments = sorted(tree.neurite_segments(), key=lambda edge: edge_rank[edge[1].id])
    for parent, child in initial_segments:
        length = _precondition_length(parent.position, child.position, precondition_norm_mode)
        if not isfinite(length):
            raise ValueError(f"edge {parent.id}-{child.id} length exceeds float range")
        max_radius = max(parent.radius, child.radius)
        if length <= gamma * max_radius:
            continue
        subdivision_ratio = length/max_radius
        if not isfinite(subdivision_ratio) or subdivision_ratio > max_splits_per_segment:
            raise ValueError(f"edge {parent.id}-{child.id} requires testing more than "
                             f"{max_splits_per_segment} subdivisions, exceeding the limit")
        nmax = max(2, ceil(subdivision_ratio))
        displacement = tuple(b-a for a, b in zip(parent.position, child.position))
        radius_delta = child.radius-parent.radius
        # Keep the source's arithmetic at ordinary scales, where it affects
        # tie-breaking. At extreme scales its squared penalties can all become
        # zero or infinity; dividing every error by the same length preserves
        # their mathematical ordering and keeps the objective representable.
        scaled_penalty = length < 2.**-500 or length > 2.**500
        def penalty(k):
            # The source evaluates each rounded interpolated vertex and sums
            # the individual squared errors. An algebraic closed form can
            # select a different k when candidate penalties are nearly tied.
            total, previous_position, previous_radius = 0., parent.position, parent.radius
            for i in range(1, k+1):
                if i == k:
                    position, radius = child.position, child.radius
                else:
                    fraction = i/k
                    position = tuple(a+fraction*d for a, d in zip(parent.position, displacement))
                    radius = parent.radius+fraction*radius_delta
                if precondition_norm_mode == "robust":
                    delta = tuple(b-a for a, b in zip(previous_position, position))
                    squared = sum(x*x for x in delta)
                    segment_length = sqrt(squared) if squared > 0 and isfinite(squared) else dist(previous_position, position)
                else:
                    segment_length = _precondition_length(previous_position, position, precondition_norm_mode)
                if scaled_penalty:
                    error = segment_length/length-gamma*(max(previous_radius, radius)/length)
                else:
                    error = segment_length-gamma*max(previous_radius, radius)
                total += error*error
                previous_position, previous_radius = position, radius
            return total
        k = min(range(2, nmax+1), key=penalty)
        previous = parent.id
        heappush(free_edge_ids, edge_rank.pop(child.id))
        for i in range(1, k):
            t = i/k
            pos = tuple(a+t*(b-a) for a, b in zip(parent.position, child.position))
            nodes[next_id] = Node(next_id, child.kind, *pos, parent.radius+t*(child.radius-parent.radius), previous)
            edge_rank[next_id] = allocate_edge_id()
            previous = next_id
            next_id += 1
        edge_rank[child.id] = allocate_edge_id()
        nodes[child.id] = replace(nodes[child.id], parent=previous)
        split_count += 1

    children = {node_id: set() for node_id in nodes}
    for node in nodes.values():
        if node.parent != -1:
            children[node.parent].add(node.id)
    def neurite_children(node_id):
        return [nodes[i] for i in children[node_id] if nodes[i].kind in neurites]
    def is_neurite_edge(child_id):
        return (child_id in nodes and nodes[child_id].kind in neurites and nodes[child_id].parent in nodes and
                nodes[nodes[child_id].parent].kind in neurites)
    def edge_values(child_id):
        v = nodes[child_id]
        u = nodes[v.parent]
        u_children = neurite_children(u.id)
        v_children = neurite_children(v.id)
        root = u.parent == -1 or nodes[u.parent].kind not in neurites
        ru = 0. if root else max([u.radius, nodes[u.parent].radius] +
                                 [n.radius for n in u_children if n.id != v.id])
        # The checked-in getSMDVRadii uses u's outgoing neighbours here.
        rv = max([v.radius]+[n.radius for n in u_children]) if v_children else 0.
        length = _precondition_length(u.position, v.position, precondition_norm_mode)
        neighbor_sum = ru+rv
        secondary = beta*neighbor_sum if isfinite(neighbor_sum) else beta*ru+beta*rv
        weight = min(length-alpha*max(u.radius, v.radius), length-secondary)
        return u, v, u_children, v_children, root, length, weight

    collapsed = 0
    while True:
        queue = _SourcePriorityQueue()
        for node_id in sorted((i for i in nodes if is_neurite_edge(i)), key=edge_rank.__getitem__):
            queue.insert(node_id, edge_values(node_id)[5])
        changed = False
        while queue:
            _, child_id = queue.pop()
            if not is_neurite_edge(child_id):
                continue
            u, v, u_children, v_children, root, _, weight = edge_values(child_id)
            if weight > 0:
                continue
            u_branch, v_branch, v_terminal = len(u_children) > 1, len(v_children) > 1, not v_children
            if (root and (v_branch or v_terminal)) or (u_branch and v_terminal):
                continue
            if root or (u_branch and not v_branch):
                pos, radius = u.position, u.radius
            elif not u_branch and (v_branch or v_terminal):
                pos, radius = v.position, v.radius
            else:
                def midpoint(a, b):
                    total = a+b
                    return total/2 if isfinite(total) else a/2+b/2
                pos = tuple(midpoint(a, b) for a, b in zip(u.position, v.position))
                radius = midpoint(u.radius, v.radius)
            nodes[u.id] = replace(u, x=pos[0], y=pos[1], z=pos[2], radius=radius)
            children[u.id].remove(v.id)
            heappush(free_edge_ids, edge_rank.pop(v.id))
            for child in children[v.id]:
                nodes[child] = replace(nodes[child], parent=u.id)
                children[u.id].add(child)
            del children[v.id]
            del nodes[v.id]
            collapsed += 1
            changed = True
            # Match changeKey/reinsert semantics for all incident edges.
            affected_edges = [u.id] + sorted(children[u.id], key=edge_rank.__getitem__)
            for affected in affected_edges:
                if is_neurite_edge(affected):
                    _, _, _, _, _, length, weight = edge_values(affected)
                    if not queue.change_key(affected, length) and weight <= 0:
                        queue.insert(affected, length)
        if not changed:
            order = {node_id: tuple(sorted(child_ids, key=edge_rank.__getitem__))
                     for node_id, child_ids in children.items()}
            edges = tuple((nodes[child].parent, child) for child in sorted(edge_rank, key=edge_rank.__getitem__))
            state = (SourceEdgeState(tuple((edge_rank[child], nodes[child].parent, child)
                                           for child in sorted(edge_rank, key=edge_rank.__getitem__)),
                                     tuple(sorted(free_edge_ids)), next_edge_rank)
                     if _initial_edge_state is not None else None)
            return PreconditioningResult(Morphology(list(nodes.values()), child_order=order), split_count, collapsed, edges, state)
