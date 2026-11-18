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

"""Track source-style graph edge IDs during morphology preparation."""
from collections import deque
from dataclasses import dataclass
from heapq import heapify, heappop, heappush


@dataclass(frozen=True, slots=True)
class SourceEdgeState:
    """Live edges, available IDs below ``next_id``, and the next fresh ID."""
    entries: tuple[tuple[int, int, int], ...]  # (edge ID, source node, destination node)
    free_ids: tuple[int, ...]
    next_id: int

    @property
    def edge_order(self):
        return tuple((parent, child) for _, parent, child in sorted(self.entries))


def _initial_source_edges(tree):
    edges = []
    for root in tree.roots:
        queue = deque([root])
        while queue:
            parent = queue.popleft()
            for child in tree.children[parent]:
                edges.append((parent, child))
                queue.append(child)
    return SourceEdgeState(tuple((rank, parent, child) for rank, (parent, child) in enumerate(edges)),
                           (), len(edges))


def source_edge_state_after_root_adjustment(initial, prepared, state=None):
    """Track root collapses and insertions using the source's reusable IDs.

    ``initial`` is the tree immediately before root adjustment. Existing edge
    IDs survive endpoint redirection during collapse; deleted IDs are reused
    in ascending order. Return None if the observed change cannot be matched
    to those operations. This tracks the Python preparation and does not
    assert that every numerical root decision matches C++.
    """
    state = _initial_source_edges(initial) if state is None else state
    by_child = {child: (rank, parent) for rank, parent, child in state.entries}
    expected = {(node.parent, node.id) for node in initial.nodes.values() if node.parent != -1}
    if len(by_child) != len(expected) or {(parent, child) for child, (_, parent) in by_child.items()} != expected:
        return None
    children = {node: set(values) for node, values in initial.children.items()}
    free = list(state.free_ids)
    heapify(free)
    next_id = state.next_id

    def allocate():
        nonlocal next_id
        if free:
            return heappop(free)
        result = next_id
        next_id += 1
        return result

    for soma in initial.roots:
        if initial.nodes[soma].kind != 1:
            continue
        # Source adjacency lists are sorted by edge ID. Reused IDs from an
        # earlier cell can change this order before a later root adjustment.
        for root in sorted(initial.children[soma], key=lambda node: by_child[node][0]):
            if initial.nodes[root].kind not in (2, 3, 4):
                continue
            if root not in prepared.nodes or root not in by_child:
                return None
            # Root collapse deletes a child, redirects its outgoing edges to
            # the surviving root, and frees the collapsed edge's ID.
            while True:
                outgoing = children[root]
                removed = [child for child in outgoing if child not in prepared.nodes]
                if not removed:
                    break
                if len(outgoing) != 1 or len(removed) != 1:
                    return None
                child = removed[0]
                rank, _ = by_child.pop(child)
                heappush(free, rank)
                outgoing.remove(child)
                for descendant in children.pop(child):
                    edge_id, _ = by_child[descendant]
                    by_child[descendant] = (edge_id, root)
                    outgoing.add(descendant)
            parent = prepared.nodes[root].parent
            if parent != soma:
                if (parent in initial.nodes or parent not in prepared.nodes or
                        prepared.nodes[parent].parent != soma or by_child[root][1] != soma):
                    return None
                old_id, _ = by_child.pop(root)
                heappush(free, old_id)
                by_child[parent] = (allocate(), soma)
                by_child[root] = (allocate(), parent)
                children[soma].remove(root)
                children[soma].add(parent)
                children[parent] = {root}
    actual = {(node.parent, node.id) for node in prepared.nodes.values() if node.parent != -1}
    inferred = {(parent, child) for child, (_, parent) in by_child.items()}
    if len(by_child) != len(actual) or inferred != actual:
        return None
    entries = tuple(sorted((rank, parent, child) for child, (rank, parent) in by_child.items()))
    return SourceEdgeState(entries, tuple(sorted(free)), next_id)
