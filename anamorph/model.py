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

"""Prepare SWC samples for the nonlinear AnaMorph model.

Based on AnaMorph (https://github.com/NeuroBox3D/AnaMorph).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from .precondition import precondition
from .roots import adjust_soma_roots, normalize_soma_sections
from .source_order import source_edge_state_after_root_adjustment
from .swc import Morphology


@dataclass(frozen=True, slots=True)
class PreparedMorphology:
    morphology: Morphology
    split_segments: int
    collapsed_segments: int
    source_edge_order: tuple[tuple[int, int], ...] | None = None


def _apply_source_child_order(tree, state):
    """Keep adjacency lists in the source graph's edge-ID order."""
    if state is None:
        return tree
    rank = {child: edge_id for edge_id, _, child in state.entries}
    order = {node: tuple(sorted(children, key=rank.__getitem__)) for node, children in tree.children.items()}
    if all(order[node] == tree.children[node] for node in tree.nodes):
        return tree
    return Morphology(list(tree.nodes.values()), child_order=order)


def prepare_morphology(tree: Morphology, *, preconditioning: bool = True,
                       alpha: float = 3., beta: float = 1.5, gamma: float = 10.,
                       scale_radius: float = 1., preserve_surface_roots: bool = True,
                       source_soma_centroid: bool = False,
                       precondition_norm_mode: str = "robust") -> PreparedMorphology:
    """Consolidate soma sections, position root vertices, and precondition.

    This reproduces the source's stage order. Its greedy queue update order and
    some root/format corner cases still require comparison with the C++ build.
    ``source_soma_centroid`` selects source-order rather than stable summation.
    """
    if not isfinite(scale_radius) or scale_radius < .1:
        raise ValueError("radius scale must be finite and at least 0.1")
    if precondition_norm_mode not in ("robust", "source", "source-x87"):
        raise ValueError("precondition_norm_mode must be robust, source, or source-x87")
    for node in tree.nodes.values():
        if node.kind not in (1, 2, 3, 4):
            raise ValueError(f"node {node.id} has unsupported SWC type {node.kind}")
        if node.radius < 1e-8:
            raise ValueError(f"node {node.id} has a radius below the source's 1e-8 minimum")
        if node.parent != -1:
            parent = tree.nodes[node.parent]
            if parent.kind in (2, 3, 4) and node.kind != parent.kind:
                raise ValueError(f"edge {parent.id}-{node.id} changes neurite type")
    initial = normalize_soma_sections(tree, source_soma_centroid=source_soma_centroid)
    tree = adjust_soma_roots(initial, preserve_surface_roots=preserve_surface_roots)
    source_state = source_edge_state_after_root_adjustment(initial, tree)
    tree = _apply_source_child_order(tree, source_state)
    split_segments = collapsed_segments = 0
    if preconditioning:
        result = precondition(tree, alpha, beta, gamma, precondition_norm_mode=precondition_norm_mode,
                              _initial_edge_state=source_state)
        tree = result.morphology
        split_segments, collapsed_segments = result.split_segments, result.collapsed_segments
        source_state = result.source_edge_state
    if scale_radius != 1.:
        tree = Morphology([replace(node, radius=node.radius*scale_radius) for node in tree.nodes.values()],
                          child_order=tree.children)
        # The source repeats root adjustment after scaling, so root geometry
        # remains attached to the resized soma sphere.
        initial = tree
        tree = adjust_soma_roots(initial, preserve_surface_roots=preserve_surface_roots)
        if source_state is not None:
            source_state = source_edge_state_after_root_adjustment(initial, tree, source_state)
        tree = _apply_source_child_order(tree, source_state)
    return PreparedMorphology(tree, split_segments, collapsed_segments,
                              None if source_state is None else source_state.edge_order)
