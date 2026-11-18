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

"""SWC input and basic morphology-tree operations.

Based on AnaMorph (https://github.com/NeuroBox3D/AnaMorph).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import dist, isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True, slots=True)
class Node:
    id: int
    kind: int
    x: float
    y: float
    z: float
    radius: float
    parent: int

    @property
    def position(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z


@dataclass(frozen=True, slots=True)
class DistanceViolation:
    edge: tuple[int, int]
    length: float
    primary: bool
    severe_primary: bool
    secondary: bool
    source_neighbor_radius: float
    destination_neighbor_radius: float
    exact_length_squared: tuple[int, int] | None = None


def _exact_distance_squared(first, second):
    """Squared distance for the exact values represented by input floats."""
    return sum(((Fraction(a)-Fraction(b))**2 for a, b in zip(first, second)), Fraction(0))


class Morphology:
    """Validated SWC forest, retaining the input's soma samples and identifiers."""

    def __init__(self, nodes: list[Node], *, child_order: Mapping[int, tuple[int, ...]] | None = None):
        if not nodes:
            raise ValueError("SWC file contains no nodes")
        by_id: dict[int, Node] = {}
        for node in nodes:
            if node.id <= 0 or node.id in by_id:
                raise ValueError(f"invalid or duplicate node id: {node.id}")
            if node.kind <= 0:
                raise ValueError(f"node {node.id} has invalid type {node.kind}")
            if not all(isfinite(x) for x in (*node.position, node.radius)) or node.radius <= 0:
                raise ValueError(f"node {node.id} must have finite coordinates and a positive radius")
            by_id[node.id] = node
        children: dict[int, list[int]] = {node.id: [] for node in nodes}
        roots: list[int] = []
        for node in nodes:
            if node.parent == -1:
                roots.append(node.id)
            elif node.parent not in by_id:
                raise ValueError(f"node {node.id} references missing parent {node.parent}")
            else:
                children[node.parent].append(node.id)
        if child_order is not None:
            if set(child_order) != set(children):
                raise ValueError("child ordering must contain every node")
            for node_id, order in child_order.items():
                if len(order) != len(children[node_id]) or set(order) != set(children[node_id]):
                    raise ValueError(f"child ordering for node {node_id} does not match its edges")
                children[node_id] = list(order)
        if not roots:
            raise ValueError("SWC file has no root (parent -1)")
        # Each node has at most one parent, so any nodes unreachable from roots
        # must belong to a cycle. Use an iterative traversal for deep neurons.
        seen: set[int] = set()
        stack = roots.copy()
        while stack:
            current = stack.pop()
            if current in seen:
                raise ValueError(f"cycle detected at node {current}")
            seen.add(current)
            stack.extend(children[current])
        if len(seen) != len(nodes):
            raise ValueError("SWC contains a cycle disconnected from its roots")
        self.nodes: Mapping[int, Node] = MappingProxyType(by_id)
        self.children: Mapping[int, tuple[int, ...]] = MappingProxyType(
            {key: tuple(value) for key, value in children.items()}
        )
        self.roots = tuple(roots)

    def neurite_segments(self):
        """Yield edges with a neurite at both ends, as pairs of nodes."""
        for node in self.nodes.values():
            if node.kind in (2, 3, 4) and node.parent != -1:
                parent = self.nodes[node.parent]
                if parent.kind in (2, 3, 4):
                    yield parent, node

    def primary_distance_violations(self, alpha: float = 3.0):
        """Return neurite edges with length <= alpha times their larger radius.

        This is a basic tree diagnostic, not the full geometric analysis.
        """
        if not isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be positive and finite")
        alpha = Fraction(alpha)
        return [(a.id, b.id) for a, b in self.neurite_segments()
                if _exact_distance_squared(a.position, b.position) <= (alpha*Fraction(max(a.radius, b.radius)))**2]

    def secondary_distance_radii(self, u: Node, v: Node):
        """Return source and destination radii used by the C++ SMDV test.

        The destination-side calculation follows the checked-in source, which
        scans u's children rather than v's children. This differs from the
        symmetric neighborhood described in the paper.
        """
        if v.parent != u.id or u.id not in self.nodes or v.id not in self.nodes:
            raise ValueError("nodes must form a directed neurite segment")
        neurite = (2, 3, 4)
        u_children = [self.nodes[i] for i in self.children[u.id] if self.nodes[i].kind in neurite]
        v_children = [self.nodes[i] for i in self.children[v.id] if self.nodes[i].kind in neurite]
        root = u.parent == -1 or self.nodes[u.parent].kind not in neurite
        r_u = 0. if root else max([u.radius, self.nodes[u.parent].radius]+
                                  [n.radius for n in u_children if n.id != v.id])
        r_v = max([v.radius]+[n.radius for n in u_children]) if v_children else 0.
        return r_u, r_v

    def minimum_distance_violations(self):
        """Report the source thresholds using exact represented input values.

        An unrepresentable length is infinity; its exact squared value is also
        returned so that overflow does not destroy the distance evidence.
        """
        violations = []
        for u, v in self.neurite_segments():
            length = dist(u.position, v.position)
            length_squared = _exact_distance_squared(u.position, v.position)
            radius = Fraction(max(u.radius, v.radius))
            primary = length_squared <= (2*radius)**2
            severe = length_squared <= radius**2
            r_u, r_v = self.secondary_distance_radii(u, v)
            secondary = length_squared <= (Fraction(r_u)+Fraction(r_v))**2
            if primary or secondary:
                exact = (length_squared.numerator, length_squared.denominator) if not isfinite(length) else None
                violations.append(DistanceViolation((u.id, v.id), length, primary, severe, secondary, r_u, r_v, exact))
        return violations


def read_swc(path: str | Path) -> Morphology:
    nodes: list[Node] = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            fields = line.split("#", 1)[0].split()
            if not fields:
                continue
            if len(fields) != 7:
                raise ValueError(f"line {number}: expected seven SWC fields")
            try:
                nodes.append(Node(int(fields[0]), int(fields[1]), *(float(x) for x in fields[2:6]), int(fields[6])))
            except ValueError as error:
                raise ValueError(f"line {number}: invalid SWC value") from error
    return Morphology(nodes)
