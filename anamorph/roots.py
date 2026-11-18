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

"""Placement of neurite roots on their soma spheres.

Based on AnaMorph (https://github.com/NeuroBox3D/AnaMorph).
"""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from math import dist, fsum, hypot, isfinite, sqrt

from .swc import Morphology, Node


def normalize_soma_sections(tree: Morphology, *, source_soma_centroid: bool = False) -> Morphology:
    """Collapse each connected soma-section tree to the C++ single-point model.

    Its position is the centroid of the section positions and its radius the
    maximum section radius. Neurites attached to any section are reparented to
    that soma root. Soma-free neurite components are retained unchanged.
    Soma sections below a neurite are rejected. The default
    uses a stable sum; ``source_soma_centroid`` uses C++ sequential addition.
    """
    if type(source_soma_centroid) is not bool:
        raise ValueError("source_soma_centroid must be boolean")
    nodes = tree.nodes
    if any(nodes[root].kind not in (1, 2, 3, 4) for root in tree.roots):
        raise ValueError("each morphology root must be a soma or neurite")
    soma_roots = [root for root in tree.roots if nodes[root].kind == 1]
    owner = {}
    groups = {root: [] for root in soma_roots}
    soma_children = {}
    for root in soma_roots:
        owner[root] = root
        groups[root].append(nodes[root])
        # The source first inserts all direct soma sections, then traverses
        # those sections in child order. Neurites directly attached to the
        # root precede neurites redirected from the section tree.
        direct = [child for child in tree.children[root] if nodes[child].kind == 1]
        soma_children[root] = [child for child in tree.children[root] if nodes[child].kind != 1]
        for child in direct:
            owner[child] = root
            groups[root].append(nodes[child])
        stack = list(reversed(direct))
        while stack:
            current = stack.pop()
            for child in tree.children[current]:
                if nodes[child].kind == 1:
                    owner[child] = root
                    groups[root].append(nodes[child])
                    stack.append(child)
                else:
                    soma_children[root].append(child)
    for node in nodes.values():
        if node.kind == 1 and node.id not in owner:
            raise ValueError(f"soma section {node.id} appears below a neurite")
    result = []
    for node in nodes.values():
        if node.kind == 1:
            if node.id not in groups:
                continue
            sections = groups[node.id]
            center = []
            for i in range(3):
                if source_soma_centroid:
                    total = 0.
                    for section in sections:
                        total += section.position[i]
                    if not isfinite(total):
                        raise ValueError("source soma centroid sum exceeds float range")
                    center.append(total/len(sections))
                else:
                    try:
                        total = fsum(section.position[i] for section in sections)
                    except OverflowError:
                        total = float("inf")
                    if isfinite(total):
                        center.append(total/len(sections))
                    else:
                        # The mean can be finite even when the sum overflows.
                        exact = sum((Fraction(s.position[i]) for s in sections), Fraction(0))
                        center.append(float(exact/len(sections)))
            result.append(replace(node, x=center[0], y=center[1], z=center[2],
                                  radius=max(s.radius for s in sections)))
        elif node.parent in owner:
            result.append(replace(node, parent=owner[node.parent]))
        else:
            result.append(node)
    child_order = {node.id: (tuple(soma_children[node.id]) if node.id in soma_children
                             else tree.children[node.id]) for node in result}
    return Morphology(result, child_order=child_order)


def adjust_soma_roots(tree: Morphology, *, preserve_surface_roots: bool = True) -> Morphology:
    """Move or insert each neurite root at the surface of its parent soma.

    This follows the C++ single-soma-sphere procedure. Multi-sample soma
    reconstruction is intentionally not inferred: each parent soma node is
    treated as its own sphere. A root exactly on the sphere is preserved by
    default to avoid inserting a coincident point; set preserve_surface_roots
    to False to reproduce the source's strict-containment insertion behavior.
    With preservation enabled, a root moved to a child's boundary position
    absorbs that child rather than leaving a zero-length first edge.
    A neurite that branches or ends inside its parent sphere cannot be adjusted
    and raises ValueError.
    """
    nodes = dict(tree.nodes)
    child_order = {node_id: list(children) for node_id, children in tree.children.items()}
    next_id = max(nodes) + 1
    def absorb_child(root, child):
        # Preserve the existing root ID and the child's ordered descendants.
        merged = replace(root, x=child.x, y=child.y, z=child.z, radius=child.radius)
        nodes[root.id] = merged
        del nodes[child.id]
        child_order[root.id] = child_order.pop(child.id)
        for child_id in child_order[root.id]:
            nodes[child_id] = replace(nodes[child_id], parent=root.id)
        return merged
    roots = [(p, n.id) for n in nodes.values() if n.kind in (2, 3, 4)
             and n.parent != -1 and (p := nodes[n.parent]).kind == 1]
    for soma, root_id in roots:
        root = nodes[root_id]
        radius = soma.radius
        distance = dist(soma.position, root.position)
        # The source's strict containment test inserts a duplicate root on an
        # exact boundary hit, which can make the spline impossible to build.
        if preserve_surface_roots and distance == radius:
            continue
        if distance >= radius:
            fraction = radius/distance
            if isfinite(distance) and fraction:
                position = tuple(c+fraction*(p-c) for c, p in zip(soma.position, root.position))
            else:
                # Halving before subtraction preserves a finite direction
                # when opposite-sign coordinates have an unrepresentable gap.
                # It also avoids losing the displacement when radius/distance
                # underflows despite a representable surface position.
                delta = tuple(p*.5-c*.5 for c, p in zip(soma.position, root.position))
                scale = max(abs(x) for x in delta)
                unit = tuple(x/scale for x in delta)
                norm = hypot(*unit)
                position = tuple(c+radius*(x/norm) for c, x in zip(soma.position, unit))
            nodes[next_id] = Node(next_id, root.kind, *position, root.radius, soma.id)
            nodes[root.id] = replace(root, parent=next_id)
            siblings = child_order[soma.id]
            siblings[siblings.index(root.id)] = next_id
            child_order[next_id] = [root.id]
            next_id += 1
            continue
        while True:
            children = [nodes[child_id] for child_id in child_order[root.id]]
            if len(children) != 1 or children[0].kind not in (2, 3, 4):
                raise ValueError(f"neurite rooted at node {root_id} branches or ends inside soma {soma.id}")
            child = children[0]
            child_distance = dist(soma.position, child.position)
            if child_distance < radius:
                # Keep the current root's identifier while adopting the child
                # position and reconnecting its descendants.
                root = absorb_child(root, child)
                continue
            if preserve_surface_roots and child_distance == radius:
                # Keeping both points would create a zero-length first edge.
                # The boundary node can have multiple outgoing branches.
                root = absorb_child(root, child)
                break
            d = tuple(b-a for a, b in zip(root.position, child.position))
            v = tuple(a-c for a, c in zip(root.position, soma.position))
            # Intersect the exiting ray with the sphere in radius-relative
            # coordinates. Squaring the unscaled inputs can overflow or
            # underflow even though the intersection itself is representable.
            length, offset_length = hypot(*d), hypot(*v)
            if not isfinite(length) or not isfinite(offset_length) or not length:
                raise ValueError(f"neurite root {root_id} has unrepresentable segment geometry")
            along = fsum((x/radius)*(y/length) for x, y in zip(v, d))
            deficit = ((radius-offset_length)/radius)*(1+offset_length/radius)
            radical = sqrt(max(0., along*along+deficit))
            # Rationalization avoids cancellation when the root is close to
            # the sphere and the segment points outward.
            travel = deficit/(radical+along) if along > 0 else radical-along
            scale = max(radius, length)
            t = ((radius/scale)*travel)/(length/scale)
            if not 0 <= t <= 1:
                raise ValueError(f"could not place neurite root {root_id} on soma sphere")
            position = tuple(a+t*x for a, x in zip(root.position, d))
            if preserve_surface_roots and position == child.position:
                # An outward intersection can also round to the child's
                # coordinates even when the distance comparison is not exact.
                root = absorb_child(root, child)
                break
            nodes[root.id] = replace(root, x=position[0], y=position[1], z=position[2],
                                     radius=root.radius+t*(child.radius-root.radius))
            break
    return Morphology(list(nodes.values()), child_order=child_order)
