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

"""Experimental neuron assembly from analytic canals and red-blue unions.

Based on AnaMorph (https://github.com/NeuroBox3D/AnaMorph). This assembles
closed neurite stumps with red-blue union and then appends their tails.
"""
from __future__ import annotations

from collections import deque
from contextlib import ExitStack
from dataclasses import replace
from itertools import chain, count
from math import cos, isfinite, pi, sqrt
import random

from .analytic_intersections import CanalSegment, _curve_roots
from .analytic_network import analyze_analytic_partial, _bounding_box
from .canal_mesh import (CanalPathMesh, _segment_tangent, append_canal_path, append_terminal_hemisphere,
                         canal_path_mesh, ico_sphere)
from .geometry import Spline, order_paths_breadth_first, partition_paths
from .mesh import Mesh, cross, dot, sub
from .polynomial import add, derivative, evaluate, multiply
from .bernstein import BernsteinPolynomial
from .redblue import (RedBlueBoundaryInteraction, RedBlueDisjoint, RedBlueError, RedBlueMultipleLoops,
                      RedBlueNumericalEdgeCase, RedBlueRetryLimit, _face_bvh, _point_inside_surface,
                      redblue_union_with_retries)
from .partial_obj import PartialObjWriter
from .model import prepare_morphology
from .source_random import make_random
from .source_float import x87_sqrt
from .source_render import source_render_vector_score
from .source_spline import SourceCanalSegment, source_root_cylinder, source_spline_controls
from .swc import Morphology


class PartialAnalysisError(ValueError):
    """Generation was stopped because partial checks found or could not resolve issues."""

    def __init__(self, report, *, gate="all"):
        self.report = report
        self.blocking_findings = report.source_job_findings if gate == "source" else report.findings
        label = "source-selected intersection checks" if gate == "source" else "partial analytic checks"
        super().__init__(f"{label} found {len(self.blocking_findings)} finding(s) and "
                         f"{len(report.unresolved)} unresolved check(s); run analyze-analytic for details")


class RedBlueTerminalRingConsumed(RedBlueError):
    """A stump cannot grow further after its end ring or cap was removed."""


def _render_vector_score(segment, axis, *, tolerance=1e-6):
    """Find min |axis × gamma'|² / |gamma'|² via Bézier clipping."""
    controls = getattr(segment, "source_controls", None)
    if controls is None:
        velocity = [derivative(p) for p in segment.power]
    else:
        # The selector uses a power-basis polynomial internally, but it must
        # describe the same Bézier derivative that will render the surface.
        velocity = []
        for component in controls:
            d = [3*(b-a) for a, b in zip(component, component[1:])]
            velocity.append([d[0], 2*(d[1]-d[0]), d[2]-2*d[1]+d[0]])
    scale = max(abs(coefficient) for component in velocity for coefficient in component)
    if not scale:
        raise ValueError("zero-speed centerline prevents a stable render frame")
    # The ratio is invariant under uniform scaling of all velocity components.
    # Normalize before forming fourth-order products to avoid absolute-epsilon
    # failures and underflow for very small coordinate units.
    velocity = [[coefficient/scale for coefficient in component] for component in velocity]
    cross_components = [add([axis[i]*c for c in velocity[j]], [-axis[j]*c for c in velocity[i]])
                        for i, j in ((1, 2), (2, 0), (0, 1))]
    def squared_norm(components):
        result = [0.]
        for component in components:
            result = add(result, multiply(component, component))
        return result
    p, q = squared_norm(cross_components), squared_norm(velocity)
    numerator = add(multiply(derivative(p), q), [-x for x in multiply(p, derivative(q))])
    roots = _curve_roots(BernsteinPolynomial.from_power(numerator, degree=7), tolerance=tolerance)
    parameters = [0., 1.] + [root.midpoint for root in roots]
    values = []
    for t in parameters:
        tangent = tuple(evaluate(component, t) for component in velocity)
        speed = dot(tangent, tangent)
        if speed <= 1e-24:
            raise ValueError("zero-speed centerline prevents a stable render frame")
        transverse = cross(axis, tangent)
        values.append(dot(transverse, transverse)/speed)
    return min(values)


def _render_vector(segments, *, rng=None, source_style=False):
    """Use the source's z-first, then random unit-vector selection strategy."""
    def permissible(axis):
        score = source_render_vector_score if source_style else _render_vector_score
        return min(score(segment, axis) for segment in segments) > .01
    if permissible((0., 0., 1.)):
        return (0., 0., 1.)
    rng = rng if rng is not None else random.Random(0)
    for _ in range(1024):
        # Aux::VecMat::randUnitVec3 samples a cube before normalizing.
        values = tuple(rng.uniform(-1.1, 1.1) for _ in range(3))
        length = x87_sqrt(dot(values, values)) if source_style else sqrt(dot(values, values))
        if not length:
            continue
        candidate = tuple(value/length for value in values)
        if permissible(candidate):
            return candidate
    raise ValueError("could not find a permissible render vector for the neurite path")


def _path_curves(tree, path, parametrization, *, source_curve_evaluation=False):
    """Build source-style canal segments, including the root cylinder."""
    nodes = [tree.nodes[i] for i in path]
    parent = tree.nodes.get(nodes[0].parent)
    root = parent is not None and parent.kind == 1
    tangent = sub(nodes[0].position, parent.position) if root else None
    spline = Spline.interpolate([node.position for node in nodes], parametrization=parametrization,
                                start_direction=tangent)
    curves = [CanalSegment.from_spline(spline, i, a.radius, b.radius)
              for i, (a, b) in enumerate(zip(nodes, nodes[1:]))]
    if source_curve_evaluation:
        _, controls = source_spline_controls([node.position for node in nodes],
                                             parametrization=parametrization, start_direction=tangent)
        curves = [SourceCanalSegment(curve.power, curve.radius0, curve.radius1, controls[i])
                  for i, curve in enumerate(curves)]
    if root:
        displacement = sub(nodes[0].position, parent.position)
        power = tuple((parent.position[i], displacement[i], 0., 0.) for i in range(3))
        if source_curve_evaluation:
            cylinder = source_root_cylinder(parent.position, nodes[0].position, nodes[0].radius)
        else:
            cylinder = CanalSegment(power, nodes[0].radius, nodes[0].radius)
        curves.insert(0, cylinder)
    return nodes, parent, root, curves


def _require_neurite_segments(tree):
    # A lone neurite root does not define a canal segment. Without this check,
    # partition_paths returns no path and assembly silently drops the neurite.
    for node in tree.nodes.values():
        parent = tree.nodes.get(node.parent)
        if (node.kind in (2, 3, 4) and (parent is None or parent.kind == 1) and
                not any(tree.nodes[child].kind in (2, 3, 4) for child in tree.children[node.id])):
            raise ValueError(f"neurite root {node.id} has no segment to render")


def _manifold_safe_flush_selection(mesh, selected):
    """Keep extra faces until every remaining vertex has one connected fan.

    The input is an oriented vertex-manifold mesh, possibly with boundary.
    Faces are only removed from the proposed flush set, preserving its halo.
    """
    selected = set(selected)
    if not selected:
        return ()
    incident = [[] for _ in mesh.vertices]
    edges = {}
    for fid, face in enumerate(mesh.faces):
        for vertex in face:
            incident[vertex].append(fid)
        for a, b in zip(face, face[1:]+face[:1]):
            edge = (a, b) if a < b else (b, a)
            edges.setdefault(edge, []).append(fid)
    queue = deque(vertex for vertex, faces in enumerate(incident) if faces)
    queued = set(queue)
    while queue:
        vertex = queue.popleft()
        queued.remove(vertex)
        remaining = [fid for fid in incident[vertex] if fid not in selected]
        if not remaining:
            continue
        visited, stack = set(), [remaining[0]]
        while stack:
            fid = stack.pop()
            if fid in visited:
                continue
            visited.add(fid)
            for other in mesh.faces[fid]:
                if other == vertex:
                    continue
                edge = (vertex, other) if vertex < other else (other, vertex)
                stack.extend(neighbor for neighbor in edges[edge]
                             if neighbor not in selected and neighbor not in visited)
        if len(visited) == len(remaining):
            continue
        for fid in incident[vertex]:
            if fid not in selected:
                continue
            selected.remove(fid)
            for neighbor in mesh.faces[fid]:
                if neighbor not in queued:
                    queue.append(neighbor)
                    queued.add(neighbor)
    return tuple(sorted(selected))


def _flushable_faces(mesh, boxes):
    """Select faces away from remaining canal boxes, retaining a vertex halo.

    The boxes are numerical conservative estimates, not certified enclosures.
    A one-ring halo protects already-written boundary vertices from the next
    union; the writer independently checks that this protection held.
    """
    if not mesh.faces or not boxes:
        return ()
    tree = _face_bvh(mesh)
    protected = set()
    def overlaps(a, b):
        return all(a[i][0] <= b[i][1] and b[i][0] <= a[i][1] for i in range(3))
    for box in boxes:
        stack = [tree]
        while stack:
            bounds, left, right, entries = stack.pop()
            if not overlaps(bounds, box):
                continue
            if entries:
                protected.update(fid for fid, _, face_bounds in entries if overlaps(face_bounds, box))
            else:
                stack.extend((left, right))
    if not protected:
        # Leave the intact mesh available for the normal disjoint diagnostic.
        return ()
    vertices = {v for fid in protected for v in mesh.faces[fid]}
    selected = (i for i, face in enumerate(mesh.faces) if not vertices.intersection(face))
    return _manifold_safe_flush_selection(mesh, selected)


def _future_canal_boxes(path_data):
    boxes = []
    for _, _, _, curves in path_data:
        for curve in curves:
            box = _bounding_box(curve)
            # Match the intent of the source's generous relative and absolute
            # padding, retaining additional geometry around future paths.
            boxes.append(tuple((lo-(hi-lo)*.1-.01, hi+(hi-lo)*.1+.01) for lo, hi in box))
    return boxes


def render_individual_surfaces(tree: Morphology, *, angular_segments: int = 6, soma_refinements: int = 3,
                               strategy: str = "max-chordal-depth", parametrization: str = "chord-length",
                               triangle_height_factor: float = 1., preconditioning: bool = True,
                               pc_alpha: float = 3., pc_beta: float = 1.5, pc_gamma: float = 10.,
                               scale_radius: float = 1., filter_angle: float = 90.,
                               radius_ratio: float = 10., preserve_crease_edges: bool = False,
                               arc_length_mode: str = "source", preserve_surface_roots: bool = True,
                               random_seed: int = 0, random_generator: str = "python",
                               quad_sqrt_mode: str = "native", render_trig_mode: str = "native",
                               source_curve_evaluation: bool = False, source_render_vector: bool = False,
                               source_frame: bool = False, source_soma_centroid: bool = False,
                               precondition_norm_mode: str = "robust") -> Mesh:
    """Render separate soma/junction spheres and capped canal paths.

    These components intentionally overlap; this output is for inspecting the
    individual modeling surfaces and is not a stitched cell boundary.
    """
    if (type(source_curve_evaluation) is not bool or type(source_render_vector) is not bool or
            type(source_frame) is not bool or type(source_soma_centroid) is not bool or angular_segments < 3 or
            not isfinite(triangle_height_factor) or triangle_height_factor <= 0 or
            quad_sqrt_mode not in ("native", "windows-x87") or render_trig_mode not in ("native", "windows-x87")):
        raise ValueError("invalid meshing settings")
    source_curve_evaluation = source_curve_evaluation or source_render_vector
    tree = prepare_morphology(tree, preconditioning=preconditioning, alpha=pc_alpha,
                              beta=pc_beta, gamma=pc_gamma, scale_radius=scale_radius,
                              preserve_surface_roots=preserve_surface_roots,
                              source_soma_centroid=source_soma_centroid,
                              precondition_norm_mode=precondition_norm_mode).morphology
    _require_neurite_segments(tree)
    rng = make_random(random_seed, random_generator)
    roots = [tree.nodes[node_id] for node_id in tree.roots]
    owners = {}
    for root_node in roots:
        stack = [root_node.id]
        while stack:
            node_id = stack.pop()
            owners[node_id] = root_node.id
            stack.extend(tree.children[node_id])
    paths_by_root = {root_node.id: [] for root_node in roots}
    for path in partition_paths(tree, strategy, filter_angle, radius_ratio):
        paths_by_root[owners[path[0]]].append(path)
    pieces = []
    for root_node in roots:
        if root_node.kind == 1 or len(tree.children[root_node.id]) > 1:
            pieces.append(ico_sphere(root_node.position, root_node.radius, soma_refinements))
        for path in paths_by_root[root_node.id]:
            _, _, root, curves = _path_curves(tree, path, parametrization,
                                               source_curve_evaluation=source_curve_evaluation)
            # The source chooses the frame from the actual neurite path rather
            # than the extra root cylinder and uses phi=0 for this output.
            render_vector = _render_vector(curves[1:] if root else curves, rng=rng, source_style=source_render_vector)
            pieces.append(canal_path_mesh(curves, angular_segments=angular_segments,
                                          triangle_height_factor=triangle_height_factor,
                                          render_vector=render_vector, phi0=0.,
                                          preserve_crease_edges=preserve_crease_edges,
                                          arc_length_mode=arc_length_mode, quad_sqrt_mode=quad_sqrt_mode,
                                          render_trig_mode=render_trig_mode, source_frame=source_frame))
    vertices, faces = [], []
    for piece in pieces:
        offset = len(vertices)
        vertices.extend(piece.vertices)
        faces.extend(tuple(offset+i for i in face) for face in piece.faces)
    return Mesh(tuple(vertices), tuple(faces))


def assemble_redblue_neuron(tree: Morphology, *, angular_segments: int = 6, soma_refinements: int = 3,
                            strategy: str = "max-chordal-depth", parametrization: str = "chord-length",
                            triangle_height_factor: float = 1., max_outer_attempts: int | None = None,
                            outer_loop_maxiter: int = 16, radius_factor_decrement: float = .01,
                            max_inner_attempts: int = 8, max_complex_edge_growth_factor: float = 2.,
                            preconditioning: bool = True,
                            pc_alpha: float = 3., pc_beta: float = 1.5, pc_gamma: float = 10.,
                            scale_radius: float = 1., filter_angle: float = 90., radius_ratio: float = 10.,
                            preserve_crease_edges: bool = False, arc_length_mode: str = "source",
                            preserve_surface_roots: bool = True, random_seed: int = 0, random_generator: str = "python",
                            source_ear_clipping: bool = False, source_delaunay: bool = False,
                            source_delaunay_allow_limit: bool = False,
                            quad_sqrt_mode: str = "native",
                            render_trig_mode: str = "native", source_retry_semantics: bool = False,
                            source_curve_evaluation: bool = False,
                            source_render_vector: bool = False, source_frame: bool = False,
                            source_soma_centroid: bool = False,
                            precondition_norm_mode: str = "robust",
                            _partial_output=None,
                            _flush_face_limit: int = 100000, _partial_stack=None):
    """Mesh a morphology forest using canal surfaces and red-blue union.

    The consistency classifier is incomplete; callers must treat this as
    experimental meshing. Multiple-loop
    retry details and final self-intersection validation are not fully implemented.
    By default retries continue until the source-style safe radius bound is
    reached. Set max_outer_attempts to a positive integer to impose a limit.
    ``source_delaunay`` uses the source's floating-point cleanup and implies
    source ear clipping; it is a comparison mode, not a mesh-quality guarantee.
    ``source_delaunay_allow_limit`` permits its bounded output with a warning.
    """
    if (type(source_ear_clipping) is not bool or type(source_delaunay) is not bool or
            type(source_delaunay_allow_limit) is not bool or
            type(source_retry_semantics) is not bool or
            type(source_curve_evaluation) is not bool or type(source_render_vector) is not bool or
            type(source_frame) is not bool or type(source_soma_centroid) is not bool or
            quad_sqrt_mode not in ("native", "windows-x87") or
            render_trig_mode not in ("native", "windows-x87") or angular_segments < 3 or
            (max_outer_attempts is not None and (type(max_outer_attempts) is not int or max_outer_attempts < 1)) or
            max_inner_attempts < 1 or outer_loop_maxiter < 1 or
            not isfinite(radius_factor_decrement) or not 0 < radius_factor_decrement <= .1 or
            not isfinite(max_complex_edge_growth_factor) or max_complex_edge_growth_factor <= 0):
        raise ValueError("invalid meshing settings")
    source_curve_evaluation = source_curve_evaluation or source_render_vector
    source_delaunay = source_delaunay or source_delaunay_allow_limit
    source_ear_clipping = source_ear_clipping or source_delaunay
    if ((_partial_output is None) != (_partial_stack is None) or
            type(_flush_face_limit) is not int or _flush_face_limit < 1):
        raise ValueError("invalid partial OBJ output settings")
    tree = prepare_morphology(tree, preconditioning=preconditioning, alpha=pc_alpha,
                              beta=pc_beta, gamma=pc_gamma, scale_radius=scale_radius,
                              preserve_surface_roots=preserve_surface_roots,
                              source_soma_centroid=source_soma_centroid,
                              precondition_norm_mode=precondition_norm_mode).morphology
    _require_neurite_segments(tree)
    somas = [node for node in tree.nodes.values() if node.kind == 1]
    if any(soma.parent != -1 for soma in somas):
        raise ValueError("soma nodes must be morphology roots")
    if any(node.kind not in (1, 2, 3, 4) for node in tree.nodes.values()):
        raise ValueError("analytic red-blue assembly supports SWC types 1-4 only")
    paths = order_paths_breadth_first(tree, partition_paths(tree, strategy, filter_angle, radius_ratio))
    rng = make_random(random_seed, random_generator)
    vertices, faces = [], []
    junction_roots = [tree.nodes[node_id] for node_id in tree.roots
                      if tree.nodes[node_id].kind in (2, 3, 4) and len(tree.children[node_id]) > 1]
    junction_surfaces = {}
    for soma in somas + junction_roots:
        sphere = ico_sphere(soma.position, soma.radius, soma_refinements)
        if soma.kind != 1:
            junction_surfaces[soma.id] = sphere
        offset = len(vertices)
        vertices.extend(sphere.vertices)
        faces.extend(tuple(offset+i for i in face) for face in sphere.faces)
    result = Mesh(tuple(vertices), tuple(faces))
    writer = _partial_stack.enter_context(PartialObjWriter(result)) if _partial_stack is not None else None
    path_data = ([_path_curves(tree, path, parametrization, source_curve_evaluation=source_curve_evaluation)
                  for path in paths] if writer is not None else None)
    path_boxes = [None]*len(path_data) if path_data is not None else None
    flushes, peak_active_faces = 0, len(result.faces)
    def complete_path(curves, rvec, phi):
        boundary = canal_path_mesh(curves, angular_segments=angular_segments,
                                   triangle_height_factor=triangle_height_factor,
                                   render_vector=rvec, phi0=phi, return_boundary=True,
                                   preserve_crease_edges=preserve_crease_edges,
                                   arc_length_mode=arc_length_mode, quad_sqrt_mode=quad_sqrt_mode,
                                   render_trig_mode=render_trig_mode, source_frame=source_frame)
        last = curves[-1]
        return append_terminal_hemisphere(boundary, start=last.point(1.),
                                          direction=_segment_tangent(last, 1.), radius=last.radius1,
                                          angular_segments=angular_segments, render_vector=rvec, phi0=phi,
                                          quad_sqrt_mode=quad_sqrt_mode, render_trig_mode=render_trig_mode,
                                          source_frame=source_frame)
    for path_index, path in enumerate(paths):
        nodes, parent, root, curves = (path_data[path_index] if path_data is not None else
                                       _path_curves(tree, path, parametrization,
                                                    source_curve_evaluation=source_curve_evaluation))
        if writer is not None and len(result.faces) > _flush_face_limit:
            for remaining_index in range(path_index, len(path_data)):
                if path_boxes[remaining_index] is None:
                    path_boxes[remaining_index] = tuple(_future_canal_boxes((path_data[remaining_index],)))
            selected = _flushable_faces(result, chain.from_iterable(path_boxes[path_index:]))
            if selected:
                result = writer.flush(selected)
                flushes += 1
        if parent is None and len(tree.children[nodes[0].id]) == 1:
            # A soma-free component starts with its first complete canal path.
            # Its proximal end retains the flat cap; the other end is rounded
            # exactly like the existing neurite terminals.
            rvec = _render_vector(curves, rng=rng, source_style=source_render_vector)
            phi = rng.uniform(0., 2*pi/angular_segments)
            seed = complete_path(curves, rvec, phi)
            offset = len(result.vertices)
            merged = Mesh(result.vertices+seed.vertices,
                          result.faces+tuple(tuple(offset+i for i in face) for face in seed.faces))
            if writer is not None:
                writer.replace_mesh(merged, tuple(range(offset)))
            result = merged
            peak_active_faces = max(peak_active_faces, len(result.faces))
            continue
        radius_factor = cos(pi/angular_segments)
        neighbor_ids = list(tree.children[nodes[0].id])
        if parent is not None and parent.kind in (2, 3, 4):
            neighbor_ids.append(parent.id)
        safe_lower_bound = min([.5]+[tree.nodes[i].radius/(2*nodes[0].radius)
                                       for i in neighbor_ids if tree.nodes[i].kind in (2, 3, 4)])
        def decrease_radius_factor(factor):
            reduced = max(safe_lower_bound, factor-radius_factor_decrement)
            if factor > safe_lower_bound and reduced == factor:
                raise ValueError("radius factor decrement is too small to change the radius factor")
            return reduced
        rvec = _render_vector(curves[1:] if root else curves, rng=rng, source_style=source_render_vector)
        last_error = None
        merged = None
        merged_vertex_map = None
        # The source joins a root cylinder + its first canal segment, or the
        # first segment of a branch. It grows the stump when the terminal ring
        # is consumed by the union before appending the remaining tail.
        stump_size = min(len(curves), 2 if root else 1)
        retry_red = result
        attempts = count(1) if max_outer_attempts is None else range(1, max_outer_attempts+1)
        for attempt in attempts:
            if attempt % outer_loop_maxiter == 0:
                radius_factor = decrease_radius_factor(radius_factor)
            phi = rng.uniform(0., 2*pi/angular_segments)
            try:
                junction = junction_surfaces.get(nodes[0].id) if parent is None else None
                if junction is not None:
                    # Join the complete closed root path, including its tip.
                    # This avoids a temporary flat end cap coinciding with the
                    # rounded junction when the first segment is very short.
                    full_curves = list(curves)
                    full_curves[0] = replace(full_curves[0], radius0=full_curves[0].radius0*radius_factor)
                    full_path = complete_path(full_curves, rvec, phi)
                    union = redblue_union_with_retries(retry_red, full_path, max_attempts=max_inner_attempts,
                                                       max_growth_factor=max_complex_edge_growth_factor,
                                                       return_vertex_maps=True,
                                                       _allow_red_boundary=writer is not None,
                                                       _source_ear_rng=rng if source_ear_clipping else None,
                                                       _source_delaunay=source_delaunay,
                                                       source_delaunay_allow_limit=source_delaunay_allow_limit)
                    if source_retry_semantics and union.inner_attempts == max_inner_attempts:
                        last_error = RedBlueRetryLimit("source inner iteration limit reached")
                        radius_factor = decrease_radius_factor(radius_factor)
                        retry_red = result
                    else:
                        merged = union.mesh
                        if writer is not None:
                            merged_vertex_map = tuple(union.red_vertex_map[:len(result.vertices)])
                else:
                    stump_curves = list(curves[:stump_size])
                    if not root:
                        initial = stump_curves[0]
                        stump_curves[0] = replace(initial, radius0=initial.radius0*radius_factor)
                    stump = canal_path_mesh(stump_curves, angular_segments=angular_segments,
                                            triangle_height_factor=triangle_height_factor,
                                            render_vector=rvec, phi0=phi, return_boundary=True,
                                            preserve_crease_edges=preserve_crease_edges,
                                            arc_length_mode=arc_length_mode, quad_sqrt_mode=quad_sqrt_mode,
                                            render_trig_mode=render_trig_mode, source_frame=source_frame)
                    union = redblue_union_with_retries(retry_red, stump.mesh, max_attempts=max_inner_attempts,
                                                       max_growth_factor=max_complex_edge_growth_factor,
                                                       return_vertex_maps=True,
                                                       _allow_red_boundary=writer is not None,
                                                       _source_ear_rng=rng if source_ear_clipping else None,
                                                       _source_delaunay=source_delaunay,
                                                       source_delaunay_allow_limit=source_delaunay_allow_limit)
                    mapped_ring = tuple(union.blue_vertex_map[i] for i in stump.end_ring)
                    mapped_cap = union.blue_vertex_map[stump.end_cap]
                    if mapped_cap is None or any(i is None for i in mapped_ring):
                        last_error = RedBlueError("red-blue union consumed the terminal ring or cap")
                        if stump_size == len(curves):
                            raise RedBlueTerminalRingConsumed(str(last_error))
                        # The source carries its outer iteration counter and
                        # radius factor forward when the initial stump grows.
                        stump_size += 1
                        retry_red = result
                        if source_retry_semantics and union.inner_attempts == max_inner_attempts:
                            radius_factor = decrease_radius_factor(radius_factor)
                    elif source_retry_semantics and union.inner_attempts == max_inner_attempts:
                        # The checked-in source restarts whenever the inner loop
                        # reaches its limit, even when the last union succeeded.
                        last_error = RedBlueRetryLimit("source inner iteration limit reached")
                        radius_factor = decrease_radius_factor(radius_factor)
                        retry_red = result
                    else:
                        mapped = CanalPathMesh(union.mesh, -1, mapped_ring, mapped_cap, stump.end_circle_offset)
                        tail_result = append_canal_path(union.mesh, mapped, curves[stump_size:],
                                                        angular_segments=angular_segments,
                                                        triangle_height_factor=triangle_height_factor,
                                                        render_vector=rvec, phi0=phi, return_boundary=True,
                                                        preserve_crease_edges=preserve_crease_edges,
                                                        arc_length_mode=arc_length_mode,
                                                        return_vertex_map=writer is not None, quad_sqrt_mode=quad_sqrt_mode,
                                                        render_trig_mode=render_trig_mode, source_frame=source_frame)
                        terminal, tail_map = tail_result if writer is not None else (tail_result, None)
                        last = curves[-1]
                        direction = _segment_tangent(last, 1.)
                        hemisphere = append_terminal_hemisphere(terminal, start=last.point(1.), direction=direction,
                                                               radius=last.radius1, angular_segments=angular_segments,
                                                               render_vector=rvec, phi0=phi,
                                                               return_vertex_map=writer is not None, quad_sqrt_mode=quad_sqrt_mode,
                                                               render_trig_mode=render_trig_mode, source_frame=source_frame)
                        if writer is None:
                            merged = hemisphere
                        else:
                            merged, hemisphere_map = hemisphere
                            def composed(i):
                                if i is None:
                                    return None
                                i = tail_map[i]
                                return None if i is None else hemisphere_map[i]
                            # Repairs append vertices, but the partial writer maps
                            # only the original active mesh from before retries.
                            merged_vertex_map = tuple(composed(i) for i in union.red_vertex_map[:len(result.vertices)])
            except RedBlueDisjoint:
                junction = junction_surfaces.get(nodes[0].id) if parent is None else None
                if junction is None:
                    raise
                # A terminal path wholly inside the original convex junction
                # contributes no exterior faces. Include the terminal cap in
                # the numerical containment check before retaining the junction.
                try:
                    contained = all(_point_inside_surface(point, junction, 1e-10) for point in full_path.vertices)
                except RedBlueNumericalEdgeCase:
                    contained = False
                if not contained:
                    raise
                merged = result
                if writer is not None:
                    merged_vertex_map = tuple(range(len(result.vertices)))
            except (RedBlueBoundaryInteraction, RedBlueTerminalRingConsumed):
                raise
            except (RedBlueMultipleLoops, RedBlueRetryLimit) as error:
                last_error = error
                radius_factor = decrease_radius_factor(radius_factor)
                retry_red = getattr(error, "_retry_red", result)
                if source_retry_semantics and getattr(error, "_source_last_iteration", False):
                    radius_factor = decrease_radius_factor(radius_factor)
                    retry_red = result
            except RedBlueNumericalEdgeCase as error:
                # A new angular offset can resolve numerical contacts. Other
                # RedBlueError instances indicate unexpected topology or logic
                # failures and must propagate instead of being silently retried.
                last_error = error
                retry_red = getattr(error, "_retry_red", result)
                if source_retry_semantics and getattr(error, "_source_last_iteration", False):
                    radius_factor = decrease_radius_factor(radius_factor)
                    retry_red = result
            if merged is not None:
                if writer is not None:
                    writer.replace_mesh(merged, merged_vertex_map)
                result = merged
                peak_active_faces = max(peak_active_faces, len(result.faces))
                break
            if radius_factor <= safe_lower_bound:
                raise RedBlueError(f"could not merge neurite path {path}: reached safe radius bound after {last_error}") from last_error
        if merged is None:
            raise RedBlueError(f"could not merge neurite path {path}: {last_error}") from last_error
    if writer is not None:
        writer.finalize(_partial_output)
        return {"output": str(_partial_output), "vertices": writer.vertex_count, "faces": writer.face_count,
                "flushes": flushes, "peak_active_faces": peak_active_faces}
    return result


def assemble_redblue_to_obj(tree: Morphology, output, *, flush_face_limit: int = 100000, **meshing_options):
    """Experimental source-style partial flushing to an OBJ during assembly.

    The meshing and box calculations are numerical, and the complete analytic
    consistency classifier has not yet been ported. The output is replaced
    only after all unions and boundary checks succeed. Returns output counts
    and the peak number of active faces; it does not return a full Mesh.
    """
    with ExitStack() as stack:
        return assemble_redblue_neuron(tree, _partial_output=output, _flush_face_limit=flush_face_limit,
                                       _partial_stack=stack, **meshing_options)


def assemble_redblue_checked(tree: Morphology, *, analysis_max_pairs: int = 100000,
                             analysis_max_rectangles: int = 10000, analysis_max_intervals: int = 100000,
                             analysis_univariate_tolerance: float = 1e-6,
                             analysis_bivariate_tolerance: float = 1e-4,
                             check_adjacent_end_start: bool = False, analysis_job_mode: str = "extended",
                             analysis_source_job_limit: int = 1000000, analysis_gate: str = "all",
                             **meshing_options):
    """Run partial checks before assembly and return ``(mesh, report)``.

    Raises PartialAnalysisError, with the full report attached, if any check
    reports a finding or remains unresolved. A successful report is still
    incomplete and does not certify the resulting mesh or network. The
    source gate uses only source-selected intersection jobs, like the C++
    clean decision, and also rejects unresolved jobs.
    """
    if analysis_gate not in ("all", "source"):
        raise ValueError("analysis gate must be all or source")
    if analysis_gate == "source" and analysis_job_mode != "source":
        raise ValueError("the source analysis gate requires source job mode")
    shared_options = ("strategy", "parametrization", "preconditioning", "pc_alpha", "pc_beta", "pc_gamma",
                      "scale_radius", "filter_angle", "radius_ratio", "preserve_surface_roots", "source_soma_centroid",
                      "precondition_norm_mode")
    analysis_options = {name: meshing_options[name] for name in shared_options if name in meshing_options}
    report = analyze_analytic_partial(tree, prepare=True, max_pairs=analysis_max_pairs,
                                      max_rectangles=analysis_max_rectangles, max_intervals=analysis_max_intervals,
                                      univariate_tolerance=analysis_univariate_tolerance,
                                      bivariate_tolerance=analysis_bivariate_tolerance,
                                      check_adjacent_end_start=check_adjacent_end_start,
                                      job_mode=analysis_job_mode, source_job_limit=analysis_source_job_limit,
                                      **analysis_options)
    blocked = (bool(report.findings or report.unresolved) if analysis_gate == "all"
               else report.source_jobs_clean is not True)
    if blocked:
        raise PartialAnalysisError(report, gate=analysis_gate)
    return assemble_redblue_neuron(tree, **meshing_options), report


def assemble_redblue_checked_to_obj(tree: Morphology, output, *, flush_face_limit: int = 100000, **options):
    """Run partial checks, then assemble to OBJ using experimental flushing."""
    with ExitStack() as stack:
        return assemble_redblue_checked(tree, _partial_output=output, _flush_face_limit=flush_face_limit,
                                        _partial_stack=stack, **options)
