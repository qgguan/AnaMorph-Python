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

"""Analytic soma and canal-segment tessellation building blocks.

Based on AnaMorph (https://github.com/NeuroBox3D/AnaMorph), especially
generateIcoSphere and CanalSurface::generateMesh. This module builds capped
individual segments and paths as well as terminal hemisphere meshes.
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from math import cos, dist, hypot, isfinite, pi, sin, sqrt, ulp
from sys import float_info

from .analytic_intersections import CanalSegment
from .mesh import Mesh, cross, dot
from .polynomial import derivative, evaluate
from .source_float import x87_sincos, x87_sqrt


@dataclass(frozen=True, slots=True)
class CanalPathMesh:
    mesh: Mesh
    start_cap: int
    end_ring: tuple[int, ...]
    end_cap: int
    end_circle_offset: bool


def _normalize(vector):
    length = hypot(*vector)
    if not isfinite(length) or length == 0:
        raise ValueError("centerline tangent and render vector do not define a stable frame")
    return tuple(x/length for x in vector)


def _render_frame(direction, render_vector):
    """Construct a frame using a scale-independent parallelism check."""
    tangent = _normalize(direction)
    render_axis = _normalize(render_vector)
    transverse = cross(render_axis, tangent)
    if hypot(*transverse) <= 1e-14:
        raise ValueError("centerline tangent and render vector do not define a stable frame")
    y = _normalize(transverse)
    return tangent, y, _normalize(cross(tangent, y))


def _source_render_frame(direction, render_vector, *, hemisphere=False, sqrt_mode="native"):
    """Use source-order vector normalization, with a scaled extreme-value fallback."""
    fallback = _render_frame(direction, render_vector)
    def normalize(vector):
        square = vector[0]*vector[0] + vector[1]*vector[1] + vector[2]*vector[2]
        if not isfinite(square) or square <= 0:
            raise ArithmeticError
        length = x87_sqrt(square) if sqrt_mode == "windows-x87" else sqrt(square)
        result = tuple(value/length for value in vector)
        if not all(isfinite(value) for value in result):
            raise ArithmeticError
        return result
    try:
        tangent = normalize(direction)
        # The terminal routine normalizes its direction, then normalizes a
        # copy again to form the ring frame. Its third axis is not normalized.
        axis = normalize(tangent) if hemisphere else tangent
        y = normalize(cross(render_vector, axis))
        z = cross(axis, y)
        if not hemisphere:
            z = normalize(z)
        if not all(isfinite(value) for value in z):
            raise ArithmeticError
        return tangent, y, z
    except (ArithmeticError, OverflowError):
        return fallback


def _join_tolerance(first, second, scale, relative):
    """Relative geometric tolerance plus rounding at the given coordinates."""
    return relative*scale + 16*max(ulp(value) for value in (*first, *second))


def _validate_generated_triangles(vertices, faces, description):
    """Reject faces lost to coordinate rounding before they reach stitching."""
    for face in faces:
        a, b, c = (vertices[index] for index in face)
        ab = tuple(y-x for x, y in zip(a, b))
        ac = tuple(y-x for x, y in zip(a, c))
        if not all(isfinite(value) for edge in (ab, ac) for value in edge):
            raise ValueError(f"{description} triangle is not representable at this coordinate precision")
        scale = max(abs(value) for edge in (ab, ac) for value in edge)
        if not scale:
            raise ValueError(f"{description} triangle is not representable at this coordinate precision")
        normal = cross(tuple(value/scale for value in ab), tuple(value/scale for value in ac))
        if not hypot(*normal):
            raise ValueError(f"{description} triangle is not representable at this coordinate precision")


def _triangle_edge_ratio(vertices, x, y, z, description, *, quad_sqrt_mode="native"):
    def source_length(first, second):
        delta = tuple(vertices[first][i]-vertices[second][i] for i in range(3))
        squared = delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2]
        # The source uses this unscaled expression; retain a scaled fallback
        # when its intermediate squares overflow or underflow.
        if not isfinite(squared) or squared < float_info.min:
            return hypot(*delta)
        return x87_sqrt(squared) if quad_sqrt_mode == "windows-x87" else sqrt(squared)
    lengths = (source_length(x, y), source_length(y, z), source_length(z, x))
    if not all(isfinite(length) and length > 0 for length in lengths):
        raise ValueError(f"{description} triangle is not representable at this coordinate precision")
    return max(lengths)/min(lengths)


def _validate_terminal_cap(mesh, boundary, angular_segments, context):
    """Verify the ordered ring is exactly the oriented cap fan to remove."""
    ring, cap = boundary.end_ring, boundary.end_cap
    if (len(ring) != angular_segments or len(set(ring)) != angular_segments or cap in ring or
            any(type(i) is not int or i < 0 or i >= len(mesh.vertices) for i in (*ring, cap))):
        raise ValueError(f"terminal ring or cap {context}")
    cap_faces = [face for face in mesh.faces if cap in face]
    if len(cap_faces) != angular_segments:
        raise ValueError(f"terminal cap {context}")
    expected = {(cap, ring[i], ring[(i+1)%angular_segments]) for i in range(angular_segments)}
    actual = set()
    for face in cap_faces:
        if len(face) != 3 or face.count(cap) != 1:
            raise ValueError(f"terminal cap {context}")
        index = face.index(cap)
        actual.add((cap, face[(index+1)%3], face[(index+2)%3]))
    if actual != expected:
        raise ValueError(f"terminal cap {context}")


def _segment_tangent(segment, t):
    """Use a segment's native derivative when it supplies one."""
    if hasattr(segment, "tangent"):
        return segment.tangent(t)
    return tuple(evaluate(derivative(p), t) for p in segment.power)


def _source_arc_length_function(segment: CanalSegment, step: float, *, windows_x87: bool = False):
    """Reproduce the source's sampled arc length, including its last-step rule.

    The original sums each full interval and then measures the final distance
    from the *penultimate* sampled point, so the last full interval is counted
    twice. If no full interval is visited its `last` vector is uninitialized;
    we use the starting point in that undefined case.
    """
    if not isfinite(step) or not 0 < step <= 1 or type(windows_x87) is not bool:
        raise ValueError("invalid source arc-length sampling settings")
    def distance(a, b):
        if not windows_x87:
            return dist(a, b)
        delta = tuple(x-y for x, y in zip(a, b))
        square = delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2]
        if not isfinite(square) or (square == 0 and any(delta)):
            raise ValueError("source arc-length distance is not representable")
        return x87_sqrt(square)
    times, points, cumulative = [0.], [segment.point(0.)], [0.]
    t = step
    while t < 1.:
        point = segment.point(t)
        times.append(t)
        points.append(point)
        cumulative.append(cumulative[-1]+distance(point, points[-2]))
        t += step
    def arc(end):
        if end == 0:
            return 0.
        last_index = bisect_left(times, end)-1
        if last_index <= 0:
            return distance(segment.point(end), points[0])
        return cumulative[last_index]+distance(segment.point(end), points[last_index-1])
    return arc


def ico_sphere(center, radius: float, refinements: int = 3) -> Mesh:
    """Tessellate the soma from the same oriented icosahedron as the source."""
    if (len(center) != 3 or not all(isfinite(x) for x in center) or
            not isfinite(radius) or radius <= 0 or not 0 <= refinements <= 8):
        raise ValueError("invalid soma sphere or refinement count")
    phi = (1+sqrt(5))/2
    norm = sqrt(2+phi)
    vertices = [tuple(x/norm for x in p) for p in (
        (0,-1,-phi), (0,-1,phi), (0,1,-phi), (0,1,phi),
        (-phi,0,-1), (-phi,0,1), (phi,0,-1), (phi,0,1),
        (-1,-phi,0), (-1,phi,0), (1,-phi,0), (1,phi,0))]
    faces = [(0,8,4),(0,2,6),(1,7,3),(1,5,8),(2,0,4),(2,11,6),
             (3,7,11),(3,5,1),(4,8,5),(6,11,7),(9,5,3),(9,3,11),
             (9,11,2),(9,2,4),(9,4,5),(10,7,1),(10,1,8),(10,8,0),
             (10,0,6),(10,6,7)]
    for _ in range(refinements):
        cache, new_faces = {}, []
        def midpoint(a, b):
            edge = tuple(sorted((a, b)))
            if edge not in cache:
                cache[edge] = len(vertices)
                vertices.append(_normalize(tuple((x+y)/2 for x, y in zip(vertices[a], vertices[b]))))
            return cache[edge]
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces.extend(((a, ab, ca), (ca, ab, bc), (bc, ab, b), (bc, c, ca)))
        faces = new_faces
    vertices = [tuple(center[i]+radius*p[i] for i in range(3)) for p in vertices]
    _validate_generated_triangles(vertices, faces, "soma")
    return Mesh(tuple(vertices), tuple(faces))


def canal_segment_mesh(segment: CanalSegment, *, angular_segments: int = 6,
                       triangle_height_factor: float = 1.0, render_vector=(0., 0., 1.),
                       phi0: float = 0., preserve_crease_edges: bool = False,
                       arc_samples: int = 1000, max_rings: int = 100000,
                       start_circle_offset: bool = False, arc_length_mode: str = "source",
                       quad_sqrt_mode: str = "native", render_trig_mode: str = "native",
                       source_frame: bool = False) -> Mesh:
    """Tessellate one arc-length-radius canal segment with capped ends.

    Ring spacing, frame, alternating angular offset and quad diagonal choice
    follow the source. `arc_length_mode="source"` follows its last-interval
    rule; `"source-x87"` also follows the local Windows distance arithmetic;
    `"accurate"` uses a conventional cumulative polyline approximation.
    """
    if (angular_segments < 3 or not isfinite(triangle_height_factor) or triangle_height_factor <= 0 or
            len(render_vector) != 3 or not all(isfinite(x) for x in render_vector) or
            not isfinite(phi0) or arc_samples < 2 or max_rings < 2 or
            arc_length_mode not in ("source", "source-x87", "accurate") or quad_sqrt_mode not in ("native", "windows-x87") or
            render_trig_mode not in ("native", "windows-x87") or type(source_frame) is not bool):
        raise ValueError("invalid canal mesh parameters")
    sine = (lambda angle: x87_sincos(angle)[0]) if render_trig_mode == "windows-x87" else sin
    cosine = (lambda angle: x87_sincos(angle)[1]) if render_trig_mode == "windows-x87" else cos
    if arc_length_mode in ("source", "source-x87"):
        source_arc = _source_arc_length_function(segment, 1/arc_samples, windows_x87=arc_length_mode == "source-x87")
        length = source_arc(1.)
    else:
        samples = [segment.point(i/arc_samples) for i in range(arc_samples+1)]
        lengths = [0.]
        for a, b in zip(samples, samples[1:]):
            lengths.append(lengths[-1]+dist(a, b))
        length = lengths[-1]
    if not isfinite(length) or length <= 0:
        raise ValueError("canal segment has zero arc length")
    if arc_length_mode in ("source", "source-x87"):
        radius_arc = _source_arc_length_function(segment, .01, windows_x87=arc_length_mode == "source-x87")
        radius_length = radius_arc(1.)
        if radius_length <= 0:
            raise ValueError("canal segment has zero sampled radius-interpolation arc length")
        def radius(t):
            ratio = min(1., max(0., radius_arc(t)/radius_length))
            return segment.radius0+(segment.radius1-segment.radius0)*ratio
    else:
        def radius(t):
            sample = min(arc_samples-1, int(t*arc_samples))
            frac = t*arc_samples-sample
            arclength = lengths[sample] + frac*(lengths[sample+1]-lengths[sample])
            return segment.radius0+(segment.radius1-segment.radius0)*arclength/length
    dphi = 2*pi/angular_segments
    times = [0.]
    while times[-1] < 1:
        square = 2*(1-cosine(dphi))
        chord = radius(times[-1])*(x87_sqrt(square) if arc_length_mode == "source-x87" else sqrt(square))
        if arc_length_mode == "source-x87" and not preserve_crease_edges:
            height = triangle_height_factor*x87_sqrt(3.)*chord/2
        else:
            height = triangle_height_factor*chord*(1 if preserve_crease_edges else sqrt(3)/2)
        next_t = times[-1]+height/length
        if next_t <= times[-1]:
            raise ValueError("canal ring step is too small to represent")
        times.append(next_t)
        if len(times) > max_rings+1:
            raise ValueError("canal segment exceeds ring count limit")
    slack = times[-1]-1
    times.pop()
    count = len(times)
    times = [t-i*slack/count for i, t in enumerate(times)] + [1.]
    vertices, rings = [], []
    for i, t in enumerate(times):
        direction = _segment_tangent(segment, t)
        tangent, y, z = (_source_render_frame(direction, render_vector, sqrt_mode=quad_sqrt_mode)
                         if source_frame else _render_frame(direction, render_vector))
        point, r = segment.point(t), radius(t)
        offset_bit = start_circle_offset != bool(i % 2)
        offset = phi0+dphi/2 if not preserve_crease_edges and offset_bit else phi0
        ring = []
        for j in range(angular_segments):
            angle = (j*(2*pi))/angular_segments+offset
            ring.append(len(vertices))
            vertices.append(tuple((point[k]+y[k]*(r*cosine(angle)))+z[k]*(r*sine(angle)) for k in range(3)))
        rings.append(ring)
    faces = []
    for first, second in zip(rings, rings[1:]):
        for j in range(angular_segments):
            a, b, c, d = first[j], first[(j+1)%angular_segments], second[(j+1)%angular_segments], second[j]
            def quality(x, y, z):
                return _triangle_edge_ratio(vertices, x, y, z, "canal", quad_sqrt_mode=quad_sqrt_mode)
            if quality(a,b,c)+quality(a,c,d) < quality(a,b,d)+quality(b,c,d):
                faces.extend(((a,b,c), (a,c,d)))
            else:
                faces.extend(((a,b,d), (b,c,d)))
    start, end = len(vertices), len(vertices)+1
    vertices.extend((segment.point(0.), segment.point(1.)))
    for j in range(angular_segments):
        faces.append((start, rings[0][(j+1)%angular_segments], rings[0][j]))
        faces.append((end, rings[-1][j], rings[-1][(j+1)%angular_segments]))
    _validate_generated_triangles(vertices, faces, "canal")
    return Mesh(tuple(vertices), tuple(faces))


def canal_path_mesh(segments, *, angular_segments: int = 6, triangle_height_factor: float = 1.0,
                    render_vector=(0., 0., 1.), phi0: float = 0.,
                    preserve_crease_edges: bool = False, arc_samples: int = 1000,
                    start_circle_offset: bool = False, return_boundary: bool = False,
                    arc_length_mode: str = "source", quad_sqrt_mode: str = "native",
                    render_trig_mode: str = "native", source_frame: bool = False) -> Mesh | CanalPathMesh:
    """Mesh a C1-continuous sequence, sharing its intermediate cross sections.

    The result has caps only at the two path ends. Segment meshes use a common
    render vector and carry the alternating-circle parity across boundaries.
    """
    segments = tuple(segments)
    if not segments:
        raise ValueError("a canal path needs at least one segment")
    vertices, faces = [], []
    previous_ring = None
    circle_offset = start_circle_offset
    previous_segment = None
    first_cap = None
    for part_index, segment in enumerate(segments):
        if previous_segment is not None:
            previous_end, next_start = previous_segment.point(1.), segment.point(0.)
            radius_scale = max(segment.radius0, previous_segment.radius1)
            if dist(previous_end, next_start) > _join_tolerance(previous_end, next_start, radius_scale, 1e-9):
                raise ValueError("consecutive canal segments must meet")
            if abs(previous_segment.radius1-segment.radius0) > (1e-9*radius_scale +
                    16*max(ulp(segment.radius0), ulp(previous_segment.radius1))):
                raise ValueError("consecutive canal segments must have the same shared radius")
            a = _normalize(_segment_tangent(previous_segment, 1.))
            b = _normalize(_segment_tangent(segment, 0.))
            if dot(a, b) < 1-1e-8:
                raise ValueError("consecutive canal segments must have matching tangent directions")
        part = canal_segment_mesh(segment, angular_segments=angular_segments,
                                  triangle_height_factor=triangle_height_factor, render_vector=render_vector,
                                  phi0=phi0, preserve_crease_edges=preserve_crease_edges,
                                  arc_samples=arc_samples, start_circle_offset=circle_offset,
                                  arc_length_mode=arc_length_mode, quad_sqrt_mode=quad_sqrt_mode,
                                  render_trig_mode=render_trig_mode, source_frame=source_frame)
        ring_count = (len(part.vertices)-2)//angular_segments
        start_cap, end_cap = len(part.vertices)-2, len(part.vertices)-1
        mapping = {}
        for i, point in enumerate(part.vertices):
            if i == start_cap and part_index != 0:
                continue
            if i == end_cap and part_index != len(segments)-1:
                continue
            if previous_ring is not None and i < angular_segments:
                old_point = vertices[previous_ring[i]]
                if dist(old_point, point) > _join_tolerance(old_point, point, segment.radius0, 1e-8):
                    raise ValueError("adjacent canal render frames do not match")
                mapping[i] = previous_ring[i]
            else:
                mapping[i] = len(vertices)
                vertices.append(point)
        for face in part.faces:
            if (start_cap in face and part_index != 0) or (end_cap in face and part_index != len(segments)-1):
                continue
            faces.append(tuple(mapping[i] for i in face))
        if part_index == 0:
            first_cap = mapping[start_cap]
        previous_ring = [mapping[(ring_count-1)*angular_segments+j] for j in range(angular_segments)]
        if not preserve_crease_edges:
            circle_offset = circle_offset != bool((ring_count-1) % 2)
        previous_segment = segment
    mesh = Mesh(tuple(vertices), tuple(faces))
    if return_boundary:
        return CanalPathMesh(mesh, first_cap, tuple(previous_ring), mapping[end_cap], circle_offset)
    return mesh


def append_canal_path(mesh: Mesh, boundary: CanalPathMesh, segments, *, angular_segments: int = 6,
                      triangle_height_factor: float = 1.0, render_vector=(0., 0., 1.), phi0: float = 0.,
                      preserve_crease_edges: bool = False, arc_samples: int = 1000,
                      return_boundary: bool = False, arc_length_mode: str = "source",
                      return_vertex_map: bool = False, quad_sqrt_mode: str = "native",
                      render_trig_mode: str = "native", source_frame: bool = False):
    """Reopen a surviving end cap and append a continuous tail mesh.

    `boundary.end_ring` and `boundary.end_cap` refer to `mesh`, not necessarily
    to boundary.mesh; after red-blue union the caller supplies mapped indices.
    """
    if (quad_sqrt_mode not in ("native", "windows-x87") or render_trig_mode not in ("native", "windows-x87") or
            type(source_frame) is not bool):
        raise ValueError("invalid rendering arithmetic mode")
    _validate_terminal_cap(mesh, boundary, angular_segments, "was changed by red-blue union")
    segments = tuple(segments)
    if not segments:
        result = boundary if return_boundary else mesh
        return (result, tuple(range(len(mesh.vertices)))) if return_vertex_map else result
    tail_info = canal_path_mesh(segments, angular_segments=angular_segments,
                                triangle_height_factor=triangle_height_factor, render_vector=render_vector,
                                phi0=phi0, preserve_crease_edges=preserve_crease_edges,
                                arc_samples=arc_samples, start_circle_offset=boundary.end_circle_offset,
                                return_boundary=True, arc_length_mode=arc_length_mode, quad_sqrt_mode=quad_sqrt_mode,
                                render_trig_mode=render_trig_mode, source_frame=source_frame)
    tail = tail_info.mesh
    vertices = list(mesh.vertices)
    mapping = {}
    tail_start_cap = tail_info.start_cap
    for i, point in enumerate(tail.vertices):
        if i == tail_start_cap:
            continue
        if i < angular_segments:
            existing = boundary.end_ring[i]
            old_point = mesh.vertices[existing]
            if dist(old_point, point) > _join_tolerance(old_point, point, segments[0].radius0, 1e-8):
                raise ValueError("tail and surviving terminal ring do not match")
            mapping[i] = existing
        else:
            mapping[i] = len(vertices)
            vertices.append(point)
    faces = [face for face in mesh.faces if boundary.end_cap not in face]
    faces.extend(tuple(mapping[i] for i in face) for face in tail.faces if tail_start_cap not in face)
    used = sorted({i for face in faces for i in face})
    reindex = {old: new for new, old in enumerate(used)}
    result = Mesh(tuple(vertices[i] for i in used), tuple(tuple(reindex[i] for i in face) for face in faces))
    if return_boundary:
        result = CanalPathMesh(result, -1, tuple(reindex[mapping[i]] for i in tail_info.end_ring),
                               reindex[mapping[tail_info.end_cap]], tail_info.end_circle_offset)
    return (result, tuple(reindex.get(i) for i in range(len(mesh.vertices)))) if return_vertex_map else result


def append_terminal_hemisphere(boundary: CanalPathMesh, *, start, direction, radius: float,
                               angular_segments: int, render_vector=(0., 0., 1.), phi0: float = 0.,
                               return_vertex_map: bool = False, quad_sqrt_mode: str = "native",
                               render_trig_mode: str = "native", source_frame: bool = False):
    """Replace a path's end cap with the source's hemispherical terminal mesh."""
    mesh = boundary.mesh
    if (angular_segments < 3 or not isfinite(radius) or radius <= 0 or
            len(start) != 3 or not all(isfinite(value) for value in start) or
            len(direction) != 3 or not all(isfinite(value) for value in direction) or not isfinite(phi0) or
            quad_sqrt_mode not in ("native", "windows-x87") or render_trig_mode not in ("native", "windows-x87") or
            type(source_frame) is not bool):
        raise ValueError("invalid hemisphere radius or terminal ring")
    sine = (lambda angle: x87_sincos(angle)[0]) if render_trig_mode == "windows-x87" else sin
    cosine = (lambda angle: x87_sincos(angle)[1]) if render_trig_mode == "windows-x87" else cos
    _validate_terminal_cap(mesh, boundary, angular_segments, "was changed before hemisphere construction")
    old_center = mesh.vertices[boundary.end_cap]
    if dist(old_center, start) > _join_tolerance(old_center, start, radius, 1e-8):
        raise ValueError("hemisphere start does not match the terminal cap")
    tangent, y, z = (_source_render_frame(direction, render_vector, hemisphere=True, sqrt_mode=quad_sqrt_mode)
                     if source_frame else _render_frame(direction, render_vector))
    for index in boundary.end_ring:
        point = mesh.vertices[index]
        displacement = tuple(point[i]-start[i] for i in range(3))
        tolerance = _join_tolerance(point, start, radius, 1e-8)
        if (not all(isfinite(value) for value in displacement) or
                abs(hypot(*displacement)-radius) > tolerance or abs(dot(displacement, tangent)) > tolerance):
            raise ValueError("hemisphere radius or direction does not match the terminal ring")
    if angular_segments//2 > 2:
        # Canal rings alternate between phi0 and a half-step offset. The new
        # hemisphere rings use phi0; a different frame or phase can twist the
        # first strip into a self-intersecting surface.
        dphi = 2*pi/angular_segments
        def matches_offset(offset):
            ring_phase = phi0+offset
            for j, index in enumerate(boundary.end_ring):
                angle = j*dphi+ring_phase
                expected = tuple(start[k]+radius*(y[k]*cosine(angle)+z[k]*sine(angle)) for k in range(3))
                point = mesh.vertices[index]
                if dist(point, expected) > _join_tolerance(point, expected, radius, 1e-8):
                    return False
            return True
        if not (matches_offset(0.) or matches_offset(dphi/2)):
            raise ValueError("hemisphere phase or frame does not match the terminal ring")
    vertices = list(mesh.vertices)
    faces = [face for face in mesh.faces if boundary.end_cap not in face]
    old_face_count = len(faces)
    last_ring = list(boundary.end_ring)
    n_meridians = angular_segments//2
    for i in range(1, n_meridians-1):
        # Preserve the source's operation order: equivalent sin/cos formulas
        # can differ by an ulp, changing a nearly tied quad diagonal.
        alpha = pi/2-i*((pi/2)/n_meridians)
        t = cosine(alpha)
        ring_radius = radius*sqrt(1.-t*t)
        center = tuple(start[j]+tangent[j]*(t*radius) for j in range(3))
        ring = []
        for j in range(angular_segments):
            phi = 2*pi*j/angular_segments+phi0
            ring.append(len(vertices))
            vertices.append(tuple((center[k]+y[k]*(ring_radius*cosine(phi)))+z[k]*(ring_radius*sine(phi))
                                  for k in range(3)))
        for j in range(angular_segments):
            a, b, c, d = last_ring[j], last_ring[(j+1)%angular_segments], ring[(j+1)%angular_segments], ring[j]
            def quality(x, y0, z0):
                return _triangle_edge_ratio(vertices, x, y0, z0, "hemisphere", quad_sqrt_mode=quad_sqrt_mode)
            if quality(a,b,c)+quality(a,c,d) < quality(a,b,d)+quality(b,c,d):
                faces.extend(((a,b,c), (a,c,d)))
            else:
                faces.extend(((a,b,d), (b,c,d)))
        last_ring = ring
    tip = len(vertices)
    vertices.append(tuple(start[i]+radius*tangent[i] for i in range(3)))
    for j in range(angular_segments):
        faces.append((tip, last_ring[j], last_ring[(j+1)%angular_segments]))
    _validate_generated_triangles(vertices, faces[old_face_count:], "hemisphere")
    used = sorted({i for face in faces for i in face})
    reindex = {old: new for new, old in enumerate(used)}
    result = Mesh(tuple(vertices[i] for i in used), tuple(tuple(reindex[i] for i in face) for face in faces))
    return (result, tuple(reindex.get(i) for i in range(len(mesh.vertices)))) if return_vertex_map else result
