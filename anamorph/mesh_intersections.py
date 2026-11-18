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

"""Triangle-level self-intersection diagnostics for generated surface meshes.

Uses a sweep-and-prune broad phase and floating-point triangle tests. This is
not a formally certified exact-predicate intersection checker.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import hypot, isfinite, sqrt, ulp

from .mesh import Mesh, cross, dot, sub


@dataclass(frozen=True, slots=True)
class MeshIntersection:
    first_face: int
    second_face: int
    coplanar: bool


def _orient2(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def _unit_normal(triangle):
    edges = (sub(triangle[1], triangle[0]), sub(triangle[2], triangle[0]))
    if not all(isfinite(value) for edge in edges for value in edge):
        raise ValueError("mesh triangle coordinate differences exceed float range")
    scale = max(abs(value) for edge in edges for value in edge)
    if not scale:
        raise ValueError("mesh contains a degenerate triangle")
    normal = cross(*(tuple(value/scale for value in edge) for edge in edges))
    length = hypot(*normal)
    if not length or not isfinite(length):
        raise ValueError("mesh contains a degenerate or numerically invalid triangle")
    return tuple(value/length for value in normal), scale


def _coplanar_overlap(first, second, normal, tolerance):
    drop = max(range(3), key=lambda i: abs(normal[i]))
    axes = [i for i in range(3) if i != drop]
    origin = first[0]
    a = [(point[axes[0]]-origin[axes[0]], point[axes[1]]-origin[axes[1]]) for point in first]
    b = [(point[axes[0]]-origin[axes[0]], point[axes[1]]-origin[axes[1]]) for point in second]
    scale = max(abs(coord) for point in a+b for coord in point)
    if not isfinite(scale) or not scale:
        raise ValueError("coplanar triangle coordinates cannot be represented")
    a = [(x/scale, y/scale) for x, y in a]
    b = [(x/scale, y/scale) for x, y in b]
    coordinate_eps = orientation_eps = tolerance
    def on_segment(p, q, r):
        return (min(p[0], q[0])-coordinate_eps <= r[0] <= max(p[0], q[0])+coordinate_eps and
                min(p[1], q[1])-coordinate_eps <= r[1] <= max(p[1], q[1])+coordinate_eps)
    def segments_intersect(p, q, r, s):
        o1, o2, o3, o4 = _orient2(p,q,r), _orient2(p,q,s), _orient2(r,s,p), _orient2(r,s,q)
        if ((o1 > orientation_eps and o2 < -orientation_eps) or (o1 < -orientation_eps and o2 > orientation_eps)) and \
           ((o3 > orientation_eps and o4 < -orientation_eps) or (o3 < -orientation_eps and o4 > orientation_eps)):
            return True
        return ((abs(o1) <= orientation_eps and on_segment(p,q,r)) or
                (abs(o2) <= orientation_eps and on_segment(p,q,s)) or
                (abs(o3) <= orientation_eps and on_segment(r,s,p)) or
                (abs(o4) <= orientation_eps and on_segment(r,s,q)))
    def in_triangle(p, triangle):
        orientation = [_orient2(triangle[i], triangle[(i+1)%3], p) for i in range(3)]
        return all(x >= -orientation_eps for x in orientation) or all(x <= orientation_eps for x in orientation)
    return (in_triangle(a[0], b) or in_triangle(b[0], a) or
            any(segments_intersect(a[i], a[(i+1)%3], b[j], b[(j+1)%3]) for i in range(3) for j in range(3)))


def _exact_segment_intersects_triangle(start, end, triangle):
    """Resolve a near-parallel segment against the exact input float values."""
    a, b, c = (tuple(Fraction(value) for value in point) for point in triangle)
    start, end = (tuple(Fraction(value) for value in point) for point in (start, end))
    normal = cross(sub(b, a), sub(c, a))
    if not any(normal):
        raise ValueError("mesh contains a degenerate triangle")
    first, second = dot(sub(start, a), normal), dot(sub(end, a), normal)
    if (first > 0 and second > 0) or (first < 0 and second < 0):
        return False
    if first == second == 0:
        drop = max(range(3), key=lambda i: abs(normal[i]))
        axes = tuple(i for i in range(3) if i != drop)
        def project(point):
            return tuple(point[i] for i in axes)
        tri = tuple(project(point) for point in (a, b, c))
        u, v = project(start), project(end)
        def inside(point):
            signs = [_orient2(tri[i], tri[(i+1)%3], point) for i in range(3)]
            return all(value >= 0 for value in signs) or all(value <= 0 for value in signs)
        def on_segment(p, q, r):
            return min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and min(p[1], q[1]) <= r[1] <= max(p[1], q[1])
        def intersects(p, q, r, s):
            o1, o2, o3, o4 = _orient2(p, q, r), _orient2(p, q, s), _orient2(r, s, p), _orient2(r, s, q)
            opposite1 = (o1 < 0 < o2) or (o2 < 0 < o1)
            opposite2 = (o3 < 0 < o4) or (o4 < 0 < o3)
            return ((opposite1 and opposite2) or (o1 == 0 and on_segment(p, q, r)) or
                    (o2 == 0 and on_segment(p, q, s)) or (o3 == 0 and on_segment(r, s, p)) or
                    (o4 == 0 and on_segment(r, s, q)))
        return inside(u) or inside(v) or any(intersects(u, v, tri[i], tri[(i+1)%3]) for i in range(3))
    parameter = first/(first-second)
    point = tuple(start[i]+parameter*(end[i]-start[i]) for i in range(3))
    return all(dot(cross(sub(v, u), sub(point, u)), normal) >= 0 for u, v in ((a, b), (b, c), (c, a)))


def _segment_intersects_triangle(start, end, triangle, tolerance):
    a, b, c = triangle
    vectors = (sub(end, start), sub(b, a), sub(c, a), sub(start, a))
    if not all(isfinite(value) for vector in vectors for value in vector):
        raise ValueError("segment-triangle coordinate differences exceed float range")
    coordinate_scale = max(abs(value) for vector in vectors for value in vector)
    if not coordinate_scale:
        raise ValueError("degenerate segment or triangle")
    direction, e1, e2, s = (tuple(value/coordinate_scale for value in vector) for vector in vectors)
    if any(original and not value for original_vector, scaled_vector in zip(vectors, (direction, e1, e2, s))
           for original, value in zip(original_vector, scaled_vector)):
        raise ValueError("segment-triangle coordinate scale ratio exceeds float range")
    normal = cross(e1, e2)
    h = cross(direction, e2)
    determinant = dot(e1, h)
    scale = hypot(*direction)*hypot(*normal)
    if scale == 0 or tolerance*scale == 0:
        raise ValueError("segment-triangle orientation is numerically unrepresentable")
    if abs(determinant) <= tolerance*scale:
        return _exact_segment_intersects_triangle(start, end, triangle)
    inverse = 1/determinant
    if not isfinite(inverse):
        raise ValueError("segment-triangle determinant is numerically unrepresentable")
    u = inverse*dot(s, h)
    if u < -tolerance or u > 1+tolerance:
        return False
    q = cross(s, e1)
    v = inverse*dot(direction, q)
    if v < -tolerance or u+v > 1+tolerance:
        return False
    t = inverse*dot(e2, q)
    return -tolerance <= t <= 1+tolerance


def _triangles_intersect(first, second, tolerance):
    n1, scale1 = _unit_normal(first)
    n2, scale2 = _unit_normal(second)
    scale = max(scale1, scale2)
    parallel = hypot(*cross(n1, n2)) <= tolerance
    # A near-parallel triangle crossing the plane can extend farther than one
    # coordinate-wise edge extent: its Euclidean diameter may be larger.
    # Keep those pairs for the exact plane-sign and projection checks.
    near_plane = parallel and all(abs(dot(tuple(value/scale for value in sub(point, first[0])), n1)) <= 4*tolerance
                                  for point in second)
    if near_plane:
        # A near-zero floating-point plane distance is not evidence of actual
        # coplanarity. Resolve the signs exactly for the input binary floats;
        # a triangle strictly on one side cannot intersect the other.
        normal = _exact_normal(first)
        origin = tuple(Fraction(x) for x in first[0])
        distances = [dot(tuple(Fraction(point[i])-origin[i] for i in range(3)), normal) for point in second]
        if all(value > 0 for value in distances) or all(value < 0 for value in distances):
            return False, False
        coplanar = all(value == 0 for value in distances)
        # For a genuinely noncoplanar near-parallel pair, overlapping 2-D
        # projections remain a conservative numerical candidate.
        return _coplanar_overlap(first, second, n1, tolerance), coplanar
    return (any(_segment_intersects_triangle(first[i], first[(i+1)%3], second, tolerance) for i in range(3)) or
            any(_segment_intersects_triangle(second[i], second[(i+1)%3], first, tolerance) for i in range(3))), False


def _exact_normal(triangle):
    """Compute a triangle normal exactly for the given binary floats."""
    points = [tuple(Fraction(x) for x in point) for point in triangle]
    return cross(sub(points[1], points[0]), sub(points[2], points[0]))


def _near_parallel_axis(first, second):
    """Return an exact-normal crossing direction, or None if parallel."""
    direction = cross(_exact_normal(first), _exact_normal(second))
    maximum = max(abs(x) for x in direction)
    if maximum == 0:
        return None
    scaled = tuple(float(x/maximum) for x in direction)
    length = sqrt(dot(scaled, scaled))
    return tuple(x/length for x in scaled)


def _shared_vertex_intersection(first, second, shared, tolerance):
    """Test a pair sharing one vertex for overlap beyond that vertex."""
    n1, _ = _unit_normal(first)
    n2, _ = _unit_normal(second)
    l1 = l2 = 1.
    directions = cross(n1, n2)
    direction_length = hypot(*directions)
    # The overlap threshold must scale with these faces. A unit-size floor
    # silently hides genuine intersections on meshes smaller than tolerance.
    scale = max(hypot(*sub(p, shared)) for p in first+second)
    if not scale or not isfinite(scale):
        raise ValueError("shared-vertex triangle scale cannot be represented")
    eps = tolerance*scale
    if direction_length <= tolerance*l1*l2:
        axis = _near_parallel_axis(first, second)
    else:
        axis = tuple(x/direction_length for x in directions)
    if axis is None:
        # Clip one projected triangle against the other. Any point of the
        # resulting polygon away from the common vertex indicates an overlap
        # beyond the legitimate shared point, including a coincident segment.
        drop = max(range(3), key=lambda i: abs(n1[i]))
        axes = [i for i in range(3) if i != drop]
        project = lambda p: ((p[axes[0]]-shared[axes[0]])/scale, (p[axes[1]]-shared[axes[1]])/scale)
        polygon, clip = [project(p) for p in first], [project(p) for p in second]
        orientation = _orient2(*clip)
        sign = 1 if orientation > 0 else -1
        for i in range(3):
            a, b = clip[i], clip[(i+1)%3]
            if not polygon:
                break
            result = []
            prev = polygon[-1]
            prev_value = sign*_orient2(a, b, prev)
            for cur in polygon:
                value = sign*_orient2(a, b, cur)
                if (prev_value >= 0) != (value >= 0):
                    fraction = prev_value/(prev_value-value)
                    result.append((prev[0]+fraction*(cur[0]-prev[0]),
                                   prev[1]+fraction*(cur[1]-prev[1])))
                if value >= 0:
                    result.append(cur)
                prev, prev_value = cur, value
            polygon = result
        return any(x*x+y*y > (16*tolerance)**2 for x, y in polygon), True
    def slice_interval(triangle, other_triangle, other_normal, other_length):
        other = [p for p in triangle if p != shared]
        distances = [dot(sub(p, shared), other_normal)/other_length for p in other]
        if any(abs(value) <= eps for value in distances):
            # Do not treat a merely close vertex as lying in the other plane:
            # that can create a false overlap of arbitrary length. Resolve
            # near-zero signs using exact rational arithmetic on input floats.
            normal = _exact_normal(other_triangle)
            origin = tuple(Fraction(x) for x in shared)
            distances = [dot(tuple(Fraction(p[j])-origin[j] for j in range(3)), normal) for p in other]
        positions = [0.]
        for p, distance in zip(other, distances):
            if distance == 0:
                positions.append(dot(sub(p, shared), axis))
        if (distances[0] < 0 < distances[1]) or (distances[1] < 0 < distances[0]):
            fraction = float(distances[0]/(distances[0]-distances[1]))
            point = tuple(other[0][j]+fraction*(other[1][j]-other[0][j]) for j in range(3))
            positions.append(dot(sub(point, shared), axis))
        return min(positions), max(positions)
    a0, a1 = slice_interval(first, second, n2, l2)
    b0, b1 = slice_interval(second, first, n1, l1)
    return min(a1, b1)-max(a0, b0) > 16*eps, False


def _shared_edge_intersection(first, second, common, tolerance):
    """Adjacent faces sharing an edge overlap beyond it only when coplanar."""
    n1, _ = _unit_normal(first)
    n2, _ = _unit_normal(second)
    direction = cross(n1, n2)
    if hypot(*direction) > tolerance:
        return False, False
    if _near_parallel_axis(first, second) is not None:
        return False, False
    drop = max(range(3), key=lambda i: abs(n1[i]))
    axes = [i for i in range(3) if i != drop]
    origin = common[0]
    scale = max(abs(p[axis]-origin[axis]) for p in first+second for axis in axes)
    if not scale or not isfinite(scale):
        raise ValueError("shared-edge triangle coordinates cannot be represented")
    project = lambda p: ((p[axes[0]]-origin[axes[0]])/scale, (p[axes[1]]-origin[axes[1]])/scale)
    edge = [project(p) for p in common]
    a = next(p for p in first if p not in common)
    b = next(p for p in second if p not in common)
    # Same-side third vertices mean the planar interiors overlap near the edge.
    return _orient2(*edge, project(a))*_orient2(*edge, project(b)) > 0, True


def find_self_intersections(mesh: Mesh, *, tolerance: float = 1e-9, max_hits: int = 1000,
                            include_adjacent: bool = False):
    """Find intersections between triangles, optionally including adjacent faces.

    By default, triangles sharing any vertex are excluded. With
    ``include_adjacent``, intersections extending beyond the shared vertex or
    edge are also reported. The result remains a floating-point diagnostic, not
    a proof of global non-self-intersection. Tolerance is relative to local
    triangle geometry, with additional coordinate-rounding allowances in the
    broad phase. `max_hits`
    prevents unbounded output; hitting the cap is explicitly reported.
    Returns (hits, truncated).
    """
    if not isfinite(tolerance) or tolerance <= 0 or max_hits < 1:
        raise ValueError("tolerance and max_hits must be positive")
    records = []
    for fid, face in enumerate(mesh.faces):
        if len(face) != 3 or len(set(face)) != 3 or any(i < 0 or i >= len(mesh.vertices) for i in face):
            raise ValueError("mesh contains an invalid triangular face")
        points = tuple(mesh.vertices[i] for i in face)
        if any(len(point) != 3 or not all(isfinite(x) for x in point) for point in points):
            raise ValueError("mesh contains a non-finite or invalid vertex")
        _unit_normal(points)
        bounds = tuple((min(p[i] for p in points), max(p[i] for p in points)) for i in range(3))
        extent = max(hi-lo for lo, hi in bounds)
        rounding = tuple(max(ulp(point[axis]) for point in points) for axis in range(3))
        records.append((bounds[0][0], fid, face, points, bounds, extent, rounding))
    records.sort(key=lambda item: item[0])
    max_extent = max((record[5] for record in records), default=0.)
    max_x_rounding = max((record[6][0] for record in records), default=0.)
    sweep_padding = 8*tolerance*max_extent+8*max_x_rounding
    active, hits = [], []
    for xmin, fid, face, points, bounds, extent, rounding in records:
        # A global upper bound is needed when expiring sweep entries: a later
        # face can be larger than the current one. Pair tests use tighter
        # local geometry and coordinate-rounding allowances.
        active = [record for record in active if record[4][0][1]+sweep_padding >= xmin]
        for _, other_id, other_face, other_points, other_bounds, other_extent, other_rounding in active:
            common_ids = set(face) & set(other_face)
            if common_ids and not include_adjacent:
                continue
            padding = tuple(8*tolerance*max(extent, other_extent)+8*max(rounding[i], other_rounding[i])
                            for i in range(3))
            if any(bounds[i][0] > other_bounds[i][1]+padding[i] or
                   other_bounds[i][0] > bounds[i][1]+padding[i] for i in range(3)):
                continue
            if len(common_ids) == 3:
                intersects, coplanar = True, True
            elif len(common_ids) == 2:
                common = tuple(mesh.vertices[i] for i in common_ids)
                intersects, coplanar = _shared_edge_intersection(points, other_points, common, tolerance)
            elif len(common_ids) == 1:
                shared = mesh.vertices[next(iter(common_ids))]
                intersects, coplanar = _shared_vertex_intersection(points, other_points, shared, tolerance)
            else:
                intersects, coplanar = _triangles_intersect(points, other_points, tolerance)
            if intersects:
                hits.append(MeshIntersection(other_id, fid, coplanar))
                if len(hits) >= max_hits:
                    return hits, True
        active.append((xmin, fid, face, points, bounds, extent, rounding))
    return hits, False
