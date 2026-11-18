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

"""Red-blue mesh intersection primitives, ported from AnaMorph.

Based on AnaMorph (https://github.com/NeuroBox3D/AnaMorph), including tuple
generation, cyclic ordering, and single-circle cutting and stitching. Several
source retry and geometric edge cases remain incomplete; see PORT_STATUS.md.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from collections import Counter, deque
from fractions import Fraction
from math import atan2, fsum, hypot, isfinite, pi, sqrt, ulp
from sys import float_info
import warnings

from .mesh import Mesh, cross, dot, sub
from .source_polygon import source_delaunay2d, source_ear_triangulate
from .source_random import make_random


class RedBlueError(RuntimeError):
    pass


class RedBlueNumericalEdgeCase(RedBlueError):
    """An intersection touches a vertex/edge or is coplanar and ambiguous."""


class RedBlueTriangulationError(RedBlueNumericalEdgeCase):
    """A cut polygon could not be triangulated safely."""


class SourceDelaunayLimitWarning(RuntimeWarning):
    """Source-style Delaunay cleanup returned before convergence."""


class RedBlueDisjoint(RedBlueError):
    """No surface intersection was found; containment is not determined."""


class RedBlueComplexEdges(RedBlueError):
    def __init__(self, edges):
        super().__init__("one or more edges intersect multiple opposite faces")
        self.edges = tuple(edges)


class RedBlueAffectedCircleTrivial(RedBlueError):
    """One colour has no edge intersections and needs face subdivision."""

    def __init__(self, message, *, red=None, face=None, position=None):
        super().__init__(message)
        self.red, self.face, self.position = red, face, position


class RedBlueMultipleLoops(RedBlueError):
    pass


class RedBlueBoundaryInteraction(RedBlueError):
    """A proposed operation would change a previously flushed boundary."""


class RedBlueRetryLimit(RedBlueError):
    """The inner complex-edge/face-subdivision retry loop did not converge."""


@dataclass(frozen=True, slots=True)
class ComplexEdge:
    red: bool
    edge: tuple[int, int]
    parameters: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class IntersectionTuple:
    red: bool
    position: tuple[float, float, float]
    edge_parameter: float
    barycentric_s: float
    barycentric_t: float
    edge_out: int
    edge_in: int
    opposite_face: int
    backward_face: int
    forward_face: int

    @property
    def incident_faces(self):
        return (self.backward_face, self.forward_face)


@dataclass(frozen=True, slots=True)
class RedBlueUnionResult:
    mesh: Mesh
    red_vertex_map: tuple[int | None, ...]
    blue_vertex_map: tuple[int | None, ...]
    inner_attempts: int = 1


def _exact_triangle_normal(a, b, c):
    """Normalize the exact cross product of the represented coordinates."""
    points = [tuple(Fraction(value) for value in point) for point in (a, b, c)]
    normal = cross(sub(points[1], points[0]), sub(points[2], points[0]))
    scale = max(map(abs, normal))
    if not scale:
        raise ValueError("mesh contains a degenerate face")
    values = tuple(float(value/scale) for value in normal)
    length = hypot(*values)
    return tuple(value/length for value in values)


def _triangle_unit_normal(a, b, c):
    """Compute a triangle normal without squaring world-scale coordinates."""
    ab, ac = sub(b, a), sub(c, a)
    if not all(isfinite(value) for edge in (ab, ac) for value in edge):
        raise ValueError("triangle coordinate differences exceed float range")
    scale = max(abs(value) for edge in (ab, ac) for value in edge)
    if not scale:
        raise ValueError("mesh contains a degenerate face")
    scaled_ab = tuple(value/scale for value in ab)
    scaled_ac = tuple(value/scale for value in ac)
    normal = cross(scaled_ab, scaled_ac)
    length = hypot(*normal)
    if not isfinite(length):
        raise ValueError("mesh contains a numerically invalid face")
    if length <= 64*ulp(1.)*hypot(*scaled_ab)*hypot(*scaled_ac):
        return _exact_triangle_normal(a, b, c)
    return tuple(value/length for value in normal)


def _edge_faces(mesh: Mesh, *, allow_boundary: bool = False):
    if any(len(vertex) != 3 or not all(isfinite(value) for value in vertex) for vertex in mesh.vertices):
        raise ValueError("mesh contains a non-finite or invalid vertex")
    edges: dict[tuple[int, int], list[int]] = {}
    incident_faces = [[] for _ in mesh.vertices]
    face_keys = set()
    for fid, face in enumerate(mesh.faces):
        if len(face) != 3 or len(set(face)) != 3 or any(v < 0 or v >= len(mesh.vertices) for v in face):
            raise ValueError("mesh contains an invalid triangular face")
        face_key = tuple(sorted(face))
        if face_key in face_keys:
            raise ValueError("red-blue intersection requires distinct triangular faces")
        face_keys.add(face_key)
        for vertex in face:
            incident_faces[vertex].append(fid)
        points = [mesh.vertices[v] for v in face]
        _triangle_unit_normal(*points)
        for u, v in zip(face, face[1:]+face[:1]):
            key = (u, v) if u < v else (v, u)
            # Encode orientation without a second face scan. The negative
            # form -fid-1 also distinguishes the orientation of face zero.
            edges.setdefault(key, []).append(fid if u < v else -fid-1)
    if not edges or any(len(faces) not in ((1, 2) if allow_boundary else (2,)) for faces in edges.values()):
        raise ValueError("red-blue intersection requires " +
                         ("edge-manifold" if allow_boundary else "closed edge-manifold") + " triangular meshes")
    boundary_edges = [0]*len(mesh.vertices)
    for edge, adjacent in edges.items():
        if len(adjacent) == 2 and (adjacent[0] >= 0) == (adjacent[1] >= 0):
            raise ValueError("red-blue intersection requires consistently oriented faces")
        if len(adjacent) == 1:
            for vertex in edge:
                boundary_edges[vertex] += 1
        edges[edge] = [index if index >= 0 else -index-1 for index in adjacent]
    # Edge incidence alone permits separate sheets that meet only at a
    # vertex. At every used vertex the incident faces must form one fan,
    # closed in the interior or bounded by exactly two boundary edges.
    for vertex, incident in enumerate(incident_faces):
        if not incident:
            continue
        if boundary_edges[vertex] not in (0, 2):
            raise ValueError("red-blue intersection requires vertex-manifold meshes")
        visited, stack = set(), [incident[0]]
        while stack:
            face_id = stack.pop()
            if face_id in visited:
                continue
            visited.add(face_id)
            for other in mesh.faces[face_id]:
                if other == vertex:
                    continue
                edge = (vertex, other) if vertex < other else (other, vertex)
                stack.extend(i for i in edges[edge] if i not in visited)
        if len(visited) != len(incident):
            raise ValueError("red-blue intersection requires vertex-manifold meshes")
    return edges


def _boxes_overlap(points_a, points_b, eps):
    return all(max(p[i] for p in points_a)+eps >= min(p[i] for p in points_b) and
               max(p[i] for p in points_b)+eps >= min(p[i] for p in points_a) for i in range(3))


def _face_bvh(mesh: Mesh, leaf_size: int = 12):
    """Build a static bounding-volume hierarchy over triangle faces."""
    entries = []
    for fid, face in enumerate(mesh.faces):
        triangle = tuple(mesh.vertices[v] for v in face)
        bounds = tuple((min(p[i] for p in triangle), max(p[i] for p in triangle)) for i in range(3))
        entries.append((fid, triangle, bounds))
    def build(items):
        bounds = tuple((min(item[2][i][0] for item in items), max(item[2][i][1] for item in items)) for i in range(3))
        if len(items) <= leaf_size:
            return bounds, None, None, tuple(items)
        axis = max(range(3), key=lambda i: bounds[i][1]-bounds[i][0])
        items.sort(key=lambda item: item[2][axis][0]+item[2][axis][1])
        middle = len(items)//2
        return bounds, build(items[:middle]), build(items[middle:]), ()
    return build(entries)


class _PreparedMesh:
    """Immutable-mesh search data, with the face hierarchy built lazily."""
    def __init__(self, mesh, *, allow_boundary=False):
        self.mesh = mesh
        self.edges = _edge_faces(mesh, allow_boundary=allow_boundary)
        self._faces = None

    def faces(self):
        if self._faces is None:
            self._faces = _face_bvh(self.mesh)
        return self._faces


class _RedBlueCache:
    """Keep only the most recent red and blue immutable-mesh indices."""
    def __init__(self, *, allow_red_boundary=False):
        self.red = None
        self.blue = None
        self.allow_red_boundary = allow_red_boundary

    def clear(self):
        self.red = None
        self.blue = None

    def prepare(self, mesh, *, red):
        # Mesh is frozen, but callers can still construct one with mutable
        # lists despite the tuple annotations. Never cache such operands.
        name = "red" if red else "blue"
        immutable = (type(mesh.vertices) is tuple and type(mesh.faces) is tuple and
                     all(type(vertex) is tuple for vertex in mesh.vertices) and
                     all(type(face) is tuple for face in mesh.faces))
        if not immutable:
            setattr(self, name, None)
            return _PreparedMesh(mesh, allow_boundary=red and self.allow_red_boundary)
        prepared = getattr(self, name)
        if prepared is None or prepared.mesh is not mesh:
            # Drop the previous, potentially large hierarchy and incidence
            # map before allocating a replacement for the next mesh.
            setattr(self, name, None)
            prepared = None
            prepared = _PreparedMesh(mesh, allow_boundary=red and self.allow_red_boundary)
            setattr(self, name, prepared)
        return prepared


def _query_face_bvh(tree, start, end, eps):
    bounds = tuple((min(start[i], end[i]), max(start[i], end[i])) for i in range(3))
    edge_extent = max(hi-lo for lo, hi in bounds)
    edge_magnitude = max(abs(value) for interval in bounds for value in interval)
    root_bounds = tree[0]
    root_extent = max(hi-lo for lo, hi in root_bounds)
    root_magnitude = max(abs(value) for interval in root_bounds for value in interval)
    # The narrow phase compares its tolerance against Euclidean edge lengths,
    # and permits small barycentric and edge-parameter excursions. An AABB
    # extent is measured per coordinate, so its padding must be larger than
    # eps*extent to avoid discarding near-boundary cases before classification.
    upper_padding = 8*eps*max(edge_extent, root_extent) + 8*ulp(max(edge_magnitude, root_magnitude))
    def intersects(b):
        # The narrow-phase epsilon is relative to local geometry. A fixed
        # world-unit expansion makes every tiny face a broad-phase candidate.
        if all(bounds[i][0] <= b[i][1] and b[i][0] <= bounds[i][1] for i in range(3)):
            return True
        if any(bounds[i][0] > b[i][1]+upper_padding or b[i][0] > bounds[i][1]+upper_padding for i in range(3)):
            return False
        extent = max(edge_extent, b[0][1]-b[0][0], b[1][1]-b[1][0], b[2][1]-b[2][0])
        magnitude = max(edge_magnitude, *(abs(value) for interval in b for value in interval))
        padding = 8*eps*extent + 8*ulp(magnitude)
        return all(bounds[i][0] <= b[i][1]+padding and b[i][0] <= bounds[i][1]+padding for i in range(3))
    stack, result = [tree], []
    while stack:
        node_bounds, left, right, entries = stack.pop()
        if not intersects(node_bounds):
            continue
        if entries:
            result.extend((fid, triangle) for fid, triangle, face_bounds in entries if intersects(face_bounds))
        else:
            stack.append(left)
            stack.append(right)
    result.sort(key=lambda item: item[0])
    return result


def _projected_segment_disjoint(start, end, triangle, normal, eps):
    """Prove a segment and triangle disjoint in their dominant 2-D projection."""
    drop = max(range(3), key=lambda i: abs(normal[i]))
    axes = tuple(i for i in range(3) if i != drop)
    origin = triangle[0]
    def project(point):
        return tuple(point[i]-origin[i] for i in axes)
    segment = (project(start), project(end))
    tri = tuple(project(point) for point in triangle)
    scale = max(abs(value) for point in segment+tri for value in point)
    for a, b in tuple(zip(tri, tri[1:]+tri[:1]))+(segment,):
        axis = (a[1]-b[1], b[0]-a[0])
        axis_length = sqrt(axis[0]**2+axis[1]**2)
        if axis_length == 0:
            continue
        first = [p[0]*axis[0]+p[1]*axis[1] for p in segment]
        second = [p[0]*axis[0]+p[1]*axis[1] for p in tri]
        margin = eps*scale*axis_length*4
        if max(first)+margin < min(second) or max(second)+margin < min(first):
            return True
    return False


def _exact_edge_face_hit(start, end, triangle, eps):
    """Resolve an ill-conditioned segment/triangle hit on input floats."""
    a, b, c, start_exact, end_exact = (
        tuple(Fraction(value) for value in point) for point in (*triangle, start, end))
    direction, ab, ac, offset = sub(end_exact, start_exact), sub(b, a), sub(c, a), sub(start_exact, a)
    if not any(direction):
        raise ValueError("zero-length edge or degenerate triangle")
    normal = cross(ab, ac)
    if not any(normal):
        raise ValueError("zero-length edge or degenerate triangle")
    h = cross(direction, ac)
    determinant = dot(ab, h)
    if not determinant:
        # Preserve the tolerance-based parallel handling while using exact
        # plane separation. A 2-D separating axis can also rule out contact.
        distance = dot(offset, normal)
        normal_squared = dot(normal, normal)
        local_squared = max(dot(direction, direction), dot(ab, ab), dot(ac, ac))
        if distance*distance > Fraction(eps)**2*normal_squared*local_squared:
            return None
        drop = max(range(3), key=lambda i: abs(normal[i]))
        axes = tuple(i for i in range(3) if i != drop)
        def project(point):
            return tuple(point[i]-a[i] for i in axes)
        projected_triangle = tuple(map(project, (a, b, c)))
        segment = tuple(map(project, (start_exact, end_exact)))
        scale = max(abs(value) for point in projected_triangle+segment for value in point)
        for first, second in tuple(zip(projected_triangle, projected_triangle[1:]+projected_triangle[:1]))+(segment,):
            axis = (first[1]-second[1], second[0]-first[0])
            axis_squared = axis[0]*axis[0]+axis[1]*axis[1]
            if not axis_squared:
                continue
            segment_values = [point[0]*axis[0]+point[1]*axis[1] for point in segment]
            triangle_values = [point[0]*axis[0]+point[1]*axis[1] for point in projected_triangle]
            gap = max(min(triangle_values)-max(segment_values), min(segment_values)-max(triangle_values))
            if gap > 0 and gap*gap > 16*Fraction(eps)**2*scale*scale*axis_squared:
                return None
        raise RedBlueNumericalEdgeCase("ill-conditioned parallel edge and face require perturbation")
    s = dot(offset, h)/determinant
    perpendicular = cross(offset, ab)
    t = dot(direction, perpendicular)/determinant
    lam = dot(ac, perpendicular)/determinant
    margin = Fraction(eps)
    if s < -margin or s > 1+margin or t < -margin or s+t > 1+margin or lam < -margin or lam > 1+margin:
        return None
    if min(abs(lam), abs(lam-1), abs(s), abs(t), abs(1-s-t)) <= margin:
        raise RedBlueNumericalEdgeCase("intersection falls on an edge or vertex")
    point = tuple(float(start_exact[i]+lam*direction[i]) for i in range(3))
    rounded = tuple(Fraction(value) for value in point)
    if rounded == start_exact or rounded == end_exact:
        raise RedBlueNumericalEdgeCase("rounded intersection coincides with an edge endpoint")
    # The exact intersection can be interior while the nearest representable
    # point lies outside a very thin face or off the crossing edge. Check the
    # rounded point against the face and edge using the face's minimum altitude as
    # a local scale; otherwise the subsequent cut can create a malformed seam.
    point_offset = sub(rounded, a)
    normal_squared = dot(normal, normal)
    projected_s = dot(cross(point_offset, ac), normal)/normal_squared
    projected_t = dot(cross(ab, point_offset), normal)/normal_squared
    if projected_s <= 0 or projected_t <= 0 or projected_s+projected_t >= 1:
        raise RedBlueNumericalEdgeCase("rounded intersection is outside the face interior")
    bc = sub(c, b)
    max_edge_squared = max(dot(ab, ab), dot(ac, ac), dot(bc, bc))
    # Allow unavoidable rounding up to a small fraction of local altitude,
    # even when a world-coordinate ulp exceeds the predicate epsilon. This
    # keeps ordinary translated meshes usable while rejecting material seam
    # displacement. The floor is a heuristic, not an error certificate.
    position_margin = max(margin, Fraction(1, 65536))
    plane_error = dot(point_offset, normal)
    if plane_error*plane_error*max_edge_squared > position_margin*position_margin*normal_squared*normal_squared:
        raise RedBlueNumericalEdgeCase("rounded intersection is too far from the face plane")
    line_error = cross(sub(rounded, start_exact), direction)
    if (dot(line_error, line_error)*max_edge_squared >
            position_margin*position_margin*normal_squared*dot(direction, direction)):
        raise RedBlueNumericalEdgeCase("rounded intersection is too far from the crossing edge")
    normal_scale = max(map(abs, normal))
    scaled_normal = tuple(float(value/normal_scale) for value in normal)
    length = hypot(*scaled_normal)
    return float(lam), float(s), float(t), point, tuple(value/length for value in scaled_normal)


def _edge_face_hit(start, end, triangle, eps):
    """Möller–Trumbore segment/triangle intersection with explicit edge cases."""
    a, b, c = triangle
    raw_direction, raw_ab, raw_ac, raw_offset = sub(end, start), sub(b, a), sub(c, a), sub(start, a)
    vectors = (raw_direction, raw_ab, raw_ac, raw_offset)
    if not all(isfinite(value) for vector in vectors for value in vector):
        raise RedBlueNumericalEdgeCase("edge-face coordinate differences exceed float range")
    coordinate_scale = max(abs(value) for vector in vectors for value in vector)
    if coordinate_scale == 0:
        raise ValueError("zero-length edge or degenerate triangle")
    direction, ab, ac, origin_offset = (tuple(value/coordinate_scale for value in vector) for vector in vectors)
    if any(original and not scaled for vector, normalized in zip(vectors, (direction, ab, ac, origin_offset))
           for original, scaled in zip(vector, normalized)):
        raise RedBlueNumericalEdgeCase("edge-face coordinate scale ratio exceeds float range")
    normal = cross(ab, ac)
    h = cross(direction, ac)
    determinant = dot(ab, h)
    direction_length, normal_length = hypot(*direction), hypot(*normal)
    if direction_length == 0:
        raise ValueError("zero-length edge or degenerate triangle")
    ab_length, ac_length = hypot(*ab), hypot(*ac)
    normal_error = 64*ulp(1.)*ab_length*ac_length
    determinant_error = normal_error*direction_length
    if normal_length <= normal_error or (determinant_error and abs(determinant) <= determinant_error):
        return _exact_edge_face_hit(start, end, triangle, eps)
    scale = direction_length*normal_length
    if scale == 0:
        raise RedBlueNumericalEdgeCase("edge-face orientation underflows float range")
    threshold = eps*scale
    if threshold == 0:
        raise RedBlueNumericalEdgeCase("edge-face tolerance underflows float range")
    if abs(determinant) <= threshold:
        # Parallel coplanar candidates cannot be classified by this primitive.
        plane_offset = abs(dot(origin_offset, normal))
        local_scale = max(direction_length, hypot(*ab), hypot(*ac))
        if plane_offset <= eps*normal_length*local_scale:
            local_end = tuple(origin_offset[i]+direction[i] for i in range(3))
            if _projected_segment_disjoint(origin_offset, local_end, ((0., 0., 0.), ab, ac), normal, eps):
                return None
            raise RedBlueNumericalEdgeCase("coplanar edge and face require perturbation or subdivision")
        return None
    # Even with a well-conditioned triangle and determinant, a long edge or
    # distant start can make the three barycentric numerators the difference
    # of much larger terms. Bound that cancellation before accepting either a
    # hit or a disjoint result; use the represented coordinates exactly when
    # the expected numerator error can exceed the requested tolerance.
    offset_length = hypot(*origin_offset)
    error_scale = 64*ulp(1.)*offset_length
    numerator_error = max(error_scale*direction_length*ac_length,
                          error_scale*direction_length*ab_length,
                          error_scale*ab_length*ac_length)
    if numerator_error > eps*abs(determinant):
        return _exact_edge_face_hit(start, end, triangle, eps)
    inverse = 1/determinant
    if not isfinite(inverse):
        raise RedBlueNumericalEdgeCase("edge-face determinant exceeds float precision")
    s = inverse*dot(origin_offset, h)
    if s < -eps or s > 1+eps:
        return None
    perpendicular = cross(origin_offset, ab)
    t = inverse*dot(direction, perpendicular)
    if t < -eps or s+t > 1+eps:
        return None
    lam = inverse*dot(ac, perpendicular)
    if lam < -eps or lam > 1+eps:
        return None
    if min(abs(lam), abs(lam-1), abs(s), abs(t), abs(1-s-t)) <= eps:
        raise RedBlueNumericalEdgeCase("intersection falls on an edge or vertex")
    point = tuple(start[i]+lam*raw_direction[i] for i in range(3))
    if not all(isfinite(value) for value in point):
        raise RedBlueNumericalEdgeCase("edge-face intersection point exceeds float range")
    # Even an accurate parameter can round to a point far from the original
    # edge after a large translation. Resolve and validate coordinates exactly
    # whenever an ulp is significant relative to the face's minimum altitude.
    bc_length = hypot(*(ab[i]-ac[i] for i in range(3)))
    altitude = normal_length/max(ab_length, ac_length, bc_length)
    rounding = max(ulp(value) for value in point)/coordinate_scale
    if rounding > eps*altitude/8:
        return _exact_edge_face_hit(start, end, triangle, eps)
    return lam, s, t, point, tuple(value/normal_length for value in normal)


def generate_intersection_tuples(edge_mesh: Mesh, face_mesh: Mesh, *, red: bool,
                                 eps: float = 1e-10):
    """Generate one colour's edge-face tuples and report complex edges.

    The bounding-volume hierarchy performs the C++ broad phase's role without
    reproducing its exact octree data structure or traversal order.
    """
    if not isfinite(eps) or eps <= 0:
        raise ValueError("epsilon must be positive and finite")
    edge_incidence = _edge_faces(edge_mesh)
    _edge_faces(face_mesh)
    face_index = _face_bvh(face_mesh)
    return _generate_intersection_tuples(edge_mesh, edge_incidence, face_index, red=red, eps=eps)


def _edge_start_side(start, point, triangle, normal):
    """Classify the start relative to a face, resolving uncertain signs exactly."""
    to_start = sub(start, point)
    scale = max(abs(value) for value in to_start)
    if not isfinite(scale) or not scale:
        raise RedBlueNumericalEdgeCase("intersection point is indistinguishable from edge endpoint")
    scaled = tuple(value/scale for value in to_start)
    terms = tuple(scaled[i]*normal[i] for i in range(3))
    side = sum(terms)
    # Intersection-point rounding can be significant when the mesh is far
    # from the origin; include coordinate ulps in addition to dot-product
    # cancellation before deciding whether the floating sign is trustworthy.
    rounding = sum(abs(normal[i])*(ulp(start[i])/scale+ulp(point[i])/scale)
                   for i in range(3) if normal[i])
    uncertainty = 64*ulp(1.)*sum(abs(term) for term in terms)+8*rounding
    if abs(side) > uncertainty:
        return 1 if side > 0 else -1
    a, b, c, origin = (tuple(Fraction(value) for value in item) for item in (*triangle, start))
    exact = dot(sub(origin, a), cross(sub(b, a), sub(c, a)))
    if not exact:
        raise RedBlueNumericalEdgeCase("edge-face orientation is numerically ambiguous")
    return 1 if exact > 0 else -1


def _generate_intersection_tuples(edge_mesh, edge_incidence, face_index, *, red, eps):
    """Generate tuples using validated edge incidence and face search data."""
    hits, complex_edges = [], []
    for edge, adjacent in edge_incidence.items():
        start, end = (edge_mesh.vertices[v] for v in edge)
        candidates = []
        for fid, triangle in _query_face_bvh(face_index, start, end, eps):
            hit = _edge_face_hit(start, end, triangle, eps)
            if hit is None:
                continue
            if len(adjacent) != 2:
                raise RedBlueBoundaryInteraction("intersection touches an open mesh boundary")
            lam, s, t, point, normal = hit
            side = _edge_start_side(start, point, triangle, normal)
            outside, inside = edge if side > 0 else (edge[1], edge[0])
            candidates.append(IntersectionTuple(red, point, lam, s, t, outside, inside, fid,
                                                adjacent[0], adjacent[1]))
        if len(candidates) > 1:
            complex_edges.append(ComplexEdge(red, edge, tuple(c.edge_parameter for c in candidates)))
        hits.extend(candidates)
    hits.sort(key=lambda x: (x.edge_out, x.edge_in, x.opposite_face))
    # The C++ broad phase sorts candidate pairs by the two undirected edge
    # IDs before it records complex edges. Repairing those edges in that order
    # determines the IDs of newly inserted vertices and, in turn, the cyclic
    # starting point and random consumption of later cut polygons.
    complex_edges.sort(key=lambda item: item.edge)
    return hits, complex_edges


def cyclically_order_tuples(mesh: Mesh, tuples: list[IntersectionTuple], *, red: bool):
    """Order one colour's tuples around the affected face circle.

    Index tuples by incident face so traversal examines local neighbors
    instead of scanning the remaining list at every step. Original input
    indices preserve the same tie order as the source-style linear scan.
    """
    if not tuples:
        raise RedBlueAffectedCircleTrivial("one colour has no edge intersections")
    if any(t.red != red for t in tuples):
        raise ValueError("tuple colour does not match mesh colour")
    by_face = {}
    for index, hit in enumerate(tuples):
        for face in set(hit.incident_faces):
            by_face.setdefault(face, []).append(index)
    remaining = set(range(len(tuples)-1))
    ordered = [tuples[-1]]
    while remaining:
        current = ordered[-1]
        choice = None
        for shared_face in set(current.incident_faces):
            for index in by_face[shared_face]:
                if index not in remaining or (choice is not None and index >= choice[0]):
                    continue
                candidate = tuples[index]
                shared = set(current.incident_faces) & set(candidate.incident_faces)
                if len(shared) != 1:
                    continue
                face = mesh.faces[shared_face]
                forward = any(face[j] == current.edge_out and face[(j+1)%3] == current.edge_in for j in range(3))
                if forward == red:
                    choice = index, shared_face
        if choice is None:
            if set(ordered[-1].incident_faces) & set(ordered[0].incident_faces):
                raise RedBlueMultipleLoops("intersection produces multiple loops")
            raise RedBlueError("intersection tuples cannot be ordered into a face circle")
        index, shared_face = choice
        current = ordered[-1]
        if current.forward_face != shared_face:
            ordered[-1] = replace(current, forward_face=current.backward_face, backward_face=current.forward_face)
        remaining.remove(index)
        candidate = tuples[index]
        if candidate.backward_face != shared_face:
            candidate = replace(candidate, backward_face=candidate.forward_face, forward_face=candidate.backward_face)
        ordered.append(candidate)
    if not set(ordered[-1].incident_faces) & set(ordered[0].incident_faces):
        raise RedBlueError("intersection face circle is not closed")
    return ordered


def intersection_face_circles(red_mesh: Mesh, blue_mesh: Mesh, *, eps: float = 1e-10, _cache=None,
                              _allow_red_boundary: bool = False):
    """Produce ordered red and blue circles before the cutting/stitching phase."""
    if not isfinite(eps) or eps <= 0:
        raise ValueError("epsilon must be positive and finite")
    cache = _cache if _cache is not None else _RedBlueCache(allow_red_boundary=_allow_red_boundary)
    if cache.allow_red_boundary != _allow_red_boundary:
        raise ValueError("red-blue cache boundary policy does not match the operation")
    red_prepared = cache.prepare(red_mesh, red=True)
    blue_prepared = cache.prepare(blue_mesh, red=False)
    red_tuples, red_complex = _generate_intersection_tuples(red_mesh, red_prepared.edges, blue_prepared.faces(), red=True, eps=eps)
    blue_tuples, blue_complex = _generate_intersection_tuples(blue_mesh, blue_prepared.edges, red_prepared.faces(), red=False, eps=eps)
    if red_complex or blue_complex:
        raise RedBlueComplexEdges(red_complex+blue_complex)
    if not red_tuples and not blue_tuples:
        raise RedBlueDisjoint("no edge-face intersections; containment was not checked")
    if not red_tuples or not blue_tuples:
        other = blue_tuples if not red_tuples else red_tuples
        if len({hit.opposite_face for hit in other}) != 1:
            raise RedBlueError("trivial affected circle spans more than one face")
        # The ordinary sum can overflow for valid, similarly translated
        # coordinates even though their mean is representable. This branch
        # is an exceptional repair path, so exact averaging is inexpensive.
        position = tuple(float(sum((Fraction(hit.position[i]) for hit in other), Fraction(0))/len(other))
                         for i in range(3))
        raise RedBlueAffectedCircleTrivial("affected face needs subdivision", red=not red_tuples,
                                           face=other[0].opposite_face, position=position)
    return (cyclically_order_tuples(red_mesh, red_tuples, red=True),
            cyclically_order_tuples(blue_mesh, blue_tuples, red=False))


def zip_intersection_circles(red_circle: list[IntersectionTuple], blue_circle: list[IntersectionTuple]):
    """Merge ordered colour circles into the complete intersection polygon.

    Follows the face-transition scan in RedBlueAlgorithm. Input lists are not
    mutated. Unlike the source's unbounded rotations, inconsistent circles
    raise RedBlueError instead of looping indefinitely.
    """
    if len(red_circle) < 2 or not blue_circle or any(not t.red for t in red_circle) or any(t.red for t in blue_circle):
        raise RedBlueError("ordered red and blue circles are required")
    red, blue = deque(red_circle), deque(blue_circle)
    for _ in range(len(red)):
        if red[0].opposite_face != red[1].opposite_face:
            break
        red.rotate(-1)
    else:
        raise RedBlueAffectedCircleTrivial("red circle intersects only one blue face")
    for _ in range(len(blue)):
        if blue[0].backward_face == red[0].opposite_face:
            break
        blue.rotate(-1)
    else:
        raise RedBlueError("red and blue circles have no matching starting face")
    polygon = []
    while red:
        current_face = red[0].opposite_face
        while red and red[0].opposite_face == current_face:
            polygon.append(red.popleft())
        if not red:
            polygon.extend(blue)
            break
        next_face = red[0].opposite_face
        if not blue or blue[0].backward_face != current_face:
            raise RedBlueError("red-blue circle face transition is inconsistent")
        while blue and blue[0].forward_face != next_face:
            polygon.append(blue.popleft())
        if not blue:
            raise RedBlueError("blue circle ended before the next red face")
        polygon.append(blue.popleft())
    return polygon


def _orient2_sign(a, b, c):
    """Sign of the orientation of the represented 2-D coordinates."""
    left = (b[0]-a[0])*(c[1]-a[1])
    right = (b[1]-a[1])*(c[0]-a[0])
    value = left-right
    magnitude = abs(left)+abs(right)
    if magnitude >= float_info.min and abs(value) > 16*ulp(1.)*magnitude:
        return 1 if value > 0 else -1
    exact = ((Fraction(b[0])-Fraction(a[0]))*(Fraction(c[1])-Fraction(a[1])) -
             (Fraction(b[1])-Fraction(a[1]))*(Fraction(c[0])-Fraction(a[0])))
    return (exact > 0)-(exact < 0)


def _validate_polygon_boundary(indices, points):
    """Reject non-simple projected boundaries before ear clipping."""
    def orientation(a, b, c):
        return _orient2_sign(points[a], points[b], points[c])
    def on_segment(a, b, p):
        a, b, p = points[a], points[b], points[p]
        return (min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and
                min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))
    def intersects(a, b, c, d):
        pa, pb, pc, pd = (points[i] for i in (a, b, c, d))
        if any(max(pa[k], pb[k]) < min(pc[k], pd[k]) or max(pc[k], pd[k]) < min(pa[k], pb[k])
               for k in (0, 1)):
            return False
        s1, s2, s3, s4 = orientation(a, b, c), orientation(a, b, d), orientation(c, d, a), orientation(c, d, b)
        return ((s1*s2 < 0 and s3*s4 < 0) or (not s1 and on_segment(a, b, c)) or
                (not s2 and on_segment(a, b, d)) or (not s3 and on_segment(c, d, a)) or
                (not s4 and on_segment(c, d, b)))
    n = len(indices)
    for i in range(n):
        a, b, c = indices[i-1], indices[i], indices[(i+1)%n]
        if points[a] == points[b]:
            raise RedBlueTriangulationError("cut polygon has a zero-length boundary edge")
        if not orientation(a, b, c):
            ab = tuple(Fraction(points[a][k])-Fraction(points[b][k]) for k in (0, 1))
            cb = tuple(Fraction(points[c][k])-Fraction(points[b][k]) for k in (0, 1))
            if ab[0]*cb[0]+ab[1]*cb[1] > 0:
                raise RedBlueTriangulationError("cut polygon boundary overlaps itself")
        for j in range(i+1, n):
            if j == i+1 or (i == 0 and j == n-1):
                continue
            if intersects(indices[i], indices[(i+1)%n], indices[j], indices[(j+1)%n]):
                raise RedBlueTriangulationError("cut polygon boundary intersects itself")


def _triangulate_polygon(indices, vertices, normal, basis=None, *, source_rng=None, source_delaunay=False,
                         source_delaunay_allow_limit=False):
    """Ear-clip a simple planar boundary, preserving the original face normal."""
    if len(indices) < 3 or len(set(indices)) != len(indices):
        raise RedBlueTriangulationError("invalid cut polygon")
    normal_length = hypot(*normal)
    if not isfinite(normal_length) or normal_length == 0:
        raise RedBlueTriangulationError("cut polygon has no finite plane normal")
    unit_normal = tuple(value/normal_length for value in normal)
    origin = vertices[indices[0]]
    if basis is None:
        # Choose a long chord to avoid magnifying noise in a tiny first edge.
        basis = max((sub(vertices[i], origin) for i in indices[1:]), key=lambda value: hypot(*value))
    basis_length = hypot(*basis)
    if not isfinite(basis_length) or basis_length == 0:
        raise RedBlueTriangulationError("cut polygon has no finite plane basis")
    candidate_axis = tuple(value/basis_length for value in basis)
    normal_component = dot(unit_normal, candidate_axis)
    tangent = tuple(candidate_axis[i]-normal_component*unit_normal[i] for i in range(3))
    tangent_length = hypot(*tangent)
    if not isfinite(tangent_length) or tangent_length <= 1e-14:
        raise RedBlueTriangulationError("cut polygon plane basis is degenerate")
    x_axis = tuple(value/tangent_length for value in tangent)
    transverse = cross(unit_normal, x_axis)
    transverse_length = hypot(*transverse)
    if not isfinite(transverse_length) or transverse_length <= 1e-14:
        raise RedBlueTriangulationError("cut polygon plane basis is degenerate")
    y_axis = tuple(value/transverse_length for value in transverse)
    # An orthonormal basis preserves Euclidean angles and circumcircles;
    # dropping a coordinate distorts Delaunay flips on tilted triangles.
    projected = {i: (dot(sub(vertices[i], origin), x_axis), dot(sub(vertices[i], origin), y_axis))
                 for i in indices}
    span = max(max(projected[i][j] for i in indices)-min(projected[i][j] for i in indices) for j in (0, 1))
    if not isfinite(span) or span <= 0:
        raise RedBlueTriangulationError("cut polygon has no finite projected extent")
    for index in indices:
        point = vertices[index]
        displacement = sub(point, origin)
        rounding = fsum(abs(unit_normal[i])*max(ulp(origin[i]), ulp(point[i])) for i in range(3))
        if abs(dot(displacement, unit_normal)) > 1e-9*span+32*rounding:
            raise RedBlueTriangulationError("cut polygon vertices are not on the face plane")
    origin = projected[indices[0]]
    # Work in local, unit-scale coordinates so area and in-circle predicates
    # are not dominated by absolute translations or a fixed world-unit floor.
    points = {i: tuple((projected[i][j]-origin[j])/span for j in (0, 1)) for i in indices}
    _validate_polygon_boundary(indices, points)
    def orient(a, b, c):
        return _orient2_sign(points[a], points[b], points[c])
    poly = list(indices)
    terms = [(points[a][0]*points[b][1], points[b][0]*points[a][1])
             for a, b in zip(poly, poly[1:]+poly[:1])]
    signed_area = fsum(left-right for left, right in terms)
    area_magnitude = fsum(abs(left)+abs(right) for left, right in terms)
    if area_magnitude < float_info.min or abs(signed_area) <= 16*ulp(1.)*area_magnitude:
        signed_area = sum((Fraction(points[a][0])*Fraction(points[b][1]) -
                           Fraction(points[b][0])*Fraction(points[a][1])
                           for a, b in zip(poly, poly[1:]+poly[:1])), Fraction(0))
    if signed_area == 0:
        raise RedBlueTriangulationError("cut polygon has zero projected area")
    sign = 1 if signed_area > 0 else -1
    triangles = []
    if source_rng is not None:
        # The source projects absolute world positions rather than local,
        # normalized coordinates before its randomized ear test.
        source_points = [(dot(vertices[i], x_axis), dot(vertices[i], y_axis)) for i in indices]
        try:
            triangles = [tuple(indices[i] for i in face) for face in source_ear_triangulate(source_points, source_rng)]
        except (ValueError, OverflowError) as error:
            raise RedBlueTriangulationError(f"source-style polygon triangulation failed: {error}") from error
        # Source ears can contain a nearly collinear intermediate triangle.
        # Let Delaunay flipping try to resolve it; the completed triangulation
        # still must pass the face-normal and area checks below.
        exact_points = {index: tuple(Fraction(value) for value in points[index]) for index in indices}
        def exact_cross(a, b, c):
            a, b, c = exact_points[a], exact_points[b], exact_points[c]
            return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
        polygon_area = sum((exact_points[a][0]*exact_points[b][1]-exact_points[b][0]*exact_points[a][1]
                            for a, b in zip(indices, indices[1:]+indices[:1])), Fraction(0))
        if sum((exact_cross(*face) for face in triangles), Fraction(0)) != polygon_area:
            raise RedBlueTriangulationError("source-style polygon triangulation does not cover the boundary")
    else:
        while len(poly) > 3:
            for i in range(len(poly)):
                a, b, c = poly[i-1], poly[i], poly[(i+1)%len(poly)]
                if sign*orient(a, b, c) <= 0:
                    continue
                if any(sign*orient(a, b, p) >= 0 and sign*orient(b, c, p) >= 0 and
                       sign*orient(c, a, p) >= 0 for p in poly if p not in (a, b, c)):
                    continue
                triangles.append((a, b, c))
                del poly[i]
                break
            else:
                raise RedBlueTriangulationError("could not triangulate the cut polygon")
        if sign*orient(*poly) <= 0:
            raise RedBlueTriangulationError("cut polygon produced a degenerate triangle")
        triangles.append(tuple(poly))
    # Improve the ear-clipped triangulation while keeping every polygon edge
    # constrained. The source similarly runs a 2-D Delaunay edge-flip pass.
    boundary = {tuple(sorted((a, b))) for a, b in zip(indices, indices[1:]+indices[:1])}
    def in_circle(a, b, c, d):
        ax, ay = (points[a][j]-points[d][j] for j in (0, 1))
        bx, by = (points[b][j]-points[d][j] for j in (0, 1))
        cx, cy = (points[c][j]-points[d][j] for j in (0, 1))
        alift, blift, clift = ax*ax+ay*ay, bx*bx+by*by, cx*cx+cy*cy
        value = (alift*(bx*cy-by*cx) - blift*(ax*cy-ay*cx) + clift*(ax*by-ay*bx))
        permanent = (alift*(abs(bx*cy)+abs(by*cx)) +
                     blift*(abs(ax*cy)+abs(ay*cx)) +
                     clift*(abs(ax*by)+abs(ay*bx)))
        if permanent >= float_info.min and abs(value) > 32*ulp(1.)*permanent:
            return 1 if value > 0 else -1
        # Evaluate the determinant of the actual projected floats, including
        # their coordinate differences, without cancellation or underflow.
        exact_points = [tuple(Fraction(value) for value in points[i]) for i in (a, b, c, d)]
        (ax, ay), (bx, by), (cx, cy) = (
            (p[0]-exact_points[3][0], p[1]-exact_points[3][1]) for p in exact_points[:3])
        exact = ((ax*ax+ay*ay)*(bx*cy-by*cx) -
                 (bx*bx+by*by)*(ax*cy-ay*cx) +
                 (cx*cx+cy*cy)*(ax*by-ay*bx))
        return (exact > 0)-(exact < 0)
    if source_delaunay:
        # The source determinant assumes counter-clockwise triangles. Inside
        # cuts can have the opposite projected orientation; reflect the
        # second coordinate for the cleanup, retaining the same vertex IDs and
        # restoring the requested face orientation below.
        source_points = {index: (dot(vertices[index], x_axis), sign*dot(vertices[index], y_axis))
                         for index in indices}
        try:
            result = source_delaunay2d(source_points, triangles)
        except (ValueError, OverflowError) as error:
            raise RedBlueTriangulationError(f"source-style Delaunay cleanup failed: {error}") from error
        if not result.converged and not source_delaunay_allow_limit:
            raise RedBlueTriangulationError("source-style Delaunay cleanup did not converge")
        triangles = list(result.triangles)
        if not result.converged:
            # The C++ code keeps the current triangles at its iteration limit.
            # Check their combinatorial boundary and exact projected area before
            # accepting them; the normal checks below still apply as well.
            ids = set(indices)
            if (len(triangles) != len(indices)-2 or
                    any(len(face) != 3 or len(set(face)) != 3 or not set(face) <= ids for face in triangles)):
                raise RedBlueTriangulationError("source-style bounded cleanup produced an invalid patch")
            directed = Counter((a, b) for face in triangles for a, b in zip(face, face[1:]+face[:1]))
            edges = Counter(tuple(sorted(edge)) for edge, count in directed.items() for _ in range(count))
            if (any(edges[edge] != 1 for edge in boundary) or
                    any(count != (1 if edge in boundary else 2) for edge, count in edges.items()) or
                    any(directed[(a, b)] != 1 or directed[(b, a)] != 1
                        for (a, b), count in edges.items() if count == 2)):
                raise RedBlueTriangulationError("source-style bounded cleanup changed the patch boundary")
            exact_points = {i: tuple(Fraction(value) for value in points[i]) for i in indices}
            def area(a, b, c):
                a, b, c = (exact_points[i] for i in (a, b, c))
                return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
            polygon_area = sum((exact_points[a][0]*exact_points[b][1]-exact_points[b][0]*exact_points[a][1]
                                for a, b in zip(indices, indices[1:]+indices[:1])), Fraction(0))
            if abs(sum((area(*face) for face in triangles), Fraction(0))) != abs(polygon_area):
                raise RedBlueTriangulationError("source-style bounded cleanup does not cover the boundary")
            warnings.warn("source-style Delaunay cleanup reached its iteration limit; using its current triangulation",
                          SourceDelaunayLimitWarning, stacklevel=2)
    max_flips = max(1, 16*len(triangles)**2)
    flips = 0
    while not source_delaunay:
        edge_triangles = {}
        for fid, face in enumerate(triangles):
            for a, b in zip(face, face[1:]+face[:1]):
                edge_triangles.setdefault(tuple(sorted((a, b))), []).append(fid)
        changed = False
        for (u, v), adjacent in edge_triangles.items():
            if (u, v) in boundary or len(adjacent) != 2:
                continue
            first, second = (triangles[i] for i in adjacent)
            if not any(first[i] == u and first[(i+1)%3] == v for i in range(3)):
                u, v = v, u
            c = next(x for x in first if x not in (u, v))
            d = next(x for x in second if x not in (u, v))
            # The alternative diagonal must stay inside the quadrilateral.
            if (sign*orient(c, d, v) <= 0 or sign*orient(d, c, u) <= 0 or
                    sign*in_circle(u, v, c, d) <= 0):
                continue
            triangles[adjacent[0]] = (c, d, v)
            triangles[adjacent[1]] = (d, c, u)
            flips += 1
            if flips > max_flips:
                raise RedBlueTriangulationError("cut-polygon Delaunay flipping did not converge")
            changed = True
            break
        if not changed:
            break
    a, b, c = (vertices[i] for i in triangles[0])
    try:
        first_normal = _triangle_unit_normal(a, b, c)
    except ValueError as error:
        raise RedBlueTriangulationError("cut polygon produced a numerically invalid triangle") from error
    if dot(first_normal, unit_normal) < 0:
        triangles = [(a, c, b) for a, b, c in triangles]
    for face in triangles:
        try:
            face_normal = _triangle_unit_normal(*(vertices[i] for i in face))
        except ValueError as error:
            raise RedBlueTriangulationError("cut polygon produced a numerically invalid triangle") from error
        if dot(face_normal, unit_normal) <= 0:
            raise RedBlueTriangulationError("cut polygon produced a reversed triangle")
    return triangles


def _vertex_components(mesh: Mesh):
    """Return the connected vertex sets of a triangular surface."""
    neighbors = [set() for _ in mesh.vertices]
    for a, b, c in mesh.faces:
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    unseen = {v for face in mesh.faces for v in face}
    components = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        component, stack = {start}, [start]
        while stack:
            vertex = stack.pop()
            new = neighbors[vertex] & unseen
            component.update(new)
            unseen.difference_update(new)
            stack.extend(new)
        components.append(component)
    return components


def _point_inside_surface(point, mesh: Mesh, eps: float):
    """Classify a point by oriented solid angle, raising for ambiguous cases."""
    angles = []
    for face in mesh.faces:
        p0, p1, p2 = (mesh.vertices[i] for i in face)
        ab, ac = sub(p1, p0), sub(p2, p0)
        try:
            normal = _triangle_unit_normal(p0, p1, p2)
        except ValueError as error:
            raise RedBlueNumericalEdgeCase("cannot classify against a degenerate or unrepresentable face") from error
        face_scale = max(abs(value) for edge in (ab, ac) for value in edge)
        # A point well outside the expanded face box cannot touch this face.
        # Per-coordinate ulps avoid a large translation on one axis inflating
        # the tolerance on unrelated axes.
        near_box = True
        for axis in range(3):
            coordinates = (p0[axis], p1[axis], p2[axis])
            padding = 8*eps*face_scale + 8*max(ulp(point[axis]), *(ulp(value) for value in coordinates))
            if point[axis] < min(coordinates)-padding or point[axis] > max(coordinates)+padding:
                near_box = False
                break
        if near_box:
            ap = sub(point, p0)
            if not all(isfinite(value) for value in ap):
                raise RedBlueNumericalEdgeCase("point-face coordinate differences exceed float range")
            ab, ac, ap = (tuple(value/face_scale for value in vector) for vector in (ab, ac, ap))
            local_scale = max(hypot(*ab), hypot(*ac))
        if near_box and abs(dot(ap, normal)) <= eps*local_scale:
            aa, ab_ac, cc = dot(ab, ab), dot(ab, ac), dot(ac, ac)
            ap_ab, ap_ac = dot(ap, ab), dot(ap, ac)
            denominator = aa*cc-ab_ac*ab_ac
            if not isfinite(denominator) or denominator <= 0:
                raise RedBlueNumericalEdgeCase("cannot classify point against a nearly degenerate face")
            u = (cc*ap_ab-ab_ac*ap_ac)/denominator
            v = (aa*ap_ac-ab_ac*ap_ab)/denominator
            if u >= -eps and v >= -eps and u+v <= 1+eps:
                raise RedBlueNumericalEdgeCase("component touches or nearly touches the other surface")
        vectors = tuple(sub(mesh.vertices[i], point) for i in face)
        if not all(isfinite(value) for vector in vectors for value in vector):
            vectors = tuple(tuple(mesh.vertices[i][j]*.5-point[j]*.5 for j in range(3)) for i in face)
        scale = max(abs(value) for vector in vectors for value in vector)
        if not scale or not isfinite(scale):
            raise RedBlueNumericalEdgeCase("cannot represent point-face displacement")
        a, b, c = (tuple(value/scale for value in vector) for vector in vectors)
        la, lb, lc = (hypot(*v) for v in (a, b, c))
        if min(la, lb, lc) == 0:
            raise RedBlueNumericalEdgeCase("component touches a vertex of the other surface")
        numerator = dot(a, cross(b, c))
        denominator = la*lb*lc + dot(a, b)*lc + dot(b, c)*la + dot(c, a)*lb
        angles.append(2*atan2(numerator, denominator))
    winding = abs(fsum(angles))/(4*pi)
    # A point on a surface, or a numerically unstable winding, does not give
    # the near-integer result expected for a closed oriented surface.
    if not isfinite(winding):
        raise RedBlueNumericalEdgeCase("non-finite winding while classifying a disconnected component")
    nearest = round(winding)
    if abs(winding-nearest) > 1e-5:
        raise RedBlueNumericalEdgeCase("could not classify a disconnected component against the other surface")
    return nearest != 0


def _excluded_components(mesh: Mesh, other: Mesh, polygon, *, red: bool, keep_outside: bool, eps: float):
    """Classify components untouched by the intersection loop."""
    components = _vertex_components(mesh)
    if len(components) <= 1:
        return set()
    index = {vertex: i for i, component in enumerate(components) for vertex in component}
    affected = {index[hit.edge_out] for hit in polygon if hit.red == red}
    if len(affected) != 1:
        raise RedBlueMultipleLoops("intersection loop spans multiple disconnected components")
    affected_id = next(iter(affected))
    excluded = set()
    for i, component in enumerate(components):
        if i != affected_id:
            inside = _point_inside_surface(mesh.vertices[min(component)], other, eps)
            if inside == keep_outside:
                excluded.update(component)
    return excluded


def _cut_side(mesh: Mesh, polygon: list[IntersectionTuple], *, red: bool, keep_outside: bool,
              mesh_offset: int, seam_offset: int, all_vertices, excluded_vertices=frozenset(), source_rng=None,
              source_delaunay=False, source_delaunay_allow_limit=False):
    """Cut and retain the requested side of the intersection on one mesh."""
    matching = [i for i, point in enumerate(polygon) if point.red == red]
    if len(matching) < 2:
        raise RedBlueAffectedCircleTrivial("surface needs at least two edge crossings")
    seam_id = lambda i: seam_offset+i
    face_patches = []
    first = polygon[matching[0]]
    initial_kept = first.edge_out if keep_outside else first.edge_in
    outside = initial_kept
    for position, current_index in enumerate(matching):
        next_index = matching[(position+1) % len(matching)]
        current, following = polygon[current_index], polygon[next_index]
        between = []
        k = (current_index+1) % len(polygon)
        while k != next_index:
            between.append(k)
            k = (k+1) % len(polygon)
        if outside not in (current.edge_out, current.edge_in):
            raise RedBlueError("outside vertex is inconsistent with the next crossed edge")
        inside = current.edge_in if outside == current.edge_out else current.edge_out
        shared = {current.edge_out, current.edge_in} & {following.edge_out, following.edge_in}
        if len(shared) != 1:
            raise RedBlueError("successive same-colour intersection edges do not share a vertex")
        shared = next(iter(shared))
        shared_faces = set(current.incident_faces) & set(following.incident_faces)
        if len(shared_faces) != 1:
            raise RedBlueError("successive same-colour intersections do not share a face")
        face_id = next(iter(shared_faces))
        old_face = mesh.faces[face_id]
        remaining = set(old_face) - {current.edge_out, current.edge_in}
        if len(remaining) != 1:
            raise RedBlueError("intersection edge is not incident to the expected face")
        remaining = next(iter(remaining))
        if red:
            boundary = [mesh_offset+outside, seam_id(current_index)]
            boundary.extend(seam_id(i) for i in between)
            boundary.append(seam_id(next_index))
            if shared == inside:
                boundary.append(mesh_offset+remaining)
        else:
            boundary = [mesh_offset+outside]
            if shared == inside:
                boundary.append(mesh_offset+remaining)
            boundary.append(seam_id(next_index))
            boundary.extend(seam_id(i) for i in reversed(between))
            boundary.append(seam_id(current_index))
        a, b, c = (mesh.vertices[i] for i in old_face)
        face_patches.extend(_triangulate_polygon(boundary, all_vertices, _triangle_unit_normal(a, b, c),
                                                 basis=sub(b, a), source_rng=source_rng,
                                                 source_delaunay=source_delaunay,
                                                 source_delaunay_allow_limit=source_delaunay_allow_limit))
        if shared == inside:
            outside = remaining
    if outside != initial_kept:
        raise RedBlueError("outside vertex did not close around the intersection")
    # Block the endpoints on the retained side and flood from the opposite
    # side. This handles either side using the source's cut-hole traversal.
    blocked = {(polygon[i].edge_out if keep_outside else polygon[i].edge_in) for i in matching}
    neighbors = [set() for _ in mesh.vertices]
    for a, b, c in mesh.faces:
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    deleted = set()
    stack = [first.edge_in if keep_outside else first.edge_out]
    while stack:
        current = stack.pop()
        if current in blocked or current in deleted:
            continue
        deleted.add(current)
        stack.extend(neighbors[current] - blocked - deleted)
    kept = [tuple(mesh_offset+i for i in face) for face in mesh.faces
            if not any(i in deleted or i in excluded_vertices for i in face)]
    return kept+face_patches


def _source_polygon_options(source_ear_clipping, source_delaunay, random_seed, random_generator,
                            internal_rng, internal_delaunay):
    if (type(source_ear_clipping) is not bool or type(source_delaunay) is not bool or
            type(internal_delaunay) is not bool):
        raise ValueError("source polygon options must be boolean")
    delaunay = source_delaunay or internal_delaunay
    rng = internal_rng
    if rng is None and (source_ear_clipping or delaunay):
        rng = make_random(random_seed, random_generator)
    return rng, delaunay


def redblue_boolean(red_mesh: Mesh, blue_mesh: Mesh, *, operation: str = "union", eps: float = 1e-10,
                    return_vertex_maps: bool = False, source_ear_clipping: bool = False,
                    source_delaunay: bool = False, source_delaunay_allow_limit: bool = False,
                    random_seed: int = 0, random_generator: str = "python", _cache=None,
                    _allow_red_boundary: bool = False, _source_ear_rng=None,
                    _source_delaunay: bool = False) -> Mesh | RedBlueUnionResult:
    """Combine two transversely intersecting, outward-oriented closed meshes.

    ``operation`` is ``union``, ``intersection``, or ``difference`` (red
    minus blue). This implements the red-blue single-circle path. Complex
    edges, tangencies, multiple loops, containment, and the source's automatic
    retries/subdivision are not handled here; they raise RedBlueError. The
    planar polygons use ear clipping followed by constrained Delaunay flips.
    Source-style ear clipping and Delaunay cleanup are optional comparison
    modes; the latter implies the former. The selected RNG is local to the call.
    ``source_delaunay_allow_limit`` also implies the source modes and permits
    the current triangles at the source iteration limit, with a warning and
    patch checks; it does not establish that the result is Delaunay.
    """
    if operation not in ("union", "intersection", "difference"):
        raise ValueError("operation must be union, intersection, or difference")
    if type(source_delaunay_allow_limit) is not bool or type(source_delaunay) is not bool:
        raise ValueError("source Delaunay options must be boolean")
    source_delaunay = source_delaunay or source_delaunay_allow_limit
    _source_ear_rng, _source_delaunay = _source_polygon_options(
        source_ear_clipping, source_delaunay, random_seed, random_generator, _source_ear_rng, _source_delaunay)
    if _allow_red_boundary and operation != "union":
        raise ValueError("open red meshes are only supported for experimental union")
    red_boundary = set()
    if _allow_red_boundary:
        red_boundary = {edge for edge, faces in _edge_faces(red_mesh, allow_boundary=True).items() if len(faces) == 1}
        if red_boundary and len(_vertex_components(blue_mesh)) != 1:
            raise RedBlueBoundaryInteraction("cannot classify disconnected blue components against an open red mesh")
    try:
        red_circle, blue_circle = intersection_face_circles(red_mesh, blue_mesh, eps=eps, _cache=_cache,
                                                             _allow_red_boundary=_allow_red_boundary)
        if _cache is not None:
            # The stitched result can be large; indices are no longer needed.
            _cache.clear()
        polygon = zip_intersection_circles(red_circle, blue_circle)
        red_keep_outside = operation != "intersection"
        blue_keep_outside = operation == "union"
        red_excluded = _excluded_components(red_mesh, blue_mesh, polygon, red=True,
                                            keep_outside=red_keep_outside, eps=eps)
        blue_excluded = _excluded_components(blue_mesh, red_mesh, polygon, red=False,
                                             keep_outside=blue_keep_outside, eps=eps)
    except (RedBlueNumericalEdgeCase, RedBlueMultipleLoops) as error:
        # No cutting has taken place during tuple processing or component
        # classification. Retain earlier red repairs when retrying.
        error._red_intact = True
        raise
    blue_offset = len(red_mesh.vertices)
    seam_offset = blue_offset+len(blue_mesh.vertices)
    vertices = list(red_mesh.vertices)+list(blue_mesh.vertices)+[t.position for t in polygon]
    red_faces = _cut_side(red_mesh, polygon, red=True, keep_outside=red_keep_outside,
                          mesh_offset=0, seam_offset=seam_offset, all_vertices=vertices, excluded_vertices=red_excluded,
                          source_rng=_source_ear_rng, source_delaunay=_source_delaunay,
                          source_delaunay_allow_limit=source_delaunay_allow_limit)
    blue_faces = _cut_side(blue_mesh, polygon, red=False, keep_outside=blue_keep_outside,
                           mesh_offset=blue_offset, seam_offset=seam_offset, all_vertices=vertices, excluded_vertices=blue_excluded,
                           source_rng=_source_ear_rng, source_delaunay=_source_delaunay,
                           source_delaunay_allow_limit=source_delaunay_allow_limit)
    if operation == "difference":
        # The retained blue portion bounds a cavity, so its normal points
        # into the blue solid rather than out of it.
        blue_faces = [(a, c, b) for a, b, c in blue_faces]
    faces = red_faces + blue_faces
    used = sorted({v for face in faces for v in face})
    index = {v: i for i, v in enumerate(used)}
    result = Mesh(tuple(vertices[i] for i in used), tuple(tuple(index[i] for i in face) for face in faces))
    try:
        result_edges = _edge_faces(result, allow_boundary=_allow_red_boundary)
    except ValueError as error:
        raise RedBlueError(f"invalid stitched surface: {error}") from error
    if _allow_red_boundary:
        if any(a not in index or b not in index for a, b in red_boundary):
            raise RedBlueBoundaryInteraction("union removed a flushed boundary vertex")
        expected = {tuple(sorted((index[a], index[b]))) for a, b in red_boundary}
        actual = {edge for edge, adjacent in result_edges.items() if len(adjacent) == 1}
        if actual != expected:
            raise RedBlueBoundaryInteraction("union changed the flushed boundary")
    if return_vertex_maps:
        return RedBlueUnionResult(result, tuple(index.get(i) for i in range(len(red_mesh.vertices))),
                                  tuple(index.get(blue_offset+i) for i in range(len(blue_mesh.vertices))))
    return result


def redblue_union(red_mesh: Mesh, blue_mesh: Mesh, *, eps: float = 1e-10,
                  return_vertex_maps: bool = False, source_ear_clipping: bool = False,
                  source_delaunay: bool = False, source_delaunay_allow_limit: bool = False, random_seed: int = 0,
                  random_generator: str = "python") -> Mesh | RedBlueUnionResult:
    """Construct the union using the single-circle red-blue algorithm."""
    return redblue_boolean(red_mesh, blue_mesh, operation="union", eps=eps, return_vertex_maps=return_vertex_maps,
                           source_ear_clipping=source_ear_clipping, source_delaunay=source_delaunay,
                           source_delaunay_allow_limit=source_delaunay_allow_limit,
                           random_seed=random_seed, random_generator=random_generator)


def redblue_intersection(red_mesh: Mesh, blue_mesh: Mesh, *, eps: float = 1e-10,
                         return_vertex_maps: bool = False, source_ear_clipping: bool = False,
                         source_delaunay: bool = False, source_delaunay_allow_limit: bool = False, random_seed: int = 0,
                         random_generator: str = "python") -> Mesh | RedBlueUnionResult:
    """Construct the common volume using the single-circle red-blue algorithm."""
    return redblue_boolean(red_mesh, blue_mesh, operation="intersection", eps=eps, return_vertex_maps=return_vertex_maps,
                           source_ear_clipping=source_ear_clipping, source_delaunay=source_delaunay,
                           source_delaunay_allow_limit=source_delaunay_allow_limit,
                           random_seed=random_seed, random_generator=random_generator)


def redblue_difference(red_mesh: Mesh, blue_mesh: Mesh, *, eps: float = 1e-10,
                       return_vertex_maps: bool = False, source_ear_clipping: bool = False,
                       source_delaunay: bool = False, source_delaunay_allow_limit: bool = False, random_seed: int = 0,
                       random_generator: str = "python") -> Mesh | RedBlueUnionResult:
    """Subtract blue from red using the single-circle red-blue algorithm."""
    return redblue_boolean(red_mesh, blue_mesh, operation="difference", eps=eps, return_vertex_maps=return_vertex_maps,
                           source_ear_clipping=source_ear_clipping, source_delaunay=source_delaunay,
                           source_delaunay_allow_limit=source_delaunay_allow_limit,
                           random_seed=random_seed, random_generator=random_generator)


def _split_edge_at_parameters(mesh: Mesh, edge: tuple[int, int], parameters) -> Mesh:
    """Split an edge at sorted fractions, replacing its two incident faces."""
    u, v = edge
    values = sorted(parameters)
    if not values or any(not 0 < t < 1 for t in values) or any(a >= b for a, b in zip(values, values[1:])):
        raise ValueError("edge split parameters must be distinct and inside (0,1)")
    vertices = list(mesh.vertices)
    new_ids = []
    for t in values:
        new_ids.append(len(vertices))
        vertices.append(tuple(a+t*(b-a) for a, b in zip(mesh.vertices[u], mesh.vertices[v])))
    split_points = [vertices[u]] + [vertices[i] for i in new_ids] + [vertices[v]]
    if (any(not all(isfinite(value) for value in point) for point in split_points) or
            any(a == b for a, b in zip(split_points, split_points[1:]))):
        raise RedBlueNumericalEdgeCase("edge split points are not distinct finite coordinates")
    faces, count = [], 0
    for face in mesh.faces:
        found = False
        for i in range(3):
            a, b, c = face[i], face[(i+1)%3], face[(i+2)%3]
            if {a, b} == {u, v}:
                points = [a] + (new_ids if a == u else list(reversed(new_ids))) + [b]
                children = [(x, y, c) for x, y in zip(points, points[1:])]
                _validate_split_faces(vertices, face, children)
                faces.extend(children)
                count += 1
                found = True
                break
        if not found:
            faces.append(face)
    if count != 2:
        raise RedBlueError("complex edge no longer belongs to exactly two faces")
    return Mesh(tuple(vertices), tuple(faces))


def _split_face_at_point(mesh: Mesh, face_id: int, position) -> Mesh:
    vertices = list(mesh.vertices)
    new_id = len(vertices)
    vertices.append(tuple(position))
    a, b, c = mesh.faces[face_id]
    children = ((a, b, new_id), (b, c, new_id), (c, a, new_id))
    _validate_split_faces(vertices, (a, b, c), children)
    faces = [face for i, face in enumerate(mesh.faces) if i != face_id]
    faces.extend(children)
    return Mesh(tuple(vertices), tuple(faces))


def _validate_split_faces(vertices, parent, children):
    """Reject rounded split points that collapse or reverse a child face."""
    try:
        original_normal = _triangle_unit_normal(*(vertices[i] for i in parent))
        for face in children:
            normal = _triangle_unit_normal(*(vertices[i] for i in face))
            if dot(normal, original_normal) <= 0:
                raise ValueError("split child reverses its parent face")
    except (ValueError, OverflowError) as error:
        raise RedBlueNumericalEdgeCase("split produced a degenerate or reversed face") from error


def redblue_boolean_with_retries(red_mesh: Mesh, blue_mesh: Mesh, *, operation: str = "union", eps: float = 1e-10,
                                 max_attempts: int = 8, max_growth_factor: float = 2.0,
                                 return_vertex_maps: bool = False, source_ear_clipping: bool = False,
                                 source_delaunay: bool = False, source_delaunay_allow_limit: bool = False,
                                 random_seed: int = 0,
                                 random_generator: str = "python", _cache=None,
                                 _allow_red_boundary: bool = False, _source_ear_rng=None,
                                 _source_delaunay: bool = False) -> Mesh | RedBlueUnionResult:
    """Retry a red-blue operation after splitting complex edges or trivial faces.

    Follows the source's inner-loop repair strategy. Numerical degeneracies,
    disjoint/contained components, multiple loops, and growth-limit failures
    still raise; the source's outer radius/randomization loop is not here.
    Source-style polygon options use one seeded random stream across retries.
    """
    if max_attempts < 1 or not isfinite(max_growth_factor) or max_growth_factor <= 0:
        raise ValueError("max_attempts and max_growth_factor must be positive and finite")
    if type(source_delaunay_allow_limit) is not bool or type(source_delaunay) is not bool:
        raise ValueError("source Delaunay options must be boolean")
    source_delaunay = source_delaunay or source_delaunay_allow_limit
    _source_ear_rng, _source_delaunay = _source_polygon_options(
        source_ear_clipping, source_delaunay, random_seed, random_generator, _source_ear_rng, _source_delaunay)
    red, blue = red_mesh, blue_mesh
    cache = _cache if _cache is not None else _RedBlueCache(allow_red_boundary=_allow_red_boundary)
    initial_count = None
    for attempt in range(1, max_attempts+1):
        try:
            result = redblue_boolean(red, blue, operation=operation, eps=eps,
                                     return_vertex_maps=return_vertex_maps, _cache=cache,
                                     _allow_red_boundary=_allow_red_boundary, _source_ear_rng=_source_ear_rng,
                                     _source_delaunay=_source_delaunay,
                                     source_delaunay_allow_limit=source_delaunay_allow_limit)
            if return_vertex_maps:
                return RedBlueUnionResult(result.mesh, result.red_vertex_map[:len(red_mesh.vertices)],
                                          result.blue_vertex_map[:len(blue_mesh.vertices)], attempt)
            return result
        except RedBlueComplexEdges as error:
            if initial_count is None:
                initial_count = len(error.edges)
            if len(error.edges) > max_growth_factor*initial_count:
                failure = RedBlueRetryLimit("complex-edge count exceeded the retry growth limit")
                failure._source_last_iteration = attempt == max_attempts
                raise failure from error
            for item in error.edges:
                hits = sorted(item.parameters)
                cuts = [(a+b)/2 for a, b in zip(hits, hits[1:])]
                if item.red:
                    red = _split_edge_at_parameters(red, item.edge, cuts)
                else:
                    blue = _split_edge_at_parameters(blue, item.edge, cuts)
        except RedBlueAffectedCircleTrivial as error:
            if error.red is None or error.face is None or error.position is None:
                raise
            if error.red:
                red = _split_face_at_point(red, error.face, error.position)
            else:
                blue = _split_face_at_point(blue, error.face, error.position)
        except (RedBlueNumericalEdgeCase, RedBlueMultipleLoops) as error:
            error._source_last_iteration = attempt == max_attempts
            if getattr(error, "_red_intact", False):
                # Splits append vertices without renumbering the input. This
                # mesh is safe for the assembly's next outer attempt.
                error._retry_red = red
            raise
    raise RedBlueRetryLimit("red-blue operation exceeded its retry limit")


def redblue_union_with_retries(red_mesh: Mesh, blue_mesh: Mesh, *, eps: float = 1e-10,
                               max_attempts: int = 8, max_growth_factor: float = 2.0,
                               return_vertex_maps: bool = False, source_ear_clipping: bool = False,
                               source_delaunay: bool = False, source_delaunay_allow_limit: bool = False,
                               random_seed: int = 0,
                               random_generator: str = "python", _cache=None,
                               _allow_red_boundary: bool = False, _source_ear_rng=None,
                               _source_delaunay: bool = False) -> Mesh | RedBlueUnionResult:
    """Retry red-blue union after subdividing complex edges or affected faces."""
    return redblue_boolean_with_retries(red_mesh, blue_mesh, operation="union", eps=eps,
                                        max_attempts=max_attempts, max_growth_factor=max_growth_factor,
                                        return_vertex_maps=return_vertex_maps, source_ear_clipping=source_ear_clipping,
                                        source_delaunay=source_delaunay, source_delaunay_allow_limit=source_delaunay_allow_limit,
                                        random_seed=random_seed,
                                        random_generator=random_generator, _cache=_cache,
                                        _allow_red_boundary=_allow_red_boundary, _source_ear_rng=_source_ear_rng,
                                        _source_delaunay=_source_delaunay)
