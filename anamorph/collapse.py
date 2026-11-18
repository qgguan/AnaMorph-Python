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

"""Experimental greedy edge-collapse cleanup for triangular meshes.

Based on AnaMorph (https://github.com/NeuroBox3D/AnaMorph). This is a
conservative Python reimplementation, not a bit-for-bit port of the C++ code.
"""
from __future__ import annotations

import heapq
from collections import deque
from math import dist, isfinite, sqrt, ulp
from .mesh import Mesh, cross, dot, sub


def _validate_collapse_options(alpha: float = 1.5, area_floor: float = 0.125,
                               area_ceiling: float = 0.5, depth: int = 5,
                               max_collapses: int = 100000, safety: str = "safe"):
    if not all(isfinite(v) for v in (alpha, area_floor, area_ceiling)) or alpha < 1 or area_floor <= 0 or area_ceiling <= 0:
        raise ValueError("edge collapse requires alpha >= 1 and positive finite area factors")
    if (type(depth) is not int or type(max_collapses) is not int or
            depth < 0 or depth > 20 or max_collapses < 0):
        raise ValueError("invalid edge-collapse depth or limit")
    if safety not in ("safe", "source"):
        raise ValueError("edge-collapse safety must be 'safe' or 'source'")


class _SourcePriorityQueue:
    """Min-heap with the source's reusable insertion IDs for key ties."""

    def __init__(self):
        self._heap = []
        self._active = {}
        self._free = []
        self._next = 0
        self._serial = 0

    def __bool__(self):
        return bool(self._active)

    def discard(self, face):
        entry = self._active.pop(face, None)
        if entry is not None:
            heapq.heappush(self._free, entry[0])

    def insert(self, face, priority):
        self.discard(face)
        if self._free:
            slot = heapq.heappop(self._free)
        else:
            slot, self._next = self._next, self._next+1
        self._serial += 1
        self._active[face] = (slot, self._serial)
        heapq.heappush(self._heap, (priority, slot, self._serial, face))

    def change_key(self, face, priority):
        """Change a queued key without changing the source tie-break ID."""
        entry = self._active.get(face)
        if entry is None:
            return False
        slot = entry[0]
        self._serial += 1
        self._active[face] = (slot, self._serial)
        heapq.heappush(self._heap, (priority, slot, self._serial, face))
        return True

    def pop(self):
        while self._heap:
            priority, slot, serial, face = heapq.heappop(self._heap)
            if self._active.get(face) == (slot, serial):
                del self._active[face]
                heapq.heappush(self._free, slot)
                return priority, face
        raise IndexError("pop from empty priority queue")


def _source_face_neighborhood(face_id, faces, edge_faces, depth):
    """Visit the initial face and its edge-neighbors in the source BFS order."""
    seen = {face_id}
    queue = deque(((face_id, 0),))
    ordered = []
    while queue:
        current, current_depth = queue.popleft()
        ordered.append(current)
        # The source expands depth == limit, including one further face ring.
        if current_depth <= depth:
            face = faces[current]
            for a, b in zip(face, face[1:]+face[:1]):
                edge = (a, b) if a < b else (b, a)
                for neighbor in sorted(edge_faces[edge]):
                    if neighbor not in seen:
                        seen.add(neighbor)
                        queue.append((neighbor, current_depth+1))
    return ordered


def _source_neighborhood_average(face_id, neighborhood, metrics, alpha):
    """Accumulate acceptable triangle areas sequentially, as in C++."""
    total, count = 0., 0
    for neighbor in neighborhood:
        aspect, area = metrics[neighbor]
        if aspect < alpha:
            total += area
            count += 1
    return total/count if count else metrics[face_id][1]


def greedy_edge_collapse(mesh: Mesh, alpha: float = 1.5, area_floor: float = 0.125,
                         area_ceiling: float = 0.5, depth: int = 5,
                         max_collapses: int = 100000, *, safety: str = "safe") -> Mesh:
    """Reduce small/poor triangles with topological and local orientation checks.

    Uses an original-neighborhood area reference and C++-style processing
    predicate. The default adds local orientation and duplicate-face checks.
    ``safety='source'`` uses the source's topological rule without those extra
    geometric checks, and can create inverted or coincident triangles. Neither
    mode is a global self-intersection test, and other numerical differences
    can still produce different output from C++.
    """
    _validate_collapse_options(alpha, area_floor, area_ceiling, depth, max_collapses, safety)
    vertices = list(mesh.vertices)
    faces = list(mesh.faces)
    vertex_faces = [set() for _ in vertices]
    edges = {}
    def edge_key(a, b):
        return (a, b) if a < b else (b, a)
    def face_edges(f):
        a, b, c = f
        return (edge_key(a, b), edge_key(b, c), edge_key(c, a))
    def attach(fid, face):
        for v in face:
            vertex_faces[v].add(fid)
        for e in face_edges(face):
            edges.setdefault(e, set()).add(fid)
    def detach(fid, face):
        for v in face:
            vertex_faces[v].discard(fid)
        for e in face_edges(face):
            incident = edges[e]
            incident.remove(fid)
            if not incident:
                del edges[e]
    def source_length(a, b):
        delta = sub(a, b)
        squared = dot(delta, delta)
        # The original uses sqrt(x*x + y*y + z*z). Keep hypot's scaling
        # fallback for lengths outside the squared floating-point range.
        return sqrt(squared) if squared > 0 and isfinite(squared) else dist(a, b)
    def metrics(face):
        a, b, c = (vertices[i] for i in face)
        normal = cross(sub(b, a), sub(c, a))
        area = sqrt(dot(normal, normal))/2
        if area == 0:
            raise ValueError("mesh contains a degenerate triangle")
        sides = (source_length(a, b), source_length(b, c), source_length(c, a))
        semiperimeter = .5*(sides[0]+sides[1]+sides[2])
        numerator = sides[0]*sides[1]*sides[2]
        factors = tuple(semiperimeter-side for side in sides)
        denominator = 8.*factors[0]*factors[1]*factors[2]
        if (denominator > 0 and isfinite(numerator) and isfinite(denominator) and
                min(factors) > 32*ulp(semiperimeter)):
            aspect = numerator/denominator
        else:
            # Heron's factors can collapse to zero for a very thin triangle.
            aspect = numerator*sum(sides)/(16*area*area)
        return aspect, area
    for fid, face in enumerate(faces):
        if len(set(face)) != 3 or any(i < 0 or i >= len(vertices) for i in face):
            raise ValueError("mesh contains an invalid triangle")
        attach(fid, face)
    initial = [metrics(f) for f in faces]
    references = []
    for fid in range(len(faces)):
        neighborhood = _source_face_neighborhood(fid, faces, edges, depth)
        references.append(_source_neighborhood_average(fid, neighborhood, initial, alpha))
    queue = _SourcePriorityQueue()
    def enqueue(fid):
        # The source removes an old entry before testing/reinserting a face.
        queue.discard(fid)
        face = faces[fid]
        if face is None:
            return
        ar, area = metrics(face)
        reference = references[fid]
        if area < area_ceiling*reference and (ar >= alpha or area < area_floor*reference):
            queue.insert(fid, -ar)
    for fid in range(len(faces)):
        enqueue(fid)
    count = 0
    unsafe = []
    previous_unsafe = []
    while count < max_collapses:
        if not queue:
            # Match the original fixed-point retry of edges that could not be
            # collapsed. A neighboring collapse can change their topology.
            unsafe.sort()
            if unsafe == previous_unsafe:
                break
            previous_unsafe, unsafe = unsafe, []
            for postponed in previous_unsafe:
                enqueue(postponed)
            continue
        _, fid = queue.pop()
        if faces[fid] is None:
            continue
        face = faces[fid]
        ar, area = metrics(face)
        ref = references[fid]
        if not (area < area_ceiling*ref and (ar >= alpha or area < area_floor*ref)):
            continue
        u, v = min(face_edges(face), key=lambda e: source_length(vertices[e[0]], vertices[e[1]]))
        incident = edges[(u, v)]
        if len(incident) != 2:
            unsafe.append(fid)
            continue
        neighbors_u = {w for f in vertex_faces[u] for w in faces[f] if w != u}
        neighbors_v = {w for f in vertex_faces[v] for w in faces[f] if w != v}
        opposite = {w for f in incident for w in faces[f] if w != u and w != v}
        shared = neighbors_u & neighbors_v
        topology_safe = len(shared) == 2 if safety == "source" else len(opposite) == 2 and shared == opposite
        if not topology_safe:
            unsafe.append(fid)
            continue
        target = tuple((a+b)/2 for a, b in zip(vertices[u], vertices[v]))
        affected = vertex_faces[u] | vertex_faces[v]
        safe = True
        if safety == "safe":
            proposed = set()
            for f in affected - incident:
                old_face = faces[f]
                new_face = tuple(u if i == v else i for i in old_face)
                key = tuple(sorted(new_face))
                if key in proposed:
                    safe = False
                    break
                proposed.add(key)
                old_points = [vertices[i] for i in old_face]
                new_points = [target if i == u or i == v else vertices[i] for i in old_face]
                old_n = cross(sub(old_points[1], old_points[0]), sub(old_points[2], old_points[0]))
                new_n = cross(sub(new_points[1], new_points[0]), sub(new_points[2], new_points[0]))
                if dot(old_n, new_n) <= 0 or dot(new_n, new_n) <= 1e-24*dot(old_n, old_n):
                    safe = False
                    break
        if not safe:
            unsafe.append(fid)
            continue
        removed = set(incident)
        for f in affected:
            detach(f, faces[f])
        vertices[u] = target
        for f in affected:
            if f in removed:
                faces[f] = None
            else:
                faces[f] = tuple(u if i == v else i for i in faces[f])
                attach(f, faces[f])
        count += 1
        for f in sorted(affected - removed):
            enqueue(f)
        for postponed in unsafe:
            enqueue(postponed)
        unsafe.clear()
    active_faces = [f for f in faces if f is not None]
    used = sorted({i for f in active_faces for i in f})
    new_index = {old: i for i, old in enumerate(used)}
    return Mesh(tuple(vertices[i] for i in used), tuple(tuple(new_index[i] for i in f) for f in active_faces))
