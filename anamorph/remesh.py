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

"""Experimental isotropic remeshing of a closed triangular surface.

Based on AnaMorph (https://github.com/NeuroBox3D/AnaMorph). This is a new
Python algorithm, not the red-blue mesher or cleanup from the C++ version.
"""
from __future__ import annotations

import heapq
from math import dist, isfinite, sqrt

from .mesh import Mesh, cross, dot, sub
from .collapse import greedy_edge_collapse


def _key(a, b):
    return (a, b) if a < b else (b, a)


class _Topology:
    def __init__(self, mesh):
        self.vertices = list(mesh.vertices)
        self.faces = {}
        self.vertex_faces = [set() for _ in self.vertices]
        self.edges = {}
        self.next_face = 0
        for face in mesh.faces:
            if len(set(face)) != 3 or any(i < 0 or i >= len(self.vertices) for i in face):
                raise ValueError("mesh contains an invalid triangle")
            self.add(face)
        if not self.faces or any(len(faces) != 2 for faces in self.edges.values()):
            raise ValueError("remeshing requires a closed triangular edge-manifold mesh")

    def add(self, face):
        fid = self.next_face
        self.next_face += 1
        self.faces[fid] = face
        for vertex in face:
            self.vertex_faces[vertex].add(fid)
        for a, b in zip(face, face[1:] + face[:1]):
            self.edges.setdefault(_key(a, b), set()).add(fid)
        return fid

    def remove(self, fid):
        face = self.faces.pop(fid)
        for vertex in face:
            self.vertex_faces[vertex].remove(fid)
        for a, b in zip(face, face[1:] + face[:1]):
            edge = _key(a, b)
            self.edges[edge].remove(fid)
            if not self.edges[edge]:
                del self.edges[edge]
        return face

    def length(self, edge):
        return dist(self.vertices[edge[0]], self.vertices[edge[1]])

    def normal(self, face, replacement=None):
        p = [replacement[1] if replacement and x in replacement[0] else self.vertices[x] for x in face]
        return cross(sub(p[1], p[0]), sub(p[2], p[0]))

    def split(self, edge):
        if edge not in self.edges or len(self.edges[edge]) != 2:
            return None
        u, v = edge
        midpoint = tuple((a+b)/2 for a, b in zip(self.vertices[u], self.vertices[v]))
        m = len(self.vertices)
        self.vertices.append(midpoint)
        self.vertex_faces.append(set())
        old_faces = [self.remove(fid) for fid in tuple(self.edges[edge])]
        for face in old_faces:
            for i in range(3):
                a, b, c = face[i], face[(i+1)%3], face[(i+2)%3]
                if {a, b} == {u, v}:
                    self.add((a, m, c))
                    self.add((m, b, c))
                    break
        return m

    def collapse(self, edge, target):
        if edge not in self.edges or len(self.edges[edge]) != 2:
            return False
        u, v = edge
        incident = set(self.edges[edge])
        nu = {w for f in self.vertex_faces[u] for w in self.faces[f] if w != u}
        nv = {w for f in self.vertex_faces[v] for w in self.faces[f] if w != v}
        opposite = {w for f in incident for w in self.faces[f] if w not in edge}
        if len(opposite) != 2 or nu & nv != opposite:
            return False
        affected = self.vertex_faces[u] | self.vertex_faces[v]
        replacements, keys = [], set()
        for fid in affected - incident:
            face = self.faces[fid]
            new_face = tuple(u if x == v else x for x in face)
            canonical = tuple(sorted(new_face))
            if canonical in keys:
                return False
            keys.add(canonical)
            replacements.append(new_face)
        surviving_faces = [self.faces[fid] for fid in affected - incident]
        def triangle_quality(positions, normal):
            denominator = sum(dist(positions[i], positions[(i+1)%3])**2 for i in range(3))
            return 2*sqrt(3)*sqrt(dot(normal, normal))/denominator if denominator else 0.
        def evaluate_position(position):
            if any(dist(position, self.vertices[w]) > (4/3)*target*(1+1e-10)
                   for w in (nu | nv) - {u, v}):
                return None
            quality = 1.
            for face in surviving_faces:
                old_n = self.normal(face)
                new_n = self.normal(face, ({u, v}, position))
                if dot(old_n, new_n) <= 0 or dot(new_n, new_n) <= 1e-20*dot(old_n, old_n):
                    return None
                positions = [position if x in (u, v) else self.vertices[x] for x in face]
                quality = min(quality, triangle_quality(positions, new_n))
            return quality
        a, b = self.vertices[u], self.vertices[v]
        midpoint = tuple((x+y)/2 for x, y in zip(a, b))
        midpoint_quality = evaluate_position(midpoint)
        position = midpoint
        if midpoint_quality is None or midpoint_quality < .35:
            # A hard absolute floor can trap arbitrarily short edges beside an
            # existing poor triangle. Try positions along the original edge,
            # but only accept a sub-floor result if the worst local triangle
            # does not deteriorate. Link and orientation checks still apply.
            old_quality = min(triangle_quality([self.vertices[x] for x in self.faces[fid]],
                                               self.normal(self.faces[fid])) for fid in affected)
            choices = [(midpoint_quality, midpoint)] if midpoint_quality is not None else []
            for fraction in (0., 1., .25, .75):
                point = tuple((1-fraction)*x+fraction*y for x, y in zip(a, b))
                quality = evaluate_position(point)
                if quality is not None:
                    choices.append((quality, point))
            # Check changed faces against a second topological ring. This is
            # only a local numerical diagnostic: distant faces can still
            # intersect, so it is not a global validity certificate.
            neighbors = {vertex for fid in affected for vertex in self.faces[fid]}
            nearby = {fid for vertex in neighbors for fid in self.vertex_faces[vertex]}
            local_faces = [tuple(u if vertex == v else vertex for vertex in self.faces[fid])
                           for fid in nearby if fid not in incident]
            used = sorted({vertex for face in local_faces for vertex in face})
            index = {vertex: i for i, vertex in enumerate(used)}
            local_connectivity = tuple(tuple(index[vertex] for vertex in face) for face in local_faces)
            from .mesh_intersections import find_self_intersections
            position = None
            for quality, point in sorted(choices, key=lambda item: item[0], reverse=True):
                if quality < .35 and quality < old_quality*(1-1e-12):
                    continue
                local_vertices = tuple(point if vertex == u else self.vertices[vertex] for vertex in used)
                try:
                    hits, _ = find_self_intersections(Mesh(local_vertices, local_connectivity),
                                                      max_hits=1, include_adjacent=True)
                except ValueError:
                    continue
                if not hits:
                    position = point
                    break
            if position is None:
                return False
        for fid in tuple(affected):
            self.remove(fid)
        self.vertices[u] = position
        for face in replacements:
            self.add(face)
        return True

    def flip(self, edge, target):
        if edge not in self.edges or len(self.edges[edge]) != 2:
            return False
        u, v = edge
        f1, f2 = tuple(self.edges[edge])
        a_face, b_face = self.faces[f1], self.faces[f2]
        # Locate u -> v in the first face; the second must traverse v -> u.
        for i in range(3):
            if a_face[i] == u and a_face[(i+1)%3] == v:
                break
            if a_face[i] == v and a_face[(i+1)%3] == u:
                u, v = v, u
                break
        a = next(x for x in a_face if x != u and x != v)
        b = next(x for x in b_face if x != u and x != v)
        if not any(b_face[i] == v and b_face[(i+1)%3] == u for i in range(3)):
            return False
        if a == b or _key(a, b) in self.edges:
            return False
        new_length = dist(self.vertices[a], self.vertices[b])
        if not .6*target <= new_length <= (4/3)*target:
            return False
        new_faces = ((a, b, v), (b, a, u))
        old_normals = (self.normal(a_face), self.normal(b_face))
        for face in new_faces:
            n = self.normal(face)
            if dot(n, n) == 0 or any(dot(n, old) <= 0 for old in old_normals):
                return False
        def quality(face):
            normal = self.normal(face)
            denom = sum(self.length(_key(x, y))**2 for x, y in zip(face, face[1:] + face[:1]))
            return 2*sqrt(3)*sqrt(dot(normal, normal))/denom
        before = min(quality(a_face), quality(b_face))
        after = min(quality(new_faces[0]), quality(new_faces[1]))
        if after <= before*1.02:
            return False
        self.remove(f1)
        self.remove(f2)
        for face in new_faces:
            self.add(face)
        return True

    def finish(self):
        faces = list(self.faces.values())
        used = sorted({v for f in faces for v in f})
        index = {v: i for i, v in enumerate(used)}
        return Mesh(tuple(self.vertices[v] for v in used), tuple(tuple(index[v] for v in f) for f in faces))


def isotropic_remesh(mesh: Mesh, target: float, *, iterations: int = 5, max_operations: int = 300000) -> Mesh:
    """Improve uniformity via long-edge splits, short-edge collapses and flips.

    When a midpoint collapse fails local quality or orientation checks, the
    algorithm can try other positions on that edge. Those fallback positions
    must preserve local worst-triangle quality or clear the quality floor and
    pass a numerical self-intersection check in a two-ring neighborhood.
    The algorithm changes the surface and does not project it back to the
    original geometry or test global self-intersections. Edge lengths and
    triangle areas cannot be guaranteed to fall within strict bounds.
    """
    if not isfinite(target) or target <= 0 or iterations < 1 or max_operations < 1:
        raise ValueError("target and operation limit must be positive; iterations must be at least one")
    # Voxel surfaces can contain nearly coincident vertices. A preliminary
    # poor-triangle cleanup lets the stricter isotropic collapse rules work
    # without relaxing their normal, area, and maximum-edge safeguards.
    _Topology(mesh)  # validate the input before the cleanup mutates anything
    mesh = greedy_edge_collapse(mesh, depth=2, max_collapses=max_operations)
    topology = _Topology(mesh)
    operations = 0
    for _ in range(iterations):
        # Iterate over current edges in descending size; revisiting new edges
        # lets very long input edges be split more than once in one pass.
        long_edges = [(-topology.length(e), e) for e in topology.edges]
        heapq.heapify(long_edges)
        while long_edges:
            neg, edge = heapq.heappop(long_edges)
            if edge not in topology.edges:
                continue
            length = topology.length(edge)
            if abs(length+neg) > 1e-10*max(target, length):
                continue
            if length <= 4*target/3:
                break
            mid = topology.split(edge)
            if mid is not None:
                operations += 1
                for fid in topology.vertex_faces[mid]:
                    for v in topology.faces[fid]:
                        if v != mid:
                            e = _key(mid, v)
                            heapq.heappush(long_edges, (-topology.length(e), e))
            if operations > max_operations:
                raise ValueError("remeshing operation limit exceeded")
        short_edges = [(topology.length(e), e) for e in topology.edges]
        heapq.heapify(short_edges)
        while short_edges:
            old_length, edge = heapq.heappop(short_edges)
            if edge not in topology.edges:
                continue
            length = topology.length(edge)
            if abs(length-old_length) > 1e-10*max(target, length):
                continue
            if length >= 4*target/5:
                break
            if topology.collapse(edge, target):
                operations += 1
                u = edge[0]
                for fid in topology.vertex_faces[u]:
                    for v in topology.faces[fid]:
                        if v != u:
                            e = _key(u, v)
                            heapq.heappush(short_edges, (topology.length(e), e))
            if operations > max_operations:
                raise ValueError("remeshing operation limit exceeded")
        for edge in tuple(topology.edges):
            if topology.flip(edge, target):
                operations += 1
            if operations > max_operations:
                raise ValueError("remeshing operation limit exceeded")
    return topology.finish()
