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

"""Incremental OBJ face output with persistent boundary-vertex indices.

This is an internal building block for AnaMorph-style partial mesh flushing.
The caller selects faces for flushing; neuron assembly uses it with a
conservative selection and boundary checks.
"""
from __future__ import annotations

from math import isfinite
from pathlib import Path
import os
import shutil
import tempfile

from .mesh import Mesh


def _validate_mesh(mesh):
    # Persistent IDs depend on the input not changing behind our back.
    if (type(mesh.vertices) is not tuple or type(mesh.faces) is not tuple or
            any(type(v) is not tuple or len(v) != 3 or not all(isfinite(x) for x in v)
                for v in mesh.vertices) or
            any(type(f) is not tuple or len(f) != 3 or len(set(f)) != 3 or
                any(type(i) is not int or i < 0 or i >= len(mesh.vertices) for i in f)
                for f in mesh.faces)):
        raise ValueError("partial OBJ output requires an immutable, finite triangular mesh")


def _oriented_face(face):
    """Canonicalize cyclic rotations, retaining the orientation."""
    return min(face, face[1:]+face[:1], face[2:]+face[:2])


class PartialObjWriter:
    """Write selected faces to scratch files and retain shared vertex IDs.

    The active mesh may have a boundary after a flush. ``replace_mesh``
    accepts a vertex map after a mesh operation and checks that the complete
    one-ring of each already-written vertex was preserved. The caller must
    still ensure that removed faces cannot intersect any future operation.
    """

    def __init__(self, mesh: Mesh):
        _validate_mesh(mesh)
        self.mesh = mesh
        self._vertices = tempfile.TemporaryFile(mode="w+t", encoding="utf-8", newline="\n")
        try:
            self._faces = tempfile.TemporaryFile(mode="w+t", encoding="utf-8", newline="\n")
        except BaseException:
            self._vertices.close()
            raise
        self._written = {}  # active vertex index -> (one-based OBJ index, coordinate)
        self._vertex_count = 0
        self._face_count = 0
        self._closed = False
        self._failed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _require_open(self):
        if self._closed or self._failed:
            raise ValueError("partial OBJ writer is closed or has failed")

    def close(self):
        if not self._closed:
            self._vertices.close()
            self._faces.close()
            self._closed = True

    @property
    def vertex_count(self):
        return self._vertex_count

    @property
    def face_count(self):
        return self._face_count

    def flush(self, face_indices):
        """Flush selected face indices and return the compacted active mesh."""
        self._require_open()
        selected = set(face_indices)
        if any(type(i) is not int or i < 0 or i >= len(self.mesh.faces) for i in selected):
            raise ValueError("invalid face index for partial OBJ output")
        if not selected:
            return self.mesh
        flushed = [face for i, face in enumerate(self.mesh.faces) if i in selected]
        remaining = [face for i, face in enumerate(self.mesh.faces) if i not in selected]
        used = sorted({v for face in remaining for v in face})
        old_to_new = {old: new for new, old in enumerate(used)}
        next_mesh = Mesh(tuple(self.mesh.vertices[i] for i in used),
                         tuple(tuple(old_to_new[i] for i in face) for face in remaining))
        written = dict(self._written)
        vertex_count = self._vertex_count
        new_vertices = []
        for face in flushed:
            for vertex in face:
                if vertex not in written:
                    vertex_count += 1
                    coordinate = self.mesh.vertices[vertex]
                    written[vertex] = (vertex_count, coordinate)
                    new_vertices.append(coordinate)
        try:
            for coordinate in new_vertices:
                self._vertices.write("v {:.17g} {:.17g} {:.17g}\n".format(*coordinate))
            for face in flushed:
                self._faces.write("f {} {} {}\n".format(*(written[i][0] for i in face)))
        except BaseException:
            self._failed = True
            raise
        self._written = {old_to_new[i]: entry for i, entry in written.items() if i in old_to_new}
        self._vertex_count = vertex_count
        self._face_count += len(flushed)
        self.mesh = next_mesh
        return next_mesh

    def replace_mesh(self, mesh: Mesh, old_to_new):
        """Accept an updated active mesh while protecting written boundaries.

        ``old_to_new`` maps each old vertex to its new index or ``None``. A
        previously written vertex and all incident active faces must survive
        unchanged, including orientation; this detects modifications next to a
        flushed face before they could make the eventual OBJ inconsistent.
        """
        self._require_open()
        _validate_mesh(mesh)
        if len(old_to_new) != len(self.mesh.vertices):
            raise ValueError("vertex map length does not match the active mesh")
        boundary_map = {}
        for old, (obj_id, point) in self._written.items():
            new = old_to_new[old]
            if (type(new) is not int or new < 0 or new >= len(mesh.vertices) or
                    mesh.vertices[new] != point or new in boundary_map):
                raise ValueError("mesh operation changed an already-written boundary vertex")
            boundary_map[new] = (obj_id, point)
        old_boundary = set(self._written)
        new_boundary = set(boundary_map)
        old_ring = []
        for face in self.mesh.faces:
            if old_boundary.intersection(face):
                mapped = tuple(old_to_new[i] for i in face)
                if any(type(i) is not int or i < 0 or i >= len(mesh.vertices) for i in mapped):
                    raise ValueError("mesh operation removed a face next to the written boundary")
                old_ring.append(_oriented_face(mapped))
        new_ring = [_oriented_face(face) for face in mesh.faces if new_boundary.intersection(face)]
        if sorted(old_ring) != sorted(new_ring):
            raise ValueError("mesh operation changed a face next to the written boundary")
        self.mesh = mesh
        self._written = boundary_map

    def finalize(self, path):
        """Write all remaining faces and atomically replace the requested OBJ."""
        self._require_open()
        self.flush(range(len(self.mesh.faces)))
        target = Path(path)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                             dir=target.parent, prefix=".anamorph-", suffix=".obj",
                                             delete=False) as output:
                temporary = Path(output.name)
                output.write("# Based on AnaMorph (https://github.com/NeuroBox3D/AnaMorph).\n")
                self._vertices.seek(0)
                shutil.copyfileobj(self._vertices, output)
                self._faces.seek(0)
                shutil.copyfileobj(self._faces, output)
            os.replace(temporary, target)
        except BaseException:
            self._failed = True
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise
        self.close()
