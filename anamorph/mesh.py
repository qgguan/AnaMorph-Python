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

"""Triangular OBJ input and mesh statistics, based on AnaMorph am_meshstat.

Based on AnaMorph (https://github.com/NeuroBox3D/AnaMorph).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import frexp, fsum, hypot, isfinite, ldexp, sqrt, ulp
from pathlib import Path

Vec = tuple[float, float, float]
Face = tuple[int, int, int]


def sub(a: Vec, b: Vec) -> Vec:
    return a[0]-b[0], a[1]-b[1], a[2]-b[2]


def cross(a: Vec, b: Vec) -> Vec:
    return a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]


def dot(a: Vec, b: Vec) -> float:
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def _scaled_product(description, *values):
    """Multiply finite floats without avoidable intermediate over/underflow."""
    if any(not isfinite(value) for value in values):
        raise ValueError(f"{description} is outside float range")
    if any(value == 0 for value in values):
        return 0.
    mantissa, exponent = 1., 0
    for value in values:
        part, power = frexp(value)
        mantissa *= part
        exponent += power
    try:
        result = ldexp(mantissa, exponent)
    except OverflowError as error:
        raise ValueError(f"{description} exceeds float range") from error
    if result == 0:
        raise ValueError(f"{description} underflows float range")
    return result


def _triangle_metrics(a: Vec, b: Vec, c: Vec):
    """Return representable area, dimensionless aspect ratio and obtuse flag."""
    edges = (sub(b, a), sub(c, b), sub(a, c))
    if not all(isfinite(value) for edge in edges for value in edge):
        raise ValueError("triangle coordinate differences exceed float range")
    scale = max(abs(value) for edge in edges for value in edge)
    if not scale:
        raise ValueError("mesh contains a degenerate triangle")
    normalized = tuple(tuple(value/scale for value in edge) for edge in edges)
    if any(original and not value for edge, scaled in zip(edges, normalized)
           for original, value in zip(edge, scaled)):
        raise ValueError("triangle coordinate scale ratio exceeds float range")
    normal = cross(normalized[0], normalized[1])
    normal_length = hypot(*normal)
    if not normal_length:
        raise ValueError("mesh contains a degenerate or numerically invalid triangle")
    area = _scaled_product("triangle area", .5, normal_length, scale, scale)
    sides = tuple(hypot(*edge) for edge in normalized)
    numerator = sides[0]*sides[1]*sides[2]*sum(sides)
    aspect = ((numerator/(4*normal_length))/normal_length)
    if not isfinite(aspect):
        raise ValueError("triangle aspect ratio exceeds float range")
    if aspect <= 0:
        raise ValueError("triangle aspect ratio underflows float range")
    # Retain the original unscaled squared-edge comparison when representable
    # to avoid changing right-angle decisions due to a rounded square root.
    raw_squared = [dot(edge, edge) for edge in edges]
    if all(isfinite(value) and value > 0 for value in raw_squared):
        lengths = sorted(raw_squared)
    else:
        lengths = sorted(dot(edge, edge) for edge in normalized)
    obtuse = lengths[0]+lengths[1] < lengths[2]
    return area, aspect, obtuse


def _exact_tetrahedron_determinant(a, b, c, origin):
    reference = tuple(Fraction(value) for value in origin)
    vectors = tuple(tuple(Fraction(point[i])-reference[i] for i in range(3)) for point in (a, b, c))
    return dot(vectors[0], cross(vectors[1], vectors[2]))


def _mesh_volume(vertices, faces, find):
    """Sum signed component volumes using local, scale-safe coordinates."""
    origins, scales, determinants, component_faces, extreme_components = {}, {}, {}, {}, set()
    for face in faces:
        for index in face:
            component = find(index)
            origin = origins.setdefault(component, vertices[index])
            displacement = sub(vertices[index], origin)
            if not all(isfinite(value) for value in displacement):
                raise ValueError("mesh component coordinate differences exceed float range")
            scales[component] = max(scales.get(component, 0.), *(abs(value) for value in displacement))
    for face in faces:
        a, b, c = (vertices[i] for i in face)
        component = find(face[0])
        origin, scale = origins[component], scales[component]
        displacements = tuple(sub(point, origin) for point in (a, b, c))
        vectors = tuple(tuple(value/scale for value in displacement) for displacement in displacements)
        if any(original and (not value or abs(value) < 1e-100)
               for displacement, vector in zip(displacements, vectors)
               for original, value in zip(displacement, vector)):
            extreme_components.add(component)
        u, v, w = vectors
        terms = (u[0]*v[1]*w[2], -u[0]*v[2]*w[1],
                 u[1]*v[2]*w[0], -u[1]*v[0]*w[2],
                 u[2]*v[0]*w[1], -u[2]*v[1]*w[0])
        determinant = fsum(terms)
        magnitude = fsum(abs(value) for value in terms)
        if magnitude and abs(determinant) <= 32*ulp(1.)*magnitude:
            if _exact_tetrahedron_determinant(a, b, c, origin):
                extreme_components.add(component)
            else:
                determinant = 0.
        determinants.setdefault(component, []).append(determinant/6)
        component_faces.setdefault(component, []).append(face)
    def exact_volume(selected_faces):
        total = Fraction(0)
        for face in selected_faces:
            a, b, c = (vertices[index] for index in face)
            total += _exact_tetrahedron_determinant(a, b, c, origins[find(face[0])])/6
        return total
    def as_float(value):
        try:
            volume = float(value)
        except OverflowError as error:
            raise ValueError("mesh volume exceeds float range") from error
        if value and not volume:
            raise ValueError("mesh volume underflows float range")
        return volume
    volumes = []
    for component, values in determinants.items():
        normalized_volume = fsum(values)
        magnitude = fsum(abs(value) for value in values)
        if magnitude and abs(normalized_volume) <= 32*ulp(1.)*magnitude:
            extreme_components.add(component)
        if component in extreme_components:
            # Unusual anisotropy or cancellation can erase a representable
            # volume; use exact arithmetic on the input floats in that case.
            volumes.append(as_float(exact_volume(component_faces[component])))
        else:
            volumes.append(_scaled_product("mesh volume", normalized_volume,
                                           scales[component], scales[component], scales[component]))
    try:
        signed, magnitude = fsum(volumes), fsum(abs(value) for value in volumes)
    except OverflowError:
        return abs(as_float(exact_volume(faces)))
    if magnitude and abs(signed) <= 32*ulp(1.)*magnitude:
        return abs(as_float(exact_volume(faces)))
    return abs(signed)


@dataclass(frozen=True, slots=True)
class MeshStatistics:
    vertices: int
    edges: int
    faces: int
    euler_characteristic: int
    area: float
    volume: float | None
    aspect_ratio_mean: float
    aspect_ratio_stddev: float
    aspect_ratio_max: float
    obtuse_triangles: int
    boundary_edges: int
    nonmanifold_edges: int


@dataclass(frozen=True, slots=True)
class Mesh:
    vertices: tuple[Vec, ...]
    faces: tuple[Face, ...]

    def write_obj(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
            stream.write("# Based on AnaMorph (https://github.com/NeuroBox3D/AnaMorph).\n")
            for vertex in self.vertices:
                stream.write("v {:.17g} {:.17g} {:.17g}\n".format(*vertex))
            for face in self.faces:
                stream.write("f {} {} {}\n".format(*(i+1 for i in face)))

    def statistics(self, *, include_volume: bool = True) -> MeshStatistics:
        """Return mesh measures; omit volume when only surface quality is needed."""
        if not self.faces:
            raise ValueError("mesh has no triangular faces")
        if any(len(vertex) != 3 or not all(isfinite(value) for value in vertex) for vertex in self.vertices):
            raise ValueError("mesh contains a non-finite or invalid vertex")
        edges: dict[tuple[int, int], int] = {}
        areas, ratios = [], []
        obtuse = 0
        parent = list(range(len(self.vertices)))
        rank = [0]*len(parent)
        def find(vertex):
            while parent[vertex] != vertex:
                parent[vertex] = parent[parent[vertex]]
                vertex = parent[vertex]
            return vertex
        def connect(a, b):
            a, b = find(a), find(b)
            if a != b:
                if rank[a] < rank[b]:
                    a, b = b, a
                parent[b] = a
                if rank[a] == rank[b]:
                    rank[a] += 1
        for face in self.faces:
            if len(face) != 3 or len(set(face)) != 3 or any(i < 0 or i >= len(self.vertices) for i in face):
                raise ValueError("face contains invalid or repeated vertex indices")
            a, b, c = (self.vertices[i] for i in face)
            connect(face[0], face[1])
            connect(face[0], face[2])
            area, aspect, is_obtuse = _triangle_metrics(a, b, c)
            ratios.append(aspect)
            obtuse += is_obtuse
            areas.append(area)
            for i, j in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                edge = tuple(sorted((i, j)))
                edges[edge] = edges.get(edge, 0) + 1
        ratio_scale = max(ratios)
        mean = ratio_scale*(fsum(value/ratio_scale for value in ratios)/len(ratios))
        # The C++ implementation uses Bessel's correction; its one-face
        # result is undefined (0/0), represented here by NaN.
        if len(ratios) > 1:
            deviation_scale = max(abs(value-mean) for value in ratios)
            sigma = (deviation_scale*sqrt(fsum(((value-mean)/deviation_scale)**2 for value in ratios) /
                                           (len(ratios)-1)) if deviation_scale else 0.)
        else:
            sigma = float("nan")
        total_volume = _mesh_volume(self.vertices, self.faces, find) if include_volume else None
        try:
            total_area = fsum(areas)
        except OverflowError as error:
            raise ValueError("mesh area exceeds float range") from error
        return MeshStatistics(len(self.vertices), len(edges), len(self.faces),
                              len(self.vertices)-len(edges)+len(self.faces),
                              total_area, total_volume, mean, sigma,
                              max(ratios), obtuse,
                              sum(n == 1 for n in edges.values()),
                              sum(n > 2 for n in edges.values()))


def read_obj(path: str | Path) -> Mesh:
    vertices: list[Vec] = []
    faces: list[Face] = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            fields = line.split("#", 1)[0].split()
            if not fields:
                continue
            if fields[0] == "v":
                if len(fields) < 4:
                    raise ValueError(f"line {number}: vertex needs three coordinates")
                try:
                    vertex = tuple(float(x) for x in fields[1:4])
                except ValueError as error:
                    raise ValueError(f"line {number}: invalid vertex") from error
                if not all(isfinite(x) for x in vertex):
                    raise ValueError(f"line {number}: non-finite vertex")
                vertices.append(vertex)
            elif fields[0] == "f":
                if len(fields) != 4:
                    raise ValueError(f"line {number}: only triangular faces are supported")
                indices = []
                for token in fields[1:]:
                    try:
                        index = int(token.split("/", 1)[0])
                    except ValueError as error:
                        raise ValueError(f"line {number}: invalid face index") from error
                    index = index - 1 if index > 0 else len(vertices) + index if index < 0 else -1
                    if index < 0 or index >= len(vertices):
                        raise ValueError(f"line {number}: vertex index out of range")
                    indices.append(index)
                faces.append(tuple(indices))
    return Mesh(tuple(vertices), tuple(faces))
