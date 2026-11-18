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

"""Approximate, sampling-based warnings for neurite geometry.

Based on AnaMorph (https://github.com/NeuroBox3D/AnaMorph).
These warnings are NOT the original analytical consistency analysis.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from .geometry import Spline, partition_paths
from .mesh import cross, dot, sub
from .swc import Morphology
from .polynomial import add, derivative, evaluate, multiply, roots_unit_interval


@dataclass(frozen=True, slots=True)
class GeometryWarning:
    kind: str
    segments: tuple[tuple[int, int], ...]
    detail: str

    def as_dict(self):
        return asdict(self)


def _segment_distance(p, q, a, b):
    """Shortest distance between two line segments in three dimensions."""
    u, v, w = sub(q, p), sub(b, a), sub(p, a)
    uu, vv, uv, uw, vw = dot(u, u), dot(v, v), dot(u, v), dot(u, w), dot(v, w)
    tiny = 1e-30
    if uu <= tiny and vv <= tiny:
        return sqrt(dot(w, w))
    if uu <= tiny:
        s, t = 0., max(0., min(1., vw/vv))
    elif vv <= tiny:
        s, t = max(0., min(1., -uw/uu)), 0.
    else:
        denom = uu*vv-uv*uv
        s = max(0., min(1., (uv*vw-vv*uw)/denom)) if denom > tiny*uu*vv else 0.
        t = (uv*s+vw)/vv
        if t < 0:
            t, s = 0., max(0., min(1., -uw/uu))
        elif t > 1:
            t, s = 1., max(0., min(1., (uv-uw)/uu))
    delta = tuple(w[i]+s*u[i]-t*v[i] for i in range(3))
    return sqrt(dot(delta, delta))


def _sum_squares(vectors):
    result = [0.0]
    for vector in vectors:
        result = add(result, multiply(vector, vector))
    return result


def _numeric_curve_checks(spline, segment, radius):
    """Use polynomial stationary points to check speed and local curvature."""
    coefficients = spline.power_coefficients(segment)
    velocity = [derivative(component) for component in coefficients]
    acceleration = [derivative(component) for component in velocity]
    speed_sq = _sum_squares(velocity)
    speed_scale = sum(abs(x) for x in speed_sq)
    speed_candidates = [0.0, 1.0] + roots_unit_interval(derivative(speed_sq))
    irregular = min(evaluate(speed_sq, x) for x in speed_candidates) <= 1e-10*speed_scale
    cross_product = []
    for i, j in ((1, 2), (2, 0), (0, 1)):
        second = multiply(velocity[j], acceleration[i])
        cross_product.append(add(multiply(velocity[i], acceleration[j]), [-x for x in second]))
    cross_sq = _sum_squares(cross_product)
    curvature_check = add([radius*radius*x for x in cross_sq],
                          [-x for x in multiply(multiply(speed_sq, speed_sq), speed_sq)])
    scale = sum(abs(x) for x in curvature_check)
    candidates = [0.0, 1.0] + roots_unit_interval(derivative(curvature_check))
    curvature = max(evaluate(curvature_check, x) for x in candidates) >= -1e-10*scale
    return irregular, curvature


def approximate_analysis(tree: Morphology, *, samples: int = 8, max_pairs: int = 100000,
                         strategy: str = "max-chordal-depth", parametrization: str = "chord-length") -> list[GeometryWarning]:
    """Sample splines for local curvature and nonadjacent segment proximity.

    A warning is conservative about radii but approximate about curves. Absence
    of warnings provides no guarantee. Adjacent edges and soma interactions
    are excluded; same-segment global self-intersections are not checked.
    """
    if samples < 2 or max_pairs < 0:
        raise ValueError("samples must be at least 2 and max_pairs nonnegative")
    segments = []
    warnings = []
    for path in partition_paths(tree, strategy):
        nodes = [tree.nodes[i] for i in path]
        parent = tree.nodes.get(nodes[0].parent)
        tangent = sub(nodes[0].position, parent.position) if parent and parent.kind == 1 else None
        spline = Spline.interpolate([n.position for n in nodes], parametrization=parametrization, start_direction=tangent)
        for i, (a, b) in enumerate(zip(nodes, nodes[1:])):
            edge = (a.id, b.id)
            points = [spline.evaluate_segment(i, j/samples) for j in range(samples+1)]
            radius = max(a.radius, b.radius)
            irregular = False
            curvature = False
            for j in range(samples+1):
                velocity, acceleration = spline.derivatives_segment(i, j/samples)
                speed_sq = dot(velocity, velocity)
                if speed_sq < 1e-24:
                    irregular = True
                elif radius*radius*dot(cross(velocity, acceleration), cross(velocity, acceleration)) >= speed_sq**3:
                    curvature = True
            numerical_irregular, numerical_curvature = _numeric_curve_checks(spline, i, radius)
            irregular = irregular or numerical_irregular
            curvature = curvature or numerical_curvature
            if irregular:
                warnings.append(GeometryWarning("near_zero_tangent", (edge,), "numerical spline tangent is nearly zero"))
            if curvature:
                warnings.append(GeometryWarning("local_curvature", (edge,), "numerical curvature radius is no larger than bounding radius"))
            segments.append((edge, points, radius))
    pair_count = len(segments)*(len(segments)-1)//2
    if pair_count > max_pairs:
        raise ValueError(f"analysis would inspect {pair_count:,} segment pairs; limit is {max_pairs:,}")
    for n, (edge_a, points_a, radius_a) in enumerate(segments):
        for edge_b, points_b, radius_b in segments[n+1:]:
            if set(edge_a) & set(edge_b):
                continue
            threshold = radius_a+radius_b
            if any(_segment_distance(a, b, c, d) <= threshold
                   for a, b in zip(points_a, points_a[1:])
                   for c, d in zip(points_b, points_b[1:])):
                warnings.append(GeometryWarning("neurite_proximity", (edge_a, edge_b),
                                                "sampled centerlines are within the sum of maximum segment radii"))
    return warnings
