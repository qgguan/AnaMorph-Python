# Based on AnaMorph (https://github.com/NeuroBox3D/AnaMorph).
# Original code copyright 2013-2017 G-CSC, Goethe University Frankfurt.
# Recommended citations retained from the original source:
# Mörschel K, Breit M, Queisser G. Generating neuron geometries for detailed
# three-dimensional simulations using AnaMorph. Neuroinformatics (2017).
# Grein S, Stepniewski M, Reiter S, Knodel MM, Queisser G. 1D-3D hybrid
# modelling - from multi-compartment models to full resolution models in space
# and time. Frontiers in Neuroinformatics 8, 68 (2014).
# Breit M, Stepniewski M, Grein S, Gottmann P, Reinhardt L, Queisser G.
# Anatomically detailed and large-scale simulations studying synapse loss
# and synchrony using NeuroBox. Frontiers in Neuroanatomy 10 (2016).
# See LICENSE and NOTICE.md for attribution and license terms.
"""Source-style randomized 2-D ear clipping for reference comparisons.

This intentionally retains the source's absolute tolerances and its omitted
closing-edge check. It is numerical, platform-dependent and not a certificate
that the resulting triangles cover the input polygon.
"""
from dataclasses import dataclass
from math import isfinite, sqrt


class SourcePolygonError(ValueError):
    """The source-style polygon test failed or did not converge."""


@dataclass(frozen=True, slots=True)
class SourceDelaunayResult:
    triangles: tuple[tuple[int, int, int], ...]
    iterations: int
    converged: bool


def source_delaunay2d(points, triangles):
    """Apply the checked-in source's ordered floating-point edge-flip pass.

    This comparison routine deliberately follows its non-convex-agnostic
    flips and iteration limit. It does not certify the returned triangulation;
    callers must independently validate it before using it as a surface.
    """
    points = {index: tuple(point) for index, point in points.items()}
    if (len(points) < 3 or any(type(index) is not int or not 0 <= index <= 0xffffffff for index in points) or
            any(len(point) != 2 or not all(isfinite(x) for x in point) for point in points.values())):
        raise ValueError("source Delaunay requires at least three finite 2-D points with uint32 IDs")
    triangles = [tuple(face) for face in triangles]
    if any(len(face) != 3 or len(set(face)) != 3 or any(index not in points for index in face)
           for face in triangles):
        raise ValueError("invalid source Delaunay triangle")

    def in_circumcircle(a, b, c, d):
        dx, dy = points[d]
        dx2, dy2 = dx*dx, dy*dy
        matrix = []
        for index in (a, b, c):
            x, y = points[index]
            matrix.append((x-dx, y-dy, (x*x-dx2)+(y*y-dy2)))
        m = matrix
        determinant = (m[0][0]*m[1][1]*m[2][2] + m[0][1]*m[1][2]*m[2][0] +
                       m[0][2]*m[1][0]*m[2][1] - m[0][2]*m[1][1]*m[2][0] -
                       m[0][1]*m[1][0]*m[2][2] - m[0][0]*m[1][2]*m[2][1])
        if not isfinite(determinant):
            raise SourcePolygonError("source Delaunay determinant exceeds float range")
        return determinant > 0.

    iteration, limit, again = 0, 2*len(points)*len(points), True
    while again and iteration < limit:
        again = False
        iteration += 1
        for i in range(len(triangles)):
            for j in range(i+1, len(triangles)):
                first, second = triangles[i], triangles[j]
                shared = sorted(set(first).intersection(second))
                if len(shared) != 2:
                    continue
                a = next(index for index in first if index not in shared)
                b = next(index for index in second if index not in shared)
                if not in_circumcircle(*first, b):
                    continue
                u, v = shared
                if not any(first[k] == u and first[(k+1)%3] == v for k in range(3)):
                    u, v = v, u
                triangles[i], triangles[j] = (a, u, b), (a, b, v)
                again = True
    return SourceDelaunayResult(tuple(triangles), iteration, not again)


def _segment_test(p0, p1, q0, q1, epsilon=1e-15):
    u = (p1[0]-p0[0], p1[1]-p0[1])
    v = (q1[0]-q0[0], q1[1]-q0[1])
    w = (p0[0]-q0[0], p0[1]-q0[1])
    denominator = u[0]*v[1]-u[1]*v[0]
    if abs(denominator) < epsilon:
        return None
    lam = (v[0]*w[1]-v[1]*w[0])/denominator
    mu = (u[0]*w[1]-u[1]*w[0])/denominator
    return 0 <= lam <= 1 and 0 <= mu <= 1


def _point_in_source_polygon(points, point, rng, epsilon=1e-15):
    """Return inside/outside, or None for the source's uncertain result."""
    for _ in range(1025):
        x, y = rng.uniform(-1.1, 1.1), rng.uniform(-1.1, 1.1)
        length = sqrt(x*x+y*y)
        if length == 0:
            return None
        direction = (x/length, y/length)
        count = 0
        retry = False
        for first, second in zip(points, points[1:]+points[:1]):
            v = (second[0]-first[0], second[1]-first[1])
            w = (point[0]-first[0], point[1]-first[1])
            denominator = direction[0]*v[1]-direction[1]*v[0]
            if abs(denominator) < epsilon:
                retry = True
                break
            lam = (v[0]*w[1]-v[1]*w[0])/denominator
            mu = (direction[0]*w[1]-direction[1]*w[0])/denominator
            if mu < -epsilon or mu > 1.+epsilon:
                continue
            if abs(lam) < epsilon:
                return None
            intersection = (point[0]+direction[0]*lam, point[1]+direction[1]*lam)
            if any(sum((intersection[i]-vertex[i])**2 for i in (0, 1)) < 1e-8
                   for vertex in (first, second)):
                return None
            if lam > 0:
                count += 1
        if not retry:
            return bool(count % 2)
    return None


def source_ear_triangulate(points, rng):
    """Return ordered index triples following the source's initial ear pass.

    ``rng`` supplies a ``uniform`` method and is mutated even on failure, just
    like the C++ global random stream. The source applies a separate Delaunay
    cleanup after this pass, which this function deliberately does not do.
    """
    points = tuple(tuple(p) for p in points)
    if len(points) < 3 or any(len(p) != 2 or not all(isfinite(x) for x in p) for p in points):
        raise ValueError("expected at least three finite 2-D polygon points")
    vertices = list(range(len(points)))
    triangles = []
    while len(vertices) > 3:
        n = len(vertices)
        extended = vertices+vertices[:2]
        for k in range(1, n+1):
            a, b, c = extended[k-1:k+2]
            principal = True
            # The source loops over n-1 edges; the closing edge is omitted.
            for i in range(n-1):
                x, y = extended[i:i+2]
                if x in (a, c) or y in (a, c):
                    continue
                crossing = _segment_test(points[a], points[c], points[x], points[y])
                if crossing is None:
                    raise SourcePolygonError("source polygon diagonal test is numerically ambiguous")
                if crossing:
                    principal = False
                    break
            if not principal:
                continue
            for _ in range(1023):
                t = rng.uniform(.1, .9)
                point = tuple(points[a][i]+(points[c][i]-points[a][i])*t for i in (0, 1))
                inside = _point_in_source_polygon([points[i] for i in vertices], point, rng)
                if inside is not None:
                    break
            else:
                raise SourcePolygonError("source polygon point test did not converge")
            if inside:
                triangles.append((a, b, c))
                del vertices[k % n]
                break
        else:
            raise SourcePolygonError("source polygon has no ear")
    triangles.append(tuple(vertices))
    return tuple(triangles)
