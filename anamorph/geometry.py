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

"""Neurite path partitioning and cubic interpolating centerlines.

Based on AnaMorph (https://github.com/NeuroBox3D/AnaMorph).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import acos, dist, hypot, isfinite, pi, sqrt
from .swc import Morphology

Vec = tuple[float, float, float]


def _sub(a: Vec, b: Vec) -> Vec:
    return tuple(x-y for x, y in zip(a, b))


def _unit(v: Vec) -> Vec:
    length = hypot(*v)
    if length == 0:
        raise ValueError("coincident points cannot define a spline tangent")
    if not isfinite(length) or length < float.fromhex("0x1p-1022"):
        if not all(isfinite(x) for x in v):
            raise ValueError("spline tangent exceeds float range")
        scale = max(abs(x) for x in v)
        scaled = tuple(x/scale for x in v)
        norm = hypot(*scaled)
        return tuple(x/norm for x in scaled)
    return tuple(x/length for x in v)


def partition_paths(tree: Morphology, strategy: str = "max-chordal-depth",
                    filter_angle: float = 90.0, radius_ratio: float = 10.0) -> list[tuple[int, ...]]:
    """Partition all neurite-neurite edges, continuing one path at each branch.

    Operates directly on SWC samples; it does not yet adjust root vertices or
    combine multi-sample somata as the original C++ network reader does.
    """
    if strategy not in ("max-chordal-depth", "min-angle", "simple-neurite-paths"):
        raise ValueError("invalid partition strategy")
    if not isfinite(filter_angle) or not 0 <= filter_angle <= 180 or not isfinite(radius_ratio) or radius_ratio < 1:
        raise ValueError("invalid branch filtering parameters")
    filter_radians = (filter_angle/180.)*pi
    nodes = tree.nodes
    neurites = {i for i, node in nodes.items() if node.kind in (2, 3, 4)}
    children = {i: [c for c in tree.children[i] if c in neurites] for i in neurites}
    # Compute maximum distance from each vertex to a descendant terminal in
    # postorder. The C++ selector compares depth from each candidate child.
    depth: dict[int, float] = {}
    todo = [(i, False) for i in neurites if nodes[i].parent not in neurites]
    while todo:
        i, visited = todo.pop()
        if visited:
            depth[i] = max((dist(nodes[i].position, nodes[c].position) + depth[c] for c in children[i]), default=0.0)
        else:
            todo.append((i, True))
            todo.extend((c, False) for c in children[i])

    paths: list[list[int]] = []
    # Follow the root-edge order retained by soma/root preparation instead of
    # sorting by vertex ID: inserted root vertices receive new IDs and would
    # otherwise move their entire neurites to the end of the traversal.
    roots = []
    frontier = list(reversed(tree.roots))
    while frontier:
        i = frontier.pop()
        if i in neurites and nodes[i].parent not in neurites:
            roots.append(i)
        frontier.extend(reversed(tree.children[i]))
    for i in roots:
        stack = []
        for c in children[i]:
            path = [i, c]
            paths.append(path)
            stack.append((c, path))
        while stack:
            vertex, path = stack.pop()
            cs = children[vertex]
            chosen = None
            if len(cs) == 1:
                chosen = cs[0]
            elif len(cs) > 1 and strategy != "simple-neurite-paths":
                previous = nodes[nodes[vertex].parent]
                incoming = _unit(_sub(nodes[vertex].position, previous.position))
                candidates = []
                for c in cs:
                    outgoing = _unit(_sub(nodes[c].position, nodes[vertex].position))
                    cosine = sum(a*b for a, b in zip(incoming, outgoing))
                    # NeuriteSegment::getAngle in the source snaps nearly
                    # parallel directions to zero before applying the filter
                    # and comparing candidate angles.
                    angle = 0. if cosine > 1.-1e-13 else acos(max(-1., min(1., cosine)))
                    ratio = max(nodes[vertex].radius/nodes[c].radius, nodes[c].radius/nodes[vertex].radius)
                    if angle < filter_radians and ratio < radius_ratio:
                        candidates.append((c, angle))
                if candidates:
                    chosen = (min(candidates, key=lambda item: item[1])[0] if strategy == "min-angle"
                              else max(candidates, key=lambda item: depth[item[0]])[0])
            for c in cs:
                if c == chosen:
                    path.append(c)
                    stack.append((c, path))
                else:
                    branch = [vertex, c]
                    paths.append(branch)
                    stack.append((c, branch))
    return [tuple(p) for p in paths]


def order_paths_breadth_first(tree: Morphology, paths: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    """Order each neurite's path tree breadth-first, as in cell rendering.

    The parent of a path starting at a branch is the path containing the
    incoming edge of that branch. Depth in the original vertex tree alone is
    not sufficient: a short sub-branch can start before a more distant sibling.
    """
    edge_owner = {}
    for index, path in enumerate(paths):
        if len(path) < 2:
            raise ValueError("neurite paths must contain at least one segment")
        for start, end in zip(path, path[1:]):
            if start not in tree.nodes or end not in tree.nodes or tree.nodes[end].parent != start:
                raise ValueError("neurite path contains an invalid edge")
            if (start, end) in edge_owner:
                raise ValueError("neurite paths contain a duplicate edge")
            edge_owner[start, end] = index
    children = [[] for _ in paths]
    roots = []
    for index, path in enumerate(paths):
        start = tree.nodes[path[0]]
        parent = tree.nodes.get(start.parent)
        if parent is None or parent.kind == 1:
            roots.append(index)
        else:
            owner = edge_owner.get((parent.id, start.id))
            if owner is None or owner == index:
                raise ValueError("neurite path has no valid parent path")
            children[owner].append(index)
    result = []
    for root in roots:
        queue = deque([root])
        while queue:
            index = queue.popleft()
            result.append(paths[index])
            queue.extend(children[index])
    if len(result) != len(paths):
        raise ValueError("neurite path parent links do not form a forest")
    return result


def _solve_tridiagonal(lower: list[float], diag: list[float], upper: list[float], values: list[float]) -> list[float]:
    n = len(diag)
    diag, values = diag.copy(), values.copy()
    for i in range(1, n):
        factor = lower[i] / diag[i-1]
        diag[i] -= factor * upper[i-1]
        values[i] -= factor * values[i-1]
    solution = [0.0] * n
    solution[-1] = values[-1] / diag[-1]
    for i in range(n-2, -1, -1):
        solution[i] = (values[i] - upper[i]*solution[i+1]) / diag[i]
    return solution


@dataclass(frozen=True, slots=True)
class Spline:
    """C2 interpolating cubic spline with clamped endpoint directions."""
    parameters: tuple[float, ...]
    points: tuple[Vec, ...]
    second_derivatives: tuple[Vec, ...]

    @classmethod
    def interpolate(cls, points: list[Vec] | tuple[Vec, ...], *,
                    parametrization: str = "chord-length", start_direction: Vec | None = None) -> Spline:
        if len(points) < 2:
            raise ValueError("a spline requires at least two points")
        lengths = [dist(a, b) for a, b in zip(points, points[1:])]
        if any(not isfinite(x) or x <= 0 for x in lengths):
            raise ValueError("consecutive spline points must be finite and distinct")
        if parametrization == "chord-length":
            weights = lengths
        elif parametrization == "centripetal":
            weights = [sqrt(x) for x in lengths]
        elif parametrization == "uniform":
            # The checked-in source uses i/n for interior knots, with the
            # final knot explicitly set to 1; this is not evenly spaced.
            weights = None
        else:
            raise ValueError("unknown spline parametrization")
        if weights is None:
            params = [0.0]+[i/len(points) for i in range(1, len(points)-1)]+[1.0]
        else:
            total = sum(weights)
            if not isfinite(total):
                # The sum can overflow even when all individual chords are
                # finite. Normalize first; relative knot positions are unchanged.
                scale = max(weights)
                weights = [weight/scale for weight in weights]
                total = sum(weights)
            params = [0.0]
            for weight in weights:
                params.append(params[-1] + weight/total)
            params[-1] = 1.0
        h = [b-a for a, b in zip(params, params[1:])]
        if any(not isfinite(value) or value <= 0 for value in h):
            raise ValueError("spline knot spacing is below floating-point resolution")
        start = _unit(start_direction if start_direction is not None else _sub(points[1], points[0]))
        end = _unit(_sub(points[-1], points[-2]))
        d0 = tuple(x * lengths[0]/h[0] for x in start)
        dn = tuple(x * lengths[-1]/h[-1] for x in end)
        if not all(isfinite(value) for value in (*d0, *dn)):
            raise ValueError("spline endpoint derivative exceeds float range")
        n = len(points)
        lower, diagonal, upper = [0.0]*n, [0.0]*n, [0.0]*n
        diagonal[0], upper[0] = 2*h[0], h[0]
        lower[-1], diagonal[-1] = h[-1], 2*h[-1]
        for i in range(1, n-1):
            lower[i], diagonal[i], upper[i] = h[i-1], 2*(h[i-1]+h[i]), h[i]
        solutions = []
        for axis in range(3):
            y = [p[axis] for p in points]
            rhs = [0.0]*n
            rhs[0] = 6*((y[1]-y[0])/h[0]-d0[axis])
            rhs[-1] = 6*(dn[axis]-(y[-1]-y[-2])/h[-1])
            for i in range(1, n-1):
                rhs[i] = 6*((y[i+1]-y[i])/h[i]-(y[i]-y[i-1])/h[i-1])
            if not all(isfinite(value) for value in rhs):
                raise ValueError("spline interpolation system exceeds float range")
            solution = _solve_tridiagonal(lower, diagonal, upper, rhs)
            if not all(isfinite(value) for value in solution):
                raise ValueError("spline curvature exceeds float range")
            solutions.append(solution)
        return cls(tuple(params), tuple(points), tuple(tuple(solutions[j][i] for j in range(3)) for i in range(n)))

    def evaluate_segment(self, i: int, u: float) -> Vec:
        if i < 0 or i >= len(self.points)-1 or not 0 <= u <= 1:
            raise ValueError("spline segment index or parameter out of range")
        h = self.parameters[i+1] - self.parameters[i]
        a, b = 1-u, u
        return tuple(a*self.points[i][j] + b*self.points[i+1][j] +
                     ((a**3-a)*self.second_derivatives[i][j] + (b**3-b)*self.second_derivatives[i+1][j])*h*h/6
                     for j in range(3))

    def derivatives_segment(self, i: int, u: float) -> tuple[Vec, Vec]:
        """First and second derivatives with respect to local parameter u."""
        if i < 0 or i >= len(self.points)-1 or not 0 <= u <= 1:
            raise ValueError("spline segment index or parameter out of range")
        h = self.parameters[i+1]-self.parameters[i]
        a, b = 1-u, u
        first = tuple(self.points[i+1][j]-self.points[i][j] +
                      ((1-3*a*a)*self.second_derivatives[i][j] +
                       (3*b*b-1)*self.second_derivatives[i+1][j])*h*h/6 for j in range(3))
        second = tuple((a*self.second_derivatives[i][j]+b*self.second_derivatives[i+1][j])*h*h
                       for j in range(3))
        return first, second

    def power_coefficients(self, i: int) -> tuple[tuple[float, ...], ...]:
        """Return cubic coefficients per axis in local parameter u."""
        if i < 0 or i >= len(self.points)-1:
            raise ValueError("spline segment index out of range")
        h2 = (self.parameters[i+1]-self.parameters[i])**2
        return tuple((self.points[i][j],
                      self.points[i+1][j]-self.points[i][j]-h2*(2*self.second_derivatives[i][j]+self.second_derivatives[i+1][j])/6,
                      h2*self.second_derivatives[i][j]/2,
                      h2*(self.second_derivatives[i+1][j]-self.second_derivatives[i][j])/6)
                     for j in range(3))
