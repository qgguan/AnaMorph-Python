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

"""Experimental voxel-union mesher for Python output.

This is an alternative mesher, not a translation of AnaMorph's red-blue
stitching algorithm. Based on AnaMorph (https://github.com/NeuroBox3D/AnaMorph).
"""
from __future__ import annotations

from array import array
from math import ceil, dist, floor, isfinite, sqrt, ulp

from .geometry import Spline, partition_paths
from .mesh import Mesh, cross, dot, sub
from .swc import Morphology


def generate_voxel_mesh(tree: Morphology, spacing: float, *, max_grid_points: int = 1_500_000,
                        strategy: str = "max-chordal-depth", parametrization: str = "chord-length",
                        filter_angle: float = 90., radius_ratio: float = 10.) -> Mesh:
    """Approximate the union of soma spheres and spline neurites on a grid.

    A coarser grid may lose thin processes or change topology. The result is not
    subject to the original tool's analytic consistency or meshing guarantees.
    """
    if not isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be positive and finite")
    if max_grid_points < 8:
        raise ValueError("max_grid_points must be at least 8")
    nodes = tree.nodes
    if any(n.kind not in (1, 2, 3, 4) for n in nodes.values()):
        raise ValueError("voxel meshing supports only soma, axon, and dendrite nodes (types 1-4)")
    # Reject clearly oversized grids before allocating spline samples.
    minimum_dims = [ceil((max(n.position[j]+n.radius for n in nodes.values()) -
                          min(n.position[j]-n.radius for n in nodes.values()))/spacing)+5 for j in range(3)]
    minimum_size = minimum_dims[0]*minimum_dims[1]*minimum_dims[2]
    if minimum_size > max_grid_points:
        raise ValueError(f"grid requires at least {minimum_size:,} points; limit is {max_grid_points:,}. Increase spacing or the limit")
    # Each primitive is a line segment with interpolated radii; zero-length
    # segments represent spheres. Sampling splines at half a voxel limits (but
    # does not mathematically bound) approximation error for curved segments.
    primitives = [(n.position, n.position, n.radius, n.radius) for n in nodes.values() if n.kind == 1]
    paths = partition_paths(tree, strategy, filter_angle, radius_ratio)
    edges_covered: set[tuple[int, int]] = set()
    for path in paths:
        pts = [nodes[i].position for i in path]
        first = nodes[path[0]]
        parent = nodes.get(first.parent)
        root_tangent = sub(first.position, parent.position) if parent and parent.kind == 1 else None
        spline = Spline.interpolate(pts, parametrization=parametrization, start_direction=root_tangent)
        for j, (a, b) in enumerate(zip(path, path[1:])):
            edges_covered.add((a, b))
            # Additional subdivisions help with cubic overshoot beyond chord length.
            length = dist(nodes[a].position, nodes[b].position)
            count = max(2, ceil(2*length/spacing))
            if count > max_grid_points:
                raise ValueError("spline sampling exceeds grid point limit; increase spacing")
            prev = spline.evaluate_segment(j, 0.0)
            r0, r1 = nodes[a].radius, nodes[b].radius
            for k in range(1, count+1):
                p = spline.evaluate_segment(j, k/count)
                primitives.append((prev, p, r0+(r1-r0)*(k-1)/count, r0+(r1-r0)*k/count))
                prev = p
    # Connect soma samples to first neurite samples, and retain isolated nodes.
    for n in nodes.values():
        if n.parent != -1:
            parent = nodes[n.parent]
            if (parent.id, n.id) not in edges_covered and n.kind in (2, 3, 4):
                primitives.append((parent.position, n.position, n.radius, n.radius))
        if n.kind in (2, 3, 4) and n.parent == -1 and not tree.children[n.id]:
            primitives.append((n.position, n.position, n.radius, n.radius))
    if not primitives:
        raise ValueError("morphology does not contain meshable geometry")

    low = [min(min(a[j]-ra, b[j]-rb) for a, b, ra, rb in primitives) - 2*spacing for j in range(3)]
    high = [max(max(a[j]+ra, b[j]+rb) for a, b, ra, rb in primitives) + 2*spacing for j in range(3)]
    # Near-zero grid values can place several distinct edge intersections at
    # the same floating-point coordinate, producing zero-area triangles. Keep
    # their signs but move them slightly away from zero consistently for the
    # entire grid. Include coordinate precision for translated morphologies.
    level_margin = max(spacing*1e-7, 16*max(ulp(value) for value in low+high))
    if level_margin >= spacing*.01:
        raise ValueError("grid spacing is too small for the coordinate precision; translate or rescale the morphology")
    dims = [ceil((high[j]-low[j])/spacing)+1 for j in range(3)]
    nx, ny, nz = dims
    size = nx*ny*nz
    if size > max_grid_points:
        raise ValueError(f"grid requires {size:,} points; limit is {max_grid_points:,}. Increase spacing or the limit")
    values = array("d", [3*spacing]) * size
    def index(x, y, z):
        return (z*ny+y)*nx+x
    for a, b, ra, rb in primitives:
        ab = sub(b, a)
        squared_length = dot(ab, ab)
        radius = max(ra, rb) + 2*spacing
        starts = [max(0, floor((min(a[j], b[j])-radius-low[j])/spacing)) for j in range(3)]
        ends = [min(dims[j]-1, ceil((max(a[j], b[j])+radius-low[j])/spacing)) for j in range(3)]
        for z in range(starts[2], ends[2]+1):
            pz = low[2]+z*spacing
            for y in range(starts[1], ends[1]+1):
                py = low[1]+y*spacing
                for x in range(starts[0], ends[0]+1):
                    point = (low[0]+x*spacing, py, pz)
                    t = min(1.0, max(0.0, dot(sub(point, a), ab)/squared_length)) if squared_length else 0.0
                    center = tuple(a[j]+t*ab[j] for j in range(3))
                    value = dist(point, center) - (ra+t*(rb-ra))
                    at = index(x, y, z)
                    if value < values[at]:
                        values[at] = value

    vertices, faces, edge_vertices = [], [], {}
    # A globally consistent decomposition around the cube's 0-6 diagonal.
    offsets = ((0,0,0), (1,0,0), (1,1,0), (0,1,0), (0,0,1), (1,0,1), (1,1,1), (0,1,1))
    tetrahedra = ((0,1,2,6), (0,2,3,6), (0,3,7,6), (0,7,4,6), (0,4,5,6), (0,5,1,6))
    def crossing(a, b, ids, points, vals):
        key = tuple(sorted((ids[a], ids[b])))
        if key not in edge_vertices:
            t = vals[a]/(vals[a]-vals[b])
            edge_vertices[key] = len(vertices)
            vertices.append(tuple(points[a][j]+t*(points[b][j]-points[a][j]) for j in range(3)))
        return edge_vertices[key]
    def emit(triangle, outward):
        a, b, c = (vertices[i] for i in triangle)
        normal = cross(sub(b, a), sub(c, a))
        if dot(normal, normal) == 0:
            raise ValueError("voxel grid produced a degenerate triangle; change spacing or translate the morphology")
        if dot(normal, outward) < 0:
            triangle = (triangle[0], triangle[2], triangle[1])
        faces.append(triangle)
    for z in range(nz-1):
        for y in range(ny-1):
            for x in range(nx-1):
                ids = [index(x+dx, y+dy, z+dz) for dx, dy, dz in offsets]
                vals = [value if abs(value) >= level_margin else
                        (-level_margin if value < 0 else level_margin)
                        for value in (values[i] for i in ids)]
                if min(vals) >= 0 or max(vals) < 0:
                    continue
                points = [tuple(low[j]+(x+dx, y+dy, z+dz)[j]*spacing for j in range(3)) for dx, dy, dz in offsets]
                for tet in tetrahedra:
                    inside = [i for i in tet if vals[i] < 0]
                    outside = [i for i in tet if vals[i] >= 0]
                    if not inside or not outside:
                        continue
                    outward = tuple(sum(points[i][j] for i in outside)/len(outside) - sum(points[i][j] for i in inside)/len(inside) for j in range(3))
                    if len(inside) == 1:
                        emit(tuple(crossing(inside[0], o, ids, points, vals) for o in outside), outward)
                    elif len(outside) == 1:
                        emit(tuple(crossing(i, outside[0], ids, points, vals) for i in inside), outward)
                    else:
                        a, b = inside
                        c, d = outside
                        ac, ad = crossing(a, c, ids, points, vals), crossing(a, d, ids, points, vals)
                        bc, bd = crossing(b, c, ids, points, vals), crossing(b, d, ids, points, vals)
                        emit((ac, ad, bd), outward)
                        emit((ac, bd, bc), outward)
    if not faces:
        raise ValueError("grid did not resolve the morphology; use smaller spacing")
    return Mesh(tuple(vertices), tuple(faces))
