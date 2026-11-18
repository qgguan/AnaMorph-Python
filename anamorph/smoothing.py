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

"""Laplacian mesh smoothing based on the AnaMorph implementation.

Based on AnaMorph (https://github.com/NeuroBox3D/AnaMorph).
"""
from __future__ import annotations

from math import isfinite
from .mesh import Mesh


def _neighbors(mesh):
    neighbors = [set() for _ in mesh.vertices]
    for a, b, c in mesh.faces:
        if min(a, b, c) < 0 or max(a, b, c) >= len(neighbors) or len({a, b, c}) != 3:
            raise ValueError("invalid triangular face")
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    return [tuple(sorted(n)) for n in neighbors]


def laplacian_smooth(mesh: Mesh, smoothing_factor: float = .5, iterations: int = 1) -> Mesh:
    """Apply simultaneous umbrella Laplacian steps, matching the C++ helper.

    This changes geometry and does not check for self-intersections.
    """
    if not isfinite(smoothing_factor) or iterations < 0:
        raise ValueError("smoothing factor must be finite and iterations non-negative")
    neighbors = _neighbors(mesh)
    current = mesh.vertices
    for _ in range(iterations):
        current = tuple(tuple(current[i][axis] + smoothing_factor *
                              sum(current[j][axis]-current[i][axis] for j in ns)/len(ns)
                              for axis in range(3)) if ns else current[i]
                        for i, ns in enumerate(neighbors))
    return Mesh(tuple(current), mesh.faces)


def _validate_hc_options(alpha: float = 0.4, beta: float = 0.7, iterations: int = 10):
    if not all(isfinite(x) for x in (alpha, beta)) or alpha <= 0 or beta <= alpha:
        raise ValueError("smoothing requires finite 0 < alpha < beta")
    if type(iterations) is not int or iterations < 0:
        raise ValueError("iterations must be non-negative")


def hc_smooth(mesh: Mesh, alpha: float = 0.4, beta: float = 0.7, iterations: int = 10) -> Mesh:
    """Return a smoothed copy, following the source algorithm's update order.

    This changes geometry and does not check for new self-intersections.
    """
    _validate_hc_options(alpha, beta, iterations)
    neighbors = _neighbors(mesh)
    original = mesh.vertices
    current = original
    previous_uncorrected = original
    for _ in range(iterations):
        p = [tuple(sum(current[j][axis] for j in ns)/len(ns) for axis in range(3)) if ns else current[i]
             for i, ns in enumerate(neighbors)]
        offsets = [tuple(p[i][axis] - (alpha*original[i][axis] + (1-alpha)*previous_uncorrected[i][axis])
                         for axis in range(3)) for i in range(len(p))]
        current = tuple(tuple(p[i][axis] - beta*offsets[i][axis] -
                              (1-beta)*sum(offsets[j][axis] for j in ns)/len(ns) for axis in range(3))
                        if ns else current[i] for i, ns in enumerate(neighbors))
        previous_uncorrected = p
    return Mesh(tuple(current), mesh.faces)
