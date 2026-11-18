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
# See LICENSE and NOTICE.md for the license and additional attribution terms.
"""Size-distribution diagnostics for triangular surface meshes."""
from math import dist

from .mesh import Mesh, _triangle_metrics


def _distribution(values):
    values = sorted(values)
    def percentile(fraction):
        index = fraction*(len(values)-1)
        lower = int(index)
        upper = min(lower+1, len(values)-1)
        return values[lower]+(index-lower)*(values[upper]-values[lower])
    return {"min": values[0], "p05": percentile(.05), "median": percentile(.5),
            "p95": percentile(.95), "max": values[-1]}


def mesh_quality(mesh: Mesh):
    """Return edge-length and triangle-area distributions in input units.

    This describes triangle sizes and shapes, but does not certify that a mesh
    is quasi-uniform or that its triangles are free of self-intersections.
    """
    statistics = mesh.statistics(include_volume=False)  # Surface quality does not need enclosed volume.
    edges = {tuple(sorted((a, b))) for face in mesh.faces
             for a, b in zip(face, face[1:]+face[:1])}
    lengths = [dist(mesh.vertices[a], mesh.vertices[b]) for a, b in edges]
    areas = []
    for a, b, c in mesh.faces:
        areas.append(_triangle_metrics(mesh.vertices[a], mesh.vertices[b], mesh.vertices[c])[0])
    return {"edge_lengths": _distribution(lengths), "triangle_areas": _distribution(areas),
            "aspect_ratio_mean": statistics.aspect_ratio_mean,
            "aspect_ratio_max": statistics.aspect_ratio_max,
            "certifies_quasi_uniformity": False}
