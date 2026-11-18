# Based on AnaMorph (https://github.com/NeuroBox3D/AnaMorph).
# Recommended citations retained from the original source:
# Mörschel K, Breit M, Queisser G. Generating neuron geometries for detailed
# three-dimensional simulations using AnaMorph. Neuroinformatics (2017).
# Grein S, Stepniewski M, Reiter S, Knodel MM, Queisser G. 1D-3D hybrid
# modelling - from multi-compartment models to full resolution models in space
# and time. Frontiers in Neuroinformatics 8, 68 (2014).
# Breit M, Stepniewski M, Grein S, Gottmann P, Reinhardt L, Queisser G.
# Anatomically detailed and large-scale simulations studying synapse loss
# and synchrony using NeuroBox. Frontiers in Neuroanatomy 10 (2016).
# See LICENSE and NOTICE.md for license and attribution terms.
"""Shared source-style default surface postprocessing."""

from .collapse import greedy_edge_collapse
from .smoothing import hc_smooth


def standard_settings():
    """Return fresh copies of the C++ default parameters and Python work limit."""
    return {"edge_collapse": {"alpha": 1.5, "area_floor": .125, "area_ceiling": .5,
                              "depth": 5, "max_collapses": 100000, "safety": "source"},
            "hc_smoothing": {"alpha": .4, "beta": .7, "iterations": 10}}


def standard_postprocess(mesh):
    settings = standard_settings()
    mesh = greedy_edge_collapse(mesh, **settings["edge_collapse"])
    return hc_smooth(mesh, **settings["hc_smoothing"])
