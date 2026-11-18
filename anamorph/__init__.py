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

"""Work-in-progress Python port of AnaMorph.

Based on AnaMorph (https://github.com/NeuroBox3D/AnaMorph).
"""

from .swc import Morphology, Node, read_swc
from .mesh import Mesh, MeshStatistics, read_obj
from .geometry import Spline, partition_paths
from .precondition import PreconditioningResult, precondition
from .roots import adjust_soma_roots
from .smoothing import hc_smooth, laplacian_smooth
from .voxel import generate_voxel_mesh
from .collapse import greedy_edge_collapse
from .remesh import isotropic_remesh
from .quality import mesh_quality
from .mesh_intersections import MeshIntersection, find_self_intersections
from .analytic_intersections import CanalSegment, IntersectionSolveError
from .analytic_network import AnalyticReport, AnalyticSegmentFlags, analyze_analytic_partial
from .source_jobs import SourceAnalysisJob, SourceAnalysisPlan, plan_source_analysis_jobs
from .source_analysis import SourceAnalysisTrace, SourceJobTrace, SourceSegmentFlags, SourceTraceCandidate, trace_source_analysis
from .morphview import (MorphView, MorphViewVertex, MorphViewPath, MorphViewSegment,
                        create_morphview, read_morphview, write_morphview)
from .analytic_mesher import (PartialAnalysisError, assemble_redblue_checked, assemble_redblue_neuron,
                              assemble_redblue_to_obj, assemble_redblue_checked_to_obj, render_individual_surfaces)
from .redblue import (RedBlueError, RedBlueTriangulationError, SourceDelaunayLimitWarning, RedBlueUnionResult, redblue_boolean, redblue_boolean_with_retries,
                      redblue_union, redblue_union_with_retries, redblue_intersection, redblue_difference)

__all__ = ["Morphology", "Node", "read_swc", "Mesh", "MeshStatistics", "read_obj",
           "Spline", "partition_paths", "PreconditioningResult", "precondition",
           "adjust_soma_roots", "hc_smooth", "laplacian_smooth", "generate_voxel_mesh", "greedy_edge_collapse", "isotropic_remesh", "mesh_quality",
           "MeshIntersection", "find_self_intersections",
           "CanalSegment", "IntersectionSolveError", "AnalyticReport", "AnalyticSegmentFlags", "analyze_analytic_partial", "assemble_redblue_neuron",
           "SourceAnalysisJob", "SourceAnalysisPlan", "plan_source_analysis_jobs",
           "SourceAnalysisTrace", "SourceJobTrace", "SourceSegmentFlags", "SourceTraceCandidate", "trace_source_analysis",
           "MorphView", "MorphViewVertex", "MorphViewPath", "MorphViewSegment",
           "create_morphview", "read_morphview", "write_morphview",
           "render_individual_surfaces", "RedBlueError", "RedBlueTriangulationError", "SourceDelaunayLimitWarning", "RedBlueUnionResult", "redblue_boolean",
           "PartialAnalysisError", "assemble_redblue_checked", "assemble_redblue_to_obj", "assemble_redblue_checked_to_obj",
           "redblue_boolean_with_retries", "redblue_union", "redblue_union_with_retries",
           "redblue_intersection", "redblue_difference"]
