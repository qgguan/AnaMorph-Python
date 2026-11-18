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
# See LICENSE and NOTICE.md for the license and attribution terms.
"""Per-job source-style network analysis for comparisons with AnaMorph.

Results reproduce the source's numerical choices where implemented. They are
not geometric certificates or a substitute for the broader analysis.
"""
from collections import Counter
from dataclasses import asdict, dataclass
from math import isfinite

from .analytic_intersections import CanalSegment
from .bernstein import ClippingConvergenceError
from .geometry import Spline, partition_paths
from .mesh import sub
from .model import prepare_morphology
from .source_bivariate import (SourceBivariateFailure, source_global_self_intersection_candidates,
                               source_neurite_pair_candidates)
from .source_canal_polynomials import (source_local_self_intersection_candidates,
                                       source_regularity_candidates, source_soma_neurite_candidates)
from .source_jobs import SourceAnalysisJob, plan_source_analysis_jobs
from .source_spline import SourceCanalSegment, source_spline_controls


@dataclass(frozen=True, slots=True)
class SourceTraceCandidate:
    parameters: tuple[float, ...]
    value: float
    parameter_bounds: tuple[tuple[float, float], ...]
    source: str


@dataclass(frozen=True, slots=True)
class SourceJobTrace:
    job: SourceAnalysisJob
    result: bool | None
    candidates: tuple[SourceTraceCandidate, ...] = ()
    reason: str | None = None
    classification: str | None = None
    soma_relationship: str | None = None
    solver_failure: str | None = None


@dataclass(frozen=True, slots=True)
class SourceSegmentFlags:
    """Source-style positive flags for a prepared neurite segment.

    False means no positive result was recorded, not that an omitted or failed
    job proved the flag clear. Solver issues and truncation remain explicit.
    """
    edge: tuple[int, int]
    pmdv: bool = False
    spmdv: bool = False
    smdv: bool = False
    reg: bool = False
    lsi: bool = False
    gsi: bool = False
    rc_sons: bool = False
    ic_sons: bool = False
    rc_nsns: bool = False
    icrn_nsns: bool = False
    icin_nsns: bool = False
    solver_issues: tuple[str, ...] = ()
    source_plan_may_affect: bool = False


@dataclass(frozen=True, slots=True)
class SourceAnalysisTrace:
    segment_count: int
    path_count: int
    jobs: tuple[SourceJobTrace, ...]
    truncated: bool
    unresolved: tuple[str, ...] = ()
    segment_flags: tuple[SourceSegmentFlags, ...] = ()

    @property
    def source_jobs_clean(self):
        """False on a positive; None on incomplete coverage without a positive."""
        if any(job.result is True for job in self.jobs):
            return False
        if self.truncated or self.unresolved or any(job.result is None for job in self.jobs):
            return None
        return True

    @property
    def planned_solver_work_complete(self):
        """Whether all planned jobs finished without a solver failure.

        This concerns only selected jobs and does not certify network coverage.
        A source-style positive or clear result may coexist with a failure.
        """
        return not (self.truncated or self.unresolved or
                    any(job.result is None or job.solver_failure is not None for job in self.jobs))

    def as_dict(self):
        result = asdict(self)
        result["source_jobs_clean"] = self.source_jobs_clean
        result["planned_solver_work_complete"] = self.planned_solver_work_complete
        result["job_counts"] = dict(sorted(Counter(item.job.kind for item in self.jobs).items()))
        result["certifies_consistency"] = False
        return result


def trace_source_analysis(tree, *, prepare=True, preconditioning=True, pc_alpha=3., pc_beta=1.5,
                          pc_gamma=10., scale_radius=1., preserve_surface_roots=False, source_soma_centroid=False,
                          precondition_norm_mode="robust",
                          strategy="max-chordal-depth", parametrization="chord-length",
                          filter_angle=90., radius_ratio=10., max_jobs=100000,
                          univariate_tolerance=1e-6, bivariate_tolerance=1e-4,
                          max_intervals=100000, max_rectangles=10000, _edge_order=None):
    """Execute every planned source-style job, subject to explicit work limits.

    Source numerical results and fallback positives are kept independently of
    the improved geometric analysis. An unresolved job has result ``None``;
    the source's bivariate pair exception becomes a positive without a witness.
    A source global-self-intersection result can be clear even when its
    bivariate solve failed; ``solver_failure`` and
    ``planned_solver_work_complete`` expose that difference.
    ``source_soma_centroid`` selects source-order soma summation when preparing.
    Preparation, spline construction and solver behavior are not certified to
    match C++ in all cases. This diagnostic must not be used as a clean-mesh
    certificate or as a replacement for the conservative generation gate.
    """
    if (not all(isfinite(value) and value > 0 for value in (univariate_tolerance, bivariate_tolerance)) or
            type(max_intervals) is not int or max_intervals < 1 or
            type(max_rectangles) is not int or max_rectangles < 1 or type(source_soma_centroid) is not bool or
            precondition_norm_mode not in ("robust", "source", "source-x87")):
        raise ValueError("invalid source trace solver settings")
    edge_order = _edge_order
    if prepare:
        prepared = prepare_morphology(tree, preconditioning=preconditioning, alpha=pc_alpha, beta=pc_beta,
                                      gamma=pc_gamma, scale_radius=scale_radius,
                                      preserve_surface_roots=preserve_surface_roots,
                                      source_soma_centroid=source_soma_centroid,
                                      precondition_norm_mode=precondition_norm_mode)
        edge_order, tree = prepared.source_edge_order, prepared.morphology
    elif scale_radius != 1.:
        raise ValueError("radius scaling requires morphology preparation")
    elif source_soma_centroid:
        raise ValueError("source soma centroid requires morphology preparation")
    elif precondition_norm_mode != "robust":
        raise ValueError("precondition norm mode requires morphology preparation")
    plan = plan_source_analysis_jobs(tree, prepare=False, max_jobs=max_jobs, strategy=strategy,
                                     parametrization=parametrization, filter_angle=filter_angle,
                                     radius_ratio=radius_ratio, _edge_order=edge_order)
    segments = {}
    for path in partition_paths(tree, strategy, filter_angle, radius_ratio):
        nodes = [tree.nodes[i] for i in path]
        parent = tree.nodes.get(nodes[0].parent)
        tangent = sub(nodes[0].position, parent.position) if parent and parent.kind == 1 else None
        spline = Spline.interpolate([node.position for node in nodes], parametrization=parametrization,
                                    start_direction=tangent)
        _, source_controls = source_spline_controls([node.position for node in nodes],
                                                     parametrization=parametrization, start_direction=tangent)
        for index, (a, b) in enumerate(zip(nodes, nodes[1:])):
            segment = CanalSegment.from_spline(spline, index, a.radius, b.radius)
            segments[(a.id, b.id)] = SourceCanalSegment(segment.power, a.radius, b.radius, source_controls[index])
    cells, neurites = {}, {}
    stack = [(root, root, None) for root in tree.roots]
    while stack:
        node_id, cell, neurite = stack.pop()
        if tree.nodes[node_id].kind != 1 and neurite is None:
            neurite = node_id
        cells[node_id], neurites[node_id] = cell, neurite
        stack.extend((child, cell, neurite) for child in tree.children[node_id])

    def classification(job):
        if job.kind == "SONS":
            # performFullAnalysis always creates IC_SONS_IsecInfo, even when
            # the tested soma belongs to a different cell.
            return "intra_cell_soma_neurite"
        if job.kind == "NSNS_ADJ":
            return "intra_cell_intra_neurite_adjacent"
        if job.kind == "NSNS_NONADJ":
            a, b = (edge[0] for edge in job.segments)
            if cells[a] != cells[b]:
                return "inter_cell"
            return "intra_cell_inter_neurite" if neurites[a] != neurites[b] else "intra_cell_intra_neurite"
        return None

    unresolved = []
    for node in tree.nodes.values():
        parent = tree.nodes.get(node.parent)
        if (node.kind in (2, 3, 4) and (parent is None or parent.kind == 1) and
                not any(tree.nodes[child].kind in (2, 3, 4) for child in tree.children[node.id])):
            unresolved.append(f"neurite root {node.id} has no segment to analyze")

    traced = []
    for job in plan.jobs:
        first = segments[job.segments[0]]
        suppressed_failures = []
        try:
            if job.kind == "REG":
                hits = source_regularity_candidates(first, tolerance=univariate_tolerance, max_intervals=max_intervals)
            elif job.kind == "LSI":
                hits = source_local_self_intersection_candidates(first, tolerance=univariate_tolerance,
                                                                  max_intervals=max_intervals)
            elif job.kind == "GSI":
                hits = source_global_self_intersection_candidates(first, univariate_tolerance=univariate_tolerance,
                                                                  bivariate_tolerance=bivariate_tolerance,
                                                                  max_intervals=max_intervals, max_rectangles=max_rectangles,
                                                                  _suppressed_failures=suppressed_failures)
            elif job.kind == "SONS":
                soma = tree.nodes[job.soma]
                hits = source_soma_neurite_candidates(first, soma.position, soma.radius,
                                                       is_root_segment=job.root_segment,
                                                       tolerance=univariate_tolerance, max_intervals=max_intervals)
            else:
                second = segments[job.segments[1]]
                hits = source_neurite_pair_candidates(first, second, adjacent=job.kind == "NSNS_ADJ",
                                                      first_end_second_start=bool(job.first_end_second_start),
                                                      univariate_tolerance=univariate_tolerance,
                                                      bivariate_tolerance=bivariate_tolerance,
                                                      max_intervals=max_intervals, max_rectangles=max_rectangles)
            if job.kind in ("REG", "LSI", "SONS"):
                candidates = tuple(SourceTraceCandidate((hit.parameter,), hit.value,
                                                        ((hit.interval.left, hit.interval.right),), "source_threshold")
                                   for hit in hits)
            else:
                candidates = tuple(SourceTraceCandidate((hit.t, hit.u), hit.distance_squared,
                                                        ((hit.rectangle.x0, hit.rectangle.x1),
                                                         (hit.rectangle.y0, hit.rectangle.y1)), hit.source)
                                   for hit in hits)
            soma_relationship = None
            if hits and job.kind == "SONS":
                soma_relationship = "intra_cell" if cells[job.segments[0][0]] == cells[job.soma] else "inter_cell"
            traced.append(SourceJobTrace(job, bool(hits), candidates,
                                         classification=classification(job) if hits else None,
                                         soma_relationship=soma_relationship,
                                         solver_failure=(str(suppressed_failures[0]) if suppressed_failures else None)))
        except SourceBivariateFailure as error:
            if job.kind in ("NSNS_ADJ", "NSNS_NONADJ"):
                traced.append(SourceJobTrace(job, True, reason=f"source pair solver fallback: {error}",
                                             classification=classification(job), solver_failure=str(error)))
            else:
                traced.append(SourceJobTrace(job, None, reason=f"source solver failure: {error}",
                                             solver_failure=str(error)))
        except ClippingConvergenceError as error:
            traced.append(SourceJobTrace(job, None, reason=f"source trace unresolved: {error}",
                                         solver_failure=str(suppressed_failures[0]) if suppressed_failures else str(error)))
    # The source updates minimum-distance flags before executing jobs, then
    # accumulates positive intersection flags on every involved segment.
    flags = {edge: set() for edge in segments}
    issues = {edge: [] for edge in segments}
    for violation in tree.minimum_distance_violations():
        if violation.edge in flags:
            flags[violation.edge].update(name for name, active in
                                         (("pmdv", violation.primary), ("spmdv", violation.severe_primary),
                                          ("smdv", violation.secondary)) if active)
    for item in traced:
        if item.result is True:
            name = {"REG": "reg", "LSI": "lsi", "GSI": "gsi", "SONS": "ic_sons"}.get(item.job.kind)
            if item.job.kind in ("NSNS_ADJ", "NSNS_NONADJ"):
                name = {"inter_cell": "rc_nsns", "intra_cell_inter_neurite": "icrn_nsns",
                        "intra_cell_intra_neurite": "icin_nsns",
                        "intra_cell_intra_neurite_adjacent": "icin_nsns"}[item.classification]
            if name:
                for edge in item.job.segments:
                    flags[edge].add(name)
        if item.result is None or item.solver_failure is not None:
            detail = item.reason or item.solver_failure or "unknown solver failure"
            message = f"{item.job.kind} {item.job.segments}: {detail}"
            for edge in item.job.segments:
                issues[edge].append(message)
    segment_flags = tuple(SourceSegmentFlags(edge, **{name: True for name in flags[edge]},
                                             solver_issues=tuple(issues[edge]),
                                             source_plan_may_affect=plan.truncated)
                          for edge in segments)
    return SourceAnalysisTrace(plan.segment_count, plan.path_count, tuple(traced), plan.truncated,
                               tuple(unresolved), segment_flags)
