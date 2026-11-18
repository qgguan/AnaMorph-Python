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

"""Partial network-level orchestration of analytic intersection primitives.

Based on AnaMorph (https://github.com/NeuroBox3D/AnaMorph).
This report is deliberately incomplete until all source classifications are
implemented and validated. No report from this module certifies cleanliness.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import heapq
from math import inf, isfinite, nextafter, ulp

from .analytic_intersections import (CanalSegment, IntersectionSolveError, global_self_intersection_candidates,
                                     local_self_intersection_candidates, neurite_neurite_candidates,
                                     regularity_candidates, soma_neurite_candidates, adjacent_neurite_candidates,
                                     _exact_distance_at, _exact_local_polynomial, _exact_speed_squared,
                                     _report_distance_squared, _report_exact_value)
from .bernstein import ClippingConvergenceError
from .geometry import Spline, partition_paths
from .mesh import sub
from .swc import Morphology
from .model import prepare_morphology
from .source_bivariate import (SourceBivariateFailure, source_neurite_pair_candidates,
                               source_global_self_intersection_candidates)
from .source_canal_polynomials import (source_regularity_candidates, source_local_self_intersection_candidates,
                                       source_soma_neurite_candidates)
from .source_spline import SourceCanalSegment, exact_source_distance_squared, source_spline_controls
from .spatial import _BoxIndex, _overlap


_SOURCE_INTERSECTION_KINDS = frozenset(("regularity", "local_self_intersection", "global_self_intersection",
                                        "soma_neurite_candidate", "adjacent_neurite_candidate",
                                        "neurite_neurite_candidate", "source_solver_fallback",
                                        "source_bivariate_candidate", "source_regularity_candidate",
                                        "source_local_self_intersection_candidate", "source_soma_neurite_candidate",
                                        "source_global_self_intersection_candidate", "source_pair_candidate"))


@dataclass(frozen=True, slots=True)
class AnalyticCandidate:
    """Intersection evidence, with rational values when floats cannot represent them.

    ``parallel_line_tangency`` uses parameter bounds as an exact existence
    certificate; its value is the proven minimum, not the midpoint distance.
    """
    parameters: tuple[float, ...]
    value: float | None
    source: str
    parameter_bounds: tuple[tuple[float, float], ...] | None = None
    exact_parameters: tuple[tuple[int, int], ...] | None = None
    exact_value: tuple[int, int] | None = None


def _curve_candidate(hit, source=None):
    return AnalyticCandidate((hit.t, hit.u), hit.distance_squared, hit.source if source is None else source,
                             hit.parameter_bounds, hit.exact_parameters, hit.exact_distance_squared)


def _scalar_candidate(hit):
    return AnalyticCandidate((hit.parameter,), hit.value, hit.source, (hit.parameter_bounds,),
                             exact_value=hit.exact_value)


def _regularity_candidate(segment, hit):
    value, exact = _report_exact_value(_exact_speed_squared(segment, hit.midpoint), "regularity squared speed")
    source = "exact_root_interval" if hit.certified else "root_interval"
    return AnalyticCandidate((hit.midpoint,), value, source, ((hit.left, hit.right),), exact_value=exact)


@dataclass(frozen=True, slots=True)
class AnalyticFinding:
    kind: str
    segments: tuple[tuple[int, int], ...]
    detail: str
    classification: str | None = None
    candidates: tuple[AnalyticCandidate, ...] = ()
    somas: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class AnalyticSegmentFlags:
    """Positive results for a segment, matching the source's status fields.

    False flags mean no positive finding was reported; incomplete coverage or
    unresolved jobs can still hide a violation.
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
    unresolved: tuple[str, ...] = ()
    pair_limit_may_affect: bool = False
    source_plan_may_affect: bool = False


@dataclass(frozen=True, slots=True)
class AnalyticReport:
    segment_count: int
    tested_nonadjacent_pairs: int
    tested_adjacent_pairs: int
    findings: tuple[AnalyticFinding, ...]
    unresolved: tuple[str, ...]
    complete: bool = False
    segment_flags: tuple[AnalyticSegmentFlags, ...] = ()
    pair_limit_reached: bool = False
    first_untested_pair: tuple[tuple[int, int], tuple[int, int]] | None = None
    job_mode: str = "extended"
    source_plan_truncated: bool = False
    source_job_counts: dict[str, int] | None = None

    @property
    def source_job_findings(self):
        """Positive intersection jobs, excluding source-independent diagnostics."""
        if self.job_mode != "source":
            return ()
        return tuple(finding for finding in self.findings if finding.kind in _SOURCE_INTERSECTION_KINDS)

    @property
    def source_jobs_clean(self):
        """Python's source-selected job decision; None means not decided.

        This is not a consistency certificate or a guarantee of C++ parity.
        """
        if self.job_mode != "source":
            return None
        if self.source_job_findings:
            return False
        if self.unresolved or self.source_plan_truncated:
            return None
        return True

    def as_dict(self):
        result = asdict(self)
        result["source_jobs_clean"] = self.source_jobs_clean
        return result


def _outward_float(value: Fraction, *, lower: bool):
    """Round an exact rational toward the requested infinite bound."""
    try:
        rounded = float(value)
    except OverflowError:
        rounded = inf if value > 0 else -inf
    if rounded == inf:
        return nextafter(inf, -inf) if lower else inf
    if rounded == -inf:
        return -inf if lower else nextafter(-inf, inf)
    represented = Fraction(rounded)
    if (lower and represented > value) or (not lower and represented < value):
        return nextafter(rounded, -inf if lower else inf)
    return rounded


def _bounding_box(segment: CanalSegment):
    # A cubic's Bézier controls bound its exact power-basis polynomial on
    # [0, 1]. Compute the conversion using the exact values represented by
    # the input floats, then round the expanded bounds outward. This also
    # avoids intermediate overflow when finite power coefficients cancel.
    boxes = []
    source_controls = getattr(segment, "source_controls", None)
    for axis, power in enumerate(segment.power):
        if source_controls is None:
            a, b, c, d = map(Fraction, power)
            coefficients = (a, a+b/3, a+2*b/3+c/3, a+b+c+d)
            magnitude = max(segment.max_radius, *(abs(value) for value in power))
        else:
            coefficients = tuple(map(Fraction, source_controls[axis]))
            magnitude = max(segment.max_radius, *(abs(value) for value in source_controls[axis]))
        # Also leave room for roundoff when downstream diagnostics evaluate
        # the polynomial using floating-point arithmetic.
        padding = Fraction(16*ulp(magnitude))
        radius = Fraction(segment.max_radius)
        boxes.append((_outward_float(min(coefficients)-radius-padding, lower=True),
                      _outward_float(max(coefficients)+radius+padding, lower=False)))
    return tuple(boxes)


def _soma_box(soma):
    return tuple((nextafter(coord-soma.radius, -inf), nextafter(coord+soma.radius, inf))
                 for coord in soma.position)


def _verified_corner_candidate(first, second, t, u, threshold, *, inclusive=True):
    distance = _exact_distance_at(first, second, t, u)
    if not (distance <= threshold if inclusive else distance < threshold):
        return None
    value, exact_value = _report_distance_squared(distance)
    return AnalyticCandidate((t, u), value, "corner_fallback", ((t, t), (u, u)), exact_value=exact_value)


def _overlapping_box_pairs(boxes):
    """Yield candidate index pairs by sweeping the x axis."""
    ordered = sorted(range(len(boxes)), key=lambda i: (boxes[i][0][0], i))
    active, by_right = set(), []
    for index in ordered:
        left = boxes[index][0][0]
        while by_right and by_right[0][0] < left:
            _, expired = heapq.heappop(by_right)
            active.remove(expired)
        for other in sorted(active):
            if _overlap(boxes[index], boxes[other]):
                yield (other, index) if other < index else (index, other)
        active.add(index)
        heapq.heappush(by_right, (boxes[index][0][1], index))


def analyze_analytic_partial(tree: Morphology, *, univariate_tolerance: float = 1e-6,
                             bivariate_tolerance: float = 1e-4,
                             max_pairs: int = 100000, max_rectangles: int = 10000, max_intervals: int = 100000,
                             strategy: str = "max-chordal-depth",
                             parametrization: str = "chord-length", prepare: bool = False,
                             preconditioning: bool = True, scale_radius: float = 1.,
                             pc_alpha: float = 3., pc_beta: float = 1.5, pc_gamma: float = 10.,
                             filter_angle: float = 90., radius_ratio: float = 10.,
                             check_adjacent_end_start: bool = False,
                             preserve_surface_roots: bool = True, source_soma_centroid: bool = False,
                             precondition_norm_mode: str = "robust",
                             job_mode: str = "extended",
                             source_job_limit: int = 1000000) -> AnalyticReport:
    """Run implemented analytic checks, retaining all unresolved failures.

    Incomplete soma and pair classifications and numerical solver limits mean
    no report establishes cleanliness. By default, consecutive edges also
    follow the source's early return; the optional check is only a heuristic.
    """
    if max_pairs < 0 or max_rectangles < 1:
        raise ValueError("max_pairs must be nonnegative and max_rectangles positive")
    if type(max_intervals) is not int or max_intervals < 1:
        raise ValueError("max_intervals must be a positive integer")
    if job_mode not in ("extended", "source") or type(source_job_limit) is not int or source_job_limit < 0:
        raise ValueError("invalid analytic job mode or source job limit")
    if type(source_soma_centroid) is not bool:
        raise ValueError("source_soma_centroid must be boolean")
    if precondition_norm_mode not in ("robust", "source", "source-x87"):
        raise ValueError("invalid precondition norm mode")
    if not all(isfinite(t) and t > 0 for t in (univariate_tolerance, bivariate_tolerance)):
        raise ValueError("solver tolerances must be positive and finite")
    source_edge_order = None
    if prepare:
        prepared = prepare_morphology(tree, preconditioning=preconditioning, scale_radius=scale_radius,
                                      alpha=pc_alpha, beta=pc_beta, gamma=pc_gamma,
                                      preserve_surface_roots=preserve_surface_roots,
                                      source_soma_centroid=source_soma_centroid,
                                      precondition_norm_mode=precondition_norm_mode)
        if job_mode == "source":
            source_edge_order = prepared.source_edge_order
        tree = prepared.morphology
    elif scale_radius != 1.:
        raise ValueError("radius scaling requires morphology preparation")
    elif source_soma_centroid:
        raise ValueError("source soma centroid requires morphology preparation")
    elif precondition_norm_mode != "robust":
        raise ValueError("precondition norm mode requires morphology preparation")
    # The C++ classifies pairwise intersections by cell and by the neurite
    # root belonging to each segment, regardless of spline path partitioning.
    cells, neurites = {}, {}
    stack = [(root, root, None) for root in tree.roots]
    while stack:
        node_id, cell, neurite = stack.pop()
        node = tree.nodes[node_id]
        if node.kind != 1 and neurite is None:
            neurite = node_id
        cells[node_id], neurites[node_id] = cell, neurite
        stack.extend((child, cell, neurite) for child in tree.children[node_id])
    segments, source_segments = [], {}
    for path in partition_paths(tree, strategy, filter_angle, radius_ratio):
        nodes = [tree.nodes[i] for i in path]
        parent = tree.nodes.get(nodes[0].parent)
        tangent = sub(nodes[0].position, parent.position) if parent and parent.kind == 1 else None
        spline = Spline.interpolate([n.position for n in nodes], parametrization=parametrization, start_direction=tangent)
        controls = None
        if job_mode == "source":
            _, controls = source_spline_controls([n.position for n in nodes], parametrization=parametrization,
                                                  start_direction=tangent)
        for i, (a, b) in enumerate(zip(nodes, nodes[1:])):
            segment = CanalSegment.from_spline(spline, i, a.radius, b.radius)
            segments.append(((a.id, b.id), segment, _bounding_box(segment)))
            if controls is not None:
                source_segments[(a.id, b.id)] = SourceCanalSegment(segment.power, a.radius, b.radius, controls[i])
    findings, unresolved = [], []
    unresolved_by_edge = {edge: [] for edge, _, _ in segments}
    source_plan = None
    source_soma_jobs = set()
    source_segment_jobs = set()
    if job_mode == "source":
        from .source_jobs import plan_source_analysis_jobs
        source_plan = plan_source_analysis_jobs(tree, prepare=False, max_jobs=source_job_limit,
                                                strategy=strategy, parametrization=parametrization,
                                                filter_angle=filter_angle, radius_ratio=radius_ratio,
                                                _edge_order=source_edge_order)
        source_soma_jobs = {(job.segments[0], job.soma) for job in source_plan.jobs if job.kind == "SONS"}
        source_segment_jobs = {(job.kind, job.segments[0]) for job in source_plan.jobs
                               if job.kind in ("REG", "LSI", "GSI")}
        if source_plan.truncated:
            unresolved.append(f"source job plan limit {source_job_limit} reached")
    for node in tree.nodes.values():
        parent = tree.nodes.get(node.parent)
        if (node.kind in (2, 3, 4) and (parent is None or parent.kind == 1) and
                not any(tree.nodes[child].kind in (2, 3, 4) for child in tree.children[node.id])):
            unresolved.append(f"neurite root {node.id} has no segment to analyze")
    # The original job builder does not schedule soma-soma checks. Check
    # spheres from distinct cells explicitly; raw multi-sample soma sections
    # within one cell are not independent spheres in the analytic model.
    somas = [node for node in tree.nodes.values() if node.kind == 1]
    soma_boxes = [_soma_box(node) for node in somas]
    soma_index = _BoxIndex(soma_boxes) if job_mode != "source" else None
    source_somas_by_edge = {}
    if source_plan is not None:
        soma_positions = {soma.id: index for index, soma in enumerate(somas)}
        for edge, soma_id in source_soma_jobs:
            source_somas_by_edge.setdefault(edge, []).append(soma_positions[soma_id])
        for indices in source_somas_by_edge.values():
            indices.sort()
    soma_pairs = () if job_mode == "source" else _overlapping_box_pairs(soma_boxes)
    for i, j in soma_pairs:
        a, b = somas[i], somas[j]
        if cells[a.id] == cells[b.id]:
            continue
        separation_squared = sum(((Fraction(x)-Fraction(y))**2 for x, y in zip(a.position, b.position)),
                                 Fraction(0))
        threshold = (Fraction(a.radius)+Fraction(b.radius))**2
        if separation_squared <= threshold:
            detail = "soma spheres overlap or touch"
            value, exact_value = _report_distance_squared(separation_squared)
            evidence = (AnalyticCandidate((), value, "soma_centers", exact_value=exact_value),)
            findings.append(AnalyticFinding("soma_soma_intersection", (),
                                            detail, "inter_cell", evidence, (a.id, b.id)))
    violations = {violation.edge: violation for violation in tree.minimum_distance_violations()}
    for violation in violations.values():
        if violation.primary:
            detail = "length is at most twice the maximum endpoint radius"
            if violation.severe_primary:
                detail += "; severe threshold also triggered"
            findings.append(AnalyticFinding("primary_distance_violation", (violation.edge,), detail))
        if violation.secondary:
            findings.append(AnalyticFinding("secondary_distance_violation", (violation.edge,),
                                            "length is at most the sum of the source-code neighborhood radii"))
    for edge, segment, box in segments:
        # AnaMorph schedules these as separate jobs. A failed numerical solve
        # must not prevent the other two independent checks from running.
        checks = (
            ("regularity", "centerline has a zero-speed candidate",
             lambda: regularity_candidates(segment, tolerance=univariate_tolerance, max_intervals=max_intervals)),
            ("local_self_intersection", "maximum-radius curvature criterion triggered",
             lambda: local_self_intersection_candidates(segment, tolerance=univariate_tolerance, with_bounds=True,
                                                         max_intervals=max_intervals)),
            ("global_self_intersection", "nontrivial same-segment proximity candidate",
             lambda: global_self_intersection_candidates(segment, univariate_tolerance=univariate_tolerance,
                                                          bivariate_tolerance=bivariate_tolerance,
                                                          max_rectangles=max_rectangles, max_intervals=max_intervals)),
        )
        for kind, detail, check in checks:
            source_kind = {"regularity": "REG", "local_self_intersection": "LSI",
                           "global_self_intersection": "GSI"}[kind]
            if source_plan is not None and (source_kind, edge) not in source_segment_jobs:
                continue
            evidence = ()
            try:
                hits = check()
                if kind == "regularity":
                    evidence = tuple(_regularity_candidate(segment, hit) for hit in hits)
                elif kind == "local_self_intersection":
                    evidence = tuple(_scalar_candidate(hit) for hit in hits)
                else:
                    evidence = tuple(_curve_candidate(hit) for hit in hits)
                if evidence:
                    findings.append(AnalyticFinding(kind, (edge,), detail, candidates=evidence))
            except ClippingConvergenceError as error:
                message = f"segment {edge}, {kind}: {error}"
                unresolved.append(message)
                unresolved_by_edge[edge].append(message)
                evidence = ([_scalar_candidate(hit) for hit in error.candidates]
                            if kind == "local_self_intersection" and isinstance(error, IntersectionSolveError)
                            else [])
                if kind == "regularity" and isinstance(error, IntersectionSolveError):
                    for hit in error.candidates:
                        evidence.append(_regularity_candidate(segment, hit))
                for t in (0., 1.):
                    if (kind == "regularity" and _exact_speed_squared(segment, t) == 0 and
                            not any(candidate.parameters == (t,) for candidate in evidence)):
                        evidence.append(AnalyticCandidate((t,), 0., "endpoint_fallback", ((t, t),)))
                    elif kind == "local_self_intersection":
                        value = _exact_local_polynomial(segment, t)
                        if value > 0 and not any(candidate.parameters == (t,) for candidate in evidence):
                            reported_value, exact = _report_exact_value(value, "local self-intersection value")
                            evidence.append(AnalyticCandidate((t,), reported_value, "endpoint_fallback", ((t, t),),
                                                              exact_value=exact))
                if kind == "global_self_intersection":
                    if isinstance(error, IntersectionSolveError):
                        evidence.extend(_curve_candidate(hit, "corner_fallback" if hit.source == "corner" else hit.source)
                                        for hit in error.candidates)
                    elif 2*max(univariate_tolerance, bivariate_tolerance) < 1:
                        witness = _verified_corner_candidate(segment, segment, 0., 1.,
                                                             (2*Fraction(segment.max_radius))**2, inclusive=False)
                        if witness is not None:
                            evidence.append(witness)
                if evidence:
                    findings.append(AnalyticFinding(kind, (edge,),
                                                    "partial witness found; full segment solve unresolved",
                                                    candidates=tuple(evidence)))
            if source_plan is not None and not evidence:
                suppressed_failures = []
                try:
                    if kind == "global_self_intersection":
                        source_hits = source_global_self_intersection_candidates(
                            source_segments[edge], univariate_tolerance=univariate_tolerance,
                            bivariate_tolerance=bivariate_tolerance, max_rectangles=max_rectangles,
                            max_intervals=max_intervals, _suppressed_failures=suppressed_failures)
                    else:
                        source_check = (source_regularity_candidates if kind == "regularity"
                                        else source_local_self_intersection_candidates)
                        source_hits = source_check(source_segments[edge], tolerance=univariate_tolerance,
                                                   max_intervals=max_intervals)
                except ClippingConvergenceError as error:
                    message = f"segment {edge}, {kind}: source solver trace unresolved: {error}"
                    unresolved.append(message)
                    unresolved_by_edge[edge].append(message)
                else:
                    if suppressed_failures:
                        message = (f"segment {edge}, {kind}: source bivariate solver failed and the source "
                                   f"continued with partial results: {suppressed_failures[0]}")
                        unresolved.append(message)
                        unresolved_by_edge[edge].append(message)
                    if source_hits:
                        if kind == "global_self_intersection":
                            source_evidence = []
                            threshold = (2*Fraction(segment.max_radius))**2
                            for hit in source_hits:
                                exact_distance = exact_source_distance_squared(source_segments[edge],
                                                                                source_segments[edge], hit.t, hit.u)
                                verified = exact_distance < threshold
                                value, exact = (_report_distance_squared(exact_distance) if verified
                                                else (hit.distance_squared, None))
                                bounds = ((hit.rectangle.x0, hit.rectangle.x1),
                                          (hit.rectangle.y0, hit.rectangle.y1))
                                source_evidence.append(AnalyticCandidate(
                                    (hit.t, hit.u), value,
                                    "source_root_witness" if verified else "source_threshold", bounds,
                                    exact_value=exact))
                            source_evidence = tuple(source_evidence)
                        else:
                            source_evidence = tuple(AnalyticCandidate((hit.parameter,), hit.value, "source_threshold",
                                                                     ((hit.interval.left, hit.interval.right),))
                                                    for hit in source_hits)
                        findings.append(AnalyticFinding(f"source_{kind}_candidate", (edge,),
                                                        "source-style solver threshold triggered; candidates are numerical",
                                                        candidates=source_evidence))
        soma_indices = (source_somas_by_edge.get(edge, ()) if source_plan is not None
                        else soma_index.query(box))
        for soma_index_in_tree in soma_indices:
            soma = somas[soma_index_in_tree]
            parent = tree.nodes.get(tree.nodes[edge[0]].parent)
            # The intended junction exemption applies only to the soma this
            # neurite is attached to. The source applies it to every soma,
            # which can hide a collision with a different cell at t=0.
            is_root = parent is not None and (parent.kind == 1 if job_mode == "source" else parent.id == soma.id)
            evidence = ()
            try:
                hits = soma_neurite_candidates(segment, soma.position, soma.radius,
                                               is_root_segment=is_root, tolerance=univariate_tolerance,
                                               with_bounds=True, max_intervals=max_intervals)
                if hits:
                    evidence = tuple(_scalar_candidate(hit) for hit in hits)
                    findings.append(AnalyticFinding("soma_neurite_candidate", (edge,),
                                                    f"bounding pipe overlaps soma node {soma.id}",
                                                    "intra_cell" if cells[edge[0]] == cells[soma.id] else "inter_cell",
                                                    evidence,
                                                    (soma.id,)))
            except ClippingConvergenceError as error:
                message = f"segment {edge}, soma {soma.id}: {error}"
                unresolved.append(message)
                unresolved_by_edge[edge].append(message)
                threshold = (Fraction(soma.radius)+Fraction(segment.max_radius))**2
                evidence = ([_scalar_candidate(hit) for hit in error.candidates]
                            if isinstance(error, IntersectionSolveError) else [])
                for t in (0., 1.):
                    if is_root and t <= 2*univariate_tolerance:
                        continue
                    distance_squared = sum((sum((Fraction(c)*Fraction(t)**i for i, c in enumerate(axis)),
                                                Fraction(0))-Fraction(coordinate))**2
                                           for axis, coordinate in zip(segment.power, soma.position))
                    if distance_squared <= threshold and not any(candidate.parameters == (t,) for candidate in evidence):
                        value, exact_value = _report_distance_squared(distance_squared)
                        evidence.append(AnalyticCandidate((t,), value, "endpoint_fallback", ((t, t),),
                                                          exact_value=exact_value))
                if evidence:
                    findings.append(AnalyticFinding("soma_neurite_candidate", (edge,),
                                                    f"endpoint witness overlaps soma node {soma.id}; full solve unresolved",
                                                    "intra_cell" if cells[edge[0]] == cells[soma.id] else "inter_cell",
                                                    tuple(evidence), (soma.id,)))
            if source_plan is not None and not evidence:
                try:
                    source_hits = source_soma_neurite_candidates(source_segments[edge], soma.position, soma.radius,
                                                                   is_root_segment=is_root,
                                                                   tolerance=univariate_tolerance,
                                                                   max_intervals=max_intervals)
                except ClippingConvergenceError as error:
                    message = f"segment {edge}, soma {soma.id}: source univariate trace unresolved: {error}"
                    unresolved.append(message)
                    unresolved_by_edge[edge].append(message)
                else:
                    if source_hits:
                        source_evidence = tuple(AnalyticCandidate((hit.parameter,), hit.value, "source_threshold",
                                                                 ((hit.interval.left, hit.interval.right),))
                                                for hit in source_hits)
                        findings.append(AnalyticFinding("source_soma_neurite_candidate", (edge,),
                                                        f"source-style soma threshold triggered for soma node {soma.id}; "
                                                        "candidates are not verified overlaps",
                                                        "intra_cell" if cells[edge[0]] == cells[soma.id] else "inter_cell",
                                                        source_evidence, (soma.id,)))
    tested = 0
    tested_adjacent = 0
    def pair_classification(edge_a, edge_b):
        if cells[edge_a[0]] != cells[edge_b[0]]:
            return "inter_cell"
        if neurites[edge_a[0]] != neurites[edge_b[0]]:
            return "intra_cell_inter_neurite"
        return "intra_cell_intra_neurite"

    def report(pair_limit_reached=False, first_untested_pair=None, cutoff_x=None):
        flags = {edge: set() for edge, _, _ in segments}
        for edge, violation in violations.items():
            if violation.primary:
                flags[edge].add("pmdv")
            if violation.severe_primary:
                flags[edge].add("spmdv")
            if violation.secondary:
                flags[edge].add("smdv")
        for finding in findings:
            name = {"regularity": "reg", "local_self_intersection": "lsi",
                    "source_regularity_candidate": "reg", "source_local_self_intersection_candidate": "lsi",
                    "source_global_self_intersection_candidate": "gsi",
                    "global_self_intersection": "gsi"}.get(finding.kind)
            if finding.kind in ("soma_neurite_candidate", "source_soma_neurite_candidate"):
                name = "ic_sons" if job_mode == "source" or finding.classification == "intra_cell" else "rc_sons"
            elif finding.kind in ("adjacent_neurite_candidate", "neurite_neurite_candidate", "source_solver_fallback",
                                  "source_bivariate_candidate", "source_pair_candidate"):
                name = {"inter_cell": "rc_nsns", "intra_cell_inter_neurite": "icrn_nsns",
                        "intra_cell_intra_neurite": "icin_nsns",
                        "intra_cell_intra_neurite_adjacent": "icin_nsns"}[finding.classification]
            if name:
                for edge in finding.segments:
                    flags[edge].add(name)
        statuses = tuple(AnalyticSegmentFlags(edge, **{name: True for name in flags[edge]},
                                             unresolved=tuple(unresolved_by_edge[edge]),
                                             pair_limit_may_affect=(pair_limit_reached and box[0][1] >= cutoff_x),
                                             source_plan_may_affect=bool(source_plan and source_plan.truncated))
                         for edge, _, box in segments)
        return AnalyticReport(len(segments), tested, tested_adjacent, tuple(findings), tuple(unresolved),
                              segment_flags=statuses, pair_limit_reached=pair_limit_reached,
                              first_untested_pair=first_untested_pair, job_mode=job_mode,
                              source_plan_truncated=bool(source_plan and source_plan.truncated),
                              source_job_counts=source_plan.as_dict()["job_counts"] if source_plan else None)

    if source_plan is None:
        pair_indices = _overlapping_box_pairs([segment[2] for segment in segments])
    else:
        indices = {segment[0]: index for index, segment in enumerate(segments)}
        pair_indices = ((indices[job.segments[0]], indices[job.segments[1]]) for job in source_plan.jobs
                        if job.kind in ("NSNS_ADJ", "NSNS_NONADJ"))
    def mark_source_fallback(edge_a, edge_b, first, second, classification):
        adjacent = classification == "intra_cell_intra_neurite_adjacent"
        try:
            source_hits = source_neurite_pair_candidates(
                source_segments[edge_a], source_segments[edge_b], adjacent=adjacent,
                univariate_tolerance=univariate_tolerance,
                bivariate_tolerance=bivariate_tolerance, max_rectangles=max_rectangles,
                max_intervals=max_intervals)
        except SourceBivariateFailure as error:
            findings.append(AnalyticFinding(
                "source_solver_fallback", (edge_a, edge_b),
                f"source-style bivariate clipping failed ({error}); source defaults to a positive result without a geometric witness",
                classification))
            return
        except ClippingConvergenceError as error:
            message = f"segments {edge_a} and {edge_b}: source pair trace unresolved: {error}"
        else:
            threshold = (Fraction(first.max_radius)+Fraction(second.max_radius))**2
            grouped = {"interior": [], "boundary": []}
            for hit in source_hits:
                distance = exact_source_distance_squared(source_segments[edge_a], source_segments[edge_b],
                                                         hit.t, hit.u)
                verified = distance <= threshold
                value, exact = _report_distance_squared(distance) if verified else (hit.distance_squared, None)
                rectangle = hit.rectangle
                evidence = AnalyticCandidate((hit.t, hit.u), value,
                                             "source_root_witness" if verified else "source_threshold",
                                             ((rectangle.x0, rectangle.x1), (rectangle.y0, rectangle.y1)),
                                             exact_value=exact)
                grouped["interior" if hit.source == "interior" else "boundary"].append(evidence)
            for group, kind in (("interior", "source_bivariate_candidate"), ("boundary", "source_pair_candidate")):
                if grouped[group]:
                    findings.append(AnalyticFinding(kind, (edge_a, edge_b),
                                                    f"source-style pair {group} threshold triggered; "
                                                    "candidates are numerical", classification, tuple(grouped[group])))
            return
        unresolved.append(message)
        unresolved_by_edge[edge_a].append(message)
        unresolved_by_edge[edge_b].append(message)
    for i, j in pair_indices:
        edge_a, curve_a, _ = segments[i]
        edge_b, curve_b, _ = segments[j]
        if tested+tested_adjacent >= max_pairs:
            unresolved.append(f"candidate-pair limit {max_pairs} reached")
            # The sweep orders events by their left x-bound. Any segment
            # ending before this event cannot participate in a later pair.
            # Remaining segments are conservatively marked: some may have no
            # unchecked partner, but we do not enumerate the rest of a
            # potentially quadratic pair set after reaching the work limit.
            cutoff_x = (-inf if source_plan is not None else
                        max(segments[i][2][0][0], segments[j][2][0][0]))
            return report(pair_limit_reached=True, first_untested_pair=(edge_a, edge_b), cutoff_x=cutoff_x)
        try:
            if set(edge_a) & set(edge_b):
                tested_adjacent += 1
                if edge_a[0] == edge_b[0]:
                    first, second, end_start = curve_a, curve_b, False
                    ordered_edges = edge_a, edge_b
                elif edge_a[1] == edge_b[0]:
                    first, second, end_start = curve_a, curve_b, True
                    ordered_edges = edge_a, edge_b
                elif edge_b[1] == edge_a[0]:
                    first, second, end_start = curve_b, curve_a, True
                    ordered_edges = edge_b, edge_a
                else:
                    raise ValueError("neurite edges unexpectedly share a destination")
                hits = adjacent_neurite_candidates(first, second, first_end_second_start=end_start,
                                                   univariate_tolerance=univariate_tolerance,
                                                   bivariate_tolerance=bivariate_tolerance,
                                                   max_rectangles=max_rectangles,
                                                   check_end_start=check_adjacent_end_start, max_intervals=max_intervals)
                if hits:
                    findings.append(AnalyticFinding("adjacent_neurite_candidate", ordered_edges,
                                                    "adjacent bounding pipes overlap beyond the junction",
                                                    "intra_cell_intra_neurite_adjacent",
                                                    tuple(_curve_candidate(hit) for hit in hits)))
                elif source_plan is not None and not end_start:
                    mark_source_fallback(*ordered_edges, first, second, "intra_cell_intra_neurite_adjacent")
            else:
                tested += 1
                hits = neurite_neurite_candidates(curve_a, curve_b, univariate_tolerance=univariate_tolerance,
                                                  bivariate_tolerance=bivariate_tolerance,
                                                  max_rectangles=max_rectangles, max_intervals=max_intervals)
                if hits:
                    findings.append(AnalyticFinding("neurite_neurite_candidate", (edge_a, edge_b),
                                                    "nonadjacent bounding pipes overlap", pair_classification(edge_a, edge_b),
                                                    tuple(_curve_candidate(hit) for hit in hits)))
                elif source_plan is not None:
                    mark_source_fallback(edge_a, edge_b, curve_a, curve_b, pair_classification(edge_a, edge_b))
        except ClippingConvergenceError as error:
            message = f"segments {edge_a} and {edge_b}: {error}"
            unresolved.append(message)
            unresolved_by_edge[edge_a].append(message)
            unresolved_by_edge[edge_b].append(message)
            if isinstance(error, IntersectionSolveError):
                evidence = tuple(_curve_candidate(hit, "corner_fallback" if hit.source == "corner" else hit.source)
                                 for hit in error.candidates)
                if evidence:
                    adjacent = bool(set(edge_a) & set(edge_b))
                    findings.append(AnalyticFinding(
                        "adjacent_neurite_candidate" if adjacent else "neurite_neurite_candidate",
                        ordered_edges if adjacent else (edge_a, edge_b),
                        "partial witness overlaps; full pair solve unresolved",
                        "intra_cell_intra_neurite_adjacent" if adjacent else pair_classification(edge_a, edge_b),
                        evidence))
                elif source_plan is not None:
                    if set(edge_a) & set(edge_b):
                        if not end_start:
                            mark_source_fallback(*ordered_edges, first, second, "intra_cell_intra_neurite_adjacent")
                    else:
                        mark_source_fallback(edge_a, edge_b, curve_a, curve_b, pair_classification(edge_a, edge_b))
                continue
            if set(edge_a) & set(edge_b):
                # Exclude the allowed junction corner using the same parameter
                # region as the adjacent-pair check, but retain other endpoint
                # witnesses when the full solve cannot complete.
                threshold = (Fraction(first.max_radius)+Fraction(second.max_radius))**2
                corners = []
                for t in (0., 1.):
                    for u in (0., 1.):
                        if (t >= .99 if end_start else t <= .01) and u <= .01:
                            continue
                        witness = _verified_corner_candidate(first, second, t, u, threshold)
                        if witness is not None:
                            corners.append(witness)
                if corners:
                    findings.append(AnalyticFinding("adjacent_neurite_candidate", ordered_edges,
                                                    "endpoint witness overlaps; full adjacent solve unresolved",
                                                    "intra_cell_intra_neurite_adjacent", tuple(corners)))
                elif source_plan is not None and not end_start:
                    mark_source_fallback(*ordered_edges, first, second, "intra_cell_intra_neurite_adjacent")
            else:
                # Even when the bivariate solve is unresolved, an endpoint
                # pair can independently prove that the bounding pipes meet.
                # Retain both the positive witness and the unresolved job.
                threshold = (Fraction(curve_a.max_radius)+Fraction(curve_b.max_radius))**2
                corners = []
                for t in (0., 1.):
                    for u in (0., 1.):
                        witness = _verified_corner_candidate(curve_a, curve_b, t, u, threshold)
                        if witness is not None:
                            corners.append(witness)
                if corners:
                    findings.append(AnalyticFinding("neurite_neurite_candidate", (edge_a, edge_b),
                                                    "endpoint witness overlaps; full pair solve unresolved",
                                                    pair_classification(edge_a, edge_b), tuple(corners)))
                elif source_plan is not None:
                    mark_source_fallback(edge_a, edge_b, curve_a, curve_b, pair_classification(edge_a, edge_b))
    return report()
