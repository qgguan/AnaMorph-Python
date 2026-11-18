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

"""Source-style network intersection job construction.

These utilities reproduce the C++ job builder's selection rules for
comparison. They do not execute jobs or certify network consistency.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass

from .analytic_intersections import CanalSegment
from .bernstein import BernsteinPolynomial
from .geometry import Spline, partition_paths
from .mesh import sub
from .model import prepare_morphology
from .source_canal_polynomials import _components
from .source_spline import SourceCanalSegment, source_root_cylinder, source_spline_controls
from .spatial import _BoxIndex, _overlap
from .swc import Morphology


@dataclass(frozen=True, slots=True)
class SourceAnalysisJob:
    """A source-style intersection job, identified by SWC/prepared node IDs."""
    kind: str
    segments: tuple[tuple[int, int], ...]
    soma: int | None = None
    root_segment: bool | None = None
    first_end_second_start: bool | None = None


@dataclass(frozen=True, slots=True)
class SourceAnalysisPlan:
    segment_count: int
    path_count: int
    jobs: tuple[SourceAnalysisJob, ...]
    truncated: bool

    def as_dict(self):
        result = asdict(self)
        result["job_counts"] = dict(sorted(Counter(job.kind for job in self.jobs).items()))
        result["executes_checks"] = False
        result["certifies_consistency"] = False
        return result


def _extend(box, percentage, minimum):
    result = []
    for lo, hi in box:
        padding = max(minimum, abs(hi-lo)*percentage)
        result.append((lo-padding, hi+padding))
    return tuple(result)


def _source_segment_box(segment, *, subdivision_depth=8):
    """Reproduce the source's subdivided curve and canal bounding box."""
    box = []
    for coefficients in _components(segment):
        pieces = [BernsteinPolynomial(coefficients)]
        for _ in range(subdivision_depth):
            pieces = [child for piece in pieces for child in piece.split(.5)]
        coefficients = [value for piece in pieces for value in piece.coefficients]
        box.append((min(coefficients), max(coefficients)))
    # BezierCurve::getBoundingBox pads the spine; BezierCanalSurface then
    # adds the maximum radius and pads the resulting box a second time.
    box = _extend(tuple(box), .025, .001)
    box = tuple((lo-segment.max_radius, hi+segment.max_radius) for lo, hi in box)
    return _extend(box, .025, .001)


def _source_soma_box(soma):
    return _extend(tuple((value-soma.radius, value+soma.radius) for value in soma.position), .025, .001)


def _union_boxes(boxes):
    return tuple((min(box[axis][0] for box in boxes), max(box[axis][1] for box in boxes))
                 for axis in range(3))


def plan_source_analysis_jobs(tree: Morphology, *, max_jobs: int = 100000, prepare: bool = True,
                              preconditioning: bool = True, pc_alpha: float = 3., pc_beta: float = 1.5,
                              pc_gamma: float = 10., scale_radius: float = 1.,
                              strategy: str = "max-chordal-depth", parametrization: str = "chord-length",
                              filter_angle: float = 90., radius_ratio: float = 10.,
                              preserve_surface_roots: bool = False, source_soma_centroid: bool = False,
                              precondition_norm_mode: str = "robust",
                              _edge_order=None) -> SourceAnalysisPlan:
    """Plan the checked-in C++ network analysis jobs without executing them.

    This intentionally retains the source's omission of soma checks on each
    path's final segment and its root-segment exemption against every soma.
    The bounds use the source's subdivision and padding rules. Preparation
    tracks source-style edge order when its initial allocation can be recovered;
    other cases use an approximate order. Floating-point and path-order
    differences can still change selections. ``prepare=False`` is a diagnostic
    extension; ``source_soma_centroid`` applies only when preparation is enabled.
    """
    if type(max_jobs) is not int or max_jobs < 0:
        raise ValueError("max_jobs must be a nonnegative integer")
    if type(source_soma_centroid) is not bool:
        raise ValueError("source_soma_centroid must be boolean")
    if precondition_norm_mode not in ("robust", "source", "source-x87"):
        raise ValueError("invalid precondition norm mode")
    if prepare:
        prepared = prepare_morphology(tree, preconditioning=preconditioning, alpha=pc_alpha, beta=pc_beta,
                                      gamma=pc_gamma, scale_radius=scale_radius,
                                      preserve_surface_roots=preserve_surface_roots,
                                      source_soma_centroid=source_soma_centroid,
                                      precondition_norm_mode=precondition_norm_mode)
        _edge_order = prepared.source_edge_order
        tree = prepared.morphology
    elif scale_radius != 1.:
        raise ValueError("radius scaling requires morphology preparation")
    elif source_soma_centroid:
        raise ValueError("source soma centroid requires morphology preparation")
    elif precondition_norm_mode != "robust":
        raise ValueError("precondition norm mode requires morphology preparation")
    paths = []
    segments_by_edge = {}
    for path in partition_paths(tree, strategy, filter_angle, radius_ratio):
        nodes = [tree.nodes[i] for i in path]
        parent = tree.nodes.get(nodes[0].parent)
        root = parent is not None and parent.kind == 1
        tangent = sub(nodes[0].position, parent.position) if root else None
        spline = Spline.interpolate([node.position for node in nodes], parametrization=parametrization,
                                    start_direction=tangent)
        _, controls = source_spline_controls([node.position for node in nodes],
                                             parametrization=parametrization, start_direction=tangent)
        path_segments = []
        for i, (first, second) in enumerate(zip(nodes, nodes[1:])):
            local = CanalSegment.from_spline(spline, i, first.radius, second.radius)
            segment = SourceCanalSegment(local.power, first.radius, second.radius, controls[i])
            item = ((first.id, second.id), segment, _source_segment_box(segment))
            path_segments.append(item)
            segments_by_edge[item[0]] = item
        boxes = [item[2] for item in path_segments]
        if root:
            cylinder = source_root_cylinder(parent.position, nodes[0].position, nodes[0].radius)
            boxes.append(_source_segment_box(cylinder))
        paths.append((path_segments, _extend(_union_boxes(boxes), .025, .001)))
    somas = [(node, _source_soma_box(node)) for node in tree.nodes.values() if node.kind == 1]
    soma_index = _BoxIndex([box for _, box in somas])
    path_index = _BoxIndex([box for _, box in paths])
    # The source schedules single-segment jobs in graph-edge ID order.
    edges = ([edge for edge in _edge_order if edge in segments_by_edge]
             if _edge_order is not None else None)
    if edges is not None and (len(edges) != len(segments_by_edge) or set(edges) != set(segments_by_edge)):
        raise ValueError("prepared source edge order does not match the neurite segments")
    if edges is None:
        edges = []
        for root in tree.roots:
            queue = deque([root])
            while queue:
                parent = queue.popleft()
                for child in tree.children[parent]:
                    edge = (parent, child)
                    if edge in segments_by_edge:
                        edges.append(edge)
                    queue.append(child)
    jobs = []
    class _Limit(Exception):
        pass
    def add(job):
        if len(jobs) >= max_jobs:
            raise _Limit
        jobs.append(job)
    try:
        for edge in edges:
            for kind in ("REG", "LSI", "GSI"):
                add(SourceAnalysisJob(kind, (edge,)))
        for index, (first_path, first_box) in enumerate(paths):
            for i in range(len(first_path)-1):
                edge, _, box = first_path[i]
                parent = tree.nodes.get(tree.nodes[edge[0]].parent)
                root_segment = parent is not None and parent.kind == 1
                for soma_index_in_tree in soma_index.query(box):
                    soma, _ = somas[soma_index_in_tree]
                    add(SourceAnalysisJob("SONS", (edge,), soma=soma.id, root_segment=root_segment))
                add(SourceAnalysisJob("NSNS_ADJ", (edge, first_path[i+1][0]), first_end_second_start=True))
                for second in first_path[i+2:]:
                    if _overlap(box, second[2]):
                        add(SourceAnalysisJob("NSNS_NONADJ", (edge, second[0])))
            for other_path_index in path_index.query(first_box):
                if other_path_index <= index:
                    continue
                second_path, _ = paths[other_path_index]
                for edge, _, box in first_path:
                    for other, _, other_box in second_path:
                        if edge[0] == other[0]:
                            add(SourceAnalysisJob("NSNS_ADJ", (edge, other), first_end_second_start=False))
                        elif edge[1] == other[0]:
                            add(SourceAnalysisJob("NSNS_ADJ", (edge, other), first_end_second_start=True))
                        elif other[1] == edge[0]:
                            add(SourceAnalysisJob("NSNS_ADJ", (other, edge), first_end_second_start=True))
                        elif edge[1] == other[1]:
                            raise ValueError("neurite segments share a destination")
                        elif _overlap(box, other_box):
                            add(SourceAnalysisJob("NSNS_NONADJ", (edge, other)))
    except _Limit:
        return SourceAnalysisPlan(len(edges), len(paths), tuple(jobs), True)
    return SourceAnalysisPlan(len(edges), len(paths), tuple(jobs), False)
