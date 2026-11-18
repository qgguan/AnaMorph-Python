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
"""Optional AnaMorph MorphView binary export and inspection.

The source's writer stores native-endian uint32/double data and only the first
soma. This module uses the little-endian layout of common x86 builds; the
viewer itself is not included in the checked-in source.
"""
from collections import Counter
from dataclasses import asdict, dataclass
from math import isfinite, isnan
import os
from pathlib import Path
import struct
import tempfile

from .analytic_intersections import CanalSegment
from .geometry import Spline, partition_paths
from .mesh import sub
from .model import prepare_morphology
from .source_analysis import trace_source_analysis
from .source_jobs import _extend, _source_segment_box, _source_soma_box, _union_boxes
from .source_spline import SourceCanalSegment, source_root_cylinder, source_spline_controls


_FLAGS = ("pmdv", "smdv", "lsi", "gsi", "ic_sons", "icrn_nsns", "icin_nsns")


@dataclass(frozen=True, slots=True)
class MorphViewVertex:
    position: tuple[float, float, float]
    radius: float
    branching: bool


@dataclass(frozen=True, slots=True)
class MorphViewSegment:
    coefficients: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]
    box_min: tuple[float, float, float]
    box_size: tuple[float, float, float]
    flags: tuple[bool, bool, bool, bool, bool, bool, bool]


@dataclass(frozen=True, slots=True)
class MorphViewPath:
    parameters: tuple[float, ...]
    radii: tuple[float, ...]
    box_min: tuple[float, float, float]
    box_size: tuple[float, float, float]
    segments: tuple[MorphViewSegment, ...]


@dataclass(frozen=True, slots=True)
class MorphView:
    vertices: tuple[MorphViewVertex, ...]
    paths: tuple[MorphViewPath, ...]
    soma_position: tuple[float, float, float]
    soma_radius: float
    soma_box_min: tuple[float, float, float]
    soma_box_size: tuple[float, float, float]

    def as_dict(self):
        result = asdict(self)
        result["flag_names"] = _FLAGS
        return result

    def summary(self):
        counts = Counter(name for path in self.paths for segment in path.segments
                         for name, enabled in zip(_FLAGS, segment.flags) if enabled)
        return {"vertices": len(self.vertices), "paths": len(self.paths),
                "segments": sum(len(path.segments) for path in self.paths),
                "flag_counts": dict(sorted(counts.items())), "soma_position": self.soma_position,
                "soma_radius": self.soma_radius, "flag_names": _FLAGS}


def read_morphview(path, *, windows_text_mode=False):
    """Read little-endian double .amv data, optionally undoing Windows text output.

    The source opens its binary writer with ``"w"``; Windows text mode then
    expands every newline byte. Specify this option only for such a file: an
    undamaged binary file may legitimately contain carriage-return/newline.
    """
    if type(windows_text_mode) is not bool:
        raise ValueError("windows_text_mode must be a boolean")
    data = Path(path).read_bytes()
    if windows_text_mode:
        data = data.replace(b"\r\n", b"\n")
    offset = 0
    def unpack(fmt):
        nonlocal offset
        layout = struct.Struct("<"+fmt)
        if len(data)-offset < layout.size:
            raise ValueError("truncated MorphView file")
        result = layout.unpack_from(data, offset)
        offset += layout.size
        return result
    def numbers(count, *, allow_infinite=False):
        if count > (len(data)-offset)//8:
            raise ValueError("truncated MorphView numeric array")
        result = unpack(f"{count}d")
        if any(isnan(value) or (not allow_infinite and not isfinite(value)) for value in result):
            raise ValueError("MorphView file contains non-finite values")
        return result
    def box():
        values = numbers(6, allow_infinite=True)
        if any(value < 0 for value in values[3:]):
            raise ValueError("MorphView box size must be nonnegative")
        return values[:3], values[3:]
    count, = unpack("I")
    if count > (len(data)-offset)//33:
        raise ValueError("truncated MorphView vertex list")
    vertices = []
    for _ in range(count):
        values = numbers(4)
        branching, = unpack("B")
        if values[3] <= 0 or branching not in (0, 1):
            raise ValueError("invalid MorphView vertex")
        vertices.append(MorphViewVertex(values[:3], values[3], bool(branching)))
    count, = unpack("I")
    if count > (len(data)-offset)//4:
        raise ValueError("truncated MorphView path list")
    paths = []
    for _ in range(count):
        segments_count, = unpack("I")
        if segments_count > (len(data)-offset)//(3*4+3*8+6*8+7):
            raise ValueError("truncated MorphView path")
        parameters, radii = numbers(segments_count+1), numbers(segments_count+1)
        if any(radius <= 0 for radius in radii):
            raise ValueError("invalid MorphView path radius")
        box_min, box_size = box()
        segments = []
        for _ in range(segments_count):
            components = []
            for _ in range(3):
                degree, = unpack("I")
                components.append(numbers(degree+1))
            seg_min, seg_size = box()
            flags = unpack("7B")
            if any(flag not in (0, 1) for flag in flags):
                raise ValueError("invalid MorphView segment flag")
            segments.append(MorphViewSegment(tuple(components), seg_min, seg_size, tuple(map(bool, flags))))
        paths.append(MorphViewPath(parameters, radii, box_min, box_size, tuple(segments)))
    soma = numbers(4)
    if soma[3] <= 0:
        raise ValueError("invalid MorphView soma radius")
    soma_min, soma_size = box()
    if offset != len(data):
        raise ValueError("unexpected trailing MorphView data")
    return MorphView(tuple(vertices), tuple(paths), soma[:3], soma[3], soma_min, soma_size)


def _box_components(box):
    return tuple(pair[0] for pair in box), tuple(pair[1]-pair[0] for pair in box)


def create_morphview(tree, *, preconditioning=True, pc_alpha=3., pc_beta=1.5, pc_gamma=10.,
                     scale_radius=1., preserve_surface_roots=False, source_soma_centroid=False,
                     precondition_norm_mode="robust",
                     strategy="max-chordal-depth",
                     parametrization="chord-length", filter_angle=90., radius_ratio=10.,
                     max_jobs=100000, max_intervals=100000, max_rectangles=10000,
                     univariate_tolerance=1e-6, bivariate_tolerance=1e-4):
    """Build source-style .amv data after bounded numerical source analysis.

    Rejects truncated or unresolved analysis and a negative GSI result after
    a suppressed solver failure because .amv cannot represent an unknown flag.
    Positive source jobs are encoded, including pair-solver
    fallbacks. The format writes only the first soma and omits several source
    flags; it does not certify the morphology or match every C++ rounding case.
    """
    prepared = prepare_morphology(tree, preconditioning=preconditioning, alpha=pc_alpha, beta=pc_beta,
                                  gamma=pc_gamma, scale_radius=scale_radius,
                                  preserve_surface_roots=preserve_surface_roots,
                                  source_soma_centroid=source_soma_centroid,
                                  precondition_norm_mode=precondition_norm_mode)
    tree = prepared.morphology
    somas = [node for node in tree.nodes.values() if node.kind == 1]
    if not somas:
        raise ValueError("MorphView export requires a soma")
    trace = trace_source_analysis(tree, prepare=False, strategy=strategy, parametrization=parametrization,
                                  filter_angle=filter_angle, radius_ratio=radius_ratio, max_jobs=max_jobs,
                                  max_intervals=max_intervals, max_rectangles=max_rectangles,
                                  univariate_tolerance=univariate_tolerance, bivariate_tolerance=bivariate_tolerance,
                                  _edge_order=prepared.source_edge_order)
    if trace.unresolved:
        raise ValueError("MorphView export: " + "; ".join(trace.unresolved))
    if trace.truncated or any(item.result is None for item in trace.jobs):
        raise ValueError("MorphView export requires all source-style analysis jobs to complete")
    if any(item.job.kind == "GSI" and item.result is False and item.solver_failure is not None
           for item in trace.jobs):
        raise ValueError("MorphView export cannot encode an unresolved global self-intersection result")
    flags = {item.edge: item for item in trace.segment_flags}
    vertices = tuple(MorphViewVertex(node.position, node.radius,
                                     sum(tree.nodes[child].kind in (2, 3, 4) for child in tree.children[node.id]) > 1)
                     for node in tree.nodes.values() if node.kind in (2, 3, 4))
    paths = []
    for path in partition_paths(tree, strategy, filter_angle, radius_ratio):
        nodes = [tree.nodes[index] for index in path]
        parent = tree.nodes.get(nodes[0].parent)
        root = parent is not None and parent.kind == 1
        tangent = sub(nodes[0].position, parent.position) if root else None
        spline = Spline.interpolate([node.position for node in nodes], parametrization=parametrization,
                                    start_direction=tangent)
        parameters, controls = source_spline_controls([node.position for node in nodes],
                                                       parametrization=parametrization, start_direction=tangent)
        segments, boxes = [], []
        for i, (a, b) in enumerate(zip(nodes, nodes[1:])):
            local = CanalSegment.from_spline(spline, i, a.radius, b.radius)
            segment = SourceCanalSegment(local.power, a.radius, b.radius, controls[i])
            coefficients = controls[i]
            box = _source_segment_box(segment)
            boxes.append(box)
            box_min, box_size = _box_components(box)
            segment_flags = tuple(getattr(flags[(a.id, b.id)], name) for name in _FLAGS)
            segments.append(MorphViewSegment(coefficients, box_min, box_size, segment_flags))
        if root:
            cylinder = source_root_cylinder(parent.position, nodes[0].position, nodes[0].radius)
            boxes.append(_source_segment_box(cylinder))
        box_min, box_size = _box_components(_extend(_union_boxes(boxes), .025, .001))
        paths.append(MorphViewPath(parameters, tuple(node.radius for node in nodes), box_min, box_size,
                                   tuple(segments)))
    soma = somas[0]
    soma_min, soma_size = _box_components(_source_soma_box(soma))
    return MorphView(vertices, tuple(paths), soma.position, soma.radius, soma_min, soma_size)


def write_morphview(model, path):
    """Write .amv data via an atomic replacement in the output directory."""
    if not isinstance(model, MorphView):
        raise ValueError("expected MorphView data")
    target, temporary = Path(path), None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=target.parent, prefix=".anamorph-", suffix=".amv",
                                         delete=False) as output:
            temporary = Path(output.name)
            def pack(fmt, *values):
                output.write(struct.pack("<"+fmt, *values))
            def numbers(values):
                pack(f"{len(values)}d", *values)
            pack("I", len(model.vertices))
            for vertex in model.vertices:
                numbers((*vertex.position, vertex.radius))
                pack("B", vertex.branching)
            pack("I", len(model.paths))
            for path_item in model.paths:
                pack("I", len(path_item.segments))
                numbers(path_item.parameters)
                numbers(path_item.radii)
                numbers((*path_item.box_min, *path_item.box_size))
                for segment in path_item.segments:
                    for coefficients in segment.coefficients:
                        pack("I", len(coefficients)-1)
                        numbers(coefficients)
                    numbers((*segment.box_min, *segment.box_size))
                    pack("7B", *segment.flags)
            numbers((*model.soma_position, model.soma_radius, *model.soma_box_min, *model.soma_box_size))
        if read_morphview(temporary) != model:
            raise ValueError("MorphView data failed binary round-trip validation")
        os.replace(temporary, target)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
