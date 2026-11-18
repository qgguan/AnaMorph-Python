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
# See LICENSE and NOTICE.md for license and attribution terms.
"""One-command analytic meshing, remeshing, and reporting workflow."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import inspect
import json
from math import isfinite
from pathlib import Path
import platform
import shutil
from statistics import median
import tempfile

from .analytic_mesher import assemble_redblue_neuron
from .analytic_network import analyze_analytic_partial
from .mesh_intersections import find_self_intersections
from .quality import mesh_quality
from .postprocess import standard_postprocess, standard_settings
from .redblue import _edge_faces
from .remesh import isotropic_remesh
from .swc import read_swc


def _sha256(path):
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mesh_report(mesh, check_intersections):
    statistics = asdict(mesh.statistics())
    try:
        _edge_faces(mesh)
    except ValueError as error:
        topology = {"valid": False, "detail": str(error)}
    else:
        topology = {"valid": True, "detail": None}
    if check_intersections:
        try:
            hits, truncated = find_self_intersections(mesh, max_hits=10, include_adjacent=True)
        except (ValueError, OverflowError) as error:
            intersections = {"status": "unresolved", "detail": str(error), "candidates": [], "truncated": False}
        else:
            intersections = {"status": "candidates_found" if hits else "none_found",
                             "detail": None, "candidates": [asdict(hit) for hit in hits], "truncated": truncated}
    else:
        intersections = {"status": "not_checked", "detail": None, "candidates": [], "truncated": False}
    intersections.update({"includes_adjacent": True if check_intersections else None,
                          "certifies_nonintersection": False})
    return {"statistics": statistics, "quality": mesh_quality(mesh), "topology": topology,
            "self_intersections": intersections}


def _readable_report(data):
    raw, remeshed = data["raw"], data["remeshed"]
    before, after = raw["statistics"], remeshed["statistics"]
    raw_edges, new_edges = raw["quality"]["edge_lengths"], remeshed["quality"]["edge_lengths"]
    settings = data["settings"]
    def number(value):
        return "n/a" if value is None else str(value) if isinstance(value, int) else f"{value:.6g}"
    def intersection_summary(mesh):
        check = mesh["self_intersections"]
        if check["status"] == "candidates_found":
            return f"{len(check['candidates'])} candidate(s)" + (" (limit reached)" if check["truncated"] else "")
        if check["status"] == "unresolved":
            return "unresolved: " + check["detail"]
        return check["status"].replace("_", " ")
    analysis = data["analysis"]
    if analysis["performed"]:
        diagnostic = analysis["report"]
        analysis_summary = (f"Optional geometric analysis: {len(diagnostic['findings'])} findings; "
                            f"{len(diagnostic['unresolved'])} unresolved. This analysis is incomplete "
                            "and does not block meshing.")
    else:
        analysis_summary = "Pre-meshing geometric analysis was not requested."
    remesh = settings["remeshing"]
    lines = ["# Mesh workflow report", "", f"Input: `input.swc` (SHA-256 `{data['input']['sha256']}`)",
             "Meshing method: analytic canal surfaces and red-blue stitching."]
    if remesh["method"] == "standard":
        lines.extend(["Postprocessing: source-style edge collapse followed by HC smoothing.",
                      "Edge collapse: alpha 1.5, lambda 0.125, mu 0.5, depth 5; source safety rule.",
                      "HC smoothing: alpha 0.4, beta 0.7, 10 iterations."])
    else:
        lines.extend(["Postprocessing: isotropic remeshing.",
                      f"Target edge length: {number(remesh['target'])} input units ({remesh['target_selection']}).",
                      f"Remeshing iterations: {remesh['iterations']}."])
    lines.extend([analysis_summary, "", "| Measurement | Raw | Remeshed |", "| --- | ---: | ---: |"])
    for title, key in (("Vertices", "vertices"), ("Faces", "faces"), ("Surface area", "area"),
                       ("Volume", "volume"), ("Boundary edges", "boundary_edges"),
                       ("Nonmanifold edges", "nonmanifold_edges"), ("Maximum aspect ratio", "aspect_ratio_max")):
        lines.append(f"| {title} | {number(before[key])} | {number(after[key])} |")
    for title, key in (("Shortest edge", "min"), ("Edge length p05", "p05"), ("Median edge length", "median"),
                       ("Edge length p95", "p95"), ("Longest edge", "max")):
        lines.append(f"| {title} | {number(raw_edges[key])} | {number(new_edges[key])} |")
    lines.extend(["", f"Area change: {number(data['change_percent']['area'])}%; "
                  f"volume change: {number(data['change_percent']['volume'])}%.",
                  f"Topology checks: raw {'passed' if raw['topology']['valid'] else 'failed'}; "
                  f"remeshed {'passed' if remeshed['topology']['valid'] else 'failed'}.",
                  f"Numerical intersection check: raw {intersection_summary(raw)}; "
                  f"remeshed {intersection_summary(remeshed)}.", "",
                  "Postprocessing changes the surface and does not guarantee quasi-uniformity. "
                  "Numerical intersection checks do not certify geometric validity.",
                  "All settings and detailed results are recorded in `report.json`.", ""])
    return "\n".join(lines)


def run_workflow(swc_path, output_dir, *, target_edge_length=None, iterations=None,
                 angular_segments=6, soma_refinements=2, force=False,
                 match_cpp=False, check_intersections=True, analyze=False, progress=None,
                 postprocess_method="standard"):
    """Generate raw/remeshed OBJs and a reproducible diagnostic report.

    Existing result files are never overwritten. The default is source-style
    edge collapse and HC smoothing. For isotropic remeshing, a missing target
    uses the median input neurite radius as a heuristic.
    Optional geometric analysis reports findings without blocking meshing.
    ``force`` remains accepted as a compatibility alias for the default.
    """
    if target_edge_length is not None and (not isfinite(target_edge_length) or target_edge_length <= 0):
        raise ValueError("target edge length must be positive and finite")
    if iterations is not None and (type(iterations) is not int or iterations < 1):
        raise ValueError("iterations must be a positive integer")
    if postprocess_method not in ("standard", "isotropic"):
        raise ValueError("postprocess method must be standard or isotropic")
    if postprocess_method == "standard" and (target_edge_length is not None or iterations is not None):
        raise ValueError("target edge length and remeshing iterations require isotropic postprocessing (--isotropic)")
    if postprocess_method == "isotropic" and iterations is None:
        iterations = 5
    if type(angular_segments) is not int or angular_segments < 3:
        raise ValueError("angular segments must be an integer of at least 3")
    if type(soma_refinements) is not int or soma_refinements < 0:
        raise ValueError("soma refinements must be a nonnegative integer")
    if any(type(value) is not bool for value in (force, match_cpp, check_intersections, analyze)):
        raise ValueError("workflow switches must be boolean")
    if force and analyze:
        raise ValueError("force cannot be combined with geometric analysis")
    source = Path(swc_path).resolve(strict=True)
    if not source.is_file():
        raise ValueError("input SWC must be a file")
    destination = Path(output_dir).resolve()
    names = ("input.swc", "raw.obj", "remeshed.obj", "report.md", "report.json")
    for name in names:
        target = destination/name
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"result already exists: {target}")
    destination.mkdir(parents=True, exist_ok=True)
    notify = progress if progress is not None else lambda message: None
    with tempfile.TemporaryDirectory(prefix=".anamorph-workflow-", dir=destination) as directory:
        stage = Path(directory)
        shutil.copyfile(source, stage/"input.swc")
        tree = read_swc(stage/"input.swc")
        target_selection = None
        if postprocess_method == "isotropic" and target_edge_length is None:
            radii = [node.radius for node in tree.nodes.values() if node.kind in (2, 3, 4)]
            if not radii:
                raise ValueError("cannot estimate edge length without neurite radii; set --target-edge-length")
            target_edge_length = median(radii)
            target_selection = "median input neurite radius (heuristic)"
        elif postprocess_method == "isotropic":
            target_selection = "user supplied"
        if postprocess_method == "isotropic" and (not isfinite(target_edge_length) or target_edge_length <= 0):
            raise ValueError("estimated target edge length is not positive and finite")
        meshing = {name: parameter.default for name, parameter in
                   inspect.signature(assemble_redblue_neuron).parameters.items()
                   if name != "tree" and not name.startswith("_") and parameter.default is not inspect.Parameter.empty}
        meshing.update(angular_segments=angular_segments, soma_refinements=soma_refinements)
        if match_cpp:
            meshing.update(source_delaunay=True, source_render_vector=True, source_frame=True,
                           source_soma_centroid=True, preserve_surface_roots=False,
                           precondition_norm_mode="source-x87", arc_length_mode="source-x87",
                           random_generator="windows-c", quad_sqrt_mode="windows-x87",
                           render_trig_mode="windows-x87", source_retry_semantics=True)
        analysis_settings = {"mode": "extended", "blocks_meshing": False, "check_adjacent_end_start": False,
                             "max_pairs": 100000, "max_rectangles": 10000, "max_intervals": 100000,
                             "univariate_tolerance": 1e-6, "bivariate_tolerance": 1e-4}
        if postprocess_method == "isotropic":
            notify(f"Target edge length: {target_edge_length:.6g} ({target_selection})")
        if analyze:
            notify("Running optional geometric analysis")
            shared = ("strategy", "parametrization", "preconditioning", "pc_alpha", "pc_beta", "pc_gamma",
                      "scale_radius", "filter_angle", "radius_ratio", "preserve_surface_roots",
                      "source_soma_centroid", "precondition_norm_mode")
            report = analyze_analytic_partial(tree, prepare=True, job_mode="extended",
                                              **{name: meshing[name] for name in shared})
            analysis = {"performed": True, "report": report.as_dict(), "complete": report.complete}
        else:
            analysis = {"performed": False, "reason": "not requested", "complete": False}
        notify("Generating the analytic surface")
        raw = assemble_redblue_neuron(tree, **meshing)
        if postprocess_method == "standard":
            notify("Applying source-style edge collapse and HC smoothing")
            remesh_settings = {"method": "standard", **standard_settings()}
            processed = standard_postprocess(raw)
        else:
            notify("Applying isotropic remeshing")
            options = {"target": target_edge_length, "iterations": iterations, "max_operations": 300000}
            remesh_settings = {"method": "isotropic", **options, "target_selection": target_selection}
            processed = isotropic_remesh(raw, **options)
        notify("Checking meshes and writing the report")
        raw_report = _mesh_report(raw, check_intersections)
        processed_report = _mesh_report(processed, check_intersections)
        before, after = raw_report["statistics"], processed_report["statistics"]
        changes = {name: (100*(after[name]/before[name]-1) if before[name] else None)
                   for name in ("area", "volume")}
        raw.write_obj(stage/"raw.obj")
        processed.write_obj(stage/"remeshed.obj")
        try:
            package_version = version("anamorph-python")
        except PackageNotFoundError:
            package_version = None
        report_data = {
            "input": {"original_path": str(source), "saved_copy": "input.swc", "sha256": _sha256(stage/"input.swc")},
            "outputs": {"raw": {"file": "raw.obj", "sha256": _sha256(stage/"raw.obj")},
                        "remeshed": {"file": "remeshed.obj", "sha256": _sha256(stage/"remeshed.obj")}},
            "settings": {"meshing_method": "analytic", "meshing": meshing, "source_comparison_preset": match_cpp,
                         "analysis": analysis_settings if analyze else {"mode": "not_requested"},
                         "remeshing": remesh_settings,
                         "check_intersections": check_intersections},
            "environment": {"python": platform.python_version(), "anamorph_version": package_version},
            "analysis": analysis, "raw": raw_report, "remeshed": processed_report,
            "change_percent": changes,
            "limitations": ["Optional geometric analysis is incomplete." if analyze
                            else "Pre-meshing geometric analysis was not requested.",
                            "Postprocessing changes the surface and does not guarantee quasi-uniformity.",
                            "Numerical intersection checks do not certify non-intersection."],
        }
        (stage/"report.md").write_text(_readable_report(report_data), encoding="utf-8")
        (stage/"report.json").write_text(json.dumps(report_data, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        created = []
        try:
            for name in names:
                target = destination/name
                with target.open("xb") as out:
                    created.append(target)
                    with (stage/name).open("rb") as src:
                        shutil.copyfileobj(src, out)
        except BaseException:
            for path in reversed(created):
                path.unlink(missing_ok=True)
            raise
    return {"output_directory": str(destination), "input": str(destination/"input.swc"),
            "method": "analytic", "postprocessing_method": postprocess_method,
            "raw_mesh": str(destination/"raw.obj"), "remeshed_mesh": str(destination/"remeshed.obj"),
            "report": str(destination/"report.md"), "detailed_report": str(destination/"report.json"),
            "target_edge_length": target_edge_length, "target_selection": target_selection,
            "change_percent": changes,
            "checks": {"raw_topology_valid": raw_report["topology"]["valid"],
                       "remeshed_topology_valid": processed_report["topology"]["valid"],
                       "raw_intersections": raw_report["self_intersections"]["status"],
                       "remeshed_intersections": processed_report["self_intersections"]["status"]}}
