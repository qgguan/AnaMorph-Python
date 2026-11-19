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

"""Command-line entry points for the currently available Python components."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from math import isfinite
import os
from pathlib import Path
import sys
import tempfile

from .mesh import read_obj
from .swc import read_swc
from .voxel import generate_voxel_mesh
from .precondition import precondition
from .smoothing import hc_smooth, laplacian_smooth, taubin_smooth, _validate_hc_options, _validate_taubin_options
from .roots import adjust_soma_roots
from .analysis import approximate_analysis
from .collapse import greedy_edge_collapse, _validate_collapse_options
from .remesh import isotropic_remesh
from .analytic_network import analyze_analytic_partial
from .source_jobs import plan_source_analysis_jobs
from .source_analysis import trace_source_analysis
from .morphview import create_morphview, read_morphview, write_morphview
from .analytic_mesher import (PartialAnalysisError, assemble_redblue_checked, assemble_redblue_neuron, assemble_redblue_to_obj,
                              assemble_redblue_checked_to_obj, render_individual_surfaces)
from .bernstein import ClippingConvergenceError
from .redblue import RedBlueError, _edge_faces, redblue_boolean_with_retries
from .mesh_intersections import find_self_intersections
from .quality import mesh_quality
from .postprocess import standard_postprocess, standard_settings
from .workflow import run_workflow


def _check_output_surface(label, mesh):
    """Check topology and intersection candidates without requiring volume statistics."""
    try:
        _edge_faces(mesh)
    except ValueError as error:
        raise ValueError(f"{label} mesh has invalid topology: {error}") from error
    hits, _ = find_self_intersections(mesh, max_hits=1, include_adjacent=True)
    if hits:
        raise ValueError(f"{label} mesh has a numerical self-intersection candidate")


def _add_collapse_arguments(parser):
    parser.add_argument("--collapse-alpha", type=float, help="edge-collapse aspect threshold (default: 1.5)")
    parser.add_argument("--collapse-area-floor", type=float, help="small-area factor, source lambda (default: 0.125)")
    parser.add_argument("--collapse-area-ceiling", type=float, help="area limit factor, source mu (default: 0.5)")
    parser.add_argument("--collapse-depth", type=int, help="area-neighborhood depth (default: 5)")
    parser.add_argument("--collapse-max-collapses", type=int, help="maximum collapse operations (default: 100000)")
    parser.add_argument("--collapse-safety", choices=("safe", "source"),
                        help="conservative checks or source rule (default: safe; source with --postprocess)")


def _add_taubin_arguments(parser):
    parser.add_argument("--taubin-iterations", type=int, help="Taubin iterations (default: 10 when selected)")
    parser.add_argument("--taubin-lambda", type=float, help="Taubin positive pass factor (default: 0.5)")
    parser.add_argument("--taubin-mu", type=float, help="Taubin negative pass factor (default: -0.53)")


def _taubin_options(args):
    return {"iterations": 10 if args.taubin_iterations is None else args.taubin_iterations,
            "lambda_factor": .5 if args.taubin_lambda is None else args.taubin_lambda,
            "mu_factor": -.53 if args.taubin_mu is None else args.taubin_mu}


def _collapse_options(args, enabled, *, default_safety="safe"):
    names = ("alpha", "area_floor", "area_ceiling", "depth", "max_collapses", "safety")
    overrides = {name: getattr(args, "collapse_"+name) for name in names
                 if getattr(args, "collapse_"+name) is not None}
    if not enabled:
        if overrides:
            raise ValueError("edge-collapse parameters require an enabled edge-collapse stage")
        return None
    options = dict(alpha=1.5, area_floor=.125, area_ceiling=.5, depth=5, max_collapses=100000, safety=default_safety)
    options.update(overrides)
    _validate_collapse_options(**options)
    return options


def _redblue_remesh_options(args):
    if args.uniform_edge_length is None:
        if args.uniform_iterations is not None or args.uniform_max_operations is not None:
            raise ValueError("uniform remeshing options require --uniform-edge-length")
        return None
    iterations = args.uniform_iterations if args.uniform_iterations is not None else 5
    max_operations = args.uniform_max_operations if args.uniform_max_operations is not None else 300000
    if not isfinite(args.uniform_edge_length) or args.uniform_edge_length <= 0 or iterations < 1 or max_operations < 1:
        raise ValueError("uniform target and operation limit must be positive; iterations must be at least one")
    return {"target": args.uniform_edge_length, "iterations": iterations, "max_operations": max_operations}


def _postprocess_redblue(model, collapse_options, hc_options, remesh_options):
    if collapse_options is not None or hc_options is not None or remesh_options is not None:
        print("Warning: post-processing changes surface area and volume.", file=sys.stderr)
    if collapse_options is not None:
        model = greedy_edge_collapse(model, **collapse_options)
    if hc_options is not None:
        model = hc_smooth(model, **hc_options)
    if remesh_options is not None:
        print("Warning: isotropic remeshing changes the surface and does not check global self-intersections.", file=sys.stderr)
        model = isotropic_remesh(model, **remesh_options)
    return model


def _generate_redblue_streamed(args, tree, meshing_options, collapse_options, hc_options, remesh_options):
    """Stage a streamed result; reload it only for requested whole-mesh work."""
    postprocessed = collapse_options is not None or hc_options is not None or remesh_options is not None
    analysis_options = {name: getattr(args, name) for name in
                        ("analysis_max_pairs", "analysis_max_rectangles", "analysis_max_intervals",
                         "analysis_univariate_tolerance", "analysis_bivariate_tolerance",
                         "analysis_job_mode", "analysis_source_job_limit")
                        if getattr(args, name) is not None}
    report = None
    output_path = Path(args.output)
    with tempfile.TemporaryDirectory(prefix=".anamorph-", dir=output_path.absolute().parent) as directory:
        staged = Path(directory)/"raw.obj"
        if args.require_no_findings or args.source_gate:
            analysis_options["analysis_gate"] = "all" if args.require_no_findings else "source"
            summary, report = assemble_redblue_checked_to_obj(
                tree, staged, flush_face_limit=args.flush_face_limit,
                check_adjacent_end_start=args.check_adjacent_end_start, **analysis_options, **meshing_options)
        else:
            summary = assemble_redblue_to_obj(tree, staged, flush_face_limit=args.flush_face_limit, **meshing_options)
        needs_mesh = args.check_output or postprocessed
        raw_model = read_obj(staged) if needs_mesh else None
        model = raw_model
        before = raw_model.statistics() if postprocessed else None
        if postprocessed:
            model = _postprocess_redblue(model, collapse_options, hc_options, remesh_options)
        output_model = raw_model if args.postprocess_output else model
        checked_surfaces = []
        if args.check_output:
            surfaces = (("raw", output_model), ("processed", model)) if args.postprocess_output else (("output", output_model),)
            for label, candidate in surfaces:
                _check_output_surface(label, candidate)
                checked_surfaces.append(label)
        if args.postprocess_output or not postprocessed:
            os.replace(staged, output_path)
        else:
            model.write_obj(output_path)
        if args.postprocess_output:
            model.write_obj(args.postprocess_output)
        result = {"output": args.output,
                  "vertices": len(output_model.vertices) if output_model is not None else summary["vertices"],
                  "faces": len(output_model.faces) if output_model is not None else summary["faces"],
                  "meshing": "experimental analytic canals and red-blue union", "method": "analytic", "forced": args.force,
                  "postprocessed": postprocessed, "analysis_performed": report is not None,
                  "partial_flush": {"face_limit": args.flush_face_limit, "flushes": summary["flushes"],
                                    "peak_active_faces": summary["peak_active_faces"],
                                    "reloaded_for_whole_mesh_work": bool(needs_mesh)}}
        if report is not None:
            result["analysis"] = {"complete": report.complete, "findings": len(report.findings),
                                  "unresolved": len(report.unresolved), "segments": report.segment_count,
                                  "checked_adjacent_end_start": args.check_adjacent_end_start,
                                  "job_mode": report.job_mode, "source_plan_truncated": report.source_plan_truncated,
                                  "source_jobs_clean": report.source_jobs_clean,
                                  "gate": "all" if args.require_no_findings else "source"}
        if args.check_output:
            result["output_check"] = {"checked_surfaces": checked_surfaces, "intersection_candidates": 0,
                                      "certifies_nonintersection": False}
        if args.postprocess_output:
            result["postprocess_output"] = {"output": args.postprocess_output,
                                            "vertices": len(model.vertices), "faces": len(model.faces)}
        if before:
            after = model.statistics()
            result["postprocess_change_percent"] = {
                "area": 100*(after.area/before.area-1) if before.area else None,
                "volume": 100*(after.volume/before.volume-1) if before.volume else None}
            result["postprocess_settings"] = {"edge_collapse": collapse_options, "hc_smoothing": hc_options}
            if remesh_options is not None:
                result["postprocess_settings"]["isotropic_remesh"] = remesh_options
                result["edge_lengths_before"] = mesh_quality(raw_model)["edge_lengths"]
                result["edge_lengths_after"] = mesh_quality(model)["edge_lengths"]
        return result


def main(argv: list[str] | None = None) -> int:
    print("Based on AnaMorph (https://github.com/NeuroBox3D/AnaMorph).", file=sys.stderr)
    parser = argparse.ArgumentParser(description="AnaMorph Python port (work in progress)")
    sub = parser.add_subparsers(dest="command", required=True)
    workflow = sub.add_parser("workflow", help="generate raw and remeshed OBJs with a quality report")
    workflow.add_argument("swc", help="input SWC morphology")
    workflow.add_argument("--output-dir", help="result folder (default: <input name>_results in the current directory)")
    workflow_method = workflow.add_mutually_exclusive_group()
    workflow_method.add_argument("--isotropic", dest="postprocess_method", action="store_const", const="isotropic",
                                 help="use isotropic remeshing (default)")
    workflow_method.add_argument("--standard-postprocess", dest="postprocess_method", action="store_const", const="standard",
                                 help="use source-style edge collapse and HC smoothing")
    workflow.set_defaults(postprocess_method="isotropic")
    workflow.add_argument("--smoothing", choices=("none", "hc", "taubin"),
                          help="smoothing method (default: taubin for isotropic, hc for standard)")
    _add_taubin_arguments(workflow)
    workflow.add_argument("--target-edge-length", type=float,
                          help="isotropic target in SWC units (default: median neurite radius)")
    workflow.add_argument("--iterations", type=int, help="isotropic remeshing iterations (default: 5)")
    workflow.add_argument("--angular-segments", type=int, default=6, help="canal ring segments (default: 6)")
    workflow.add_argument("--soma-refinements", type=int, default=2, help="soma sphere refinements (default: 2)")
    workflow.add_argument("--soma-blend-width", type=float,
                          help="junction width in neurite radii (default: 3 for isotropic, 2 for standard; 0 disables)")
    workflow.add_argument("--soma-blend-method", choices=("blend", "mortar"),
                          help="junction adjustment (default: mortar for isotropic, blend for standard; mortar is experimental)")
    workflow.add_argument("--match-cpp", action="store_true", help=argparse.SUPPRESS)
    workflow_checks = workflow.add_mutually_exclusive_group()
    workflow_checks.add_argument("--analyze", action="store_true",
                                 help="include an optional Python geometric analysis in the report")
    workflow_checks.add_argument("--force", action="store_true", help=argparse.SUPPRESS)
    workflow.add_argument("--skip-intersection-check", action="store_true",
                          help="omit the numerical mesh intersection checks from the report")
    swc = sub.add_parser("inspect", help="inspect an SWC tree and primary distance violations")
    swc.add_argument("swc")
    swc.add_argument("--alpha", type=float, default=3.0)
    analyze = sub.add_parser("analyze", help="run approximate sampled geometry diagnostics")
    analyze.add_argument("swc")
    analyze.add_argument("--samples", type=int, default=8)
    analyze.add_argument("--max-pairs", type=int, default=100000)
    analytic = sub.add_parser("analyze-analytic", help="run the incomplete analytic consistency checks")
    analytic.add_argument("swc")
    analytic.add_argument("--max-pairs", type=int, default=100000)
    analytic.add_argument("--max-rectangles", type=int, default=10000)
    analytic.add_argument("--max-intervals", type=int, default=100000,
                          help="univariate clipping interval limit per solve (default: 100000)")
    analytic.add_argument("--job-mode", choices=("extended", "source"), default="extended",
                          help="choose broader checks or the original source job selection")
    analytic.add_argument("--source-job-limit", type=int, default=1000000)
    analytic.add_argument("--check-adjacent-end-start", action="store_true",
                          help="also check consecutive segment pairs (the C++ source skips them)")
    analytic.add_argument("--source-surface-roots", action="store_true", help="reproduce the source's exact-boundary root insertion")
    analytic.add_argument("--source-soma-centroid", action="store_true", help="use source-order soma centroid summation")
    analytic.add_argument("--precondition-norm-mode", choices=("robust", "source", "source-x87"), default="robust",
                         help="distance arithmetic for preconditioning (default: robust)")
    analytic.add_argument("--univariate-tolerance", type=float, default=1e-6)
    analytic.add_argument("--bivariate-tolerance", type=float, default=1e-4)
    analytic.add_argument("--partition", choices=("max-chordal-depth", "min-angle", "simple-neurite-paths"), default="max-chordal-depth")
    analytic.add_argument("--filter-angle", type=float, default=90., help="maximum branch continuation angle in degrees")
    analytic.add_argument("--radius-ratio", type=float, default=10., help="maximum branch continuation radius ratio")
    analytic.add_argument("--parametrization", choices=("chord-length", "centripetal", "uniform"), default="chord-length")
    analytic.add_argument("--no-precondition", action="store_true")
    analytic.add_argument("--alpha", type=float, default=3.)
    analytic.add_argument("--beta", type=float, default=1.5)
    analytic.add_argument("--gamma", type=float, default=10.)
    analytic.add_argument("--scale-radius", type=float, default=1., help="scale radii after preconditioning and readjust roots")
    analytic.add_argument("--raw", action="store_true", help="analyze unadjusted SWC samples instead of the prepared model")
    source_plan = sub.add_parser("plan-source-jobs", help="list source-style analysis jobs without running solvers")
    source_plan.add_argument("swc")
    source_plan.add_argument("--max-jobs", type=int, default=100000)
    source_plan.add_argument("--raw", action="store_true", help="plan against unadjusted SWC samples")
    source_plan.add_argument("--no-precondition", action="store_true")
    source_plan.add_argument("--preserve-surface-roots", action="store_true",
                             help="keep roots already on the soma sphere instead of reproducing source insertion")
    source_plan.add_argument("--source-soma-centroid", action="store_true", help="use source-order soma centroid summation")
    source_plan.add_argument("--precondition-norm-mode", choices=("robust", "source", "source-x87"), default="robust",
                         help="distance arithmetic for preconditioning (default: robust)")
    source_plan.add_argument("--partition", choices=("max-chordal-depth", "min-angle", "simple-neurite-paths"),
                             default="max-chordal-depth")
    source_plan.add_argument("--parametrization", choices=("chord-length", "centripetal", "uniform"), default="chord-length")
    source_plan.add_argument("--filter-angle", type=float, default=90.)
    source_plan.add_argument("--radius-ratio", type=float, default=10.)
    source_plan.add_argument("--alpha", type=float, default=3.)
    source_plan.add_argument("--beta", type=float, default=1.5)
    source_plan.add_argument("--gamma", type=float, default=10.)
    source_plan.add_argument("--scale-radius", type=float, default=1.)
    source_trace = sub.add_parser("trace-source-jobs", help="execute bounded source-style analysis jobs for comparison")
    source_trace.add_argument("swc")
    source_trace.add_argument("--max-jobs", type=int, default=100000)
    source_trace.add_argument("--max-intervals", type=int, default=100000)
    source_trace.add_argument("--max-rectangles", type=int, default=10000)
    source_trace.add_argument("--univariate-tolerance", type=float, default=1e-6)
    source_trace.add_argument("--bivariate-tolerance", type=float, default=1e-4)
    source_trace.add_argument("--raw", action="store_true", help="trace against unadjusted SWC samples")
    source_trace.add_argument("--no-precondition", action="store_true")
    source_trace.add_argument("--preserve-surface-roots", action="store_true")
    source_trace.add_argument("--source-soma-centroid", action="store_true", help="use source-order soma centroid summation")
    source_trace.add_argument("--precondition-norm-mode", choices=("robust", "source", "source-x87"), default="robust",
                         help="distance arithmetic for preconditioning (default: robust)")
    source_trace.add_argument("--partition", choices=("max-chordal-depth", "min-angle", "simple-neurite-paths"),
                              default="max-chordal-depth")
    source_trace.add_argument("--parametrization", choices=("chord-length", "centripetal", "uniform"),
                              default="chord-length")
    source_trace.add_argument("--filter-angle", type=float, default=90.)
    source_trace.add_argument("--radius-ratio", type=float, default=10.)
    source_trace.add_argument("--alpha", type=float, default=3.)
    source_trace.add_argument("--beta", type=float, default=1.5)
    source_trace.add_argument("--gamma", type=float, default=10.)
    source_trace.add_argument("--scale-radius", type=float, default=1.)
    amv = sub.add_parser("export-amv", help="export experimental source-style MorphView data")
    amv.add_argument("swc")
    amv.add_argument("output")
    amv.add_argument("--max-jobs", type=int, default=100000)
    amv.add_argument("--max-intervals", type=int, default=100000)
    amv.add_argument("--max-rectangles", type=int, default=10000)
    amv.add_argument("--univariate-tolerance", type=float, default=1e-6)
    amv.add_argument("--bivariate-tolerance", type=float, default=1e-4)
    amv.add_argument("--no-precondition", action="store_true")
    amv.add_argument("--preserve-surface-roots", action="store_true")
    amv.add_argument("--source-soma-centroid", action="store_true", help="use source-order soma centroid summation")
    amv.add_argument("--precondition-norm-mode", choices=("robust", "source", "source-x87"), default="robust",
                         help="distance arithmetic for preconditioning (default: robust)")
    amv.add_argument("--partition", choices=("max-chordal-depth", "min-angle", "simple-neurite-paths"),
                     default="max-chordal-depth")
    amv.add_argument("--parametrization", choices=("chord-length", "centripetal", "uniform"), default="chord-length")
    amv.add_argument("--filter-angle", type=float, default=90.)
    amv.add_argument("--radius-ratio", type=float, default=10.)
    amv.add_argument("--alpha", type=float, default=3.)
    amv.add_argument("--beta", type=float, default=1.5)
    amv.add_argument("--gamma", type=float, default=10.)
    amv.add_argument("--scale-radius", type=float, default=1.)
    inspect_amv = sub.add_parser("inspect-amv", help="inspect a little-endian MorphView binary file")
    inspect_amv.add_argument("amv")
    inspect_amv.add_argument("--windows-text-mode", action="store_true",
                             help="undo newline expansion in files written by the Windows C++ writer")
    mesh = sub.add_parser("meshstat", help="calculate statistics for a triangular OBJ")
    mesh.add_argument("obj")
    quality = sub.add_parser("meshquality", help="measure triangle and edge size distributions in an OBJ")
    quality.add_argument("obj")
    check = sub.add_parser("meshcheck", help="check non-adjacent triangles for self-intersection candidates")
    check.add_argument("obj")
    check.add_argument("--max-hits", type=int, default=1000)
    check.add_argument("--include-adjacent", action="store_true", help="also check overlap beyond shared vertices or edges")
    boolean = sub.add_parser("mesh-boolean", help="combine two closed OBJ surfaces with red-blue stitching")
    boolean.add_argument("red_obj")
    boolean.add_argument("blue_obj")
    boolean.add_argument("output")
    boolean.add_argument("--operation", choices=("union", "intersection", "difference"), default="union",
                         help="difference means red minus blue")
    boolean.add_argument("--max-attempts", type=int, default=8)
    boolean.add_argument("--max-growth-factor", type=float, default=2.)
    boolean.add_argument("--source-ear-clipping", action="store_true",
                         help="use source-style randomized polygon ear clipping")
    boolean.add_argument("--source-delaunay", action="store_true",
                         help="use source-style Delaunay cleanup and ear clipping")
    boolean.add_argument("--source-delaunay-allow-limit", action="store_true",
                         help="use source-style cleanup and keep its current triangles if it hits the iteration limit")
    boolean.add_argument("--random-seed", type=int, default=0,
                         help="seed for source-style polygon triangulation")
    boolean.add_argument("--random-generator", choices=("python", "windows-c"), default="python",
                         help="random stream for source-style polygon triangulation")
    remesh = sub.add_parser("remesh", help="postprocess an existing triangular OBJ")
    remesh.add_argument("obj")
    remesh.add_argument("output")
    remesh.add_argument("--isotropic", action="store_true", help="use isotropic remeshing instead of standard postprocessing")
    remesh.add_argument("--smoothing", choices=("hc", "taubin"), help="smoothing method (default: hc after collapse; none for isotropic)")
    _add_taubin_arguments(remesh)
    remesh.add_argument("--target-edge-length", type=float, help="isotropic target (required with --isotropic)")
    remesh.add_argument("--iterations", type=int, help="isotropic iterations (default: 5; requires --isotropic)")
    postprocess = sub.add_parser("postprocess", help="apply edge collapse or smoothing to an existing OBJ")
    postprocess.add_argument("obj")
    postprocess.add_argument("output")
    postprocess.add_argument("--edge-collapse", action="store_true")
    _add_collapse_arguments(postprocess)
    postprocess.add_argument("--hc-iterations", type=int, default=0)
    postprocess.add_argument("--hc-alpha", type=float, default=.4)
    postprocess.add_argument("--hc-beta", type=float, default=.7)
    postprocess.add_argument("--laplacian-iterations", type=int, default=0)
    postprocess.add_argument("--laplacian-factor", type=float, default=.5)
    _add_taubin_arguments(postprocess)
    redblue = sub.add_parser("generate", aliases=["generate-redblue"],
                             help="generate an analytic canal and red-blue neuron mesh (default)")
    redblue.add_argument("swc")
    redblue.add_argument("output")
    # Accept old voxel-only switches just to give a clear migration message;
    # they never select the voxel mesher implicitly.
    redblue.add_argument("--spacing", type=float, help=argparse.SUPPRESS)
    redblue.add_argument("--max-grid-points", type=int, help=argparse.SUPPRESS)
    gate = redblue.add_mutually_exclusive_group()
    gate.add_argument("--force", action="store_true", help=argparse.SUPPRESS)
    gate.add_argument("--source-gate", action="store_true",
                      help=argparse.SUPPRESS)
    redblue.add_argument("--require-no-findings", action="store_true",
                         help="run partial analytic checks and stop on findings or unresolved checks")
    redblue.add_argument("--check-adjacent-end-start", action="store_true",
                         help="include the supplementary consecutive-segment check with --require-no-findings")
    redblue.add_argument("--analysis-max-pairs", type=int, help="partial-analysis candidate-pair limit (default: 100000)")
    redblue.add_argument("--analysis-max-rectangles", type=int, help="bivariate solver rectangle limit (default: 10000)")
    redblue.add_argument("--analysis-max-intervals", type=int,
                         help="univariate clipping interval limit per solve (default: 100000)")
    redblue.add_argument("--analysis-univariate-tolerance", type=float, help="univariate solver tolerance (default: 1e-6)")
    redblue.add_argument("--analysis-bivariate-tolerance", type=float, help="bivariate solver tolerance (default: 1e-4)")
    redblue.add_argument("--analysis-job-mode", choices=("extended", "source"),
                         help="choose broader checks or original source job selection")
    redblue.add_argument("--analysis-source-job-limit", type=int, help="maximum source jobs to plan")
    redblue.add_argument("--check-output", action="store_true",
                         help="check edge-manifold topology and numerical self-intersection candidates before saving")
    redblue.add_argument("--flush-face-limit", type=int,
                         help="experimentally flush completed mesh faces above this active-face count")
    redblue.add_argument("--angular-segments", type=int, default=6)
    redblue.add_argument("--soma-refinements", type=int, default=3)
    redblue.add_argument("--partition", choices=("max-chordal-depth", "min-angle", "simple-neurite-paths"), default="max-chordal-depth")
    redblue.add_argument("--filter-angle", type=float, default=90.)
    redblue.add_argument("--radius-ratio", type=float, default=10.)
    redblue.add_argument("--parametrization", choices=("chord-length", "centripetal", "uniform"), default="chord-length")
    redblue.add_argument("--triangle-height-factor", type=float, default=1.)
    redblue.add_argument("--arc-length-mode", choices=("source", "source-x87", "accurate"), default="source")
    redblue.add_argument("--source-surface-roots", action="store_true", help="reproduce the source's exact-boundary root insertion")
    redblue.add_argument("--source-soma-centroid", action="store_true", help="use source-order soma centroid summation")
    redblue.add_argument("--precondition-norm-mode", choices=("robust", "source", "source-x87"), default="robust",
                         help="distance arithmetic for preconditioning (default: robust)")
    redblue.add_argument("--preserve-crease-edges", action="store_true")
    redblue.add_argument("--outer-attempts", type=int, default=None,
                         help="optional limit on outer retries per path (default: retry until the safe radius bound)")
    redblue.add_argument("--random-seed", type=int, default=0, help="seed for render vectors and retry angles")
    redblue.add_argument("--random-generator", choices=("python", "windows-c"), default="python",
                         help="random stream for frames and retries (default: python)")
    redblue.add_argument("--source-ear-clipping", action="store_true",
                         help="use experimental source-style randomized cut-polygon ear clipping")
    redblue.add_argument("--source-delaunay", action="store_true",
                         help="use source-style Delaunay cleanup and ear clipping for cut polygons")
    redblue.add_argument("--source-delaunay-allow-limit", action="store_true",
                         help="use source-style cleanup and keep its current triangles if it hits the iteration limit")
    redblue.add_argument("--source-retry-semantics", action="store_true",
                         help="reproduce the source's inner-limit restart and radius decrements")
    redblue.add_argument("--source-curve-evaluation", action="store_true",
                         help="evaluate source-order Bézier controls when rendering canals")
    redblue.add_argument("--source-render-vector", action="store_true",
                         help="use the source's render-vector clipping and Bézier curves")
    redblue.add_argument("--source-frame", action="store_true",
                         help="use source-order frame arithmetic for canals and terminal caps")
    redblue.add_argument("--quad-sqrt-mode", choices=("native", "windows-x87"), default="native",
                         help="square-root arithmetic for canal and terminal quad diagonals")
    redblue.add_argument("--render-trig-mode", choices=("native", "windows-x87"), default="native",
                         help="trigonometry used for canal and terminal rendering")
    redblue.add_argument("--outer-loop-maxiter", type=int, default=16)
    redblue.add_argument("--radius-factor-decrement", type=float, default=.01)
    redblue.add_argument("--inner-attempts", type=int, default=8)
    redblue.add_argument("--complex-edge-max-growth-factor", type=float, default=2.,
                         help="maximum complex-edge count relative to the initial count")
    redblue.add_argument("--no-precondition", action="store_true", help="disable morphology preconditioning")
    redblue.add_argument("--alpha", type=float, default=3.)
    redblue.add_argument("--beta", type=float, default=1.5)
    redblue.add_argument("--gamma", type=float, default=10.)
    redblue.add_argument("--scale-radius", type=float, default=1., help="scale radii after preconditioning and readjust roots")
    redblue.add_argument("--postprocess", action="store_true", help="apply preliminary edge collapse and HC smoothing")
    redblue.add_argument("--edge-collapse", action="store_true", help="apply preliminary edge-collapse cleanup")
    _add_collapse_arguments(redblue)
    redblue.add_argument("--smooth-iterations", type=int, default=None, help="HC smoothing iterations; overrides --postprocess default")
    redblue.add_argument("--hc-alpha", type=float, help="HC smoothing alpha (default: 0.4)")
    redblue.add_argument("--hc-beta", type=float, help="HC smoothing beta (default: 0.7)")
    redblue.add_argument("--uniform-edge-length", type=float,
                         help="apply experimental isotropic remeshing at this target edge length")
    redblue.add_argument("--uniform-iterations", type=int, help="isotropic remeshing iterations (default: 5)")
    redblue.add_argument("--uniform-max-operations", type=int, help="isotropic remeshing work limit (default: 300000)")
    redblue.add_argument("--postprocess-output", help="write the processed mesh separately, keeping output as the raw mesh")
    generate = sub.add_parser("generate-voxel", help="explicitly generate an experimental voxel surface mesh")
    generate.add_argument("swc")
    generate.add_argument("output")
    generate.add_argument("--spacing", type=float, required=True, help="grid spacing in SWC coordinate units")
    generate.add_argument("--max-grid-points", type=int, default=1_500_000)
    generate.add_argument("--partition", choices=("max-chordal-depth", "min-angle", "simple-neurite-paths"), default="max-chordal-depth")
    generate.add_argument("--filter-angle", type=float, default=90.)
    generate.add_argument("--radius-ratio", type=float, default=10.)
    generate.add_argument("--parametrization", choices=("chord-length", "centripetal", "uniform"), default="chord-length")
    generate.add_argument("--precondition", action="store_true", help="apply experimental greedy SWC preconditioning")
    generate.add_argument("--alpha", type=float, default=3.0)
    generate.add_argument("--beta", type=float, default=1.5)
    generate.add_argument("--gamma", type=float, default=10.0)
    generate.add_argument("--smooth-iterations", type=int, default=0, help="apply HC Laplacian smoothing (default: 0)")
    generate.add_argument("--adjust-soma-roots", action="store_true", help="place neurite roots on their parent soma spheres")
    generate.add_argument("--source-surface-roots", action="store_true", help="reproduce the source's exact-boundary root insertion")
    generate.add_argument("--edge-collapse", action="store_true", help="apply experimental greedy edge-collapse cleanup")
    generate.add_argument("--uniform-edge-length", type=float, help="apply experimental isotropic remeshing")
    individual = sub.add_parser("generate-individual", help="render overlapping soma and canal modeling surfaces")
    individual.add_argument("swc")
    individual.add_argument("output")
    individual.add_argument("--angular-segments", type=int, default=6)
    individual.add_argument("--soma-refinements", type=int, default=3)
    individual.add_argument("--partition", choices=("max-chordal-depth", "min-angle", "simple-neurite-paths"), default="max-chordal-depth")
    individual.add_argument("--filter-angle", type=float, default=90.)
    individual.add_argument("--radius-ratio", type=float, default=10.)
    individual.add_argument("--parametrization", choices=("chord-length", "centripetal", "uniform"), default="chord-length")
    individual.add_argument("--random-seed", type=int, default=0, help="seed for random render vectors")
    individual.add_argument("--random-generator", choices=("python", "windows-c"), default="python",
                            help="random stream for render frames (default: python)")
    individual.add_argument("--quad-sqrt-mode", choices=("native", "windows-x87"), default="native",
                            help="square-root arithmetic for canal quad diagonals")
    individual.add_argument("--render-trig-mode", choices=("native", "windows-x87"), default="native",
                            help="trigonometry used for canal rendering")
    individual.add_argument("--source-curve-evaluation", action="store_true",
                            help="evaluate source-order Bézier controls when rendering canals")
    individual.add_argument("--source-render-vector", action="store_true",
                            help="use the source's render-vector clipping and Bézier curves")
    individual.add_argument("--source-frame", action="store_true",
                            help="use source-order canal frame arithmetic")
    individual.add_argument("--triangle-height-factor", type=float, default=1.)
    individual.add_argument("--arc-length-mode", choices=("source", "source-x87", "accurate"), default="source")
    individual.add_argument("--source-surface-roots", action="store_true", help="reproduce the source's exact-boundary root insertion")
    individual.add_argument("--source-soma-centroid", action="store_true", help="use source-order soma centroid summation")
    individual.add_argument("--precondition-norm-mode", choices=("robust", "source", "source-x87"), default="robust",
                         help="distance arithmetic for preconditioning (default: robust)")
    individual.add_argument("--preserve-crease-edges", action="store_true")
    individual.add_argument("--no-precondition", action="store_true")
    individual.add_argument("--alpha", type=float, default=3.)
    individual.add_argument("--beta", type=float, default=1.5)
    individual.add_argument("--gamma", type=float, default=10.)
    individual.add_argument("--scale-radius", type=float, default=1.)
    args = parser.parse_args(argv)
    try:
        if args.command == "workflow":
            output_dir = args.output_dir or str(Path.cwd()/(Path(args.swc).stem+"_results"))
            result = run_workflow(args.swc, output_dir, target_edge_length=args.target_edge_length,
                                  iterations=args.iterations, angular_segments=args.angular_segments,
                                  postprocess_method=args.postprocess_method,
                                  smoothing=args.smoothing, taubin_iterations=args.taubin_iterations,
                                  taubin_lambda=args.taubin_lambda, taubin_mu=args.taubin_mu,
                                  soma_blend_width=args.soma_blend_width,
                                  soma_blend_method=args.soma_blend_method,
                                  soma_refinements=args.soma_refinements, force=args.force, match_cpp=args.match_cpp,
                                  analyze=args.analyze,
                                  check_intersections=not args.skip_intersection_check,
                                  progress=lambda message: print(message+"...", file=sys.stderr))
        elif args.command == "inspect":
            morphology = read_swc(args.swc)
            violations = morphology.primary_distance_violations(args.alpha)
            result = {"nodes": len(morphology.nodes), "roots": list(morphology.roots),
                      "types": dict(sorted(Counter(n.kind for n in morphology.nodes.values()).items())),
                      "neurite_segments": sum(1 for _ in morphology.neurite_segments()),
                      "primary_distance_violations": violations, "alpha": args.alpha}
        elif args.command == "analyze":
            print("Warning: sampled diagnostics are incomplete and do not certify geometric consistency.", file=sys.stderr)
            warnings = approximate_analysis(read_swc(args.swc), samples=args.samples, max_pairs=args.max_pairs)
            result = {"method": "approximate sampled diagnostics", "certifies_consistency": False,
                      "warnings": [warning.as_dict() for warning in warnings]}
        elif args.command == "analyze-analytic":
            print("Warning: analytic network analysis is incomplete and does not certify geometric consistency.", file=sys.stderr)
            if args.job_mode == "source":
                print("Warning: source job selection intentionally omits some checks.", file=sys.stderr)
            result = analyze_analytic_partial(read_swc(args.swc), max_pairs=args.max_pairs,
                                              max_rectangles=args.max_rectangles, max_intervals=args.max_intervals,
                                              prepare=not args.raw,
                                              univariate_tolerance=args.univariate_tolerance,
                                              bivariate_tolerance=args.bivariate_tolerance,
                                              strategy=args.partition, parametrization=args.parametrization,
                                              filter_angle=args.filter_angle, radius_ratio=args.radius_ratio,
                                              check_adjacent_end_start=args.check_adjacent_end_start,
                                              job_mode=args.job_mode, source_job_limit=args.source_job_limit,
                                              preserve_surface_roots=not args.source_surface_roots,
                                              source_soma_centroid=args.source_soma_centroid,
                                              precondition_norm_mode=args.precondition_norm_mode,
                                              preconditioning=not args.no_precondition, scale_radius=args.scale_radius,
                                              pc_alpha=args.alpha, pc_beta=args.beta, pc_gamma=args.gamma).as_dict()
        elif args.command == "plan-source-jobs":
            result = plan_source_analysis_jobs(read_swc(args.swc), max_jobs=args.max_jobs, prepare=not args.raw,
                                               preconditioning=not args.no_precondition,
                                               preserve_surface_roots=args.preserve_surface_roots,
                                               source_soma_centroid=args.source_soma_centroid,
                                               precondition_norm_mode=args.precondition_norm_mode,
                                               strategy=args.partition, parametrization=args.parametrization,
                                               filter_angle=args.filter_angle, radius_ratio=args.radius_ratio,
                                               pc_alpha=args.alpha, pc_beta=args.beta, pc_gamma=args.gamma,
                                               scale_radius=args.scale_radius).as_dict()
        elif args.command == "trace-source-jobs":
            print("Warning: source-style numerical traces do not certify geometric consistency.", file=sys.stderr)
            result = trace_source_analysis(read_swc(args.swc), max_jobs=args.max_jobs, prepare=not args.raw,
                                           max_intervals=args.max_intervals, max_rectangles=args.max_rectangles,
                                           univariate_tolerance=args.univariate_tolerance,
                                           bivariate_tolerance=args.bivariate_tolerance,
                                           preconditioning=not args.no_precondition,
                                           preserve_surface_roots=args.preserve_surface_roots,
                                           source_soma_centroid=args.source_soma_centroid,
                                           precondition_norm_mode=args.precondition_norm_mode,
                                           strategy=args.partition, parametrization=args.parametrization,
                                           filter_angle=args.filter_angle, radius_ratio=args.radius_ratio,
                                           pc_alpha=args.alpha, pc_beta=args.beta, pc_gamma=args.gamma,
                                           scale_radius=args.scale_radius).as_dict()
        elif args.command == "export-amv":
            if Path(args.swc).resolve() == Path(args.output).resolve():
                raise ValueError("output path must differ from the input SWC file")
            model = create_morphview(read_swc(args.swc), preconditioning=not args.no_precondition,
                                     pc_alpha=args.alpha, pc_beta=args.beta, pc_gamma=args.gamma,
                                     scale_radius=args.scale_radius, preserve_surface_roots=args.preserve_surface_roots,
                                     source_soma_centroid=args.source_soma_centroid,
                                     precondition_norm_mode=args.precondition_norm_mode,
                                     strategy=args.partition, parametrization=args.parametrization,
                                     filter_angle=args.filter_angle, radius_ratio=args.radius_ratio,
                                     max_jobs=args.max_jobs, max_intervals=args.max_intervals,
                                     max_rectangles=args.max_rectangles, univariate_tolerance=args.univariate_tolerance,
                                     bivariate_tolerance=args.bivariate_tolerance)
            write_morphview(model, args.output)
            print("Warning: experimental MorphView export stores only the first soma and does not certify consistency.",
                  file=sys.stderr)
            result = {"output": args.output, **model.summary(), "first_soma_only": True,
                      "certifies_consistency": False}
        elif args.command == "inspect-amv":
            model = read_morphview(args.amv, windows_text_mode=args.windows_text_mode)
            result = {"input": args.amv, **model.summary(), "first_soma_only": True,
                      "certifies_consistency": False}
        elif args.command == "meshstat":
            result = asdict(read_obj(args.obj).statistics())
            # JSON has no representation for NaN.
            if result["aspect_ratio_stddev"] != result["aspect_ratio_stddev"]:
                result["aspect_ratio_stddev"] = None
        elif args.command == "meshquality":
            result = mesh_quality(read_obj(args.obj))
        elif args.command == "meshcheck":
            if args.include_adjacent:
                print("Warning: floating-point mesh intersection checks are not a proof of validity.", file=sys.stderr)
            else:
                print("Warning: floating-point check excludes pairs sharing vertices and is not a proof of validity.", file=sys.stderr)
            hits, truncated = find_self_intersections(read_obj(args.obj), max_hits=args.max_hits,
                                                       include_adjacent=args.include_adjacent)
            result = {"intersections": [asdict(hit) for hit in hits], "truncated": truncated,
                      "includes_adjacent": args.include_adjacent, "certifies_nonintersection": False}
        elif args.command == "mesh-boolean":
            output = Path(args.output).resolve()
            if output in (Path(args.red_obj).resolve(), Path(args.blue_obj).resolve()):
                raise ValueError("output path must differ from both input OBJ files")
            model = redblue_boolean_with_retries(read_obj(args.red_obj), read_obj(args.blue_obj),
                                                 operation=args.operation, max_attempts=args.max_attempts,
                                                 max_growth_factor=args.max_growth_factor,
                                                 source_ear_clipping=args.source_ear_clipping,
                                                 source_delaunay=args.source_delaunay,
                                                 source_delaunay_allow_limit=args.source_delaunay_allow_limit,
                                                 random_seed=args.random_seed,
                                                 random_generator=args.random_generator)
            model.write_obj(args.output)
            result = {"output": args.output, "operation": args.operation,
                      "vertices": len(model.vertices), "faces": len(model.faces)}
        elif args.command == "remesh":
            if Path(args.obj).resolve() == Path(args.output).resolve():
                raise ValueError("output path must differ from the input OBJ file")
            if not args.isotropic and (args.target_edge_length is not None or args.iterations is not None):
                raise ValueError("target edge length and remeshing iterations require --isotropic")
            if args.isotropic and args.target_edge_length is None:
                raise ValueError("--isotropic requires --target-edge-length for an existing OBJ")
            taubin_supplied = any(value is not None for value in (args.taubin_iterations, args.taubin_lambda, args.taubin_mu))
            if args.smoothing != "taubin" and taubin_supplied:
                raise ValueError("Taubin options require --smoothing taubin")
            taubin_options = _taubin_options(args)
            if args.smoothing == "taubin":
                _validate_taubin_options(**taubin_options)
            print("Warning: remeshing changes the surface and does not check global self-intersections.", file=sys.stderr)
            original = read_obj(args.obj)
            before = original.statistics()
            before_edges = mesh_quality(original)["edge_lengths"]
            if args.isotropic:
                settings = {"target": args.target_edge_length, "iterations": args.iterations if args.iterations is not None else 5,
                            "max_operations": 300000}
                model = isotropic_remesh(original, **settings)
                if args.smoothing == "taubin":
                    settings["taubin_smoothing"] = taubin_options
                    model = taubin_smooth(model, **taubin_options)
                elif args.smoothing == "hc":
                    settings["hc_smoothing"] = {"alpha": .4, "beta": .7, "iterations": 10}
                    model = hc_smooth(model, **settings["hc_smoothing"])
            else:
                smoothing = args.smoothing or "hc"
                settings = standard_settings(smoothing=smoothing, taubin_options=taubin_options)
                model = standard_postprocess(original, smoothing=smoothing, taubin_options=taubin_options)
            after = model.statistics()
            after_edges = mesh_quality(model)["edge_lengths"]
            model.write_obj(args.output)
            result = {"output": args.output, "vertices": len(model.vertices), "faces": len(model.faces),
                      "postprocessing_method": "isotropic" if args.isotropic else "standard",
                      "postprocess_settings": settings, "target_edge_length": args.target_edge_length,
                      "edge_lengths_before": before_edges, "edge_lengths_after": after_edges,
                      "change_percent": {"area": 100*(after.area/before.area-1) if before.area else None,
                                         "volume": 100*(after.volume/before.volume-1) if before.volume else None}}
        elif args.command == "postprocess":
            if Path(args.obj).resolve() == Path(args.output).resolve():
                raise ValueError("output path must differ from the input OBJ file")
            if args.hc_iterations < 0 or args.laplacian_iterations < 0:
                raise ValueError("smoothing iterations must be non-negative")
            taubin_iterations = args.taubin_iterations or 0
            if args.taubin_iterations is not None and args.taubin_iterations < 0:
                raise ValueError("Taubin iterations must be non-negative")
            if not taubin_iterations and (args.taubin_lambda is not None or args.taubin_mu is not None):
                raise ValueError("Taubin factors require --taubin-iterations greater than zero")
            if not args.edge_collapse and not args.hc_iterations and not args.laplacian_iterations and not taubin_iterations:
                raise ValueError("enable at least one post-processing operation")
            collapse_options = _collapse_options(args, args.edge_collapse)
            if args.hc_iterations:
                _validate_hc_options(args.hc_alpha, args.hc_beta, args.hc_iterations)
            taubin_options = _taubin_options(args) if taubin_iterations else None
            if taubin_options:
                _validate_taubin_options(**taubin_options)
            print("Warning: post-processing changes the surface and does not check global self-intersections.", file=sys.stderr)
            model = read_obj(args.obj)
            before = model.statistics()
            if args.edge_collapse:
                model = greedy_edge_collapse(model, **collapse_options)
            if args.laplacian_iterations:
                model = laplacian_smooth(model, smoothing_factor=args.laplacian_factor,
                                         iterations=args.laplacian_iterations)
            if args.hc_iterations:
                model = hc_smooth(model, alpha=args.hc_alpha, beta=args.hc_beta, iterations=args.hc_iterations)
            if taubin_options:
                model = taubin_smooth(model, **taubin_options)
            after = model.statistics()
            model.write_obj(args.output)
            result = {"output": args.output, "vertices": len(model.vertices), "faces": len(model.faces),
                      "postprocess_settings": {"edge_collapse": collapse_options,
                                               "hc_smoothing": {"alpha": args.hc_alpha, "beta": args.hc_beta,
                                                                "iterations": args.hc_iterations} if args.hc_iterations else None,
                                               "laplacian": {"smoothing_factor": args.laplacian_factor,
                                                             "iterations": args.laplacian_iterations} if args.laplacian_iterations else None,
                                               "taubin_smoothing": taubin_options},
                      "change_percent": {"area": 100*(after.area/before.area-1) if before.area else None,
                                         "volume": 100*(after.volume/before.volume-1) if before.volume else None}}
        elif args.command == "generate-individual":
            if Path(args.swc).resolve() == Path(args.output).resolve():
                raise ValueError("output path must differ from the input SWC file")
            print("Note: individual modeling surfaces overlap and are not a stitched cell boundary.", file=sys.stderr)
            model = render_individual_surfaces(read_swc(args.swc), angular_segments=args.angular_segments,
                                               soma_refinements=args.soma_refinements, strategy=args.partition,
                                               parametrization=args.parametrization,
                                               filter_angle=args.filter_angle, radius_ratio=args.radius_ratio,
                                               triangle_height_factor=args.triangle_height_factor,
                                               arc_length_mode=args.arc_length_mode,
                                               preserve_surface_roots=not args.source_surface_roots,
                                               preserve_crease_edges=args.preserve_crease_edges,
                                               preconditioning=not args.no_precondition, scale_radius=args.scale_radius,
                                               pc_alpha=args.alpha, pc_beta=args.beta, pc_gamma=args.gamma,
                                               random_seed=args.random_seed, random_generator=args.random_generator,
                                               quad_sqrt_mode=args.quad_sqrt_mode, render_trig_mode=args.render_trig_mode,
                                               source_curve_evaluation=args.source_curve_evaluation,
                                               source_render_vector=args.source_render_vector,
                                               source_frame=args.source_frame, source_soma_centroid=args.source_soma_centroid,
                                               precondition_norm_mode=args.precondition_norm_mode)
            model.write_obj(args.output)
            result = {"output": args.output, "vertices": len(model.vertices), "faces": len(model.faces),
                      "meshing": "separate overlapping soma and canal surfaces", "stitched": False}
        elif args.command in ("generate", "generate-redblue"):
            if args.spacing is not None or args.max_grid_points is not None:
                raise ValueError("--spacing and --max-grid-points are voxel options; use generate-voxel explicitly")
            if Path(args.swc).resolve() == Path(args.output).resolve():
                raise ValueError("output path must differ from the input SWC file")
            if args.source_gate:
                if args.analysis_job_mode not in (None, "source"):
                    raise ValueError("--source-gate requires --analysis-job-mode source")
                args.analysis_job_mode = "source"
            analysis_overrides = (args.analysis_max_pairs, args.analysis_max_rectangles, args.analysis_max_intervals,
                                  args.analysis_univariate_tolerance, args.analysis_bivariate_tolerance,
                                  args.analysis_job_mode, args.analysis_source_job_limit)
            if not (args.require_no_findings or args.source_gate) and (args.check_adjacent_end_start or any(x is not None for x in analysis_overrides)):
                raise ValueError("analysis options require --require-no-findings or --source-gate")
            do_collapse = args.postprocess or args.edge_collapse
            smoothing_iterations = (args.smooth_iterations if args.smooth_iterations is not None
                                    else 10 if args.postprocess else 0)
            if smoothing_iterations < 0:
                raise ValueError("smoothing iterations must be non-negative")
            collapse_options = _collapse_options(args, do_collapse, default_safety="source" if args.postprocess else "safe")
            remesh_options = _redblue_remesh_options(args)
            if not smoothing_iterations and (args.hc_alpha is not None or args.hc_beta is not None):
                raise ValueError("HC parameters require an enabled smoothing stage")
            hc_options = None
            if smoothing_iterations:
                hc_options = dict(alpha=args.hc_alpha if args.hc_alpha is not None else .4,
                                  beta=args.hc_beta if args.hc_beta is not None else .7,
                                  iterations=smoothing_iterations)
                _validate_hc_options(**hc_options)
            if args.postprocess_output:
                processed_path = Path(args.postprocess_output).resolve()
                if processed_path in (Path(args.swc).resolve(), Path(args.output).resolve()):
                    raise ValueError("postprocess output must differ from input SWC and raw output")
                if not do_collapse and not smoothing_iterations and remesh_options is None:
                    raise ValueError("postprocess output requires an enabled post-processing stage")
            print("Warning: experimental analytic red-blue meshing.", file=sys.stderr)
            if args.analysis_job_mode == "source":
                print("Warning: source job selection intentionally omits some checks.", file=sys.stderr)
            tree = read_swc(args.swc)
            report = None
            meshing_options = dict(angular_segments=args.angular_segments, soma_refinements=args.soma_refinements,
                                   max_outer_attempts=args.outer_attempts, outer_loop_maxiter=args.outer_loop_maxiter,
                                   radius_factor_decrement=args.radius_factor_decrement,
                                   max_inner_attempts=args.inner_attempts,
                                   max_complex_edge_growth_factor=args.complex_edge_max_growth_factor,
                                   strategy=args.partition,
                                   parametrization=args.parametrization, filter_angle=args.filter_angle,
                                   radius_ratio=args.radius_ratio, triangle_height_factor=args.triangle_height_factor,
                                   arc_length_mode=args.arc_length_mode, preserve_surface_roots=not args.source_surface_roots,
                                   preserve_crease_edges=args.preserve_crease_edges, preconditioning=not args.no_precondition,
                                   scale_radius=args.scale_radius, pc_alpha=args.alpha, pc_beta=args.beta, pc_gamma=args.gamma,
                                   random_seed=args.random_seed, random_generator=args.random_generator,
                                   source_ear_clipping=args.source_ear_clipping, source_delaunay=args.source_delaunay,
                                   source_delaunay_allow_limit=args.source_delaunay_allow_limit,
                                   quad_sqrt_mode=args.quad_sqrt_mode,
                                   render_trig_mode=args.render_trig_mode, source_retry_semantics=args.source_retry_semantics,
                                   source_curve_evaluation=args.source_curve_evaluation,
                                   source_render_vector=args.source_render_vector, source_frame=args.source_frame,
                                   source_soma_centroid=args.source_soma_centroid,
                                   precondition_norm_mode=args.precondition_norm_mode)
            if args.flush_face_limit is not None:
                result = _generate_redblue_streamed(args, tree, meshing_options, collapse_options, hc_options,
                                                    remesh_options)
                print(json.dumps(result, indent=2, allow_nan=False))
                return 0
            if args.require_no_findings or args.source_gate:
                analysis_options = {name: getattr(args, name) for name in
                                    ("analysis_max_pairs", "analysis_max_rectangles", "analysis_max_intervals",
                                     "analysis_univariate_tolerance", "analysis_bivariate_tolerance",
                                     "analysis_job_mode", "analysis_source_job_limit")
                                    if getattr(args, name) is not None}
                analysis_options["analysis_gate"] = "all" if args.require_no_findings else "source"
                model, report = assemble_redblue_checked(tree, check_adjacent_end_start=args.check_adjacent_end_start,
                                                         **analysis_options, **meshing_options)
            else:
                model = assemble_redblue_neuron(tree, **meshing_options)
            raw_model = model
            postprocessed = do_collapse or bool(smoothing_iterations) or remesh_options is not None
            before = model.statistics() if postprocessed else None
            if postprocessed:
                model = _postprocess_redblue(model, collapse_options, hc_options, remesh_options)
            output_model = raw_model if args.postprocess_output else model
            checked_surfaces = []
            if args.check_output:
                surfaces = (("raw", output_model), ("processed", model)) if args.postprocess_output else (("output", output_model),)
                for label, candidate in surfaces:
                    _check_output_surface(label, candidate)
                    checked_surfaces.append(label)
            output_model.write_obj(args.output)
            if args.postprocess_output:
                model.write_obj(args.postprocess_output)
            result = {"output": args.output, "vertices": len(output_model.vertices), "faces": len(output_model.faces),
                      "meshing": "experimental analytic canals and red-blue union", "method": "analytic", "forced": args.force,
                      "postprocessed": postprocessed, "analysis_performed": report is not None}
            if report is not None:
                result["analysis"] = {"complete": report.complete, "findings": len(report.findings),
                                      "unresolved": len(report.unresolved), "segments": report.segment_count,
                                      "checked_adjacent_end_start": args.check_adjacent_end_start,
                                      "job_mode": report.job_mode, "source_plan_truncated": report.source_plan_truncated,
                                      "source_jobs_clean": report.source_jobs_clean,
                                      "gate": "all" if args.require_no_findings else "source"}
            if args.check_output:
                result["output_check"] = {"checked_surfaces": checked_surfaces,
                                          "intersection_candidates": 0, "certifies_nonintersection": False}
            if args.postprocess_output:
                result["postprocess_output"] = {"output": args.postprocess_output,
                                                "vertices": len(model.vertices), "faces": len(model.faces)}
            if before:
                after = model.statistics()
                result["postprocess_change_percent"] = {
                    "area": 100*(after.area/before.area-1) if before.area else None,
                    "volume": 100*(after.volume/before.volume-1) if before.volume else None}
                result["postprocess_settings"] = {"edge_collapse": collapse_options, "hc_smoothing": hc_options}
                if remesh_options is not None:
                    result["postprocess_settings"]["isotropic_remesh"] = remesh_options
                    result["edge_lengths_before"] = mesh_quality(raw_model)["edge_lengths"]
                    result["edge_lengths_after"] = mesh_quality(model)["edge_lengths"]
        elif args.command == "generate-voxel":
            print("Warning: experimental voxel meshing does not perform AnaMorph's consistency analysis or red-blue stitching.", file=sys.stderr)
            if Path(args.swc).resolve() == Path(args.output).resolve():
                raise ValueError("output path must differ from the input SWC file")
            tree = read_swc(args.swc)
            if args.adjust_soma_roots:
                tree = adjust_soma_roots(tree, preserve_surface_roots=not args.source_surface_roots)
            if args.precondition:
                processed = precondition(tree, args.alpha, args.beta, args.gamma)
                tree = processed.morphology
                print(f"Preconditioning: {processed.split_segments} split, {processed.collapsed_segments} collapsed.", file=sys.stderr)
            model = generate_voxel_mesh(tree, args.spacing, max_grid_points=args.max_grid_points,
                                        strategy=args.partition, parametrization=args.parametrization,
                                        filter_angle=args.filter_angle, radius_ratio=args.radius_ratio)
            if args.edge_collapse:
                model = greedy_edge_collapse(model)
            if args.smooth_iterations:
                model = hc_smooth(model, iterations=args.smooth_iterations)
            if args.uniform_edge_length is not None:
                print("Warning: isotropic remeshing changes the surface and does not check global self-intersections.", file=sys.stderr)
                model = isotropic_remesh(model, args.uniform_edge_length)
            model.write_obj(args.output)
            result = {"output": args.output, "vertices": len(model.vertices), "faces": len(model.faces),
                      "method": "voxel", "spacing": args.spacing}
        else:
            raise ValueError(f"unsupported command: {args.command}")
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except PartialAnalysisError as error:
        print(f"error: {error}", file=sys.stderr)
        for finding in error.blocking_findings[:10]:
            subjects = [f"segments {finding.segments}"] if finding.segments else []
            if finding.somas:
                subjects.append(f"somas {finding.somas}")
            if finding.classification:
                subjects.append(finding.classification)
            detail = ", ".join(subjects)
            print(f"  {finding.kind}{': '+detail if detail else ''}: {finding.detail}", file=sys.stderr)
        if len(error.blocking_findings) > 10:
            print(f"  ... {len(error.blocking_findings)-10} additional findings", file=sys.stderr)
        for item in error.report.unresolved[:10]:
            print(f"  unresolved: {item}", file=sys.stderr)
        if len(error.report.unresolved) > 10:
            print(f"  ... {len(error.report.unresolved)-10} additional unresolved checks", file=sys.stderr)
        return 1
    except (ValueError, OSError, ClippingConvergenceError, RedBlueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
