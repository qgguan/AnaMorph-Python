import random
import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from math import cos, dist, pi, sin
from pathlib import Path
from unittest.mock import patch

from anamorph.analytic_intersections import CanalSegment
from anamorph.analytic_network import analyze_analytic_partial
from anamorph.bernstein import BernsteinPolynomial
from anamorph.geometry import partition_paths
from anamorph.model import prepare_morphology
from anamorph.roots import adjust_soma_roots
from anamorph.source_order import source_edge_state_after_root_adjustment
from anamorph.source_jobs import _source_segment_box, plan_source_analysis_jobs
from anamorph.source_analysis import SourceAnalysisTrace, SourceJobTrace, trace_source_analysis
from anamorph.swc import Morphology, Node, read_swc

ROOT = Path(__file__).resolve().parent.parent
SOLVER_PROBE = ROOT/"reference-build"/"solver_probe.exe"
JOBS_PROBE = ROOT/"reference-build"/"network_jobs_probe.exe"
OPTIONS_JOBS_PROBE = ROOT/"reference-build"/"network_jobs_options_probe.exe"
SCALE_JOBS_PROBE = ROOT/"reference-build"/"network_jobs_scale_probe.exe"
FLAGS_PROBE = ROOT/"reference-build"/"source_flags_probe.exe"


def _mock_source_controls(*segments):
    # Tests injecting artificial local power curves must inject corresponding
    # source Bézier curves too; otherwise they describe different geometry.
    values = [((0., 1.), (tuple(BernsteinPolynomial.from_power(axis).coefficients
                                for axis in segment.power),)) for segment in segments]
    return patch("anamorph.analytic_network.source_spline_controls", side_effect=values)


class SourceJobPlanTests(unittest.TestCase):
    def test_source_soma_classification_retains_actual_cell_relationship(self):
        cases = (
            ("1 1 -3 0 0 .3 -1\n2 3 -2 0 0 .1 1\n3 3 0 0 0 .1 2\n"
             "4 3 2 0 0 .1 3\n5 1 -1 .2 0 .3 -1\n", "inter_cell"),
            ("1 1 0 0 0 1 -1\n2 3 1.2 0 0 .2 1\n3 3 2 0 0 .2 2\n"
             "4 3 .5 0 0 .2 3\n5 3 2 1 0 .2 4\n", "intra_cell"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "soma.swc"
            for text, expected_relationship in cases:
                with self.subTest(relationship=expected_relationship):
                    path.write_text(text, encoding="utf-8")
                    tree = read_swc(path)
                    trace = trace_source_analysis(tree, preconditioning=False)
                    hits = [item for item in trace.jobs if item.job.kind == "SONS" and item.result]
                    self.assertTrue(hits)
                    self.assertTrue(all(item.classification == "intra_cell_soma_neurite" and
                                        item.soma_relationship == expected_relationship for item in hits))
                    statuses = {item.edge: item for item in trace.segment_flags}
                    self.assertTrue(all(statuses[item.job.segments[0]].ic_sons and
                                        not statuses[item.job.segments[0]].rc_sons for item in hits))
                    # Even an inter-cell SONS hit sets ic_sons in the source's
                    # actual full-analysis path. The broader report preserves
                    # its separate, physically correct relationship label.
                    report = analyze_analytic_partial(tree, prepare=True, preconditioning=False,
                                                      preserve_surface_roots=False, job_mode="source")
                    if expected_relationship == "inter_cell":
                        finding, = (item for item in report.findings if item.kind in
                                    ("soma_neurite_candidate", "source_soma_neurite_candidate") and
                                    item.classification == "inter_cell")
                        flag = next(item for item in report.segment_flags if item.edge == finding.segments[0])
                        self.assertTrue(flag.ic_sons)

    def test_unsegmented_root_is_unresolved_in_source_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"root.swc"
            path.write_text("1 1 0 0 0 1 -1\n2 3 1 0 0 .2 1\n", encoding="utf-8")
            tree = read_swc(path)
            for options in ({"prepare": False},
                            {"preconditioning": False, "preserve_surface_roots": True}):
                trace = trace_source_analysis(tree, **options)
                self.assertEqual(trace.segment_count, 0)
                self.assertIsNone(trace.source_jobs_clean)
                self.assertEqual(trace.unresolved, ("neurite root 2 has no segment to analyze",))
                self.assertEqual(trace.as_dict()["unresolved"], trace.unresolved)
            command = [sys.executable, "-m", "anamorph", "trace-source-jobs", str(path), "--raw"]
            output = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
            report = json.loads(output.stdout)
            self.assertIsNone(report["source_jobs_clean"])
            self.assertEqual(len(report["unresolved"]), 1)

    def test_source_trace_keeps_source_miss_separate_from_geometric_finding(self):
        rows = ("1 1 -3 -3 0 .2 -1", "2 3 -2 0 0 .1 1", "3 3 2 0 0 .1 2",
                "4 3 0 -2 0 .1 1", "5 3 0 2 0 .1 4")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"crossing.swc"
            path.write_text("\n".join(rows)+"\n", encoding="utf-8")
            tree = read_swc(path)
            trace = trace_source_analysis(tree, preconditioning=False)
            source_job, = (item for item in trace.jobs if item.job.kind == "NSNS_NONADJ" and
                           set(item.job.segments) == {(2, 3), (4, 5)})
            self.assertIs(source_job.result, False)
            source_flags = {item.edge: item for item in trace.segment_flags}
            self.assertTrue(all(not source_flags[edge].icrn_nsns for edge in source_job.job.segments))
            report = analyze_analytic_partial(tree, prepare=True, preconditioning=False,
                                              job_mode="source", preserve_surface_roots=False)
            self.assertTrue(any(set(finding.segments) == {(2, 3), (4, 5)} and finding.candidates
                                for finding in report.findings))
            geometric_flags = {item.edge: item for item in report.segment_flags}
            self.assertTrue(all(geometric_flags[edge].icrn_nsns for edge in source_job.job.segments))

    def test_source_trace_reports_every_planned_job_and_incomplete_states(self):
        from anamorph.bernstein import ClippingConvergenceError
        from anamorph.source_bivariate import SourceBivariateFailure
        tree = read_swc(ROOT/"examples"/"branched.swc")
        plan = plan_source_analysis_jobs(tree)
        trace = trace_source_analysis(tree)
        self.assertEqual(tuple(item.job for item in trace.jobs), plan.jobs)
        self.assertTrue(trace.source_jobs_clean)
        self.assertTrue(trace.planned_solver_work_complete)
        self.assertTrue(trace.as_dict()["planned_solver_work_complete"])
        self.assertEqual(len(trace.segment_flags), trace.segment_count)
        self.assertTrue(all(not item.solver_issues and not item.source_plan_may_affect for item in trace.segment_flags))
        self.assertFalse(trace.as_dict()["certifies_consistency"])
        self.assertTrue(all(item.result is False for item in trace.jobs))
        limited = trace_source_analysis(tree, max_jobs=2)
        self.assertTrue(limited.truncated)
        self.assertEqual(len(limited.jobs), 2)
        self.assertIsNone(limited.source_jobs_clean)
        self.assertFalse(limited.planned_solver_work_complete)
        self.assertEqual(len(limited.segment_flags), limited.segment_count)
        self.assertTrue(all(item.source_plan_may_affect for item in limited.segment_flags))
        with patch("anamorph.source_analysis.source_regularity_candidates",
                   side_effect=ClippingConvergenceError("injected limit")):
            failed = trace_source_analysis(tree)
        self.assertIsNone(failed.source_jobs_clean)
        self.assertTrue(any(item.result is None and "injected limit" in item.reason for item in failed.jobs))
        self.assertTrue(all(any("REG" in issue and "injected limit" in issue for issue in flag.solver_issues)
                            for flag in failed.segment_flags))
        with patch("anamorph.source_analysis.source_neurite_pair_candidates",
                   side_effect=SourceBivariateFailure("injected source exception")):
            fallback = trace_source_analysis(tree)
        positives = [item for item in fallback.jobs if item.result is True]
        self.assertTrue(positives)
        self.assertTrue(all(not item.candidates and "fallback" in item.reason for item in positives))
        self.assertFalse(fallback.source_jobs_clean)
        self.assertFalse(fallback.planned_solver_work_complete)
        self.assertTrue(all("injected source exception" in item.solver_failure for item in positives))
        fallback_flags = {item.edge: item for item in fallback.segment_flags}
        for item in positives:
            flag_name = {"inter_cell": "rc_nsns", "intra_cell_inter_neurite": "icrn_nsns",
                         "intra_cell_intra_neurite": "icin_nsns",
                         "intra_cell_intra_neurite_adjacent": "icin_nsns"}[item.classification]
            self.assertTrue(all(getattr(fallback_flags[edge], flag_name) and
                                any("injected source exception" in issue for issue in fallback_flags[edge].solver_issues)
                                for edge in item.job.segments))
        self.assertFalse(SourceAnalysisTrace(1, 1, (SourceJobTrace(plan.jobs[0], True),), True).source_jobs_clean)

    def test_source_regularities_update_only_planned_segments(self):
        from anamorph.bernstein import RootInterval
        from anamorph.source_canal_polynomials import SourceScalarCandidate
        tree = read_swc(ROOT/"examples"/"branched.swc")
        hit = SourceScalarCandidate(.5, 0., RootInterval(.5, .5))
        with patch("anamorph.source_analysis.source_regularity_candidates", return_value=[hit]):
            trace = trace_source_analysis(tree, max_jobs=4)
        flagged = {status.edge for status in trace.segment_flags if status.reg}
        self.assertEqual(flagged, {item.job.segments[0] for item in trace.jobs if item.job.kind == "REG"})
        self.assertEqual(len(flagged), 2)
        self.assertTrue(trace.truncated)
        self.assertTrue(all(status.source_plan_may_affect for status in trace.segment_flags))

    def test_suppressed_global_solver_failure_is_visible_and_blocks_conservative_gate(self):
        import anamorph.source_bivariate as bivariate
        from anamorph.analytic_mesher import PartialAnalysisError, assemble_redblue_checked
        tree = read_swc(ROOT/"examples"/"branched.swc")
        original = bivariate.source_bilinear_clip_roots
        def fail_global(p, q, **options):
            if p.degree_x == 7:
                raise bivariate.SourceBivariateFailure("injected global failure")
            return original(p, q, **options)
        with patch("anamorph.source_bivariate.source_bilinear_clip_roots", side_effect=fail_global):
            trace = trace_source_analysis(tree)
            report = analyze_analytic_partial(tree, prepare=True, job_mode="source",
                                              preserve_surface_roots=False)
            with self.assertRaises(PartialAnalysisError) as caught:
                assemble_redblue_checked(tree, analysis_job_mode="source", analysis_gate="source",
                                         preserve_surface_roots=False)
        gsi = [item for item in trace.jobs if item.job.kind == "GSI"]
        self.assertTrue(gsi)
        self.assertTrue(all(item.result is False for item in gsi))
        self.assertTrue(all("injected global failure" in item.solver_failure for item in gsi))
        # The source-style result remains clear, but it must not hide that
        # the bivariate search was abandoned.
        self.assertTrue(trace.source_jobs_clean)
        self.assertFalse(trace.planned_solver_work_complete)
        self.assertFalse(trace.as_dict()["planned_solver_work_complete"])
        self.assertTrue(all(any("injected global failure" in issue for issue in flag.solver_issues)
                            for flag in trace.segment_flags))
        self.assertIsNone(report.source_jobs_clean)
        self.assertTrue(any("injected global failure" in issue for issue in caught.exception.report.unresolved))
        self.assertTrue(any("injected global failure" in issue for issue in report.unresolved))
        self.assertTrue(any(any("injected global failure" in issue for issue in flag.unresolved)
                            for flag in report.segment_flags))

    def test_source_trace_cli_emits_job_results(self):
        source = ROOT/"examples"/"branched.swc"
        result = subprocess.run([sys.executable, "-m", "anamorph", "trace-source-jobs", str(source),
                                 "--max-jobs", "2"], text=True, capture_output=True, check=True, cwd=ROOT)
        report = json.loads(result.stdout)
        self.assertEqual(len(report["jobs"]), 2)
        self.assertTrue(report["truncated"])
        self.assertIsNone(report["source_jobs_clean"])
        self.assertFalse(report["planned_solver_work_complete"])
        self.assertTrue(all("solver_failure" in item for item in report["jobs"]))
        self.assertEqual(len(report["segment_flags"]), report["segment_count"])
        self.assertTrue(all(item["source_plan_may_affect"] for item in report["segment_flags"]))
        self.assertFalse(report["certifies_consistency"])
        self.assertIn("do not certify", result.stderr)

    def test_source_trace_cli_exposes_suppressed_global_failure(self):
        source = ROOT/"examples"/"branched.swc"
        result = subprocess.run([sys.executable, "-m", "anamorph", "trace-source-jobs", str(source),
                                 "--max-jobs", "3", "--bivariate-tolerance", "2"],
                                text=True, capture_output=True, check=True, cwd=ROOT)
        report = json.loads(result.stdout)
        gsi, = (item for item in report["jobs"] if item["job"]["kind"] == "GSI")
        self.assertIs(gsi["result"], False)
        self.assertIn("smaller than tolerance", gsi["solver_failure"])
        self.assertFalse(report["planned_solver_work_complete"])

    def test_source_pair_edge_candidates_report_witnesses_and_rounded_thresholds(self):
        for radius_a, radius_b, offset, expected_source in ((.03, .03, .05, "source_root_witness"),
                                                             (.1, .2, .1+.2, "source_threshold")):
            with self.subTest(offset=offset):
                first = CanalSegment(((-1., 1., 0., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)),
                                     radius_a, radius_a)
                second = CanalSegment(((offset, 0., 0., 0.), (-1., 2., 0., 0.), (0., 0., 0., 0.)),
                                      radius_b, radius_b)
                tree = Morphology([Node(1, 1, -3., 0., 0., .2, -1), Node(2, 3, -1., 0., 0., radius_a, 1),
                                   Node(3, 3, 0., 0., 0., radius_a, 2), Node(4, 1, offset, -3., 0., .2, -1),
                                   Node(5, 3, offset, -1., 0., radius_b, 4),
                                   Node(6, 3, offset, 1., 0., radius_b, 5)])
                with patch("anamorph.analytic_network.CanalSegment.from_spline",
                           side_effect=(first, second, first, second)), \
                        _mock_source_controls(first, second), \
                        patch("anamorph.analytic_network.neurite_neurite_candidates", return_value=[]):
                    report = analyze_analytic_partial(tree, job_mode="source")
                finding, = (item for item in report.findings if item.kind == "source_pair_candidate")
                self.assertEqual(finding.classification, "inter_cell")
                self.assertTrue(any(c.parameters == (1., .5) and c.source == expected_source
                                    for c in finding.candidates))
                self.assertFalse(report.source_jobs_clean)
                flags = {item.edge: item for item in report.segment_flags}
                self.assertTrue(flags[(2, 3)].rc_nsns and flags[(5, 6)].rc_nsns)

    def test_source_global_exception_continues_to_candidates_but_work_limit_is_unresolved(self):
        from anamorph.bernstein import ClippingConvergenceError
        from anamorph.source_bivariate import SourceBivariateFailure, source_global_self_intersection_candidates
        loop = CanalSegment(((0., 1., -1., 0.), (0., -1., 3., -2.), (0., 0., 0., 0.)), .05, .05)
        with patch("anamorph.source_bivariate.source_bilinear_clip_roots",
                   side_effect=SourceBivariateFailure("injected source exception")):
            hits = source_global_self_intersection_candidates(loop)
        self.assertTrue(any(hit.source == "corner" and hit.t == 0. and hit.u == 1. for hit in hits))
        with patch("anamorph.source_bivariate.source_bilinear_clip_roots",
                   side_effect=ClippingConvergenceError("injected work limit")):
            with self.assertRaisesRegex(ClippingConvergenceError, "work limit"):
                source_global_self_intersection_candidates(loop)

    def test_source_candidate_witness_checks_source_curve(self):
        from anamorph.bernstein import RootRectangle
        from anamorph.source_bivariate import SourcePairCandidate
        radius = .1
        tree = Morphology([Node(1, 1, -2., 0., 0., .2, -1), Node(2, 3, 0., 0., 0., radius, 1),
                           Node(3, 3, 1., 0., 0., radius, 2), Node(4, 3, -1., 0., 0., radius, 2)])
        first = CanalSegment(((0., 1., 0., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)), radius, radius)
        second = CanalSegment(((0., -1., 0., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)), radius, radius)
        zero = ((0., 0., 0., 0.),) * 3
        shifted = ((1., 1., 1., 1.), zero[1], zero[2])
        hit = SourcePairCandidate(0., 0., 0., RootRectangle(0., 0., 0., 0.), "boundary")
        # The injected local power curves coincide at t=u=0, while the source
        # controls are separated. Only the latter describe the source solve.
        with patch("anamorph.analytic_network.CanalSegment.from_spline",
                   side_effect=(first, second, first, second)), \
                patch("anamorph.analytic_network.source_spline_controls",
                      side_effect=(((0., 1.), (zero,)), ((0., 1.), (shifted,)))), \
                patch("anamorph.analytic_network.adjacent_neurite_candidates", return_value=[]), \
                patch("anamorph.analytic_network.source_neurite_pair_candidates", return_value=[hit]):
            report = analyze_analytic_partial(tree, job_mode="source")
        finding, = (item for item in report.findings if item.kind == "source_pair_candidate")
        self.assertEqual(finding.candidates[0].source, "source_threshold")

    def test_source_global_trace_retains_verified_witness(self):
        radius = .05
        loop = CanalSegment(((0., 1., -1., 0.), (0., -1., 3., -2.), (0., 0., 0., 0.)), radius, radius)
        straight = CanalSegment(((0., -1., 0., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)), radius, radius)
        tree = Morphology([Node(1, 1, -2., 0., 0., .2, -1), Node(2, 3, 0., 0., 0., radius, 1),
                           Node(3, 3, 1., 0., 0., radius, 2), Node(4, 3, -1., 0., 0., radius, 2)])
        with patch("anamorph.analytic_network.CanalSegment.from_spline", side_effect=(loop, straight, loop, straight)), \
                _mock_source_controls(loop, straight), \
                patch("anamorph.analytic_network.global_self_intersection_candidates", return_value=[]):
            report = analyze_analytic_partial(tree, job_mode="source")
        finding, = (item for item in report.findings if item.kind == "source_global_self_intersection_candidate")
        self.assertEqual(finding.segments, ((2, 3),))
        self.assertTrue(any(item.parameters == (0., 1.) and item.value == 0. and
                            item.source == "source_root_witness" for item in finding.candidates))
        self.assertTrue(next(flag for flag in report.segment_flags if flag.edge == (2, 3)).gsi)
        self.assertFalse(report.source_jobs_clean)

    def test_source_soma_threshold_is_reported_without_claiming_a_geometric_overlap(self):
        radius = 1e-9
        tree = Morphology([Node(1, 1, -3., 0., 0., .2, -1), Node(2, 3, -1., 0., 0., radius, 1),
                           Node(3, 3, 1., 0., 0., radius, 2), Node(4, 3, 2., 0., 0., radius, 3),
                           Node(5, 1, 0., 1e-6, 0., radius, -1)])
        report = analyze_analytic_partial(tree, job_mode="source")
        finding, = (item for item in report.findings if item.kind == "source_soma_neurite_candidate")
        self.assertEqual((finding.segments, finding.somas), (((2, 3),), (5,)))
        self.assertIn("not verified overlaps", finding.detail)
        self.assertEqual(finding.candidates[0].source, "source_threshold")
        self.assertAlmostEqual(finding.candidates[0].value, 1e-12)
        self.assertGreater(finding.candidates[0].value, (2*radius)**2)
        self.assertTrue(next(flag for flag in report.segment_flags if flag.edge == (2, 3)).ic_sons)
        self.assertFalse(report.source_jobs_clean)
        self.assertFalse(any(item.kind == "soma_neurite_candidate" and item.somas == (5,)
                             for item in report.findings))

    def test_source_univariate_trace_limit_is_unresolved(self):
        from anamorph.bernstein import ClippingConvergenceError
        tree = read_swc(ROOT/"examples"/"branched.swc")
        with patch("anamorph.analytic_network.source_regularity_candidates",
                   side_effect=ClippingConvergenceError("injected source work limit")):
            report = analyze_analytic_partial(tree, job_mode="source")
        self.assertTrue(any("source solver trace unresolved" in item for item in report.unresolved))
        self.assertTrue(any(any("source solver trace unresolved" in item for item in flag.unresolved)
                            for flag in report.segment_flags))
        self.assertIsNone(report.source_jobs_clean)

    def test_source_mode_does_not_claim_clean_for_singular_pair(self):
        radius = .0001
        sibling_tree = Morphology([Node(1, 1, -2., 0., 0., .2, -1), Node(2, 3, 0., 0., 0., radius, 1),
                                   Node(3, 3, 1., 0., 0., radius, 2), Node(4, 3, -1., 0., 0., radius, 2)])
        first = CanalSegment(((0., 1., 0., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)), radius, radius)
        for transverse, fallback in ((0., True), (1e-6, True), (.001, False), (1., False)):
            second = CanalSegment(((0., -1., 0., 0.), (0., transverse, 0., 0.),
                                   (0., 0., 0., 0.)), radius, radius)
            with self.subTest(transverse=transverse):
                with patch("anamorph.analytic_network.CanalSegment.from_spline",
                           side_effect=(first, second, first, second)), _mock_source_controls(first, second):
                    report = analyze_analytic_partial(sibling_tree, job_mode="source")
                source_findings = [item for item in report.findings if item.kind == "source_solver_fallback"]
                self.assertEqual(bool(source_findings), fallback)
                self.assertIs(report.source_jobs_clean, not fallback)
                if fallback:
                    self.assertEqual(source_findings[0].candidates, ())
                self.assertFalse(any(item.kind == "adjacent_neurite_candidate" for item in report.findings))
                flags = {item.edge: item for item in report.segment_flags}
                self.assertEqual(flags[(2, 3)].icin_nsns, fallback)
                self.assertEqual(flags[(2, 4)].icin_nsns, fallback)
                with patch("anamorph.analytic_network.CanalSegment.from_spline", side_effect=(first, second)):
                    extended = analyze_analytic_partial(sibling_tree, job_mode="extended")
                self.assertFalse(extended.unresolved)

        # The source's initial fit can also be singular for cubic
        # parameterizations that are not exactly affine.
        for curvature, fallback in ((1e-12, True), (1e-10, True), (1e-7, False)):
            curved_first = CanalSegment(((0., 1., curvature, 0.), (0., 0., 0., 0.),
                                         (0., 0., 0., 0.)), radius, radius)
            curved_second = CanalSegment(((0., -1., -curvature, 0.), (0., 0., 0., 0.),
                                          (0., 0., 0., 0.)), radius, radius)
            with self.subTest(curvature=curvature), \
                    patch("anamorph.analytic_network.CanalSegment.from_spline",
                          side_effect=(curved_first, curved_second, curved_first, curved_second)), \
                    _mock_source_controls(curved_first, curved_second):
                report = analyze_analytic_partial(sibling_tree, job_mode="source")
                self.assertIs(report.source_jobs_clean, not fallback)
                self.assertEqual(any(item.kind == "source_solver_fallback" for item in report.findings), fallback)

        # Source boxes include padding and can schedule two distinct,
        # separated parallel paths even though their bounding pipes miss.
        tree = Morphology([Node(1, 1, -2., 0., 0., .2, -1), Node(2, 3, 0., 0., 0., radius, 1),
                           Node(3, 3, 1., 0., 0., radius, 2), Node(4, 1, -2., .0003, 0., .2, -1),
                           Node(5, 3, 0., .0003, 0., radius, 4), Node(6, 3, 1., .0003, 0., radius, 5)])
        second = CanalSegment(((0., 1., 0., 0.), (.0003, 0., 0., 0.),
                               (0., 0., 0., 0.)), radius, radius)
        with patch("anamorph.analytic_network.CanalSegment.from_spline", side_effect=(first, second, first, second)), \
                _mock_source_controls(first, second):
            report = analyze_analytic_partial(tree, job_mode="source")
        self.assertIs(report.source_jobs_clean, False)
        finding, = (item for item in report.findings if item.kind == "source_solver_fallback")
        self.assertEqual((finding.classification, finding.candidates), ("inter_cell", ()))
        flags = {item.edge: item for item in report.segment_flags}
        self.assertTrue(flags[(2, 3)].rc_nsns and flags[(5, 6)].rc_nsns)
        self.assertFalse(any(item.kind == "neurite_neurite_candidate" for item in report.findings))

    def test_source_fallback_and_numerical_failure_are_reported_independently(self):
        from anamorph.analytic_intersections import IntersectionSolveError
        from anamorph.bernstein import ClippingConvergenceError
        radius = .0001
        tree = Morphology([Node(1, 1, -2., 0., 0., .2, -1), Node(2, 3, 0., 0., 0., radius, 1),
                           Node(3, 3, 1., 0., 0., radius, 2), Node(4, 3, -1., 0., 0., radius, 2)])
        first = CanalSegment(((0., 1., 0., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)), radius, radius)
        second = CanalSegment(((0., -1., 0., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)), radius, radius)
        with patch("anamorph.analytic_network.CanalSegment.from_spline", side_effect=(first, second, first, second)), \
             _mock_source_controls(first, second), \
             patch("anamorph.analytic_network.adjacent_neurite_candidates",
                   side_effect=IntersectionSolveError(("injected clipping failure",), ())):
            report = analyze_analytic_partial(tree, job_mode="source")
        self.assertIs(report.source_jobs_clean, False)
        self.assertTrue(any("injected clipping failure" in item for item in report.unresolved))
        fallback, = (item for item in report.findings if item.kind == "source_solver_fallback")
        self.assertEqual(fallback.candidates, ())
        self.assertTrue(all(flag.icin_nsns for flag in report.segment_flags))

        with patch("anamorph.analytic_network.CanalSegment.from_spline", side_effect=(first, second, first, second)), \
             _mock_source_controls(first, second), \
             patch("anamorph.source_bivariate.source_stationary_bernstein",
                   side_effect=ClippingConvergenceError("injected source fit overflow")):
            report = analyze_analytic_partial(tree, job_mode="source")
        self.assertIsNone(report.source_jobs_clean)
        self.assertTrue(any("injected source fit overflow" in item for item in report.unresolved))
        self.assertFalse(any(item.kind == "source_solver_fallback" for item in report.findings))

        orthogonal = CanalSegment(((0., 0., 0., 0.), (0., 1., 0., 0.),
                                   (0., 0., 0., 0.)), radius, radius)
        with patch("anamorph.analytic_network.CanalSegment.from_spline",
                   side_effect=(first, orthogonal, first, orthogonal)), _mock_source_controls(first, orthogonal):
            report = analyze_analytic_partial(tree, job_mode="source", bivariate_tolerance=1.1)
        fallback, = (item for item in report.findings if item.kind == "source_solver_fallback")
        self.assertIn("smaller than tolerance", fallback.detail)
        self.assertIs(report.source_jobs_clean, False)

    def test_source_root_midpoint_retains_independently_verified_pair_witness(self):
        radius = .1
        first = CanalSegment(((-1., 2., 0., 0.), (0., 0., 0., 0.),
                              (0., 0., 0., 0.)), radius, radius)
        second = CanalSegment(((0., 0., 0., 0.), (-1., 2., 0., 0.),
                               (0., 0., 0., 0.)), radius, radius)
        tree = Morphology([Node(1, 1, -3., 0., 0., .2, -1), Node(2, 3, -1., 0., 0., radius, 1),
                           Node(3, 3, 1., 0., 0., radius, 2), Node(4, 1, 0., -3., 0., .2, -1),
                           Node(5, 3, 0., -1., 0., radius, 4), Node(6, 3, 0., 1., 0., radius, 5)])
        with patch("anamorph.analytic_network.CanalSegment.from_spline", side_effect=(first, second, first, second)), \
             _mock_source_controls(first, second), \
             patch("anamorph.analytic_network.neurite_neurite_candidates", return_value=[]):
            report = analyze_analytic_partial(tree, job_mode="source")
        finding, = (item for item in report.findings if item.kind == "source_bivariate_candidate")
        self.assertEqual(finding.classification, "inter_cell")
        # Conversion to floating-point Bézier controls moves the crossing by
        # a tiny amount, so the exact represented-control distance is nonzero.
        self.assertTrue(any(item.source == "source_root_witness" and item.value < 1e-30
                            for item in finding.candidates))
        flags = {item.edge: item for item in report.segment_flags}
        self.assertTrue(flags[(2, 3)].rc_nsns and flags[(5, 6)].rc_nsns)
        self.assertIs(report.source_jobs_clean, False)

    def test_root_collapse_reuses_freed_edge_id_and_unknown_changes_fall_back(self):
        tree = Morphology([Node(1, 1, 0., 0., 0., 1., -1), Node(2, 3, .2, 0., 0., .1, 1),
                           Node(3, 3, .6, 0., 0., .1, 2), Node(4, 3, 1.4, 0., 0., .1, 3),
                           Node(5, 3, 2.8, 0., 0., .1, 4), Node(6, 3, 0., 1.5, 0., .1, 1),
                           Node(7, 3, 0., 3., 0., .1, 6), Node(8, 3, 0., 4., 0., .1, 7)])
        adjusted = adjust_soma_roots(tree, preserve_surface_roots=False)
        state = source_edge_state_after_root_adjustment(tree, adjusted)
        self.assertIsNotNone(state)
        self.assertEqual(state.entries, ((0, 1, 2), (1, 1, 9), (2, 9, 6), (3, 6, 7),
                                         (4, 2, 4), (5, 7, 8), (6, 4, 5)))
        self.assertEqual((state.free_ids, state.next_id), ((), 7))
        changed = Morphology([replace(node, parent=2) if node.id == 4 else node for node in tree.nodes.values()])
        self.assertIsNone(source_edge_state_after_root_adjustment(tree, changed))

    def test_preconditioning_retains_an_edge_order_for_source_jobs(self):
        for filename in ("examples/branched.swc", "examples/mixed_soma_roots.swc"):
            with self.subTest(filename=filename):
                tree = read_swc(ROOT/filename)
                prepared = prepare_morphology(tree, preserve_surface_roots=False)
                self.assertTrue(prepared.split_segments or prepared.collapsed_segments)
                self.assertIsNotNone(prepared.source_edge_order)
                actual_edges = {(node.parent, node.id) for node in prepared.morphology.nodes.values()
                                if node.parent != -1}
                self.assertEqual(set(prepared.source_edge_order), actual_edges)
                plan = plan_source_analysis_jobs(tree, preserve_surface_roots=False)
                expected = [edge for edge in prepared.source_edge_order
                            if prepared.morphology.nodes[edge[0]].kind in (2, 3, 4)]
                self.assertEqual([job.segments[0] for job in plan.jobs if job.kind == "REG"], expected)
                limited = plan_source_analysis_jobs(tree, preserve_surface_roots=False, max_jobs=1)
                self.assertEqual(limited.jobs, plan.jobs[:1])
                with patch("anamorph.analytic_network.regularity_candidates", return_value=[]) as regularity:
                    report = analyze_analytic_partial(tree, prepare=True, job_mode="source",
                                                      preserve_surface_roots=False, source_job_limit=1)
                self.assertTrue(report.source_plan_truncated)
                segment = regularity.call_args.args[0]
                self.assertLess(dist(segment.point(0.), prepared.morphology.nodes[expected[0][0]].position), 1e-12)
                self.assertLess(dist(segment.point(1.), prepared.morphology.nodes[expected[0][1]].position), 1e-12)
        inside_chain = Morphology([Node(1, 1, 0., 0., 0., 1., -1), Node(2, 3, .2, 0., 0., .1, 1),
                                   Node(3, 3, .6, 0., 0., .1, 2), Node(4, 3, 1.4, 0., 0., .1, 3),
                                   Node(5, 3, 2.5, 0., 0., .1, 4)])
        self.assertIsNotNone(prepare_morphology(inside_chain, preserve_surface_roots=False).source_edge_order)
        self.assertIsNotNone(prepare_morphology(read_swc(ROOT/"examples"/"branched.swc"),
                                                scale_radius=.8).source_edge_order)

    def test_prepared_source_job_limit_uses_preinsertion_edge_order(self):
        tree = read_swc(ROOT/"examples"/"branched.swc")
        plan = plan_source_analysis_jobs(tree, preconditioning=False, preserve_surface_roots=False)
        self.assertEqual([job.segments[0] for job in plan.jobs if job.kind == "REG"],
                         [(2, 3), (3, 4), (4, 5), (4, 6), (6, 7), (8, 2)])
        limited = plan_source_analysis_jobs(tree, preconditioning=False, preserve_surface_roots=False, max_jobs=1)
        self.assertTrue(limited.truncated)
        self.assertEqual(limited.jobs, plan.jobs[:1])
        with patch("anamorph.analytic_network.regularity_candidates", return_value=[]) as regularity:
            report = analyze_analytic_partial(tree, prepare=True, preconditioning=False, job_mode="source",
                                              preserve_surface_roots=False, source_job_limit=1)
        self.assertTrue(report.source_plan_truncated)
        self.assertEqual(regularity.call_count, 1)
        segment = regularity.call_args.args[0]
        self.assertLess(dist(segment.point(0.), tree.nodes[2].position), 1e-12)
        self.assertLess(dist(segment.point(1.), tree.nodes[3].position), 1e-12)

    def test_spatial_job_plan_matches_ordered_brute_force_with_limits(self):
        class BruteForceIndex:
            def __init__(self, boxes):
                self.boxes = tuple(boxes)

            def query(self, query):
                return [i for i, box in enumerate(self.boxes)
                        if all(a <= d and c <= b for (a, b), (c, d) in zip(box, query))]

        nodes = []
        for i in range(20):
            first = 3*i+1
            x = 3.*i
            nodes.extend((Node(first, 1, x, 0., 0., .5, -1),
                          Node(first+1, 3, x+.5, 0., 0., .1, first),
                          Node(first+2, 3, x+2., 0., 0., .1, first+1)))
        cases = (read_swc(ROOT/"examples"/"branched.swc"),
                 read_swc(ROOT/"examples"/"mixed_soma_roots.swc"), Morphology(nodes))
        for tree in cases:
            for limit in (0, 1, 7, 25, 100000):
                with self.subTest(nodes=len(tree.nodes), limit=limit):
                    with patch("anamorph.source_jobs._BoxIndex", BruteForceIndex):
                        expected = plan_source_analysis_jobs(tree, prepare=False, max_jobs=limit)
                    actual = plan_source_analysis_jobs(tree, prepare=False, max_jobs=limit)
                    self.assertEqual(actual, expected)

    def test_source_skips_final_path_segment_for_soma_jobs(self):
        tree = Morphology([Node(1, 1, -3, 0, 0, .3, -1), Node(2, 3, -2, 0, 0, .1, 1),
                           Node(3, 3, 0, 0, 0, .1, 2), Node(4, 3, 2, 0, 0, .1, 3),
                           Node(5, 1, 1, .2, 0, .3, -1)])
        plan = plan_source_analysis_jobs(tree, prepare=False)
        self.assertFalse(plan.truncated)
        self.assertEqual((plan.segment_count, plan.path_count), (2, 1))
        self.assertFalse(any(job.kind == "SONS" and job.segments == ((3, 4),) for job in plan.jobs))
        self.assertEqual(Counter(job.kind for job in plan.jobs)["NSNS_ADJ"], 1)
        extended = analyze_analytic_partial(tree)
        source = analyze_analytic_partial(tree, job_mode="source")
        self.assertTrue(any(f.kind == "soma_neurite_candidate" and f.segments == ((3, 4),)
                            and f.somas == (5,) for f in extended.findings))
        self.assertFalse(any(f.kind == "soma_neurite_candidate" and f.segments == ((3, 4),)
                             and f.somas == (5,) for f in source.findings))
        self.assertEqual(source.job_mode, "source")
        self.assertEqual(source.source_job_counts, plan.as_dict()["job_counts"])
        self.assertFalse(source.complete)

    def test_source_root_exemption_applies_to_unrelated_somas(self):
        tree = Morphology([Node(1, 1, -2, 0, 0, 1, -1), Node(2, 3, 0, 0, 0, .1, 1),
                           Node(3, 3, 1, 0, 0, .1, 2), Node(4, 3, 2, 0, 0, .1, 3),
                           Node(5, 1, 0, .2, 0, .2, -1)])
        plan = plan_source_analysis_jobs(tree, prepare=False)
        self.assertTrue(any(job.kind == "SONS" and job.soma == 5 and job.root_segment for job in plan.jobs))
        extended = analyze_analytic_partial(tree)
        source = analyze_analytic_partial(tree, job_mode="source")
        self.assertTrue(any(f.kind == "soma_neurite_candidate" and f.somas == (5,) for f in extended.findings))
        self.assertFalse(any(f.kind == "soma_neurite_candidate" and f.somas == (5,) for f in source.findings))

    def test_job_limit_returns_an_explicitly_truncated_prefix(self):
        tree = read_swc(ROOT/"examples"/"branched.swc")
        complete = plan_source_analysis_jobs(tree)
        limited = plan_source_analysis_jobs(tree, max_jobs=3)
        self.assertTrue(limited.truncated)
        self.assertEqual(limited.jobs, complete.jobs[:3])
        self.assertEqual([job.kind for job in limited.jobs], ["REG", "LSI", "GSI"])
        self.assertFalse(complete.as_dict()["executes_checks"])
        report = analyze_analytic_partial(tree, job_mode="source", source_job_limit=3)
        self.assertTrue(report.source_plan_truncated)
        self.assertTrue(any("source job plan limit" in item for item in report.unresolved))
        self.assertTrue(all(status.source_plan_may_affect for status in report.segment_flags))
        pair_limited = analyze_analytic_partial(tree, job_mode="source", max_pairs=0)
        self.assertTrue(pair_limited.pair_limit_reached)
        self.assertIsNotNone(pair_limited.first_untested_pair)
        self.assertTrue(all(status.pair_limit_may_affect for status in pair_limited.segment_flags))
        with self.assertRaises(ValueError):
            plan_source_analysis_jobs(tree, max_jobs=-1)

    def test_truncated_source_plan_executes_only_its_selected_segment_jobs(self):
        tree = read_swc(ROOT/"examples"/"branched.swc")
        for limit in (0, 1, 2, 3, 4, 7):
            with self.subTest(limit=limit):
                plan = plan_source_analysis_jobs(tree, prepare=False, max_jobs=limit)
                calls = []
                def record(kind):
                    def check(segment, **kwargs):
                        calls.append((kind, segment.point(0.)))
                        return ()
                    return check
                with patch("anamorph.analytic_network.regularity_candidates", side_effect=record("REG")), \
                     patch("anamorph.analytic_network.local_self_intersection_candidates", side_effect=record("LSI")), \
                     patch("anamorph.analytic_network.global_self_intersection_candidates", side_effect=record("GSI")):
                    report = analyze_analytic_partial(tree, job_mode="source", source_job_limit=limit)
                expected = [(job.kind, tree.nodes[job.segments[0][0]].position)
                            for job in plan.jobs if job.kind in ("REG", "LSI", "GSI")]
                self.assertEqual(Counter(calls), Counter(expected))
                self.assertTrue(report.source_plan_truncated)
                self.assertTrue(all(flag.source_plan_may_affect for flag in report.segment_flags))

    def test_cli_exposes_planning_and_source_selected_analysis(self):
        source = ROOT/"examples"/"branched.swc"
        plan_run = subprocess.run([sys.executable, "-m", "anamorph", "plan-source-jobs", str(source)],
                                  text=True, capture_output=True, check=True, cwd=ROOT)
        plan = json.loads(plan_run.stdout)
        self.assertFalse(plan["executes_checks"])
        self.assertFalse(plan["truncated"])
        self.assertEqual(sum(plan["job_counts"].values()), 21)
        analysis_run = subprocess.run([sys.executable, "-m", "anamorph", "analyze-analytic", str(source),
                                       "--job-mode", "source", "--source-surface-roots"],
                                      text=True, capture_output=True, check=True, cwd=ROOT)
        report = json.loads(analysis_run.stdout)
        self.assertEqual(report["job_mode"], "source")
        self.assertEqual(report["source_job_counts"], plan["job_counts"])
        self.assertIn("intentionally omits", analysis_run.stderr)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/"mesh.obj"
            command = [sys.executable, "-m", "anamorph", "generate-redblue", str(source), str(output),
                       "--force", "--require-no-findings", "--analysis-job-mode", "source"]
            for streaming in ([], ["--flush-face-limit", "1"]):
                run = subprocess.run(command+streaming+["--analysis-source-job-limit", "3"],
                                     text=True, capture_output=True, cwd=ROOT)
                self.assertNotEqual(run.returncode, 0)
                self.assertFalse(output.exists())
                self.assertIn("unresolved", run.stderr)
            run = subprocess.run(command+["--flush-face-limit", "1", "--angular-segments", "8",
                                          "--soma-refinements", "2", "--outer-attempts", "20",
                                          "--inner-attempts", "20"],
                                 text=True, capture_output=True, check=True, cwd=ROOT)
            result = json.loads(run.stdout)
            self.assertTrue(output.is_file())
            self.assertEqual(result["analysis"]["job_mode"], "source")
            self.assertFalse(result["analysis"]["source_plan_truncated"])
            self.assertFalse(result["analysis"]["complete"])
            self.assertGreaterEqual(result["partial_flush"]["flushes"], 2)


@unittest.skipUnless(sys.platform == "win32" and SOLVER_PROBE.is_file() and JOBS_PROBE.is_file(),
                     "optional Windows C++ source job probes are not available")
class CppSourceJobPlanTests(unittest.TestCase):
    @unittest.skipUnless(FLAGS_PROBE.is_file(), "optional C++ source flag probe is not built")
    def test_full_source_segment_flags_match_cpp_analysis(self):
        names = ("pmdv", "spmdv", "smdv", "reg", "lsi", "gsi", "rc_sons", "ic_sons",
                 "rc_nsns", "icrn_nsns", "icin_nsns")
        cases = ("examples/branched.swc", "examples/mixed_soma_roots.swc",
                 "tests/fixtures/branch_curvature_collision.swc")
        extra = ("1 1 -3 0 0 .3 -1\n2 3 -2 0 0 .1 1\n3 3 0 0 0 .1 2\n"
                 "4 3 2 0 0 .1 3\n5 1 -1 .2 0 .3 -1\n",
                 "1 1 -3 -3 0 .2 -1\n2 3 -2 0 0 .1 1\n3 3 2 0 0 .1 2\n"
                 "4 3 0 -2 0 .1 1\n5 3 0 2 0 .1 4\n",
                 "1 1 -3 -3 0 .2 -1\n2 3 -2 0 0 .1 1\n3 3 2 0 0 .1 2\n"
                 "4 1 3 -3 0 .2 -1\n5 3 0 -2 0 .1 4\n6 3 0 2 0 .1 5\n",
                 "1 1 -2 0 0 .2 -1\n2 3 -1 0 0 .1 1\n3 3 1 0 0 .1 2\n4 3 1 .1 0 .1 2\n")
        seen = set()
        with tempfile.TemporaryDirectory() as directory:
            paths = [ROOT/name for name in cases]
            for index, contents in enumerate(extra):
                path = Path(directory)/f"case{index}.swc"
                path.write_text(contents, encoding="utf-8")
                paths.append(path)
            for path in paths:
                for preconditioning in (False, True):
                    with self.subTest(path=path.name, preconditioning=preconditioning):
                        run = subprocess.run([str(FLAGS_PROBE), str(path), str(int(preconditioning))],
                                             cwd=directory, capture_output=True, text=True, timeout=30, check=True)
                        expected = Counter((tuple(round(float(value), 7) for value in fields[1:7]),
                                            tuple(map(int, fields[7:18])))
                                           for line in run.stdout.splitlines()
                                           if (fields := line.split()) and fields[0] == "FLAG")
                        tree = read_swc(path)
                        prepared = prepare_morphology(tree, preconditioning=preconditioning,
                                                      preserve_surface_roots=False).morphology
                        trace = trace_source_analysis(tree, preconditioning=preconditioning,
                                                      preserve_surface_roots=False)
                        actual = Counter((tuple(round(value, 7) for node in flag.edge
                                                for value in prepared.nodes[node].position),
                                          tuple(int(getattr(flag, name)) for name in names))
                                         for flag in trace.segment_flags)
                        self.assertEqual(actual, expected)
                        seen.update(name for flag in trace.segment_flags for name in names if getattr(flag, name))
        self.assertTrue({"pmdv", "spmdv", "smdv", "lsi", "gsi", "ic_sons", "rc_nsns",
                         "icrn_nsns", "icin_nsns"} <= seen, seen)

    @unittest.skipUnless(FLAGS_PROBE.is_file(), "optional C++ source flag probe is not built")
    def test_seeded_curved_tree_segment_flags_match_cpp(self):
        names = ("pmdv", "spmdv", "smdv", "reg", "lsi", "gsi", "rc_sons", "ic_sons",
                 "rc_nsns", "icrn_nsns", "icin_nsns")
        rng = random.Random(182811)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"sample.swc"
            for case in range(12):
                rows = ["1 1 0 0 0 .4 -1", "2 3 .65 0 0 .1 1"]
                coords = {2: (.65, 0., 0.)}
                for node in range(3, rng.randrange(6, 10)):
                    parent = rng.randrange(2, node) if rng.random() < .35 else node-1
                    point = tuple(coords[parent][axis]+rng.uniform(-1.25, 1.25) for axis in range(3))
                    coords[node] = point
                    rows.append(f"{node} 3 {point[0]:.17g} {point[1]:.17g} {point[2]:.17g} "
                                f"{rng.uniform(.06, .3):.17g} {parent}")
                path.write_text("\n".join(rows)+"\n", encoding="utf-8")
                with self.subTest(case=case):
                    run = subprocess.run([str(FLAGS_PROBE), str(path), "0"], cwd=directory,
                                         capture_output=True, text=True, timeout=30, check=True)
                    expected = Counter((tuple(round(float(value), 7) for value in fields[1:7]),
                                        tuple(map(int, fields[7:18])))
                                       for line in run.stdout.splitlines()
                                       if (fields := line.split()) and fields[0] == "FLAG")
                    tree = read_swc(path)
                    prepared = prepare_morphology(tree, preconditioning=False,
                                                  preserve_surface_roots=False).morphology
                    trace = trace_source_analysis(tree, preconditioning=False, preserve_surface_roots=False)
                    self.assertTrue(trace.planned_solver_work_complete)
                    actual = Counter((tuple(round(value, 7) for node in flag.edge
                                            for value in prepared.nodes[node].position),
                                      tuple(int(getattr(flag, name)) for name in names))
                                     for flag in trace.segment_flags)
                    self.assertEqual(actual, expected)

    @unittest.skipUnless(OPTIONS_JOBS_PROBE.is_file(), "optional C++ parameterized job probe is not built")
    def test_seeded_multicell_job_classifications_across_parameterizations(self):
        rng = random.Random(84621)
        kinds = {"REG": 0, "LSI": 1, "GSI": 2, "SONS": 3, "NSNS_ADJ": 4, "NSNS_NONADJ": 5}
        classifications = {"inter_cell": 6, "intra_cell_inter_neurite": 7,
                           "intra_cell_intra_neurite": 8, "intra_cell_intra_neurite_adjacent": 9}
        positives = 0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cells.swc"
            for case in range(4):
                rows, positions, node = [], {}, 0
                for cell in range(2):
                    node += 1
                    soma = node
                    center = (cell*1.8-1., 0., 0.)
                    rows.append(f"{soma} 1 {center[0]} 0 0 .3 -1")
                    node += 1
                    position = (center[0]+(.5 if cell == 0 else -.5), 0., 0.)
                    positions[node] = position
                    rows.append(f"{node} 3 {position[0]} 0 0 .12 {soma}")
                    for index in range(4):
                        node += 1
                        parent = (rng.choice(list(positions)[-(index+1):]) if rng.random() < .35 else node-1)
                        position = tuple(positions[parent][axis]+rng.uniform(-1.4, 1.4) for axis in range(3))
                        positions[node] = position
                        rows.append(f"{node} 3 {position[0]:.17g} {position[1]:.17g} {position[2]:.17g} "
                                    f"{rng.uniform(.06, .35):.17g} {parent}")
                path.write_text("\n".join(rows)+"\n", encoding="utf-8")
                for parameter in ("chord-length", "centripetal", "uniform"):
                    with self.subTest(case=case, parameter=parameter):
                        run = subprocess.run([str(OPTIONS_JOBS_PROBE), str(path), "0", "1", parameter],
                                             cwd=directory, capture_output=True, text=True, check=True, timeout=30)
                        expected = []
                        for line in run.stdout.splitlines():
                            parts = line.split()
                            if not parts or parts[0] != "JOB":
                                continue
                            kind, result = int(parts[1]), bool(int(parts[-1]))
                            expected.append((kind, result, int(parts[-2]) if kind in (4, 5) else -1))
                            positives += result
                        trace = trace_source_analysis(read_swc(path), preconditioning=False,
                                                      parametrization=parameter)
                        actual = [(kinds[item.job.kind], item.result,
                                   classifications[item.classification] if item.result and
                                   item.job.kind in ("NSNS_ADJ", "NSNS_NONADJ") else -1)
                                  for item in trace.jobs]
                        self.assertEqual(actual, expected)
        self.assertGreater(positives, 0)

    def test_per_job_source_trace_matches_cpp_execution(self):
        kinds = {"REG": 0, "LSI": 1, "GSI": 2, "SONS": 3, "NSNS_ADJ": 4, "NSNS_NONADJ": 5}
        classifications = {"inter_cell": 6, "intra_cell_inter_neurite": 7,
                           "intra_cell_intra_neurite": 8, "intra_cell_intra_neurite_adjacent": 9}
        cases = (ROOT/"examples"/"branched.swc", ROOT/"examples"/"mixed_soma_roots.swc",
                 ROOT/"tests"/"fixtures"/"branch_curvature_collision.swc")
        with tempfile.TemporaryDirectory() as directory:
            crossing = Path(directory)/"crossing.swc"
            crossing.write_text("1 1 -4 0 0 .25 -1\n2 3 -3 0 0 .1 1\n3 3 -1 0 0 .1 2\n"
                                "4 3 1 0 0 .1 3\n5 3 3 0 0 .1 4\n6 1 0 -4 0 .25 -1\n"
                                "7 3 0 -3 0 .1 6\n8 3 0 -1 0 .1 7\n9 3 0 1 0 .1 8\n10 3 0 3 0 .1 9\n",
                                encoding="utf-8")
            for path in (*cases, crossing):
                with self.subTest(path=path.name):
                    run = subprocess.run([str(JOBS_PROBE), str(path), "1", "1"], text=True, capture_output=True,
                                         timeout=30, check=True, cwd=directory)
                    reference = Counter()
                    for text_line in run.stdout.splitlines():
                        parts = text_line.split()
                        if not parts or parts[0] != "JOB":
                            continue
                        kind = int(parts[1])
                        edge = tuple(round(float(value), 7) for value in parts[2:8])
                        tail = ()
                        if kind == 3:
                            tail = tuple(round(float(value), 7) for value in parts[8:11])+(int(parts[11]),)
                        elif kind in (4, 5):
                            tail = tuple(round(float(value), 7) for value in parts[8:14])
                            if kind == 4:
                                tail += (int(parts[14]),)
                            tail += (int(parts[-2]),)
                        reference[(kind, edge, tail, bool(int(parts[-1])))] += 1
                    raw = read_swc(path)
                    prepared = prepare_morphology(raw, preserve_surface_roots=False).morphology
                    trace = trace_source_analysis(raw)
                    self.assertFalse(trace.truncated)
                    self.assertFalse(any(item.result is None for item in trace.jobs))
                    def coordinates(edge):
                        return tuple(round(value, 7) for node in edge for value in prepared.nodes[node].position)
                    actual = Counter()
                    for item in trace.jobs:
                        job, kind = item.job, kinds[item.job.kind]
                        edge, tail = coordinates(job.segments[0]), ()
                        if kind == 3:
                            tail = tuple(round(value, 7) for value in prepared.nodes[job.soma].position)+(int(job.root_segment),)
                        elif kind in (4, 5):
                            tail = coordinates(job.segments[1])
                            if kind == 4:
                                tail += (int(job.first_end_second_start),)
                            tail += (classifications[item.classification] if item.classification else -1,)
                        actual[(kind, edge, tail, item.result)] += 1
                    self.assertEqual(actual, reference)

    @unittest.skipUnless(SCALE_JOBS_PROBE.is_file(), "optional C++ partition probe is not built")
    def test_partition_filter_degree_conversion_boundary_matches_cpp(self):
        contents = ("1 1 -2 0 0 .25 -1\n2 3 -1 0 0 .05 1\n3 3 0 0 0 .05 2\n"
                    "4 3 .5652931960631212 -.5610766494745429 0 .05 3\n5 3 0 1 0 .05 3\n")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"case.swc"
            path.write_text(contents, encoding="utf-8")
            tree = read_swc(path)
            prepared = prepare_morphology(tree, preconditioning=False, preserve_surface_roots=False).morphology
            for mode, strategy in ((0, "max-chordal-depth"), (1, "min-angle")):
                with self.subTest(strategy=strategy):
                    angle = 44.785515214127635
                    run = subprocess.run([str(SCALE_JOBS_PROBE), str(path), "0", "3", "1", str(mode),
                                          repr(angle), "10"], text=True, capture_output=True, timeout=30, cwd=ROOT)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    reference = [tuple(round(float(value), 10) for value in parts[1:])
                                 for line in run.stdout.splitlines() if (parts := line.split())
                                 and parts[0] == "PATH"]
                    actual = [tuple(round(value, 10) for node in item for value in prepared.nodes[node].position)
                              for item in partition_paths(prepared, strategy, filter_angle=angle)]
                    self.assertEqual(actual, reference)

    @unittest.skipUnless(SCALE_JOBS_PROBE.is_file(), "optional C++ partition probe is not built")
    def test_seeded_partition_options_match_cpp_paths(self):
        rng = random.Random(928351)
        strategies = ("max-chordal-depth", "min-angle", "simple-neurite-paths")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"case.swc"
            for case in range(18):
                rows = ["1 1 0 0 0 1 -1", "2 3 1.25 0 0 .18 1"]
                coords = {2: (1.25, 0., 0.)}
                for node in range(3, 17):
                    parent = rng.randrange(2, node) if rng.random() < .55 else node-1
                    old = coords[parent]
                    point = (old[0]+rng.uniform(.3, 2.), old[1]+rng.uniform(-1., 1.),
                             old[2]+rng.uniform(-.3, .3))
                    coords[node] = point
                    radius = rng.uniform(.1, .3)
                    rows.append(f"{node} 3 {point[0]:.17g} {point[1]:.17g} {point[2]:.17g} {radius:.17g} {parent}")
                path.write_text("\n".join(rows)+"\n", encoding="utf-8")
                mode = case % 3
                angle = rng.choice((10, 30, 90, 150, 180))
                ratio = rng.choice((1.1, 1.5, 2., 10.))
                preconditioning = rng.choice((False, True))
                scale = rng.choice((.7, 1., 1.2))
                with self.subTest(case=case, strategy=strategies[mode], angle=angle, ratio=ratio,
                                  preconditioning=preconditioning, scale=scale):
                    run = subprocess.run([str(SCALE_JOBS_PROBE), str(path), str(int(preconditioning)), "3",
                                          str(scale), str(mode), str(angle), str(ratio)], text=True,
                                         capture_output=True, timeout=30, cwd=ROOT)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    reference = [tuple(round(float(value), 6) for value in parts[1:])
                                 for line in run.stdout.splitlines() if (parts := line.split())
                                 and parts[0] == "PATH"]
                    tree = read_swc(path)
                    prepared = prepare_morphology(tree, preconditioning=preconditioning, scale_radius=scale,
                                                  preserve_surface_roots=False).morphology
                    actual = [tuple(round(value, 6) for node in item for value in prepared.nodes[node].position)
                              for item in partition_paths(prepared, strategies[mode], angle, ratio)]
                    self.assertEqual(actual, reference)

    @unittest.skipUnless(SCALE_JOBS_PROBE.is_file(), "optional C++ partition probe is not built")
    def test_nearly_parallel_partition_filters_and_ties_match_cpp(self):
        contents = ("1 1 -2 0 0 .25 -1\n2 3 -1 0 0 .05 1\n3 3 0 0 0 .05 2\n"
                    "4 3 1 .0000002 0 .05 3\n5 3 1 -.0000001 0 .05 3\n")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"case.swc"
            path.write_text(contents, encoding="utf-8")
            tree = read_swc(path)
            prepared = prepare_morphology(tree, preconditioning=False, preserve_surface_roots=False).morphology
            for mode, strategy in ((0, "max-chordal-depth"), (1, "min-angle")):
                for angle in (0., 1e-5, 90.):
                    with self.subTest(strategy=strategy, angle=angle):
                        run = subprocess.run([str(SCALE_JOBS_PROBE), str(path), "0", "3", "1", str(mode),
                                              str(angle), "10"], text=True, capture_output=True, timeout=30, cwd=ROOT)
                        self.assertEqual(run.returncode, 0, run.stderr)
                        reference = [tuple(round(float(value), 10) for value in parts[1:])
                                     for line in run.stdout.splitlines() if (parts := line.split())
                                     and parts[0] == "PATH"]
                        actual = [tuple(round(value, 10) for node in item for value in prepared.nodes[node].position)
                                  for item in partition_paths(prepared, strategy, filter_angle=angle)]
                        self.assertEqual(actual, reference)

    @unittest.skipUnless(SCALE_JOBS_PROBE.is_file(), "optional C++ radius-scaling job probe is not built")
    def test_seeded_multicell_root_and_job_order_after_scaling(self):
        rng = random.Random(592717)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"case.swc"
            for case in range(3):
                rows, node = [], 1
                for cell in range(2):
                    soma = node
                    center = (cell*12., 0., 0.)
                    rows.append(f"{node} 1 {center[0]} 0 0 1 -1")
                    node += 1
                    for branch in range(3):
                        angle = branch*2*pi/3+rng.uniform(-.08, .08)
                        direction = (cos(angle), sin(angle), rng.uniform(-.05, .05))
                        parent = soma
                        lengths = (.2, .5, 1.5, 2.7, 3.4) if rng.random() < .5 else (1.5, 2.7, 3.4)
                        for length in lengths:
                            x, y, z = (center[i]+direction[i]*length for i in range(3))
                            radius = rng.uniform(.08, .16)
                            rows.append(f"{node} 3 {x:.17g} {y:.17g} {z:.17g} {radius:.17g} {parent}")
                            parent, node = node, node+1
                path.write_text("\n".join(rows)+"\n", encoding="utf-8")
                for preconditioning in (False, True):
                    for scale in (.7, 1., 1.2):
                        with self.subTest(case=case, preconditioning=preconditioning, scale=scale):
                            run = subprocess.run([str(SCALE_JOBS_PROBE), str(path),
                                                  str(int(preconditioning)), "3", str(scale)],
                                                 text=True, capture_output=True, timeout=30, cwd=ROOT)
                            self.assertEqual(run.returncode, 0, run.stderr)
                            reference_roots = [tuple(round(float(value), 6) for value in parts[1:7])
                                               for line in run.stdout.splitlines() if (parts := line.split())
                                               and parts[0] == "ROOT"]
                            reference_jobs = [tuple(round(float(value), 6) for value in parts[2:8])
                                              for line in run.stdout.splitlines() if (parts := line.split())
                                              and parts[:2] == ["JOB", "0"]]
                            tree = read_swc(path)
                            prepared = prepare_morphology(tree, preconditioning=preconditioning,
                                                          scale_radius=scale, preserve_surface_roots=False)
                            self.assertIsNotNone(prepared.source_edge_order)
                            actual_roots = [tuple(round(value, 6) for node in (soma, root)
                                                  for value in prepared.morphology.nodes[node].position)
                                            for soma in prepared.morphology.roots
                                            for root in prepared.morphology.children[soma]]
                            plan = plan_source_analysis_jobs(tree, preconditioning=preconditioning,
                                                            scale_radius=scale, preserve_surface_roots=False)
                            actual_jobs = [tuple(round(value, 6) for node in job.segments[0]
                                                 for value in prepared.morphology.nodes[node].position)
                                           for job in plan.jobs if job.kind == "REG"]
                            self.assertEqual(actual_roots, reference_roots)
                            self.assertEqual(actual_jobs, reference_jobs)

    @unittest.skipUnless(SCALE_JOBS_PROBE.is_file(), "optional C++ radius-scaling job probe is not built")
    def test_seeded_multisoma_root_collapse_and_scaling_order(self):
        rng = random.Random(83912)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"case.swc"
            for case in range(4):
                rows = []
                for node in range(1, 7):
                    xyz = tuple(rng.uniform(-.1, .1) for _ in range(3))
                    radius = rng.uniform(.85, 1.05)
                    parent = -1 if node == 1 else rng.randrange(1, node)
                    rows.append(f"{node} 1 {xyz[0]:.17g} {xyz[1]:.17g} {xyz[2]:.17g} {radius:.17g} {parent}")
                node = 7
                for branch in range(4):
                    angle = branch*pi/2+rng.uniform(-.08, .08)
                    direction = (cos(angle), sin(angle), rng.uniform(-.03, .03))
                    parent = rng.randrange(1, 7)
                    lengths = (.2, .5, 1.5, 2.7, 3.4) if rng.random() < .5 else (1.5, 2.7, 3.4)
                    for length in lengths:
                        x, y, z = (value*length for value in direction)
                        radius = rng.uniform(.08, .16)
                        rows.append(f"{node} 3 {x:.17g} {y:.17g} {z:.17g} {radius:.17g} {parent}")
                        parent, node = node, node+1
                path.write_text("\n".join(rows)+"\n", encoding="utf-8")
                for preconditioning in (False, True):
                    for scale in (.7, 1., 1.2):
                        with self.subTest(case=case, preconditioning=preconditioning, scale=scale):
                            run = subprocess.run([str(SCALE_JOBS_PROBE), str(path),
                                                  str(int(preconditioning)), "3", str(scale)],
                                                 text=True, capture_output=True, timeout=30, cwd=ROOT)
                            self.assertEqual(run.returncode, 0, run.stderr)
                            reference = [tuple(round(float(value), 6) for value in parts[2:8])
                                         for line in run.stdout.splitlines() if (parts := line.split())
                                         and parts[:2] == ["JOB", "0"]]
                            tree = read_swc(path)
                            prepared = prepare_morphology(tree, preconditioning=preconditioning,
                                                          scale_radius=scale, preserve_surface_roots=False)
                            self.assertIsNotNone(prepared.source_edge_order)
                            reference_roots = [tuple(round(float(value), 6) for value in parts[1:7])
                                               for line in run.stdout.splitlines() if (parts := line.split())
                                               and parts[0] == "ROOT"]
                            actual_roots = [tuple(round(value, 6) for node in (soma, root)
                                                  for value in prepared.morphology.nodes[node].position)
                                            for soma in prepared.morphology.roots
                                            for root in prepared.morphology.children[soma]]
                            self.assertEqual(actual_roots, reference_roots)
                            plan = plan_source_analysis_jobs(tree, preconditioning=preconditioning,
                                                            scale_radius=scale, preserve_surface_roots=False)
                            actual = [tuple(round(value, 6) for node in job.segments[0]
                                            for value in prepared.morphology.nodes[node].position)
                                      for job in plan.jobs if job.kind == "REG"]
                            self.assertEqual(actual, reference)

    @unittest.skipUnless(SCALE_JOBS_PROBE.is_file(), "optional C++ radius-scaling job probe is not built")
    def test_source_job_order_after_root_collapse_and_radius_scaling(self):
        mixed = ("1 1 0 0 0 1 -1\n2 3 .2 0 0 .1 1\n3 3 .6 0 0 .1 2\n"
                 "4 3 1.4 0 0 .1 3\n5 3 2.8 0 0 .1 4\n6 3 0 1.5 0 .1 1\n"
                 "7 3 0 3 0 .1 6\n8 3 0 4 0 .1 7\n")
        multi_cell = ("1 1 0 0 0 1 -1\n2 3 .2 0 0 .1 1\n3 3 .6 0 0 .1 2\n"
                      "4 3 1.4 0 0 .1 3\n5 3 2.8 0 0 .1 4\n6 1 10 0 0 1 -1\n"
                      "7 3 10.2 0 0 .1 6\n8 3 11.4 0 0 .1 7\n9 3 12.8 0 0 .1 8\n"
                      "10 3 10 1.5 0 .1 6\n11 3 10 2.7 0 .1 10\n12 3 10 3.9 0 .1 11\n")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"mixed.swc"
            path.write_text(mixed, encoding="utf-8")
            multi = Path(folder)/"multi.swc"
            multi.write_text(multi_cell, encoding="utf-8")
            cases = (path, multi, ROOT/"examples"/"branched.swc", ROOT/"examples"/"mixed_soma_roots.swc")
            for input_path in cases:
                for preconditioning in (False, True):
                    for scale in (.7, 1., 1.2):
                        with self.subTest(path=input_path.name, preconditioning=preconditioning, scale=scale):
                            run = subprocess.run([str(SCALE_JOBS_PROBE), str(input_path),
                                                  str(int(preconditioning)), "3", str(scale)],
                                                 text=True, capture_output=True, timeout=30, cwd=ROOT)
                            self.assertEqual(run.returncode, 0, run.stderr)
                            reference = [tuple(round(float(value), 7) for value in parts[2:8])
                                         for line in run.stdout.splitlines() if (parts := line.split())
                                         and parts[:2] == ["JOB", "0"]]
                            tree = read_swc(input_path)
                            prepared = prepare_morphology(tree, preconditioning=preconditioning,
                                                          scale_radius=scale, preserve_surface_roots=False)
                            self.assertIsNotNone(prepared.source_edge_order)
                            reference_roots = [tuple(round(float(value), 7) for value in parts[1:7])
                                               for line in run.stdout.splitlines() if (parts := line.split())
                                               and parts[0] == "ROOT"]
                            actual_roots = [tuple(round(value, 7) for node in (soma, root)
                                                  for value in prepared.morphology.nodes[node].position)
                                            for soma in prepared.morphology.roots
                                            for root in prepared.morphology.children[soma]]
                            self.assertEqual(actual_roots, reference_roots)
                            ranks = {child: index for index, (_, child) in enumerate(prepared.source_edge_order)}
                            for children in prepared.morphology.children.values():
                                self.assertEqual(list(children), sorted(children, key=ranks.__getitem__))
                            if input_path == path and not preconditioning and scale == 1.:
                                self.assertNotIn(3, prepared.morphology.nodes)
                            if input_path == multi and not preconditioning and scale in (.7, 1.):
                                roots = prepared.morphology.children[6]
                                self.assertEqual(len(roots), 2)
                                first = prepared.morphology.nodes[roots[0]].position
                                second = prepared.morphology.nodes[roots[1]].position
                                self.assertLess(dist(first, (10., scale, 0.)), 1e-12)
                                self.assertLess(dist(second, (10.+scale, 0., 0.)), 1e-12)
                            plan = plan_source_analysis_jobs(tree, preconditioning=preconditioning,
                                                            scale_radius=scale, preserve_surface_roots=False)
                            actual = [tuple(round(value, 7) for node in job.segments[0]
                                            for value in prepared.morphology.nodes[node].position)
                                      for job in plan.jobs if job.kind == "REG"]
                            self.assertEqual(actual, reference)
                            limited = plan_source_analysis_jobs(tree, preconditioning=preconditioning,
                                                               scale_radius=scale, preserve_surface_roots=False,
                                                               max_jobs=1)
                            self.assertTrue(limited.truncated)
                            self.assertEqual(limited.jobs, plan.jobs[:1])

    def test_source_job_order_after_seeded_preconditioning(self):
        rng = random.Random(416923)
        seen_splits = seen_collapses = 0
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"case.swc"
            for case in range(12):
                rows = ["1 1 0 0 0 1 -1", "2 3 1.25 0 0 .18 1"]
                coords = {2: (1.25, 0., 0.)}
                for node in range(3, 12):
                    parent = rng.randrange(2, node) if rng.random() < .4 else node-1
                    old = coords[parent]
                    point = (old[0]+rng.uniform(.25, 2.2), old[1]+rng.uniform(-.7, .7),
                             old[2]+rng.uniform(-.3, .3))
                    coords[node] = point
                    radius = rng.uniform(.1, .3)
                    rows.append(f"{node} 3 {point[0]:.17g} {point[1]:.17g} {point[2]:.17g} {radius:.17g} {parent}")
                path.write_text("\n".join(rows)+"\n", encoding="utf-8")
                with self.subTest(case=case):
                    run = subprocess.run([str(JOBS_PROBE), str(path), "1"], text=True,
                                         capture_output=True, timeout=30, cwd=ROOT)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    reference = [tuple(round(float(value), 7) for value in parts[2:8])
                                 for line in run.stdout.splitlines() if (parts := line.split())
                                 and parts[:2] == ["JOB", "0"]]
                    tree = read_swc(path)
                    prepared = prepare_morphology(tree, preserve_surface_roots=False)
                    seen_splits += bool(prepared.split_segments)
                    seen_collapses += bool(prepared.collapsed_segments)
                    self.assertIsNotNone(prepared.source_edge_order)
                    plan = plan_source_analysis_jobs(tree, preserve_surface_roots=False)
                    actual = [tuple(round(value, 7) for node in job.segments[0]
                                    for value in prepared.morphology.nodes[node].position)
                              for job in plan.jobs if job.kind == "REG"]
                    self.assertEqual(actual, reference)
        self.assertGreater(seen_splits, 0)
        self.assertGreater(seen_collapses, 0)

    def test_source_job_order_after_soma_root_insertion(self):
        rng = random.Random(640981)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"case.swc"
            for case in range(8):
                rows = []
                for i in range(1, 9):
                    xyz = tuple(rng.uniform(-.35, .35) for _ in range(3))
                    radius = rng.uniform(.8, 1.1)
                    parent = -1 if i == 1 else rng.randrange(1, i)
                    rows.append(f"{i} 1 {xyz[0]:.17g} {xyz[1]:.17g} {xyz[2]:.17g} {radius:.17g} {parent}")
                node_id = 9
                for branch in range(4):
                    angle = branch*pi/2+rng.uniform(-.15, .15)
                    direction = (cos(angle), sin(angle), rng.uniform(-.1, .1))
                    parent = rng.randrange(1, 9)
                    for j in range(3):
                        x, y, z = (direction[k]*(2+j*.8) for k in range(3))
                        rows.append(f"{node_id} 3 {x:.17g} {y:.17g} {z:.17g} .1 {parent}")
                        parent, node_id = node_id, node_id+1
                path.write_text("\n".join(rows)+"\n", encoding="utf-8")
                with self.subTest(case=case):
                    run = subprocess.run([str(JOBS_PROBE), str(path), "0"], text=True,
                                         capture_output=True, timeout=30, cwd=ROOT)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    reference = [tuple(round(float(value), 7) for value in parts[2:8])
                                 for line in run.stdout.splitlines() if (parts := line.split())
                                 and parts[:2] == ["JOB", "0"]]
                    tree = read_swc(path)
                    prepared = prepare_morphology(tree, preconditioning=False,
                                                  preserve_surface_roots=False).morphology
                    plan = plan_source_analysis_jobs(tree, preconditioning=False, preserve_surface_roots=False)
                    actual = [tuple(round(value, 7) for node in job.segments[0]
                                    for value in prepared.nodes[node].position)
                              for job in plan.jobs if job.kind == "REG"]
                    self.assertEqual(actual, reference)
                    limited = plan_source_analysis_jobs(tree, preconditioning=False,
                                                        preserve_surface_roots=False, max_jobs=7)
                    self.assertTrue(limited.truncated)
                    self.assertEqual(limited.jobs, plan.jobs[:7])

    def test_cpp_preconditioning_norm_thresholds(self):
        cases = ((-.2582743098573599, 1.6248239331868253, 1.7301055515322439,
                  .18266480860323492, 2, 3),
                 (1.14851343105324, .8569438121008419, 2.4911683195792946,
                  .1259977526346212, 3, 2))
        with tempfile.TemporaryDirectory() as folder:
            for x, y, z, radius, expected, robust_expected in cases:
                with self.subTest(expected=expected):
                    path = Path(folder)/"norm.swc"
                    path.write_text(f"1 1 1 2 2 .1 -1\n2 3 1 2 3 {radius!r} 1\n"
                                    f"3 3 {x!r} {y!r} {z!r} {radius!r} 2\n", encoding="utf-8")
                    run = subprocess.run([str(JOBS_PROBE), str(path), "1", "2"], text=True,
                                         capture_output=True, timeout=30, cwd=ROOT)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    reference = Counter(tuple(round(float(value), 7) for value in line.split()[1:7])
                                        for line in run.stdout.splitlines() if line.startswith("MDV "))
                    self.assertEqual(sum(reference.values()), expected)
                    original = read_swc(path)
                    robust = prepare_morphology(original, preserve_surface_roots=False,
                                                source_soma_centroid=True).morphology
                    self.assertEqual(len(list(robust.neurite_segments())), robust_expected)
                    for mode in ("source", "source-x87"):
                        tree = prepare_morphology(original, preserve_surface_roots=False,
                                                  source_soma_centroid=True,
                                                  precondition_norm_mode=mode).morphology
                        actual = Counter(tuple(round(value, 7) for node in (a, b) for value in node.position)
                                         for a, b in tree.neurite_segments())
                        self.assertEqual(actual, reference)
                    for command in ("plan-source-jobs", "trace-source-jobs", "analyze-analytic"):
                        args = [sys.executable, "-m", "anamorph", command, str(path),
                                "--source-soma-centroid", "--precondition-norm-mode", "source-x87"]
                        if command == "analyze-analytic":
                            args.append("--source-surface-roots")
                        cli = subprocess.run(args, text=True, capture_output=True, timeout=30, cwd=ROOT)
                        self.assertEqual(cli.returncode, 0, cli.stderr)
                        self.assertEqual(json.loads(cli.stdout)["segment_count"], expected, command)

    def test_cpp_preconditioning_subdivision_penalty_tie(self):
        import math
        tie = 10*math.sqrt(6)
        with tempfile.TemporaryDirectory() as folder:
            for length in (math.nextafter(tie, -math.inf), tie, math.nextafter(tie, math.inf)):
                with self.subTest(length=length):
                    path = Path(folder)/"penalty.swc"
                    path.write_text(f"1 1 -1 0 0 .1 -1\n2 3 0 0 0 1 1\n"
                                    f"3 3 {length!r} 0 0 1 2\n4 3 {length+5!r} 1 0 1 3\n",
                                    encoding="utf-8")
                    run = subprocess.run([str(JOBS_PROBE), str(path), "1", "2"], text=True,
                                         capture_output=True, timeout=30, cwd=ROOT)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    reference = Counter(tuple(round(float(value), 7) for value in line.split()[1:7])
                                        for line in run.stdout.splitlines() if line.startswith("MDV "))
                    tree = prepare_morphology(read_swc(path), preserve_surface_roots=False).morphology
                    actual = Counter(tuple(round(value, 7) for node in (a, b) for value in node.position)
                                     for a, b in tree.neurite_segments())
                    self.assertEqual(actual, reference)

    def test_preconditioning_matches_cpp_on_equal_length_branches(self):
        path = ROOT/"tests"/"fixtures"/"precondition_tie.swc"
        run = subprocess.run([str(JOBS_PROBE), str(path), "1", "2"], text=True,
                             capture_output=True, timeout=30, cwd=ROOT)
        self.assertEqual(run.returncode, 0, run.stderr)
        reference = Counter(tuple(round(float(value), 7) for value in line.split()[1:7])
                            for line in run.stdout.splitlines() if line.startswith("MDV "))
        tree = prepare_morphology(read_swc(path), preserve_surface_roots=False).morphology
        actual = Counter(tuple(round(value, 7) for node in (a, b) for value in node.position)
                         for a, b in tree.neurite_segments())
        self.assertEqual(len(reference), 8)
        self.assertEqual(actual, reference)

    def test_cpp_distance_violation_decisions_match_prepared_segments(self):
        cases = (
            "1 1 -3 0 0 .2 -1\n2 3 0 0 0 .2 1\n3 3 .15 0 0 .2 2\n"
            "4 3 .6 0 0 .1 3\n5 3 .7 .15 0 .3 3\n6 3 1.2 .15 0 .1 5\n",
            "1 1 -3 0 0 .2 -1\n2 3 0 0 0 1 1\n3 3 1.5 0 0 .1 2\n"
            "4 3 2 0 0 .1 3\n5 3 3 0 0 .1 4\n",
            # Three soma sections: one neurite begins outside the consolidated
            # sphere and the other case begins inside it.
            "1 1 0 0 0 1 -1\n2 1 2 0 0 2 1\n3 1 1 3 0 1 2\n"
            "4 3 5 0 0 .2 2\n5 3 6 0 0 .2 4\n6 3 7 0 0 .2 5\n",
            "1 1 0 0 0 1 -1\n2 1 2 0 0 2 1\n3 1 1 3 0 1 2\n"
            "4 3 1.3 1 0 .2 2\n5 3 4 1 0 .3 4\n6 3 5 1 0 .3 5\n",
        )
        seen = set()
        with tempfile.TemporaryDirectory() as folder:
            for case_number, contents in enumerate(cases):
                path = Path(folder)/(str(case_number)+".swc")
                path.write_text(contents, encoding="utf-8")
                for preconditioning in (False, True):
                    with self.subTest(case=case_number, preconditioning=preconditioning):
                        run = subprocess.run([str(JOBS_PROBE), str(path), str(int(preconditioning)), "2"],
                                             text=True, capture_output=True, timeout=30, cwd=ROOT)
                        self.assertEqual(run.returncode, 0, run.stderr)
                        reference = Counter()
                        for line in run.stdout.splitlines():
                            parts = line.split()
                            if parts and parts[0] == "MDV":
                                edge = tuple(round(float(value), 7) for value in parts[1:7])
                                reference[edge, bool(int(parts[7]))] += 1
                        tree = prepare_morphology(read_swc(path), preconditioning=preconditioning,
                                                  preserve_surface_roots=False).morphology
                        violations = {v.edge: v for v in tree.minimum_distance_violations()}
                        actual = Counter()
                        for parent, child in tree.neurite_segments():
                            edge = tuple(round(value, 7) for node in (parent, child) for value in node.position)
                            violation = violations.get((parent.id, child.id))
                            actual[edge, violation is not None] += 1
                            if violation:
                                seen.add((violation.primary, violation.severe_primary, violation.secondary))
                        self.assertEqual(actual, reference)
        # The source adapter returns only its aggregate decision; these cases
        # independently exercise primary-only and secondary-only decisions.
        self.assertIn((True, False, False), seen)
        self.assertIn((False, False, True), seen)
        self.assertTrue(any(severe for _, severe, _ in seen))

    def test_cpp_pair_findings_match_edges_and_source_classifications(self):
        # Crossings exercise all four source NSNS classifications. The source
        # probe executes its solver and generateNSNSIsecInfo on each positive job.
        cases = {
            "intercell": ("1 1 -4 0 0 .25 -1\n2 3 -3 0 0 .1 1\n3 3 -1 0 0 .1 2\n"
                          "4 3 1 0 0 .1 3\n5 3 3 0 0 .1 4\n6 1 0 -4 0 .25 -1\n"
                          "7 3 0 -3 0 .1 6\n8 3 0 -1 0 .1 7\n9 3 0 1 0 .1 8\n10 3 0 3 0 .1 9\n"),
            "interneurite": ("1 1 -4 -4 0 .25 -1\n2 3 -3 0 0 .1 1\n3 3 -1 0 0 .1 2\n"
                             "4 3 1 0 0 .1 3\n5 3 3 0 0 .1 4\n6 3 0 -3 0 .1 1\n"
                             "7 3 0 -1 0 .1 6\n8 3 0 1 0 .1 7\n9 3 0 3 0 .1 8\n"),
            "adjacent": ("1 1 -2 0 0 .3 -1\n2 3 0 0 0 .1 1\n3 3 2 0 0 .1 2\n"
                         "4 3 2 .05 0 .1 2\n"),
        }
        classifications = {"inter_cell": 6, "intra_cell_inter_neurite": 7,
                           "intra_cell_intra_neurite": 8, "intra_cell_intra_neurite_adjacent": 9}
        observed = set()
        with tempfile.TemporaryDirectory() as folder:
            for name, contents in cases.items():
                with self.subTest(case=name):
                    path = Path(folder)/(name+".swc")
                    path.write_text(contents, encoding="utf-8")
                    run = subprocess.run([str(JOBS_PROBE), str(path), "1", "1"], text=True,
                                         capture_output=True, timeout=30, cwd=ROOT)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    reference = Counter()
                    for line in run.stdout.splitlines():
                        parts = line.split()
                        if len(parts) < 4 or parts[0] != "JOB" or int(parts[1]) not in (4, 5) or not int(parts[-1]):
                            continue
                        kind, classification = int(parts[1]), int(parts[-2])
                        values = tuple(round(float(value), 7) for value in parts[2:14])
                        edges = tuple(sorted((values[:6], values[6:])))
                        reference[kind, edges, classification] += 1
                        observed.add(classification)
                    tree = read_swc(path)
                    prepared = prepare_morphology(tree, preserve_surface_roots=False).morphology
                    report = analyze_analytic_partial(tree, prepare=True, job_mode="source",
                                                      preserve_surface_roots=False)
                    self.assertFalse(report.unresolved)
                    actual = Counter()
                    for finding in report.findings:
                        if finding.kind not in ("adjacent_neurite_candidate", "neurite_neurite_candidate"):
                            continue
                        def edge_coordinates(edge):
                            return tuple(round(value, 7) for node in edge for value in prepared.nodes[node].position)
                        edges = tuple(sorted(edge_coordinates(edge) for edge in finding.segments))
                        kind = 4 if finding.kind == "adjacent_neurite_candidate" else 5
                        actual[kind, edges, classifications[finding.classification]] += 1
                        flag = {6: "rc_nsns", 7: "icrn_nsns", 8: "icin_nsns", 9: "icin_nsns"}[
                            classifications[finding.classification]]
                        statuses = {entry.edge: entry for entry in report.segment_flags}
                        self.assertTrue(all(getattr(statuses[edge], flag) for edge in finding.segments))
                    self.assertEqual(actual, reference)
        self.assertEqual(observed, {6, 7, 8, 9})

    def test_cpp_execution_matches_selected_clean_and_soma_collision_cases(self):
        # Both implementations prepare the geometry before executing checks.
        # The C++ adapter executes its actual jobs; this is not just a plan check.
        with tempfile.TemporaryDirectory() as folder:
            collision = Path(folder)/"collision.swc"
            collision.write_text("1 1 -3 0 0 .3 -1\n2 3 -2 0 0 .1 1\n"
                                 "3 3 0 0 0 .1 2\n4 3 2 0 0 .1 3\n"
                                 "5 1 -1 .2 0 .3 -1\n", encoding="utf-8")
            curvature = ROOT/"tests"/"fixtures"/"branch_curvature_collision.swc"
            for path in (ROOT/"examples"/"branched.swc", ROOT/"examples"/"mixed_soma_roots.swc",
                         collision, curvature):
                with self.subTest(path=path.name):
                    run = subprocess.run([str(JOBS_PROBE), str(path), "1", "1"], text=True,
                                         capture_output=True, timeout=30, cwd=ROOT)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    reference = Counter()
                    for line in run.stdout.splitlines():
                        if line.startswith("JOB "):
                            parts = line.split()
                            if int(parts[-1]):
                                reference[int(parts[1])] += 1
                    report = analyze_analytic_partial(read_swc(path), prepare=True, job_mode="source",
                                                      preserve_surface_roots=False)
                    kinds = {"regularity": 0, "local_self_intersection": 1, "global_self_intersection": 2,
                             "soma_neurite_candidate": 3, "adjacent_neurite_candidate": 4,
                             "neurite_neurite_candidate": 5}
                    actual = Counter(kinds[f.kind] for f in report.findings if f.kind in kinds)
                    self.assertFalse(report.unresolved)
                    self.assertEqual(actual, reference)
                    if path == curvature:
                        self.assertEqual(reference[1], 2)
                        prepared = prepare_morphology(read_swc(path), preserve_surface_roots=False).morphology
                        expected_edges = Counter()
                        for line in run.stdout.splitlines():
                            parts = line.split()
                            if len(parts) >= 9 and parts[0] == "JOB" and parts[1] == "1" and int(parts[-1]):
                                expected_edges[tuple(round(float(v), 7) for v in parts[2:8])] += 1
                        actual_edges = Counter(tuple(round(v, 7) for node in finding.segments[0]
                                                     for v in prepared.nodes[node].position)
                                               for finding in report.findings
                                               if finding.kind == "local_self_intersection")
                        self.assertEqual(actual_edges, expected_edges)
                    if path == collision:
                        self.assertGreater(reference[3], 0)
                        prepared = prepare_morphology(read_swc(path), preserve_surface_roots=False).morphology
                        expected_somas = Counter()
                        for line in run.stdout.splitlines():
                            parts = line.split()
                            if len(parts) >= 13 and parts[0] == "JOB" and parts[1] == "3" and int(parts[-1]):
                                values = tuple(round(float(value), 7) for value in parts[2:11])
                                expected_somas[values] += 1
                        actual_somas = Counter()
                        statuses = {status.edge: status for status in report.segment_flags}
                        for finding in report.findings:
                            if finding.kind == "soma_neurite_candidate":
                                edge, = finding.segments
                                soma, = finding.somas
                                coords = tuple(round(value, 7) for node in (*edge, soma)
                                               for value in prepared.nodes[node].position)
                                actual_somas[coords] += 1
                                self.assertEqual(finding.classification, "inter_cell")
                                self.assertTrue(statuses[edge].ic_sons)
                        self.assertEqual(actual_somas, expected_somas)

    def test_seeded_intercell_crossings_match_cpp_pairs(self):
        rng = random.Random(74193)
        rows = [row.split() for row in (
            "1 1 -2 0 0 .4 -1", "2 3 -1.4 0 0 .12 1", "3 3 -.2 0 0 .12 2",
            "4 3 1 0 0 .1 3", "5 3 2.4 0 0 .08 4", "6 1 2 .5 0 .4 -1",
            "7 3 1.4 .5 0 .12 6", "8 3 .2 .2 0 .12 7", "9 3 -1 -.4 0 .1 8",
            "10 3 -2.4 -.6 0 .08 9")]
        classifications = {"inter_cell": 6, "intra_cell_inter_neurite": 7,
                           "intra_cell_intra_neurite": 8, "intra_cell_intra_neurite_adjacent": 9}
        positive_intercell = 0
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"crossing.swc"
            for case in range(20):
                varied = []
                for row in rows:
                    fields = row.copy()
                    if fields[1] != "1":
                        for axis in range(2, 5):
                            fields[axis] = repr(float(fields[axis])+rng.uniform(-.08, .08))
                        fields[5] = repr(float(fields[5])*rng.uniform(.8, 1.2))
                    varied.append(" ".join(fields))
                path.write_text("\n".join(varied)+"\n", encoding="utf-8")
                with self.subTest(case=case):
                    run = subprocess.run([str(JOBS_PROBE), str(path), "1", "1"], text=True,
                                         capture_output=True, timeout=30, cwd=ROOT)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    reference = Counter()
                    for line in run.stdout.splitlines():
                        parts = line.split()
                        if not parts or parts[0] != "JOB" or int(parts[1]) not in (4, 5) or not int(parts[-1]):
                            continue
                        values = tuple(round(float(value), 7) for value in parts[2:14])
                        classification = int(parts[-2])
                        edges = tuple(sorted((values[:6], values[6:])))
                        reference[int(parts[1]), edges, classification] += 1
                        positive_intercell += classification == 6
                    tree = read_swc(path)
                    prepared = prepare_morphology(tree, preserve_surface_roots=False).morphology
                    report = analyze_analytic_partial(tree, prepare=True, job_mode="source",
                                                      preserve_surface_roots=False)
                    self.assertFalse(report.unresolved)
                    actual = Counter()
                    statuses = {entry.edge: entry for entry in report.segment_flags}
                    for finding in report.findings:
                        if finding.kind not in ("adjacent_neurite_candidate", "neurite_neurite_candidate"):
                            continue
                        edges = tuple(sorted(tuple(round(value, 7) for node in edge
                                                   for value in prepared.nodes[node].position)
                                             for edge in finding.segments))
                        classification = classifications[finding.classification]
                        kind = 4 if finding.kind == "adjacent_neurite_candidate" else 5
                        actual[kind, edges, classification] += 1
                        if classification == 6:
                            self.assertTrue(all(statuses[edge].rc_nsns for edge in finding.segments))
                    self.assertEqual(actual, reference)
        self.assertGreater(positive_intercell, 0)

    def test_curved_tree_source_trace_results_match_cpp(self):
        rng = random.Random(8123019)
        kinds = {"REG": 0, "LSI": 1, "GSI": 2, "SONS": 3, "NSNS_ADJ": 4, "NSNS_NONADJ": 5}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"case.swc"
            for case in range(8):
                rows = ["1 1 0 0 0 .5 -1", "2 3 .7 0 0 .12 1"]
                coords = {2: (.7, 0., 0.)}
                for node in range(3, rng.randrange(6, 10)):
                    parent = rng.randrange(2, node) if rng.random() < .3 else node-1
                    point = tuple(coords[parent][i]+rng.uniform(-1.3, 1.3) for i in range(3))
                    coords[node] = point
                    rows.append(f"{node} 3 {point[0]:.17g} {point[1]:.17g} {point[2]:.17g} "
                                f"{rng.uniform(.07, .4):.17g} {parent}")
                path.write_text("\n".join(rows)+"\n", encoding="utf-8")
                with self.subTest(case=case):
                    run = subprocess.run([str(JOBS_PROBE), str(path), "0", "1"], cwd=directory,
                                         capture_output=True, text=True, check=True, timeout=30)
                    expected = [(int(parts[1]), bool(int(parts[-1]))) for line in run.stdout.splitlines()
                                if (parts := line.split()) and parts[0] == "JOB"]
                    trace = trace_source_analysis(read_swc(path), preconditioning=False)
                    self.assertEqual([(kinds[item.job.kind], item.result) for item in trace.jobs], expected)

    def test_seeded_branch_analysis_matches_cpp(self):
        rng = random.Random(92301)
        rows = [line.split() for line in (ROOT/"examples"/"branched.swc").read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
        kinds = {"regularity": 0, "local_self_intersection": 1, "global_self_intersection": 2,
                 "soma_neurite_candidate": 3, "adjacent_neurite_candidate": 4,
                 "neurite_neurite_candidate": 5}
        positive = 0
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"case.swc"
            for case in range(30):
                varied = []
                for row in rows:
                    fields = row.copy()
                    if fields[1] != "1":
                        for axis in range(2, 5):
                            fields[axis] = repr(float(fields[axis])+rng.uniform(-.15, .15))
                        fields[5] = repr(float(fields[5])*rng.uniform(.6, 1.4))
                    varied.append(" ".join(fields))
                path.write_text("\n".join(varied)+"\n", encoding="utf-8")
                with self.subTest(case=case):
                    run = subprocess.run([str(JOBS_PROBE), str(path), "1", "1"], text=True,
                                         capture_output=True, timeout=30, cwd=ROOT)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    jobs = [line.split() for line in run.stdout.splitlines() if line.startswith("JOB ")]
                    reference = Counter(int(job[1]) for job in jobs if int(job[-1]))
                    tree = read_swc(path)
                    report = analyze_analytic_partial(tree, prepare=True, job_mode="source",
                                                      preserve_surface_roots=False)
                    self.assertFalse(report.unresolved)
                    self.assertEqual(Counter(kinds[f.kind] for f in report.findings if f.kind in kinds),
                                     reference)
                    prepared = prepare_morphology(tree, preserve_surface_roots=False).morphology
                    reference_edges = Counter(tuple(round(float(v), 7) for v in job[2:8])
                                              for job in jobs if job[1] == "1" and int(job[-1]))
                    actual_edges = Counter(tuple(round(v, 7) for node in finding.segments[0]
                                                 for v in prepared.nodes[node].position)
                                           for finding in report.findings
                                           if finding.kind == "local_self_intersection")
                    self.assertEqual(actual_edges, reference_edges)
                    positive += reference[1]
        self.assertGreater(positive, 0)

    def test_subdivided_canal_bounds_match_cpp(self):
        rng = random.Random(42092)
        for index in range(20):
            power = tuple(tuple(rng.uniform(-10., 10.) for _ in range(4)) for _ in range(3))
            r0, r1 = rng.uniform(.01, 2.), rng.uniform(.01, 2.)
            curve = CanalSegment(power, r0, r1)
            data = " ".join(map(repr, [value for row in power for value in row]+[r0, r1]))
            run = subprocess.run([str(SOLVER_PROBE), "bounds"], input=data, text=True,
                                 capture_output=True, timeout=10, cwd=ROOT)
            self.assertEqual(run.returncode, 0, run.stderr)
            line = next(line for line in run.stdout.splitlines() if line.startswith("BOUNDS "))
            values = tuple(map(float, line.split()[1:]))
            actual = _source_segment_box(curve)
            with self.subTest(index=index):
                for axis in range(3):
                    for side in range(2):
                        self.assertAlmostEqual(actual[axis][side], values[2*axis+side], places=10)

    def test_source_job_multisets_match_cpp(self):
        names = {0: "REG", 1: "LSI", 2: "GSI", 3: "SONS", 4: "NSNS_ADJ", 5: "NSNS_NONADJ"}
        for filename in ("examples/branched.swc", "examples/mixed_soma_roots.swc",
                         "tests/fixtures/branch_tie.swc", "examples/many_branches.swc"):
            for preconditioning in (False, True):
                with self.subTest(filename=filename, preconditioning=preconditioning):
                    run = subprocess.run([str(JOBS_PROBE), str(ROOT/filename), str(int(preconditioning))],
                                         text=True, capture_output=True, timeout=30, cwd=ROOT)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    tree = read_swc(ROOT/filename)
                    prepared = prepare_morphology(tree, preconditioning=preconditioning,
                                                  preserve_surface_roots=False).morphology
                    plan = plan_source_analysis_jobs(tree, preconditioning=preconditioning)
                    self.assertFalse(plan.truncated)
                    def coordinate(node):
                        return tuple(round(value, 7) for value in node.position)
                    def edge(ids):
                        return coordinate(prepared.nodes[ids[0]])+coordinate(prepared.nodes[ids[1]])
                    def key(kind, first, second=(), soma=(), flag=None):
                        if second and (kind == "NSNS_NONADJ" or (kind == "NSNS_ADJ" and flag is False)):
                            first, second = sorted((first, second))
                        return kind, first, second, soma, flag
                    reference = Counter()
                    for line in run.stdout.splitlines():
                        if not line.startswith("JOB "):
                            continue
                        parts = line.split()
                        kind, values = names[int(parts[1])], list(map(float, parts[2:]))
                        first = tuple(round(value, 7) for value in values[:6])
                        second = tuple(round(value, 7) for value in values[6:12]) if kind.startswith("NSNS") else ()
                        soma = tuple(round(value, 7) for value in values[6:9]) if kind == "SONS" else ()
                        flag = bool(values[9]) if kind == "SONS" else bool(values[12]) if kind == "NSNS_ADJ" else None
                        reference[key(kind, first, second, soma, flag)] += 1
                    actual = Counter()
                    for job in plan.jobs:
                        first = edge(job.segments[0])
                        second = edge(job.segments[1]) if len(job.segments) > 1 else ()
                        soma = coordinate(prepared.nodes[job.soma]) if job.soma is not None else ()
                        flag = job.root_segment if job.kind == "SONS" else job.first_end_second_start
                        actual[key(job.kind, first, second, soma, flag)] += 1
                    self.assertEqual(actual, reference)


if __name__ == "__main__":
    unittest.main()
