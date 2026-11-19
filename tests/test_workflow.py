import hashlib
import json
from dataclasses import replace
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from anamorph.mesh import read_obj
from anamorph.mesh_intersections import MeshIntersection
from anamorph.canal_mesh import ico_sphere
from anamorph.analytic_network import AnalyticFinding, analyze_analytic_partial
from anamorph.swc import read_swc
from anamorph.workflow import _mesh_report, run_workflow
from anamorph.collapse import greedy_edge_collapse
from anamorph.smoothing import hc_smooth
from anamorph.smoothing import taubin_smooth
from anamorph.remesh import isotropic_remesh
from anamorph.junction import fill_and_smooth_soma_junctions


ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT/"examples"/"branched.swc"


class WorkflowTests(unittest.TestCase):
    def test_isotropic_taubin_then_mortar_order(self):
        calls = []
        def record(name, operation):
            def run(*args, **kwargs):
                calls.append((name, kwargs.copy()))
                return operation(*args, **kwargs)
            return run
        with tempfile.TemporaryDirectory() as folder, \
                patch("anamorph.workflow.isotropic_remesh", side_effect=record("isotropic", isotropic_remesh)), \
                patch("anamorph.workflow.taubin_smooth", side_effect=record("taubin", taubin_smooth)), \
                patch("anamorph.workflow.fill_and_smooth_soma_junctions", side_effect=record("mortar", fill_and_smooth_soma_junctions)):
            result = run_workflow(EXAMPLE, Path(folder)/"results", target_edge_length=.14,
                                  angular_segments=8, soma_refinements=1, check_intersections=False)
            self.assertEqual([name for name, _ in calls], ["isotropic", "taubin", "mortar"])
            self.assertEqual(calls[0][1]["iterations"], 5)
            self.assertEqual(calls[1][1]["iterations"], 10)
            self.assertEqual(calls[2][1]["width_factor"], 3.)
            report = json.loads(Path(result["detailed_report"]).read_text(encoding="utf-8"))
            self.assertEqual(report["settings"]["soma_junction_blend"]["method"], "mortar")
            self.assertIn("after smoothing", Path(result["report"]).read_text(encoding="utf-8"))

    def test_command_saves_meshes_and_reproducible_reports(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/"results"
            run = subprocess.run([sys.executable, "-m", "anamorph", "workflow", str(EXAMPLE),
                                  "--output-dir", str(target), "--target-edge-length", "0.14",
                                  "--angular-segments", "8"], cwd=ROOT,
                                 text=True, capture_output=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stderr)
            summary = json.loads(run.stdout)
            self.assertEqual(summary["method"], "analytic")
            self.assertEqual(set(path.name for path in target.iterdir()),
                             {"input.swc", "raw.obj", "remeshed.obj", "report.md", "report.json"})
            self.assertEqual(Path(summary["report"]).resolve(), (target/"report.md").resolve())
            report = json.loads((target/"report.json").read_text(encoding="utf-8"))
            self.assertFalse(report["analysis"]["performed"])
            self.assertFalse(report["analysis"]["complete"])
            self.assertEqual(report["settings"]["meshing_method"], "analytic")
            self.assertFalse(report["settings"]["source_comparison_preset"])
            self.assertEqual(report["settings"]["meshing"]["random_generator"], "python")
            self.assertEqual(report["settings"]["remeshing"]["target"], .14)
            self.assertEqual(report["settings"]["remeshing"]["target_selection"], "user supplied")
            self.assertEqual(report["settings"]["remeshing"]["taubin_smoothing"]["iterations"], 10)
            self.assertEqual(report["settings"]["soma_junction_blend"]["method"], "mortar")
            for label, filename in (("raw", "raw.obj"), ("remeshed", "remeshed.obj")):
                path = target/filename
                mesh = read_obj(path)
                self.assertEqual(len(mesh.faces), report[label]["statistics"]["faces"])
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), report["outputs"][label]["sha256"])
                self.assertTrue(report[label]["topology"]["valid"])
                self.assertEqual(report[label]["self_intersections"]["status"], "none_found")
                self.assertFalse(report[label]["self_intersections"]["certifies_nonintersection"])
            self.assertEqual(hashlib.sha256(EXAMPLE.read_bytes()).hexdigest(), report["input"]["sha256"])
            text = (target/"report.md").read_text(encoding="utf-8")
            self.assertIn("Meshing method: analytic", text)
            self.assertIn("Area change:", text)
            self.assertIn("Pre-meshing geometric analysis was not requested", text)
            self.assertIn("does not guarantee quasi-uniformity", text)
            self.assertNotEqual(report["change_percent"]["volume"], 0.)

    def test_automatic_target_and_skipped_checks_are_recorded(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch("anamorph.workflow.analyze_analytic_partial", side_effect=AssertionError("unexpected analysis")):
                result = run_workflow(EXAMPLE, Path(folder)/"results",
                                      check_intersections=False, soma_refinements=1, postprocess_method="isotropic")
            report = json.loads(Path(result["detailed_report"]).read_text(encoding="utf-8"))
            tree = read_swc(EXAMPLE)
            expected = statistics.median(node.radius for node in tree.nodes.values() if node.kind in (2, 3, 4))
            self.assertEqual(result["target_edge_length"], expected)
            self.assertIn("heuristic", report["settings"]["remeshing"]["target_selection"])
            self.assertFalse(report["analysis"]["performed"])
            self.assertEqual(report["raw"]["self_intersections"]["status"], "not_checked")
            self.assertEqual(report["remeshed"]["self_intersections"]["status"], "not_checked")
            self.assertIsNone(report["raw"]["self_intersections"]["includes_adjacent"])

    def test_optional_python_analysis_is_recorded_and_nonblocking(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/"results"
            run = subprocess.run([sys.executable, "-m", "anamorph", "workflow", str(EXAMPLE),
                                  "--output-dir", str(target), "--isotropic", "--target-edge-length", "0.14",
                                  "--analyze", "--skip-intersection-check"], cwd=ROOT,
                                 text=True, capture_output=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stderr)
            report = json.loads((target/"report.json").read_text(encoding="utf-8"))
            self.assertTrue(report["analysis"]["performed"])
            self.assertEqual(report["analysis"]["report"]["job_mode"], "extended")
            self.assertGreater(report["analysis"]["report"]["segment_count"], 0)
            self.assertFalse(report["settings"]["analysis"]["blocks_meshing"])
            self.assertIn("does not block meshing", (target/"report.md").read_text(encoding="utf-8"))

    def test_optional_analysis_findings_do_not_prevent_mesh_output(self):
        base = analyze_analytic_partial(read_swc(EXAMPLE), prepare=True)
        finding = AnalyticFinding("primary_distance_violation", ((2, 3),), "example finding")
        with tempfile.TemporaryDirectory() as folder, \
                patch("anamorph.workflow.analyze_analytic_partial", return_value=replace(base, findings=(finding,))):
            result = run_workflow(EXAMPLE, Path(folder)/"results", target_edge_length=.14,
                                  analyze=True, check_intersections=False, postprocess_method="isotropic")
            report = json.loads(Path(result["detailed_report"]).read_text(encoding="utf-8"))
            self.assertEqual(report["analysis"]["report"]["findings"][0]["kind"], "primary_distance_violation")
            self.assertTrue(Path(result["raw_mesh"]).is_file())
            self.assertTrue(Path(result["remeshed_mesh"]).is_file())

    def test_streamed_analytic_command_needs_no_analysis_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/"cell.obj"
            run = subprocess.run([sys.executable, "-m", "anamorph", "generate-redblue", str(EXAMPLE), str(output),
                                  "--angular-segments", "8", "--soma-refinements", "1", "--flush-face-limit", "100"],
                                 cwd=ROOT, text=True, capture_output=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stderr)
            report = json.loads(run.stdout)
            self.assertFalse(report["analysis_performed"])
            self.assertGreater(len(read_obj(output).faces), 0)

    def test_existing_results_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/"results"
            target.mkdir()
            existing = target/"raw.obj"
            existing.write_text("keep me", encoding="utf-8")
            run = subprocess.run([sys.executable, "-m", "anamorph", "workflow", str(EXAMPLE),
                                  "--output-dir", str(target)], cwd=ROOT, text=True, capture_output=True, timeout=30)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("result already exists", run.stderr)
            self.assertEqual(existing.read_text(encoding="utf-8"), "keep me")
            self.assertEqual([path.name for path in target.iterdir()], ["raw.obj"])

    def test_failure_does_not_publish_partial_results(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/"results"
            target.mkdir()
            sentinel = target/"unrelated.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with patch("anamorph.workflow.isotropic_remesh", side_effect=ValueError("synthetic failure")):
                with self.assertRaisesRegex(ValueError, "synthetic failure"):
                    run_workflow(EXAMPLE, target, target_edge_length=.14, postprocess_method="isotropic")
            self.assertEqual([path.name for path in target.iterdir()], ["unrelated.txt"])
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_publication_failure_removes_only_new_results(self):
        with tempfile.TemporaryDirectory() as folder:
            target = (Path(folder)/"results").resolve()
            target.mkdir()
            sentinel = target/"unrelated.txt"
            sentinel.write_text("keep", encoding="utf-8")
            original = Path.open
            def open_with_failure(path, mode="r", *args, **kwargs):
                if path == target/"remeshed.obj" and mode == "xb":
                    raise FileExistsError("simulated competing result")
                return original(path, mode, *args, **kwargs)
            with patch.object(Path, "open", open_with_failure):
                with self.assertRaisesRegex(FileExistsError, "competing"):
                    run_workflow(EXAMPLE, target, target_edge_length=.14, check_intersections=False,
                                 postprocess_method="isotropic")
            self.assertEqual([path.name for path in target.iterdir()], ["unrelated.txt"])
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_invalid_target_is_rejected_before_creating_results(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/"results"
            with self.assertRaisesRegex(ValueError, "positive and finite"):
                run_workflow(EXAMPLE, target, target_edge_length=0.)
            self.assertFalse(target.exists())

    def test_standard_option_uses_source_style_collapse_then_hc(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch("anamorph.workflow.isotropic_remesh", side_effect=AssertionError("unexpected isotropic remeshing")):
            result = run_workflow(EXAMPLE, Path(folder)/"results", check_intersections=False, soma_refinements=1,
                                  soma_blend_width=0, postprocess_method="standard")
            report = json.loads(Path(result["detailed_report"]).read_text(encoding="utf-8"))
            self.assertEqual(result["postprocessing_method"], "standard")
            options = report["settings"]["remeshing"]
            self.assertEqual(options["method"], "standard")
            self.assertEqual(options["edge_collapse"], {"alpha": 1.5, "area_floor": .125, "area_ceiling": .5,
                                                        "depth": 5, "max_collapses": 100000, "safety": "source"})
            self.assertEqual(options["hc_smoothing"], {"alpha": .4, "beta": .7, "iterations": 10})
            raw = read_obj(result["raw_mesh"])
            expected = hc_smooth(greedy_edge_collapse(raw, **options["edge_collapse"]), **options["hc_smoothing"])
            actual = read_obj(result["remeshed_mesh"])
            self.assertEqual(expected.faces, actual.faces)
            for point, expected_point in zip(actual.vertices, expected.vertices):
                for value, expected_value in zip(point, expected_point):
                    self.assertAlmostEqual(value, expected_value, places=7)
            self.assertIn("source-style edge collapse", Path(result["report"]).read_text(encoding="utf-8"))

    def test_standard_rejects_isotropic_options(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/"results"
            with self.assertRaisesRegex(ValueError, "isotropic postprocessing"):
                run_workflow(EXAMPLE, target, target_edge_length=.14, postprocess_method="standard")
            self.assertFalse(target.exists())

    def test_standard_cli_option_selects_hc_pipeline(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/"results"
            result = subprocess.run([sys.executable, "-m", "anamorph", "workflow", str(EXAMPLE),
                                     "--output-dir", str(target), "--standard-postprocess",
                                     "--skip-intersection-check"], cwd=ROOT,
                                    text=True, capture_output=True, timeout=60, check=True)
            info = json.loads(result.stdout)
            report = json.loads((target/"report.json").read_text(encoding="utf-8"))
            self.assertEqual(info["postprocessing_method"], "standard")
            self.assertEqual(report["settings"]["remeshing"]["hc_smoothing"]["iterations"], 10)
            self.assertEqual(report["settings"]["soma_junction_blend"]["method"], "blend")

    def test_isotropic_smoothing_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch("anamorph.workflow.taubin_smooth", side_effect=AssertionError("unexpected smoothing")):
            result = run_workflow(EXAMPLE, Path(folder)/"results", target_edge_length=.14,
                                  iterations=1, smoothing="none", soma_blend_width=0, check_intersections=False)
            report = json.loads(Path(result["detailed_report"]).read_text(encoding="utf-8"))
            self.assertNotIn("taubin_smoothing", report["settings"]["remeshing"])
            self.assertNotIn("hc_smoothing", report["settings"]["remeshing"])

    def test_intersection_findings_and_failures_are_not_reported_as_clean(self):
        mesh = ico_sphere((0., 0., 0.), 1., 0)
        with patch("anamorph.workflow.find_self_intersections",
                   return_value=([MeshIntersection(0, 1, False)], True)):
            report = _mesh_report(mesh, True)["self_intersections"]
        self.assertEqual(report["status"], "candidates_found")
        self.assertTrue(report["truncated"])
        self.assertEqual(report["candidates"][0]["first_face"], 0)
        with patch("anamorph.workflow.find_self_intersections", side_effect=ValueError("cannot check")):
            report = _mesh_report(mesh, True)["self_intersections"]
        self.assertEqual(report["status"], "unresolved")
        self.assertEqual(report["detail"], "cannot check")


if __name__ == "__main__":
    unittest.main()
