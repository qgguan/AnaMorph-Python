import math
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from fractions import Fraction
from unittest.mock import patch
from pathlib import Path

from anamorph.analytic_intersections import CanalSegment
from anamorph.analytic_network import AnalyticFinding, AnalyticReport
from anamorph.canal_mesh import (canal_segment_mesh, canal_path_mesh, append_canal_path,
                                 _source_arc_length_function, _triangle_edge_ratio, _render_frame, _source_render_frame,
                                 append_terminal_hemisphere, ico_sphere)
from anamorph.mesh import Mesh, cross, dot, sub
from anamorph.redblue import RedBlueError, RedBlueNumericalEdgeCase, RedBlueTriangulationError, redblue_union_with_retries
from anamorph.mesh import read_obj
from anamorph.collapse import greedy_edge_collapse
from anamorph.smoothing import hc_smooth
from anamorph.analytic_mesher import _render_vector, _render_vector_score
from anamorph.mesh_intersections import find_self_intersections
from anamorph.quality import mesh_quality
from anamorph.analytic_mesher import (PartialAnalysisError, RedBlueTerminalRingConsumed, assemble_redblue_checked,
                                     assemble_redblue_neuron, render_individual_surfaces)
from anamorph.swc import Morphology, Node, read_swc


class CanalMeshTests(unittest.TestCase):
    def test_quad_edge_ratio_handles_extreme_scales(self):
        for scale in (1e-200, 1e-160, 1., 1e200):
            points = ((0., 0., 0.), (scale, 0., 0.), (0., scale, 0.))
            self.assertAlmostEqual(_triangle_edge_ratio(points, 0, 1, 2, "test"), math.sqrt(2))

    def test_source_gate_distinguishes_intersection_jobs_from_distance_flags(self):
        distance = AnalyticFinding("primary_distance_violation", ((2, 3),), "short segment")
        hit = AnalyticFinding("local_self_intersection", ((2, 3),), "curvature")
        base = AnalyticReport(1, 0, 0, (distance,), (), job_mode="source")
        self.assertTrue(base.source_jobs_clean)
        self.assertTrue(base.as_dict()["source_jobs_clean"])
        self.assertIsNone(replace(base, job_mode="extended").source_jobs_clean)
        self.assertIsNone(replace(base, unresolved=("unfinished job",)).source_jobs_clean)
        self.assertIsNone(replace(base, source_plan_truncated=True).source_jobs_clean)
        self.assertFalse(replace(base, findings=(distance, hit), unresolved=("unfinished job",)).source_jobs_clean)
        sentinel = object()
        with patch("anamorph.analytic_mesher.analyze_analytic_partial", return_value=base), \
                patch("anamorph.analytic_mesher.assemble_redblue_neuron", return_value=sentinel):
            mesh, report = assemble_redblue_checked(None, analysis_job_mode="source", analysis_gate="source")
            self.assertIs(mesh, sentinel)
            self.assertIs(report, base)
            with self.assertRaises(PartialAnalysisError):
                assemble_redblue_checked(None, analysis_job_mode="source", analysis_gate="all")
            with self.assertRaisesRegex(ValueError, "requires source job mode"):
                assemble_redblue_checked(None, analysis_gate="source")
        for report in (replace(base, findings=(distance, hit)), replace(base, unresolved=("unfinished job",))):
            with self.subTest(report=report), \
                    patch("anamorph.analytic_mesher.analyze_analytic_partial", return_value=report), \
                    patch("anamorph.analytic_mesher.assemble_redblue_neuron") as assemble:
                with self.assertRaises(PartialAnalysisError) as caught:
                    assemble_redblue_checked(None, analysis_job_mode="source", analysis_gate="source")
                self.assertEqual(caught.exception.blocking_findings, report.source_job_findings)
                assemble.assert_not_called()

    def test_source_gate_cli_checks_before_writing_in_both_modes(self):
        root = Path(__file__).resolve().parent.parent
        clean = root/"examples"/"branched.swc"
        dirty = root/"tests"/"fixtures"/"branch_curvature_collision.swc"
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/"result.obj"
            for streamed in (False, True):
                with self.subTest(streamed=streamed):
                    common = ["--source-gate", "--angular-segments", "8", "--soma-refinements", "2"]
                    if streamed:
                        common += ["--flush-face-limit", "1"]
                    command = [sys.executable, "-m", "anamorph", "generate-redblue", str(clean), str(output)]
                    run = subprocess.run(command+common+["--check-output"], text=True, capture_output=True)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    data = json.loads(run.stdout)
                    self.assertFalse(data["forced"])
                    self.assertFalse(data["analysis"]["complete"])
                    self.assertTrue(data["analysis"]["source_jobs_clean"])
                    self.assertEqual((data["analysis"]["job_mode"], data["analysis"]["gate"]), ("source", "source"))
                    self.assertEqual(data["faces"], len(read_obj(output).faces))
                    self.assertIn("omits some checks", run.stderr)
                    if not streamed:
                        strict = subprocess.run(command+common+["--require-no-findings"],
                                                text=True, capture_output=True)
                        self.assertEqual(strict.returncode, 0, strict.stderr)
                        self.assertEqual(json.loads(strict.stdout)["analysis"]["gate"], "all")
                    for input_path, extra, expected in ((dirty, [], "local_self_intersection"),
                                                        (clean, ["--analysis-source-job-limit", "0"], "unresolved")):
                        output.write_text("untouched", encoding="utf-8")
                        run = subprocess.run([sys.executable, "-m", "anamorph", "generate-redblue",
                                              str(input_path), str(output)]+common+extra,
                                             text=True, capture_output=True)
                        self.assertNotEqual(run.returncode, 0)
                        self.assertIn(expected, run.stderr)
                        self.assertEqual(output.read_text(encoding="utf-8"), "untouched")
            output.write_text("untouched", encoding="utf-8")
            invalid = subprocess.run([sys.executable, "-m", "anamorph", "generate-redblue", str(clean), str(output),
                                      "--source-gate", "--analysis-job-mode", "extended"],
                                     text=True, capture_output=True)
            self.assertNotEqual(invalid.returncode, 0)
            self.assertIn("requires --analysis-job-mode source", invalid.stderr)
            self.assertEqual(output.read_text(encoding="utf-8"), "untouched")

    def test_curvature_violation_and_output_checks_reject_real_crossing(self):
        source = Path(__file__).resolve().parent/"fixtures"/"branch_curvature_collision.swc"
        tree = read_swc(source)
        options = dict(angular_segments=8, soma_refinements=2, max_outer_attempts=30)
        mesh = assemble_redblue_neuron(tree, **options)
        candidates, _ = find_self_intersections(mesh, include_adjacent=True)

        def exact_crossing(first_face, second_face):
            # Verify a strict edge/triangle crossing using exact rational
            # values of the generated floating-point vertex coordinates.
            def vector(i):
                return tuple(Fraction(value) for value in mesh.vertices[i])
            def segment_triangle(a, b, tri):
                a, b = vector(a), vector(b)
                p, q, r = map(vector, tri)
                direction, edge1, edge2 = sub(b, a), sub(q, p), sub(r, p)
                h = cross(direction, edge2)
                determinant = dot(edge1, h)
                if not determinant:
                    return False
                s = sub(a, p)
                u = dot(s, h)/determinant
                other = cross(s, edge1)
                v = dot(direction, other)/determinant
                t = dot(edge2, other)/determinant
                return 0 < t < 1 and 0 < u < 1 and 0 < v and u+v < 1
            for first, second in ((first_face, second_face), (second_face, first_face)):
                if any(segment_triangle(a, b, second) for a, b in zip(first, first[1:]+first[:1])):
                    return True
            return False

        self.assertTrue(any(not set(mesh.faces[hit.first_face]).intersection(mesh.faces[hit.second_face]) and
                            exact_crossing(mesh.faces[hit.first_face], mesh.faces[hit.second_face])
                            for hit in candidates))
        with self.assertRaises(PartialAnalysisError) as raised:
            assemble_redblue_checked(tree, **options)
        self.assertTrue(any(f.kind == "local_self_intersection" for f in raised.exception.report.findings))
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"existing.obj"
            command = [sys.executable, "-m", "anamorph", "generate-redblue", str(source), str(output),
                       "--force", "--angular-segments", "8", "--soma-refinements", "2"]
            for streamed in (False, True):
                for option, expected in (("--require-no-findings", "local_self_intersection"),
                                         ("--check-output", "self-intersection candidate")):
                    with self.subTest(streamed=streamed, option=option):
                        output.write_text("untouched", encoding="utf-8")
                        args = command+[option]+(["--flush-face-limit", "1"] if streamed else [])
                        result = subprocess.run(args, capture_output=True, text=True)
                        self.assertEqual(result.returncode, 1)
                        self.assertIn(expected, result.stderr)
                        self.assertEqual(output.read_text(encoding="utf-8"), "untouched")

    def test_redblue_cli_can_save_raw_and_uniform_meshes_with_or_without_flushing(self):
        example = Path(__file__).resolve().parent.parent/"examples"/"branched.swc"
        with tempfile.TemporaryDirectory() as tmp:
            for streamed in (False, True):
                with self.subTest(streamed=streamed):
                    raw, uniform = Path(tmp)/f"raw_{streamed}.obj", Path(tmp)/f"uniform_{streamed}.obj"
                    command = [sys.executable, "-m", "anamorph", "generate-redblue", str(example), str(raw),
                               "--force", "--angular-segments", "8", "--soma-refinements", "2",
                               "--uniform-edge-length", ".14", "--uniform-iterations", "8",
                               "--postprocess-output", str(uniform), "--check-output"]
                    if streamed:
                        command += ["--flush-face-limit", "1"]
                    completed = subprocess.run(command, capture_output=True, text=True, check=True)
                    report = json.loads(completed.stdout)
                    original, improved = read_obj(raw), read_obj(uniform)
                    self.assertTrue(report["postprocessed"])
                    self.assertEqual(report["output_check"]["checked_surfaces"], ["raw", "processed"])
                    self.assertFalse(report["output_check"]["certifies_nonintersection"])
                    self.assertEqual(report["postprocess_settings"]["isotropic_remesh"],
                                     {"target": .14, "iterations": 8, "max_operations": 300000})
                    before, after = mesh_quality(original)["edge_lengths"], mesh_quality(improved)["edge_lengths"]
                    self.assertEqual(report["edge_lengths_before"], before)
                    self.assertEqual(report["edge_lengths_after"], after)
                    self.assertLess(after["p95"]/after["p05"], before["p95"]/before["p05"])
                    self.assertIsNotNone(report["postprocess_change_percent"]["area"])
                    self.assertIsNotNone(report["postprocess_change_percent"]["volume"])
                    if streamed:
                        self.assertTrue(report["partial_flush"]["reloaded_for_whole_mesh_work"])
            output = Path(tmp)/"uniform_only.obj"
            completed = subprocess.run([sys.executable, "-m", "anamorph", "generate-redblue", str(example),
                                        str(output), "--force", "--angular-segments", "8",
                                        "--soma-refinements", "2", "--uniform-edge-length", ".14",
                                        "--uniform-iterations", "2"], capture_output=True, text=True, check=True)
            report = json.loads(completed.stdout)
            self.assertEqual(report["edge_lengths_after"], mesh_quality(read_obj(output))["edge_lengths"])
            self.assertNotIn("postprocess_output", report)

    def test_redblue_cli_rejects_invalid_uniform_options_before_writing(self):
        example = Path(__file__).resolve().parent.parent/"examples"/"branched.swc"
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"existing.obj"
            command = [sys.executable, "-m", "anamorph", "generate-redblue", str(example), str(output), "--force"]
            for options in (("--uniform-edge-length", "0"), ("--uniform-edge-length", "nan"),
                            ("--uniform-iterations", "2"),
                            ("--uniform-edge-length", ".14", "--uniform-max-operations", "0")):
                with self.subTest(options=options):
                    output.write_text("untouched", encoding="utf-8")
                    completed = subprocess.run(command+list(options), capture_output=True, text=True)
                    self.assertEqual(completed.returncode, 1)
                    self.assertIn("uniform", completed.stderr)
                    self.assertEqual(output.read_text(encoding="utf-8"), "untouched")

    def test_branched_assembly_scales_to_large_finite_coordinates(self):
        tree = read_swc(Path(__file__).resolve().parent.parent/"examples"/"branched.swc")
        def assemble(scale):
            scaled = Morphology([Node(node.id, node.kind, node.x*scale, node.y*scale, node.z*scale,
                                      node.radius*scale, node.parent) for node in tree.nodes.values()])
            return assemble_redblue_neuron(scaled, preconditioning=False, angular_segments=8,
                                           soma_refinements=2, max_outer_attempts=20)
        baseline, large = assemble(1.), assemble(1e200)
        self.assertEqual((len(baseline.vertices), len(baseline.faces)), (len(large.vertices), len(large.faces)))
        self.assertLess(max(math.dist(first, tuple(value/1e200 for value in second))
                            for first, second in zip(baseline.vertices, large.vertices)), 1e-12)

    def test_univariate_work_limit_is_reported_and_gates_both_mesh_paths(self):
        example = Path(__file__).resolve().parent.parent/"examples"/"branched.swc"
        analysis = subprocess.run([sys.executable, "-m", "anamorph", "analyze-analytic", str(example),
                                   "--max-intervals", "1"], text=True, capture_output=True, check=True)
        report = json.loads(analysis.stdout)
        self.assertTrue(any("interval limit" in item for item in report["unresolved"]))
        invalid = subprocess.run([sys.executable, "-m", "anamorph", "analyze-analytic", str(example),
                                  "--max-intervals", "0"], text=True, capture_output=True)
        self.assertNotEqual(invalid.returncode, 0)
        self.assertIn("max_intervals", invalid.stderr)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"mesh.obj"
            for streaming in (False, True):
                output.write_text("untouched", encoding="utf-8")
                command = [sys.executable, "-m", "anamorph", "generate-redblue", str(example), str(output),
                           "--force", "--require-no-findings", "--analysis-max-intervals", "1"]
                if streaming:
                    command += ["--flush-face-limit", "1"]
                run = subprocess.run(command, text=True, capture_output=True)
                self.assertNotEqual(run.returncode, 0)
                self.assertIn("unresolved", run.stderr)
                self.assertEqual(output.read_text(encoding="utf-8"), "untouched")

    def test_icosphere_counts_radius_and_orientation(self):
        center = (1., -2., 3.)
        for level, (vertices, faces) in enumerate(((12, 20), (42, 80), (162, 320))):
            mesh = ico_sphere(center, 2., level)
            stats = mesh.statistics()
            self.assertEqual((stats.vertices, stats.faces, stats.euler_characteristic), (vertices, faces, 2))
            self.assertEqual((stats.boundary_edges, stats.nonmanifold_edges), (0, 0))
            for point in mesh.vertices:
                self.assertAlmostEqual(math.dist(point, center), 2.)
            for a, b, c in mesh.faces:
                p, q, r = mesh.vertices[a], mesh.vertices[b], mesh.vertices[c]
                self.assertGreater(dot(cross(sub(q, p), sub(r, p)), sub(p, center)), 0.)

    def test_generated_surfaces_reject_unrepresentable_coordinates(self):
        with self.assertRaisesRegex(ValueError, "soma triangle.*coordinate precision"):
            ico_sphere((1e16, 1e16, 1e16), .01, 1)
        segment = CanalSegment(((1e16, 10., 0., 0.), (1e16, 0., 0., 0.),
                                (1e16, 0., 0., 0.)), .01, .01)
        with self.assertRaisesRegex(ValueError, "canal triangle.*coordinate precision"):
            canal_segment_mesh(segment, angular_segments=6, arc_samples=20)
        for scale in (1e-200, 1e200):
            with self.subTest(scale=scale):
                sphere = ico_sphere((0., 0., 0.), scale, 1)
                self.assertEqual(len(set(sphere.vertices)), len(sphere.vertices))
                segment = CanalSegment(((0., scale, 0., 0.), (0., 0., 0., 0.),
                                        (0., 0., 0., 0.)), .2*scale, .2*scale)
                mesh = canal_segment_mesh(segment, angular_segments=8, arc_samples=20)
                self.assertEqual(len(set(mesh.vertices)), len(mesh.vertices))

    def test_postprocess_cli_writes_a_separate_existing_mesh(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "sphere.obj", Path(tmp) / "smoothed.obj"
            ico_sphere((0., 0., 0.), 1., 1).write_obj(source)
            original = source.read_bytes()
            command = [sys.executable, "-m", "anamorph", "postprocess", str(source), str(output),
                       "--laplacian-iterations", "1", "--laplacian-factor", ".25"]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            report = json.loads(result.stdout)
            self.assertEqual(report["faces"], 80)
            self.assertLess(report["change_percent"]["area"], 0.)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(read_obj(output).statistics().boundary_edges, 0)
            rejected = subprocess.run([sys.executable, "-m", "anamorph", "postprocess", str(source),
                                       str(source), "--hc-iterations", "1"], capture_output=True, text=True)
            self.assertEqual(rejected.returncode, 1)
            self.assertEqual(source.read_bytes(), original)

    def test_postprocess_cli_passes_custom_source_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp)/"sphere.obj", Path(tmp)/"processed.obj"
            ico_sphere((0., 0., 0.), 1., 1).write_obj(source)
            collapse = dict(alpha=1.05, area_floor=2., area_ceiling=3., depth=2, max_collapses=3, safety="safe")
            hc = dict(alpha=.2, beta=.8, iterations=2)
            command = [sys.executable, "-m", "anamorph", "postprocess", str(source), str(output),
                       "--edge-collapse", "--collapse-alpha", "1.05", "--collapse-area-floor", "2",
                       "--collapse-area-ceiling", "3", "--collapse-depth", "2", "--collapse-max-collapses", "3",
                       "--hc-iterations", "2", "--hc-alpha", ".2", "--hc-beta", ".8"]
            run = subprocess.run(command, text=True, capture_output=True, check=True)
            report = json.loads(run.stdout)
            expected = hc_smooth(greedy_edge_collapse(read_obj(source), **collapse), **hc)
            self.assertEqual(read_obj(output), expected)
            self.assertLess(len(expected.faces), 80)
            self.assertEqual(report["postprocess_settings"]["edge_collapse"], collapse)
            self.assertEqual(report["postprocess_settings"]["hc_smoothing"], hc)
            output.write_text("untouched", encoding="utf-8")
            invalid = subprocess.run(command+["--collapse-alpha", ".5"], text=True, capture_output=True)
            self.assertNotEqual(invalid.returncode, 0)
            self.assertIn("alpha >= 1", invalid.stderr)
            self.assertEqual(output.read_text(encoding="utf-8"), "untouched")

    def test_redblue_postprocess_options_apply_with_and_without_flushing(self):
        example = Path(__file__).resolve().parent.parent/"examples"/"branched.swc"
        collapse = dict(alpha=1.05, area_floor=.9, area_ceiling=2., depth=1, max_collapses=3, safety="source")
        hc = dict(alpha=.2, beta=.8, iterations=2)
        with tempfile.TemporaryDirectory() as tmp:
            for streaming in (False, True):
                with self.subTest(streaming=streaming):
                    raw, processed = Path(tmp)/"raw.obj", Path(tmp)/"processed.obj"
                    command = [sys.executable, "-m", "anamorph", "generate-redblue", str(example), str(raw),
                               "--force", "--angular-segments", "8", "--soma-refinements", "2",
                               "--edge-collapse", "--collapse-alpha", "1.05", "--collapse-area-floor", ".9",
                               "--collapse-area-ceiling", "2", "--collapse-depth", "1", "--collapse-max-collapses", "3",
                               "--collapse-safety", "source",
                               "--smooth-iterations", "2", "--hc-alpha", ".2", "--hc-beta", ".8",
                               "--postprocess-output", str(processed)]
                    if streaming:
                        command += ["--flush-face-limit", "1"]
                    run = subprocess.run(command, text=True, capture_output=True, check=True)
                    info = json.loads(run.stdout)
                    expected = hc_smooth(greedy_edge_collapse(read_obj(raw), **collapse), **hc)
                    self.assertEqual(read_obj(processed), expected)
                    self.assertEqual(info["postprocess_settings"], {"edge_collapse": collapse, "hc_smoothing": hc})
            raw.write_text("untouched", encoding="utf-8")
            invalid = subprocess.run([sys.executable, "-m", "anamorph", "generate-redblue", str(example), str(raw),
                                      "--force", "--smooth-iterations", "2", "--hc-alpha", ".8", "--hc-beta", ".2"],
                                     text=True, capture_output=True)
            self.assertNotEqual(invalid.returncode, 0)
            self.assertIn("0 < alpha < beta", invalid.stderr)
            self.assertEqual(raw.read_text(encoding="utf-8"), "untouched")
            invalid = subprocess.run([sys.executable, "-m", "anamorph", "generate-redblue", str(example), str(raw),
                                      "--force", "--collapse-safety", "source"], text=True, capture_output=True)
            self.assertNotEqual(invalid.returncode, 0)
            self.assertIn("require an enabled edge-collapse stage", invalid.stderr)
            self.assertEqual(raw.read_text(encoding="utf-8"), "untouched")

    def test_remesh_cli_reports_size_and_geometry_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "sphere.obj", Path(tmp) / "uniform.obj"
            ico_sphere((0., 0., 0.), 1., 1).write_obj(source)
            original = source.read_bytes()
            result = subprocess.run([sys.executable, "-m", "anamorph", "remesh", str(source), str(output),
                                     "--isotropic", "--target-edge-length", ".4", "--iterations", "2"],
                                    capture_output=True, text=True, check=True)
            report = json.loads(result.stdout)
            self.assertEqual(source.read_bytes(), original)
            self.assertIn("p95", report["edge_lengths_before"])
            self.assertIn("p95", report["edge_lengths_after"])
            before, after = read_obj(source).statistics(), read_obj(output).statistics()
            self.assertAlmostEqual(report["change_percent"]["area"], 100*(after.area/before.area-1))
            self.assertAlmostEqual(report["change_percent"]["volume"], 100*(after.volume/before.volume-1))

    def test_remesh_defaults_to_source_style_postprocessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "sphere.obj", Path(tmp) / "processed.obj"
            ico_sphere((0., 0., 0.), 1., 1).write_obj(source)
            run = subprocess.run([sys.executable, "-m", "anamorph", "remesh", str(source), str(output)],
                                 capture_output=True, text=True, check=True)
            info = json.loads(run.stdout)
            self.assertEqual(info["postprocessing_method"], "standard")
            self.assertEqual(info["postprocess_settings"]["edge_collapse"]["safety"], "source")
            self.assertEqual(info["postprocess_settings"]["hc_smoothing"]["iterations"], 10)
            self.assertTrue(output.is_file())
            rejected = subprocess.run([sys.executable, "-m", "anamorph", "remesh", str(source),
                                       str(Path(tmp)/"invalid.obj"), "--target-edge-length", ".4"],
                                      capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("--isotropic", rejected.stderr)

    def test_straight_canal_mesh_is_closed_and_follows_radii(self):
        segment = CanalSegment(((0., 2., 0., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)), .2, .3)
        mesh = canal_segment_mesh(segment, angular_segments=8, arc_length_mode="accurate")
        stats = mesh.statistics()
        self.assertEqual((stats.boundary_edges, stats.nonmanifold_edges, stats.euler_characteristic), (0, 0, 2))
        for point in mesh.vertices[:-2]:
            expected = .2+.1*point[0]/2
            self.assertAlmostEqual(math.hypot(point[1], point[2]), expected, places=6)
        self.assertGreater(stats.volume, 0)

    def test_source_arc_length_reproduces_last_interval_rule(self):
        segment = CanalSegment(((0., 1., 0., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)), .2, .3)
        for windows_x87 in (False, True):
            with self.subTest(windows_x87=windows_x87):
                arc = _source_arc_length_function(segment, .1, windows_x87=windows_x87)
                self.assertEqual(arc(0.), 0.)
                self.assertAlmostEqual(arc(.05), .05)
                self.assertAlmostEqual(arc(.35), .45)
                self.assertAlmostEqual(arc(1.), 1.1)

    def test_canal_rejects_parallel_render_vector(self):
        segment = CanalSegment(((0., 1., 0., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)), .2, .2)
        with self.assertRaisesRegex(ValueError, "stable frame"):
            canal_segment_mesh(segment, render_vector=(1., 0., 0.))

    def test_retry_seed_is_reproducible_and_independent_of_attempt_budget(self):
        tree = Morphology([Node(1, 1, 0, 0, 0, 1, -1), Node(2, 3, 1.3, 0, 0, .2, 1),
                           Node(3, 3, 2.5, 0, 0, .2, 2)])
        settings = dict(angular_segments=8, soma_refinements=2, max_inner_attempts=20)
        first = assemble_redblue_neuron(tree, random_seed=0, max_outer_attempts=1, **settings)
        self.assertEqual(first, assemble_redblue_neuron(tree, random_seed=0, max_outer_attempts=12, **settings))
        self.assertNotEqual(first, assemble_redblue_neuron(tree, random_seed=1, max_outer_attempts=12, **settings))

    def test_source_retry_restarts_even_after_last_inner_attempt_succeeds(self):
        tree = Morphology([Node(1, 1, 0, 0, 0, 1, -1), Node(2, 3, 1.3, 0, 0, .2, 1),
                           Node(3, 3, 2.5, 0, 0, .2, 2)])
        settings = dict(angular_segments=8, soma_refinements=2, max_outer_attempts=1, max_inner_attempts=1)
        self.assertGreater(len(assemble_redblue_neuron(tree, **settings).faces), 0)
        with self.assertRaisesRegex(RedBlueError, "source inner iteration limit reached"):
            assemble_redblue_neuron(tree, source_retry_semantics=True, **settings)

    def test_default_outer_retries_are_not_cut_off_at_128(self):
        tree = Morphology([Node(1, 1, 0, 0, 0, 1, -1), Node(2, 3, 1.3, 0, 0, .2, 1),
                           Node(3, 3, 2.5, 0, 0, .2, 2)])
        calls = []
        def intermittent_contact(red, blue, **kwargs):
            calls.append(1)
            if len(calls) <= 128:
                raise RedBlueNumericalEdgeCase("injected contact")
            return redblue_union_with_retries(red, blue, **kwargs)
        settings = dict(angular_segments=8, soma_refinements=2, max_inner_attempts=20)
        with patch("anamorph.analytic_mesher.redblue_union_with_retries", side_effect=intermittent_contact):
            with self.assertRaisesRegex(RedBlueError, "injected contact"):
                assemble_redblue_neuron(tree, max_outer_attempts=128, **settings)
        self.assertEqual(len(calls), 128)
        calls.clear()
        with patch("anamorph.analytic_mesher.redblue_union_with_retries", side_effect=intermittent_contact):
            mesh = assemble_redblue_neuron(tree, **settings)
        self.assertEqual(len(calls), 129)
        self.assertEqual(mesh.statistics().boundary_edges, 0)

    def test_outer_retry_rejects_unrepresentable_radius_decrement(self):
        tree = Morphology([Node(1, 1, 0, 0, 0, 1, -1), Node(2, 3, 1.3, 0, 0, .2, 1),
                           Node(3, 3, 2.5, 0, 0, .2, 2)])
        with self.assertRaisesRegex(ValueError, "too small to change"):
            assemble_redblue_neuron(tree, outer_loop_maxiter=1, radius_factor_decrement=1e-320)

    def test_complex_edge_growth_limit_is_forwarded_to_union(self):
        tree = Morphology([Node(1, 1, 0, 0, 0, 1, -1), Node(2, 3, 1.3, 0, 0, .2, 1),
                           Node(3, 3, 2.5, 0, 0, .2, 2)])
        seen = []
        def capture(red, blue, **kwargs):
            seen.append(kwargs["max_growth_factor"])
            return redblue_union_with_retries(red, blue, **kwargs)
        with patch("anamorph.analytic_mesher.redblue_union_with_retries", side_effect=capture):
            assemble_redblue_neuron(tree, angular_segments=8, soma_refinements=2,
                                    max_complex_edge_growth_factor=3.5)
        self.assertTrue(seen)
        self.assertTrue(all(value == 3.5 for value in seen))
        with self.assertRaisesRegex(ValueError, "meshing settings"):
            assemble_redblue_neuron(tree, max_complex_edge_growth_factor=0.)

    def test_growing_a_stump_uses_the_same_outer_attempt_budget(self):
        example = Path(__file__).resolve().parent.parent / "examples" / "branched.swc"
        def run(limit):
            calls = []
            def consume_first_terminal(red, blue, **kwargs):
                result = redblue_union_with_retries(red, blue, **kwargs)
                calls.append(1)
                if len(calls) == 1:
                    return replace(result, blue_vertex_map=(None,)*len(result.blue_vertex_map))
                return result
            with patch("anamorph.analytic_mesher.redblue_union_with_retries", side_effect=consume_first_terminal):
                result = assemble_redblue_neuron(read_swc(example), angular_segments=8, soma_refinements=2,
                                                 max_outer_attempts=limit, max_inner_attempts=20)
            return result, calls
        with self.assertRaisesRegex(RedBlueError, "terminal ring or cap"):
            run(1)
        result, calls = run(2)
        self.assertGreaterEqual(len(calls), 2)
        self.assertEqual(result.statistics().boundary_edges, 0)

    def test_retry_stops_at_safe_radius_and_exhausted_stump_is_fatal(self):
        example = Path(__file__).resolve().parent.parent / "examples" / "branched.swc"
        tree = read_swc(example)
        calls = []
        def fail_numerically(*args, **kwargs):
            calls.append(1)
            raise RedBlueNumericalEdgeCase("injected")
        with patch("anamorph.analytic_mesher.redblue_union_with_retries", side_effect=fail_numerically):
            with self.assertRaisesRegex(RedBlueError, "safe radius bound"):
                assemble_redblue_neuron(tree, angular_segments=8, soma_refinements=2, max_outer_attempts=100,
                                        outer_loop_maxiter=1, radius_factor_decrement=.1)
        self.assertEqual(len(calls), 5)
        calls.clear()
        def consume_second_terminal(red, blue, **kwargs):
            result = redblue_union_with_retries(red, blue, **kwargs)
            calls.append(1)
            return (replace(result, blue_vertex_map=(None,)*len(result.blue_vertex_map))
                    if len(calls) == 2 else result)
        with patch("anamorph.analytic_mesher.redblue_union_with_retries", side_effect=consume_second_terminal):
            with self.assertRaises(RedBlueTerminalRingConsumed):
                assemble_redblue_neuron(tree, angular_segments=8, soma_refinements=2,
                                        max_outer_attempts=20, max_inner_attempts=20)
        self.assertEqual(len(calls), 2)

    def test_numerical_contact_retries_but_unexpected_failure_is_fatal(self):
        tree = Morphology([Node(1, 1, 0, 0, 0, 1, -1), Node(2, 3, 1.3, 0, 0, .2, 1),
                           Node(3, 3, 2.5, 0, 0, .2, 2)])
        calls = []
        def numerical_once(red, blue, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise RedBlueNumericalEdgeCase("injected numerical contact")
            return redblue_union_with_retries(red, blue, **kwargs)
        with patch("anamorph.analytic_mesher.redblue_union_with_retries", side_effect=numerical_once):
            result = assemble_redblue_neuron(tree, angular_segments=8, soma_refinements=2,
                                             max_outer_attempts=3, max_inner_attempts=20)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result.statistics().boundary_edges, 0)
        calls.clear()
        failure = RedBlueError("injected topology invariant failure")
        def fail_invariant(*args, **kwargs):
            calls.append(1)
            raise failure
        with patch("anamorph.analytic_mesher.redblue_union_with_retries", side_effect=fail_invariant):
            with self.assertRaises(RedBlueError) as captured:
                assemble_redblue_neuron(tree, angular_segments=8, soma_refinements=2, max_outer_attempts=3)
        self.assertIs(captured.exception, failure)
        self.assertEqual(len(calls), 1)

    def test_render_vector_check_finds_an_interior_parallel_tangent(self):
        segment = CanalSegment(((0., -.5, .5, 0.), (0., 0., 0., 0.), (0., 1., 0., 0.)), .2, .2)
        self.assertLess(_render_vector_score(segment, (0., 0., 1.)), 1e-8)
        chosen = _render_vector((segment,))
        self.assertGreater(_render_vector_score(segment, chosen), .01)

    def test_render_vector_check_is_invariant_to_coordinate_scale(self):
        for scale in (1., 1e-3, 1e-6, 1e-9):
            segment = CanalSegment(((0., scale, 0., 0.),
                                    (.23**2*scale, -.46*scale, scale, 0.),
                                    (0., 0., 0., 0.)), .2*scale, .2*scale)
            self.assertLess(_render_vector_score(segment, (1., 0., 0.)), 1e-8, scale)

    def test_analytic_soma_and_canal_join_by_redblue(self):
        soma = ico_sphere((0., 0., 0.), 1., 2)
        segment = CanalSegment(((0., 2., 0., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)), .22, .22)
        canal = canal_segment_mesh(segment, angular_segments=8, phi0=.137)
        result = redblue_union_with_retries(soma, canal, max_attempts=20, max_growth_factor=6)
        stats = result.statistics()
        self.assertEqual((stats.boundary_edges, stats.nonmanifold_edges, stats.euler_characteristic), (0, 0, 2))
        self.assertGreater(stats.volume, soma.statistics().volume)
        self.assertLess(stats.volume, soma.statistics().volume+canal.statistics().volume)

    def test_multiple_canal_segments_share_cross_section(self):
        first = CanalSegment(((0., 1., 0., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)), .2, .25)
        second = CanalSegment(((1., 1., 0., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)), .25, .3)
        path = canal_path_mesh((first, second), angular_segments=8, phi0=.13)
        stats = path.statistics()
        self.assertEqual((stats.boundary_edges, stats.nonmanifold_edges, stats.euler_characteristic), (0, 0, 2))
        # The shared cross-section appears once and has no center cap.
        cross_section = [p for p in path.vertices if abs(p[0]-1) < 1e-12]
        self.assertEqual(len(cross_section), 8)
        self.assertTrue(all(abs(math.hypot(p[1], p[2])-.25) < 1e-10 for p in cross_section))
        mismatch = CanalSegment(((1., 0., 0., 0.), (0., 1., 0., 0.), (0., 0., 0., 0.)), .25, .3)
        with self.assertRaisesRegex(ValueError, "tangent"):
            canal_path_mesh((first, mismatch), angular_segments=8)

    def test_canal_mesh_and_joins_are_scale_aware(self):
        meshes, hemispheres = [], []
        for scale in (1., 1e-9, 1e-15, 1e-18, 1e12):
            with self.subTest(scale=scale):
                first = CanalSegment(((0., scale, 0., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)),
                                     .2*scale, .2*scale)
                second = CanalSegment(((scale, scale, 0., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)),
                                      .2*scale, .2*scale)
                boundary = canal_path_mesh((first,), angular_segments=8, return_boundary=True,
                                           render_vector=(0., 0., scale))
                joined = append_canal_path(boundary.mesh, boundary, (second,), angular_segments=8,
                                           render_vector=(0., 0., scale))
                direct = canal_path_mesh((first, second), angular_segments=8, render_vector=(0., 0., scale))
                self.assertEqual(len(joined.vertices), len(direct.vertices))
                self.assertEqual(len(joined.faces), len(direct.faces))
                self.assertEqual(direct.statistics().boundary_edges, 0)
                meshes.append(tuple(tuple(x/scale for x in p) for p in direct.vertices))
                rounded = append_terminal_hemisphere(boundary, start=(scale, 0., 0.), direction=(scale, 0., 0.),
                                                     radius=.2*scale, angular_segments=8,
                                                     render_vector=(0., 0., scale))
                self.assertEqual(rounded.statistics().boundary_edges, 0)
                hemispheres.append(tuple(tuple(x/scale for x in p) for p in rounded.vertices))
                displaced = CanalSegment(((2*scale, scale, 0., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)),
                                         .2*scale, .2*scale)
                with self.assertRaisesRegex(ValueError, "must meet"):
                    canal_path_mesh((first, displaced), angular_segments=8)
                with self.assertRaisesRegex(ValueError, "terminal ring do not match"):
                    append_canal_path(boundary.mesh, boundary, (displaced,), angular_segments=8)
                wrong_radius = CanalSegment(second.power, .3*scale, .2*scale)
                with self.assertRaisesRegex(ValueError, "same shared radius"):
                    canal_path_mesh((first, wrong_radius), angular_segments=8)
        for mesh in meshes[1:]:
            self.assertEqual(len(mesh), len(meshes[0]))
            self.assertLess(max(math.dist(p, q) for p, q in zip(mesh, meshes[0])), 1e-13)
        for mesh in hemispheres[1:]:
            self.assertEqual(len(mesh), len(hemispheres[0]))
            self.assertLess(max(math.dist(p, q) for p, q in zip(mesh, hemispheres[0])), 1e-13)

    def test_canal_render_frame_rejects_near_parallel_axes_at_any_scale(self):
        segment = CanalSegment(((0., 1., 0., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)), .2, .2)
        for scale in (1e-18, 1., 1e18):
            with self.subTest(scale=scale), self.assertRaisesRegex(ValueError, "stable frame"):
                canal_segment_mesh(segment, render_vector=(scale, scale*1e-16, 0.))

    def test_join_tolerance_does_not_grow_with_coordinate_offset(self):
        first = CanalSegment(((1e6, .01, 0., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)), .001, .001)
        second = CanalSegment(((1e6+.01, .01, 0., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)), .001, .001)
        self.assertEqual(canal_path_mesh((first, second), angular_segments=8).statistics().boundary_edges, 0)
        gap = CanalSegment(((1e6+.0101, .01, 0., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)), .001, .001)
        with self.assertRaisesRegex(ValueError, "must meet"):
            canal_path_mesh((first, gap), angular_segments=8)

    def test_reopen_terminal_cap_and_append_tail(self):
        first = CanalSegment(((0., 1., 0., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)), .2, .25)
        second = CanalSegment(((1., 1., 0., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)), .25, .3)
        initial = canal_path_mesh((first,), angular_segments=8, phi0=.13, return_boundary=True)
        result = append_canal_path(initial.mesh, initial, (second,), angular_segments=8, phi0=.13)
        mapped, vertex_map = append_canal_path(initial.mesh, initial, (second,), angular_segments=8, phi0=.13,
                                                return_vertex_map=True)
        self.assertEqual(mapped, result)
        self.assertIsNone(vertex_map[initial.end_cap])
        for i, new in enumerate(vertex_map):
            if new is not None:
                self.assertEqual(initial.mesh.vertices[i], result.vertices[new])
        reference = canal_path_mesh((first, second), angular_segments=8, phi0=.13)
        self.assertEqual(len(result.vertices), len(reference.vertices))
        self.assertEqual(len(result.faces), len(reference.faces))
        self.assertAlmostEqual(result.statistics().volume, reference.statistics().volume)
        self.assertEqual(result.statistics().boundary_edges, 0)

    def test_terminal_hemisphere_extends_beyond_endpoint(self):
        segment = CanalSegment(((0., 1., 0., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)), .2, .2)
        boundary = canal_path_mesh((segment,), angular_segments=8, phi0=.13, return_boundary=True)
        rounded = append_terminal_hemisphere(boundary, start=(1., 0., 0.), direction=(1., 0., 0.),
                                             radius=.2, angular_segments=8, phi0=.13)
        mapped, vertex_map = append_terminal_hemisphere(boundary, start=(1., 0., 0.), direction=(1., 0., 0.),
                                                        radius=.2, angular_segments=8, phi0=.13,
                                                        return_vertex_map=True)
        self.assertEqual(mapped, rounded)
        self.assertIsNone(vertex_map[boundary.end_cap])
        with self.assertRaisesRegex(ValueError, "does not match the terminal cap"):
            append_terminal_hemisphere(boundary, start=(4., 0., 0.), direction=(1., 0., 0.),
                                       radius=.2, angular_segments=8, phi0=.13)
        with self.assertRaisesRegex(ValueError, "does not match the terminal ring"):
            append_terminal_hemisphere(boundary, start=(1., 0., 0.), direction=(1., 0., 0.),
                                       radius=.4, angular_segments=8, phi0=.13)
        with self.assertRaisesRegex(ValueError, "does not match the terminal ring"):
            append_terminal_hemisphere(boundary, start=(1., 0., 0.), direction=(0., 1., 0.),
                                       radius=.2, angular_segments=8, phi0=.13)
        for i, new in enumerate(vertex_map):
            if new is not None:
                self.assertEqual(boundary.mesh.vertices[i], rounded.vertices[new])
        self.assertAlmostEqual(max(p[0] for p in rounded.vertices), 1.2)
        self.assertGreater(rounded.statistics().volume, boundary.mesh.statistics().volume)
        self.assertEqual((rounded.statistics().boundary_edges, rounded.statistics().euler_characteristic), (0, 2))

    def test_source_frame_falls_back_when_squared_norm_is_unrepresentable(self):
        for scale in (1e-200, 1e200):
            with self.subTest(scale=scale):
                direction, render = (scale, scale/3, 0.), (0., scale/5, scale)
                expected = _render_frame(direction, render)
                self.assertEqual(_source_render_frame(direction, render, sqrt_mode="windows-x87"), expected)
                self.assertEqual(_source_render_frame(direction, render, hemisphere=True,
                                                       sqrt_mode="windows-x87"), expected)

    def test_tail_and_hemisphere_reject_invalid_terminal_ring_or_phase(self):
        first = CanalSegment(((0., 1., 0., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)), .2, .2)
        second = CanalSegment(((1., 1., 0., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)), .2, .2)
        boundary = canal_path_mesh((first,), angular_segments=8, phi0=.13, return_boundary=True)
        kwargs = dict(start=(1., 0., 0.), direction=(1., 0., 0.), radius=.2, angular_segments=8)
        # A half-turn twist used to be accepted and yielded self-intersections.
        with self.assertRaisesRegex(ValueError, "phase or frame"):
            append_terminal_hemisphere(boundary, phi0=.13+math.pi, **kwargs)
        with self.assertRaisesRegex(ValueError, "phase or frame"):
            append_terminal_hemisphere(boundary, phi0=.13, render_vector=(0., 1., 0.), **kwargs)
        append_terminal_hemisphere(boundary, phi0=.13+2*math.pi, **kwargs)
        # Keep the same floating-point addition order as canal ring creation.
        large_phase = canal_path_mesh((first,), angular_segments=8, phi0=1e15,
                                     start_circle_offset=True, return_boundary=True)
        append_terminal_hemisphere(large_phase, phi0=1e15, **kwargs)
        ring = list(boundary.end_ring)
        ring[0], ring[1] = ring[1], ring[0]
        unordered = replace(boundary, end_ring=tuple(ring))
        with self.assertRaisesRegex(ValueError, "terminal cap"):
            append_terminal_hemisphere(unordered, phi0=.13, **kwargs)
        with self.assertRaisesRegex(ValueError, "terminal cap"):
            append_canal_path(boundary.mesh, unordered, (second,), angular_segments=8, phi0=.13)
        faces = list(boundary.mesh.faces)
        cap_indices = [i for i, face in enumerate(faces) if boundary.end_cap in face]
        faces[cap_indices[0]] = faces[cap_indices[1]]
        corrupt_mesh = Mesh(boundary.mesh.vertices, tuple(faces))
        corrupt = replace(boundary, mesh=corrupt_mesh)
        with self.assertRaisesRegex(ValueError, "terminal cap"):
            append_terminal_hemisphere(corrupt, phi0=.13, **kwargs)
        with self.assertRaisesRegex(ValueError, "terminal cap"):
            append_canal_path(corrupt_mesh, corrupt, (second,), angular_segments=8, phi0=.13)

    def test_branched_swc_redblue_command_and_optional_postprocessing(self):
        example = Path(__file__).resolve().parent.parent / "examples" / "branched.swc"
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "redblue.obj"
            command = [sys.executable, "-m", "anamorph", "generate-redblue", str(example), str(output),
                       "--angular-segments", "8", "--soma-refinements", "2",
                       "--outer-attempts", "12", "--inner-attempts", "20"]
            generated = subprocess.run(command, text=True, capture_output=True, check=True)
            self.assertFalse(json.loads(generated.stdout)["analysis_performed"])
            self.assertTrue(output.is_file())
            result = subprocess.run(command+["--force", "--postprocess"], text=True, capture_output=True, check=True)
            self.assertIn("experimental analytic red-blue meshing", result.stderr)
            report = json.loads(result.stdout)
            self.assertTrue(report["postprocessed"])
            self.assertIn("area", report["postprocess_change_percent"])
            self.assertIn("volume", report["postprocess_change_percent"])
            stats = read_obj(output).statistics()
            self.assertEqual((stats.boundary_edges, stats.nonmanifold_edges, stats.euler_characteristic), (0, 0, 2))
            hits, truncated = find_self_intersections(read_obj(output), include_adjacent=True)
            self.assertEqual(hits, [])
            self.assertFalse(truncated)
            subprocess.run(command+["--force", "--scale-radius", ".7", "--preserve-crease-edges"],
                           text=True, capture_output=True, check=True)
            scaled = read_obj(output).statistics()
            self.assertEqual((scaled.boundary_edges, scaled.nonmanifold_edges, scaled.euler_characteristic), (0, 0, 2))
            postprocessed = Path(tmp) / "redblue_processed.obj"
            both = subprocess.run(command+["--force", "--edge-collapse", "--postprocess-output", str(postprocessed),
                                           "--check-output"],
                                  text=True, capture_output=True, check=True)
            both_report = json.loads(both.stdout)
            raw_mesh, processed_mesh = read_obj(output), read_obj(postprocessed)
            self.assertEqual(both_report["vertices"], len(raw_mesh.vertices))
            self.assertEqual(both_report["postprocess_output"]["faces"], len(processed_mesh.faces))
            self.assertGreater(len(raw_mesh.faces), len(processed_mesh.faces))
            self.assertIn("volume", both_report["postprocess_change_percent"])
            self.assertEqual(both_report["output_check"]["checked_surfaces"], ["raw", "processed"])
            rejected = subprocess.run(command+["--force", "--postprocess-output", str(postprocessed)],
                                      text=True, capture_output=True)
            self.assertEqual(rejected.returncode, 1)

    def test_redblue_optional_partial_analysis_gates_output(self):
        example = Path(__file__).resolve().parent.parent / "examples" / "branched.swc"
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "mesh.obj"
            command = [sys.executable, "-m", "anamorph", "generate-redblue", str(example), str(output),
                       "--force", "--angular-segments", "8", "--soma-refinements", "2"]
            good = subprocess.run(command+["--require-no-findings", "--check-adjacent-end-start", "--check-output"],
                                  capture_output=True, text=True, check=True)
            analysis = json.loads(good.stdout)["analysis"]
            self.assertFalse(analysis["complete"])
            self.assertEqual((analysis["findings"], analysis["unresolved"]), (0, 0))
            self.assertTrue(analysis["checked_adjacent_end_start"])
            self.assertFalse(json.loads(good.stdout)["output_check"]["certifies_nonintersection"])
            output.write_text("unchanged")
            limited = subprocess.run(command+["--require-no-findings", "--analysis-max-pairs", "0"],
                                     capture_output=True, text=True)
            self.assertEqual(limited.returncode, 1)
            self.assertIn("unresolved", limited.stderr)
            self.assertEqual(output.read_text(), "unchanged")
            crossing = Path(tmp) / "crossing.swc"
            crossing.write_text("1 1 -3 -3 0 .2 -1\n2 3 -2 0 0 .1 1\n3 3 2 0 0 .1 2\n"
                                "4 3 0 -2 0 .1 1\n5 3 0 2 0 .1 4\n")
            crossing_cmd = [sys.executable, "-m", "anamorph", "generate-redblue", str(crossing), str(output),
                            "--force", "--require-no-findings", "--no-precondition"]
            blocked = subprocess.run(crossing_cmd, capture_output=True, text=True)
            self.assertEqual(blocked.returncode, 1)
            self.assertIn("2 finding(s)", blocked.stderr)
            self.assertEqual(output.read_text(), "unchanged")
            rejected = subprocess.run(command+["--check-adjacent-end-start"], capture_output=True, text=True)
            self.assertEqual(rejected.returncode, 1)
            self.assertIn("require --require-no-findings", rejected.stderr)
            ignored_limit = subprocess.run(command+["--analysis-max-pairs", "0"], capture_output=True, text=True)
            self.assertEqual(ignored_limit.returncode, 1)
            self.assertIn("require --require-no-findings", ignored_limit.stderr)
            overlapping = Path(tmp) / "overlapping.swc"
            overlapping.write_text("1 1 0 0 0 1 -1\n2 1 .8 .13 .11 1 -1\n")
            bad_mesh = subprocess.run([sys.executable, "-m", "anamorph", "generate-redblue", str(overlapping),
                                       str(output), "--force", "--check-output", "--soma-refinements", "1"],
                                      capture_output=True, text=True)
            self.assertEqual(bad_mesh.returncode, 1)
            self.assertIn("self-intersection candidate", bad_mesh.stderr)
            self.assertEqual(output.read_text(), "unchanged")
            bad_network = subprocess.run([sys.executable, "-m", "anamorph", "generate-redblue", str(overlapping),
                                          str(output), "--force", "--require-no-findings"],
                                         capture_output=True, text=True)
            self.assertEqual(bad_network.returncode, 1)
            self.assertIn("1 finding(s)", bad_network.stderr)
            self.assertEqual(output.read_text(), "unchanged")

    def test_disconnected_soma_forest_retains_both_components(self):
        tree = Morphology([Node(1, 1, 0., 0., 0., 1., -1), Node(2, 1, 5., 0., 0., .5, -1)])
        mesh = assemble_redblue_neuron(tree, soma_refinements=1)
        stats = mesh.statistics()
        self.assertEqual((stats.vertices, stats.faces, stats.euler_characteristic), (84, 160, 4))
        self.assertEqual((stats.boundary_edges, stats.nonmanifold_edges), (0, 0))

    def test_individual_soma_and_path_remain_separate_closed_surfaces(self):
        tree = Morphology([Node(1, 1, 0, 0, 0, 1, -1), Node(2, 3, 1.3, 0, 0, .2, 1),
                           Node(3, 3, 2.5, 0, 0, .2, 2)])
        mesh = render_individual_surfaces(tree, angular_segments=8, soma_refinements=1)
        stats = mesh.statistics()
        self.assertEqual((stats.boundary_edges, stats.nonmanifold_edges, stats.euler_characteristic), (0, 0, 4))
        self.assertTrue(find_self_intersections(mesh)[0])  # Expected overlap between separate components.

    def test_exact_soma_surface_root_can_be_meshed(self):
        tree = Morphology([Node(1, 1, 0., 0., 0., 1., -1), Node(2, 3, 1., 0., 0., .2, 1),
                           Node(3, 3, 2.5, 0., 0., .2, 2)])
        mesh = assemble_redblue_neuron(tree, angular_segments=8, soma_refinements=2)
        stats = mesh.statistics()
        self.assertEqual((stats.boundary_edges, stats.nonmanifold_edges, stats.euler_characteristic), (0, 0, 2))
        self.assertEqual(find_self_intersections(mesh, include_adjacent=True)[0], [])
        with self.assertRaisesRegex(ValueError, "distinct"):
            render_individual_surfaces(tree, preserve_surface_roots=False, preconditioning=False)
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "root.swc", Path(tmp) / "mesh.obj"
            source.write_text("1 1 0 0 0 1 -1\n2 3 1 0 0 .2 1\n3 3 2.5 0 0 .2 2\n")
            command = [sys.executable, "-m", "anamorph", "generate-individual", str(source), str(output),
                       "--no-precondition", "--soma-refinements", "1"]
            subprocess.run(command, capture_output=True, text=True, check=True)
            result = subprocess.run(command+["--source-surface-roots"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("distinct", result.stderr)

    def test_interior_root_with_surface_child_can_be_meshed_without_preconditioning(self):
        tree = Morphology([Node(1, 1, 0., 0., 0., 1., -1), Node(2, 3, .5, 0., 0., .12, 1),
                           Node(3, 3, 1., 0., 0., .1, 2), Node(4, 3, 2., 0., 0., .1, 3)])
        mesh = assemble_redblue_neuron(tree, angular_segments=8, soma_refinements=2, preconditioning=False)
        stats = mesh.statistics()
        self.assertEqual((stats.boundary_edges, stats.nonmanifold_edges, stats.euler_characteristic), (0, 0, 2))
        self.assertEqual(find_self_intersections(mesh, include_adjacent=True)[0], [])
        render_individual_surfaces(tree, preconditioning=False, soma_refinements=1)
        with self.assertRaisesRegex(ValueError, "distinct"):
            render_individual_surfaces(tree, preconditioning=False, soma_refinements=1, preserve_surface_roots=False)

    def test_isolated_neurite_root_is_not_silently_dropped(self):
        tree = Morphology([Node(1, 1, 0., 0., 0., 1., -1), Node(2, 3, 1., 0., 0., .2, 1)])
        with self.assertRaisesRegex(ValueError, "no segment"):
            assemble_redblue_neuron(tree, angular_segments=8, soma_refinements=1)
        with self.assertRaisesRegex(ValueError, "no segment"):
            render_individual_surfaces(tree, angular_segments=8, soma_refinements=1)
        with self.assertRaises(PartialAnalysisError) as failure:
            assemble_redblue_checked(tree, angular_segments=8, soma_refinements=1)
        self.assertTrue(failure.exception.report.unresolved)

    def test_individual_surface_partition_filters_change_path_count(self):
        tree = Morphology([Node(1, 1, 0, 0, 0, 1, -1), Node(2, 3, 1.4, 0, 0, .2, 1),
                           Node(3, 3, 2.5, 0, 0, .2, 2), Node(4, 3, 3.5, 0, 0, .2, 3),
                           Node(5, 3, 2.5, 1, 0, .2, 3)])
        options = dict(angular_segments=8, soma_refinements=1, preconditioning=False)
        continued = render_individual_surfaces(tree, filter_angle=90., **options).statistics()
        separated = render_individual_surfaces(tree, filter_angle=0., **options).statistics()
        self.assertEqual((continued.euler_characteristic, separated.euler_characteristic), (6, 8))

    def test_individual_surfaces_cli_writes_obj(self):
        example = Path(__file__).resolve().parent.parent / "examples" / "branched.swc"
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "individual.obj"
            result = subprocess.run([sys.executable, "-m", "anamorph", "generate-individual", str(example),
                                     str(output), "--soma-refinements", "1", "--angular-segments", "8",
                                     "--filter-angle", "0"],
                                    text=True, capture_output=True, check=True)
            self.assertFalse(json.loads(result.stdout)["stitched"])
            self.assertGreater(len(read_obj(output).faces), 0)
            self.assertEqual(read_obj(output).statistics().euler_characteristic, 8)

    def test_triangulation_and_contact_failures_follow_numerical_retry(self):
        from anamorph import read_swc
        from anamorph.analytic_mesher import canal_path_mesh as real_canal_path_mesh
        example = Path(__file__).resolve().parent.parent / "examples" / "branched.swc"
        for error_type, last_iteration in ((RedBlueTriangulationError, False),
                                           (RedBlueNumericalEdgeCase, False),
                                           (RedBlueTriangulationError, True),
                                           (RedBlueNumericalEdgeCase, True)):
            with self.subTest(error_type=error_type, last_iteration=last_iteration):
                radii, calls = [], []
                def capture_stump(segments, **kwargs):
                    radii.append(segments[0].radius0)
                    return real_canal_path_mesh(segments, **kwargs)
                def fail_first_branch(red, blue, **kwargs):
                    calls.append(1)
                    if len(calls) == 2:
                        error = error_type("synthetic failure")
                        error._source_last_iteration = last_iteration
                        raise error
                    return redblue_union_with_retries(red, blue, **kwargs)
                with patch("anamorph.analytic_mesher.canal_path_mesh", side_effect=capture_stump), \
                     patch("anamorph.analytic_mesher.redblue_union_with_retries", side_effect=fail_first_branch):
                    mesh = assemble_redblue_neuron(read_swc(example), angular_segments=8, soma_refinements=2,
                                                   max_outer_attempts=32, max_inner_attempts=16,
                                                   source_retry_semantics=True)
                self.assertGreaterEqual(len(radii), 3)
                self.assertAlmostEqual(radii[1]-radii[2], int(last_iteration)*.01*read_swc(example).nodes[4].radius)
                self.assertEqual((mesh.statistics().boundary_edges, mesh.statistics().nonmanifold_edges), (0, 0))

    def test_outer_retry_preserves_only_intact_red_repairs(self):
        import anamorph.redblue as rb
        from anamorph.analytic_mesher import assemble_redblue_to_obj
        tree = Morphology([Node(1, 1, 0., 0., 0., 1., -1), Node(2, 3, 1.3, 0., 0., .2, 1),
                           Node(3, 3, 2.5, 0., 0., .2, 2)])
        original_circle, original_cut = rb.intersection_face_circles, rb._cut_side
        original_excluded = rb._excluded_components

        for mode in ("contact", "multiple_loops", "classify_contact", "classify_loops",
                     "classify_blue_contact", "classify_blue_loops",
                     "cut", "inner_limit", "flushed_contact"):
            seen, cut_calls, classify_calls = [], [], []
            def circle(red, blue, **kwargs):
                seen.append(red)
                if len(seen) == 1:
                    edges = rb._edge_faces(red, allow_boundary=True)
                    boundary = {vertex for edge, faces in edges.items() if len(faces) == 1 for vertex in edge}
                    edge = next(edge for edge, faces in edges.items() if len(faces) == 2 and
                                all(not boundary.intersection(red.faces[face]) for face in faces))
                    raise rb.RedBlueComplexEdges((rb.ComplexEdge(True, edge, (.25, .75)),))
                if len(seen) == 2 and mode in ("contact", "flushed_contact"):
                    raise rb.RedBlueNumericalEdgeCase("pre-cut contact")
                if len(seen) == 2 and mode == "multiple_loops":
                    raise rb.RedBlueMultipleLoops("pre-cut multiple loops")
                return original_circle(red, blue, **kwargs)
            def cut(*args, **kwargs):
                cut_calls.append(1)
                if mode == "cut" and len(cut_calls) == 1:
                    raise rb.RedBlueTriangulationError("cut-stage numerical failure")
                return original_cut(*args, **kwargs)
            def classify(*args, **kwargs):
                classify_calls.append(1)
                if ((len(classify_calls) == 1 and mode == "classify_contact") or
                        (len(classify_calls) == 2 and mode == "classify_blue_contact")):
                    raise rb.RedBlueNumericalEdgeCase("component contact before cutting")
                if ((len(classify_calls) == 1 and mode == "classify_loops") or
                        (len(classify_calls) == 2 and mode == "classify_blue_loops")):
                    raise rb.RedBlueMultipleLoops("component loop before cutting")
                return original_excluded(*args, **kwargs)
            settings = dict(angular_segments=8, soma_refinements=2, max_outer_attempts=8,
                            max_inner_attempts=1 if mode == "inner_limit" else 20)
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory, \
                    patch("anamorph.redblue.intersection_face_circles", side_effect=circle), \
                    patch("anamorph.redblue._cut_side", side_effect=cut), \
                    patch("anamorph.redblue._excluded_components", side_effect=classify):
                if mode == "flushed_contact":
                    path = Path(directory)/"streamed.obj"
                    info = assemble_redblue_to_obj(tree, path, flush_face_limit=1, **settings)
                    self.assertGreater(info["flushes"], 0)
                    mesh = read_obj(path)
                else:
                    mesh = assemble_redblue_neuron(tree, **settings)
                self.assertEqual((mesh.statistics().boundary_edges, mesh.statistics().nonmanifold_edges), (0, 0))
                self.assertFalse(find_self_intersections(mesh)[0])
                if mode == "inner_limit":
                    self.assertEqual(seen[1], seen[0])
                else:
                    self.assertEqual(len(seen[1].vertices), len(seen[0].vertices)+1)
                    self.assertEqual(seen[2], seen[0] if mode == "cut" else seen[1])

    def test_two_disconnected_cells_each_retain_neurite(self):
        tree = Morphology([Node(1,1,0,0,0,1,-1), Node(2,3,1.3,0,0,.2,1), Node(3,3,2.5,0,0,.2,2),
                           Node(4,1,6,0,0,1,-1), Node(5,3,7.3,0,0,.2,4), Node(6,3,8.5,0,0,.2,5)])
        mesh = assemble_redblue_neuron(tree, angular_segments=8, soma_refinements=1,
                                       max_outer_attempts=12, max_inner_attempts=16)
        stats = mesh.statistics()
        self.assertEqual((stats.boundary_edges, stats.nonmanifold_edges, stats.euler_characteristic), (0, 0, 4))
        self.assertGreater(max(p[0] for p in mesh.vertices), 8)
        self.assertEqual(find_self_intersections(mesh)[0], [])

    def test_redblue_multiple_branches(self):
        nodes = [Node(1, 1, 0, 0, 0, 1, -1)]
        for i in range(12):
            nodes.append(Node(i+2, 3, 1.25+i*.8, 0, 0, .16, 1 if i == 0 else i+1))
        next_id = 14
        for j, parent_id in enumerate((4, 6, 8, 10)):
            parent, x = parent_id, 1.25+(parent_id-2)*.8
            sign = -1 if j % 2 else 1
            for step in range(1, 4):
                nodes.append(Node(next_id, 3, x+.2*step, sign*.55*step, .15*step, .1, parent))
                parent, next_id = next_id, next_id+1
        mesh = assemble_redblue_neuron(Morphology(nodes), angular_segments=8, soma_refinements=2,
                                       max_outer_attempts=12, max_inner_attempts=16)
        stats = mesh.statistics()
        self.assertEqual((stats.boundary_edges, stats.nonmanifold_edges, stats.euler_characteristic), (0, 0, 2))
        hits, truncated = find_self_intersections(mesh)
        self.assertEqual(hits, [])
        self.assertFalse(truncated)


if __name__ == "__main__":
    unittest.main()
