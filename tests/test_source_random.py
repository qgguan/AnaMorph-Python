import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from math import dist, floor, pi
from itertools import product
from pathlib import Path

from anamorph import read_obj, read_swc
from anamorph.analytic_mesher import (PartialAnalysisError, assemble_redblue_checked, assemble_redblue_neuron,
                                      assemble_redblue_to_obj, render_individual_surfaces,
                                      _path_curves, _render_vector)
from anamorph.canal_mesh import canal_path_mesh, ico_sphere
from anamorph.geometry import order_paths_breadth_first, partition_paths
from anamorph.model import prepare_morphology
from anamorph.redblue import RedBlueComplexEdges, redblue_union
from anamorph.mesh_intersections import find_self_intersections
from anamorph.source_random import WindowsCRandom, make_random

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / "reference-build" / "individual_probe.exe"
OPTIONS_PROBE = ROOT / "reference-build" / "individual_options_probe.exe"
CELLGEN = ROOT / "reference-build" / "source" / "bin" / "am_cellgen.exe"


class SourceRandomTests(unittest.TestCase):
    def test_complex_edge_repairs_follow_source_edge_order(self):
        tree = prepare_morphology(read_swc(ROOT/"tests"/"fixtures"/"source_ear_branch_0.swc")).morphology
        path = order_paths_breadth_first(tree, partition_paths(tree))[0]
        _, _, root, curves = _path_curves(tree, path, "chord-length")
        rng = WindowsCRandom(0)
        frame = _render_vector(curves[1:] if root else curves, rng=rng)
        angle = rng.uniform(0., 2*pi/8)
        soma = next(node for node in tree.nodes.values() if node.kind == 1)
        sphere = ico_sphere(soma.position, soma.radius, 2)
        stump = canal_path_mesh(curves[:2], angular_segments=8, render_vector=frame, phi0=angle,
                                return_boundary=True)
        with self.assertRaises(RedBlueComplexEdges) as caught:
            redblue_union(sphere, stump.mesh)
        self.assertEqual([(edge.red, edge.edge) for edge in caught.exception.edges],
                         [(True, (111, 161)), (True, (160, 161))])

    def test_windows_reference_sequence_and_local_state(self):
        expected = ((0, (38, 7719, 21238, 2437, 8855, 11797)),
                    (1, (41, 18467, 6334, 26500, 19169, 15724)),
                    (12345, (7584, 19164, 25795, 22125, 5828, 23405)))
        for seed, values in expected:
            rng = WindowsCRandom(seed)
            self.assertEqual(tuple(rng.rand() for _ in values), values)
        first, second = WindowsCRandom(-1), WindowsCRandom(0xffffffff)
        self.assertEqual([first.rand() for _ in range(10)], [second.rand() for _ in range(10)])
        with self.assertRaises(ValueError):
            make_random(0, "other")

    def test_cli_individual_surface_uses_selected_generator(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory)/"case.swc", Path(directory)/"individual.obj"
            source.write_text("1 1 0 0 0 1 -1\n2 3 0 0 1.25 .2 1\n"
                              "3 3 0 0 2.25 .18 2\n4 3 .05 .03 3.25 .16 3\n", encoding="utf-8")
            command = [sys.executable, "-m", "anamorph", "generate-individual", str(source), str(target),
                       "--angular-segments", "8", "--random-generator", "windows-c",
                       "--quad-sqrt-mode", "windows-x87", "--render-trig-mode", "windows-x87",
                       "--source-render-vector", "--arc-length-mode", "source-x87"]
            subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
            expected = render_individual_surfaces(read_swc(source), angular_segments=8, random_generator="windows-c",
                                                  quad_sqrt_mode="windows-x87", render_trig_mode="windows-x87",
                                                  source_render_vector=True, arc_length_mode="source-x87")
            actual = read_obj(target)
            self.assertEqual(actual.faces, expected.faces)
            self.assertLess(max(dist(a, b) for a, b in zip(actual.vertices, expected.vertices)), 1e-12)

    def test_cli_redblue_uses_selected_generator(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory)/"case.swc", Path(directory)/"mesh.obj"
            source.write_text("1 1 0 0 0 1 -1\n2 3 0 0 1.25 .2 1\n"
                              "3 3 0 0 2.25 .18 2\n4 3 .05 .03 3.25 .16 3\n", encoding="utf-8")
            command = [sys.executable, "-m", "anamorph", "generate-redblue", str(source), str(target),
                       "--force", "--angular-segments", "8", "--soma-refinements", "2",
                       "--random-generator", "windows-c", "--source-render-vector", "--arc-length-mode", "source-x87"]
            subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
            expected = assemble_redblue_neuron(read_swc(source), angular_segments=8, soma_refinements=2,
                                               random_generator="windows-c", source_render_vector=True,
                                               arc_length_mode="source-x87")
            actual = read_obj(target)
            self.assertEqual(actual.faces, expected.faces)
            self.assertLess(max(dist(a, b) for a, b in zip(actual.vertices, expected.vertices)), 1e-12)

    def test_cli_source_ear_clipping(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = ROOT/"examples"/"branched.swc", Path(directory)/"mesh.obj"
            command = [sys.executable, "-m", "anamorph", "generate-redblue", str(source), str(target),
                       "--force", "--angular-segments", "8", "--soma-refinements", "2",
                       "--random-generator", "windows-c", "--source-ear-clipping", "--quad-sqrt-mode", "windows-x87",
                       "--render-trig-mode", "windows-x87", "--source-retry-semantics"]
            subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
            expected = assemble_redblue_neuron(read_swc(source), angular_segments=8, soma_refinements=2,
                                               random_generator="windows-c", source_ear_clipping=True,
                                               quad_sqrt_mode="windows-x87", render_trig_mode="windows-x87",
                                               source_retry_semantics=True)
            actual = read_obj(target)
            self.assertEqual(actual.faces, expected.faces)
            self.assertLess(max(dist(a, b) for a, b in zip(actual.vertices, expected.vertices)), 1e-12)

    def test_cli_source_delaunay(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = ROOT/"examples"/"branched.swc", Path(directory)/"mesh.obj"
            command = [sys.executable, "-m", "anamorph", "generate-redblue", str(source), str(target),
                       "--force", "--angular-segments", "8", "--soma-refinements", "2",
                       "--random-generator", "windows-c", "--source-delaunay"]
            subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
            expected = assemble_redblue_neuron(read_swc(source), angular_segments=8, soma_refinements=2,
                                               random_generator="windows-c", source_delaunay=True)
            actual = read_obj(target)
            self.assertEqual(actual.faces, expected.faces)
            self.assertLess(max(dist(a, b) for a, b in zip(actual.vertices, expected.vertices)), 1e-12)

    def test_cli_source_delaunay_allow_limit_implies_source_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = ROOT/"examples"/"branched.swc", Path(directory)/"mesh.obj"
            command = [sys.executable, "-m", "anamorph", "generate-redblue", str(source), str(target),
                       "--force", "--angular-segments", "8", "--soma-refinements", "2",
                       "--random-generator", "windows-c", "--source-delaunay-allow-limit"]
            run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
            self.assertNotIn("SourceDelaunayLimitWarning", run.stderr)
            expected = assemble_redblue_neuron(read_swc(source), angular_segments=8, soma_refinements=2,
                                               random_generator="windows-c", source_delaunay=True)
            self.assertEqual(read_obj(target), expected)

    def test_source_frame_cli_matches_generation_apis(self):
        with tempfile.TemporaryDirectory() as directory:
            source = ROOT/"examples"/"branched.swc"
            for command_name, generate in (("generate-redblue", assemble_redblue_neuron),
                                           ("generate-individual", render_individual_surfaces)):
                with self.subTest(command=command_name):
                    target = Path(directory)/(command_name+".obj")
                    command = [sys.executable, "-m", "anamorph", command_name, str(source), str(target),
                               "--angular-segments", "8", "--soma-refinements", "2", "--source-frame",
                               "--quad-sqrt-mode", "windows-x87", "--render-trig-mode", "windows-x87"]
                    if command_name == "generate-redblue":
                        command.append("--force")
                    subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True, timeout=30)
                    expected = generate(read_swc(source), angular_segments=8, soma_refinements=2,
                                        source_frame=True, quad_sqrt_mode="windows-x87",
                                        render_trig_mode="windows-x87")
                    actual = read_obj(target)
                    self.assertEqual(actual.faces, expected.faces)
                    self.assertLess(max(dist(a, b) for a, b in zip(actual.vertices, expected.vertices)), 1e-12)

    def test_source_ear_streamed_mesh_matches_full_assembly(self):
        tree = read_swc(ROOT/"examples"/"branched.swc")
        settings = dict(angular_segments=8, soma_refinements=2,
                        random_generator="windows-c", source_ear_clipping=True, quad_sqrt_mode="windows-x87",
                        render_trig_mode="windows-x87", source_render_vector=True, arc_length_mode="source-x87",
                        source_delaunay=True, source_frame=True)
        full = assemble_redblue_neuron(tree, **settings)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/"stream.obj"
            report = assemble_redblue_to_obj(tree, target, flush_face_limit=1, **settings)
            streamed = read_obj(target)
            def faces(mesh):
                return Counter(tuple(mesh.vertices[i] for i in face) for face in mesh.faces)
            self.assertGreater(report["flushes"], 0)
            self.assertEqual(faces(full), faces(streamed))

    @unittest.skipUnless(sys.platform == "win32" and PROBE.is_file(), "optional C++ individual probe unavailable")
    def test_vertical_individual_surface_matches_cpp_windows_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory)/"case.swc", Path(directory)/"reference"
            source.write_text("1 1 0 0 0 1 -1\n2 3 0 0 1.25 .2 1\n"
                              "3 3 0 0 2.25 .18 2\n4 3 .05 .03 3.25 .16 3\n", encoding="utf-8")
            subprocess.run([str(PROBE), str(source), str(output), "1"], cwd=directory,
                           capture_output=True, text=True, check=True, timeout=30)
            reference = read_obj(output.with_suffix(".obj"))
            model = render_individual_surfaces(read_swc(source), angular_segments=8,
                                               random_generator="windows-c", source_render_vector=True)
            self.assertEqual((len(reference.vertices), len(reference.faces)), (len(model.vertices), len(model.faces)))
            self.assertLess(max(min(dist(p, q) for q in model.vertices) for p in reference.vertices), 1e-8)
            self.assertLess(max(min(dist(p, q) for q in reference.vertices) for p in model.vertices), 1e-8)

    @unittest.skipUnless(sys.platform == "win32" and OPTIONS_PROBE.is_file(), "C++ parametrization probe unavailable")
    def test_source_curve_individual_surfaces_match_cpp(self):
        for filename in ("examples/branched.swc", "tests/fixtures/source_spline_adjacent.swc"):
            for parameter in ("chord-length", "centripetal", "uniform"):
                with self.subTest(filename=filename, parameter=parameter), tempfile.TemporaryDirectory() as directory:
                    source, output = ROOT / filename, Path(directory) / "reference"
                    subprocess.run([str(OPTIONS_PROBE), str(source), str(output), "0", parameter],
                                   cwd=directory, capture_output=True, check=True, timeout=30)
                    reference = read_obj(output.with_suffix(".obj"))
                    model = render_individual_surfaces(read_swc(source), angular_segments=8,
                                                       preconditioning=False, random_generator="windows-c",
                                                       parametrization=parameter, source_render_vector=True,
                                                       arc_length_mode="source-x87")
                    self.assertEqual((len(reference.vertices), len(reference.faces)),
                                     (len(model.vertices), len(model.faces)))
                    self.assertLess(max(min(dist(p, q) for q in model.vertices) for p in reference.vertices), 1e-8)
                    self.assertLess(max(min(dist(p, q) for q in reference.vertices) for p in model.vertices), 1e-8)

    @unittest.skipUnless(sys.platform == "win32" and CELLGEN.is_file(), "optional C++ cell generator unavailable")
    def test_vertical_stitched_surface_matches_cpp_windows_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/"case.swc"
            source.write_text("1 1 0 0 0 1 -1\n2 3 0 0 1.25 .2 1\n"
                              "3 3 0 0 2.25 .18 2\n4 3 .05 .03 3.25 .16 3\n", encoding="utf-8")
            subprocess.run([str(CELLGEN), "-i", "case", "-force-meshing", "-no-mesh-pp",
                            "-no-meshing-flush", "-meshing-soma-refs", "2",
                            "-meshing-cansurf-angularsegments", "8", "-ana-nthreads", "1"],
                           cwd=directory, capture_output=True, text=True, check=True, timeout=30)
            reference = read_obj(Path(directory)/"case.obj")
            model = assemble_redblue_neuron(read_swc(source), angular_segments=8, soma_refinements=2,
                                             random_generator="windows-c")
            self.assertEqual((len(reference.vertices), len(reference.faces)), (len(model.vertices), len(model.faces)))
            self.assertLess(max(min(dist(p, q) for q in model.vertices) for p in reference.vertices), 1e-8)
            self.assertLess(max(min(dist(p, q) for q in reference.vertices) for p in model.vertices), 1e-8)

    @unittest.skipUnless(sys.platform == "win32" and CELLGEN.is_file(), "optional C++ cell generator unavailable")
    def test_branching_stitched_surface_matches_cpp_source_ear_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/"case.swc"
            source.write_bytes((ROOT/"examples"/"branched.swc").read_bytes())
            subprocess.run([str(CELLGEN), "-i", "case", "-force-meshing", "-no-mesh-pp",
                            "-no-meshing-flush", "-meshing-soma-refs", "2",
                            "-meshing-cansurf-angularsegments", "8", "-ana-nthreads", "1"],
                           cwd=directory, capture_output=True, text=True, check=True, timeout=30)
            reference = read_obj(Path(directory)/"case.obj")
            model = assemble_redblue_neuron(read_swc(source), angular_segments=8, soma_refinements=2,
                                             random_generator="windows-c", source_ear_clipping=True,
                                             source_render_vector=True, arc_length_mode="source-x87",
                                             source_delaunay=True)
            self.assertEqual((len(reference.vertices), len(reference.faces)), (len(model.vertices), len(model.faces)))
            self.assertLess(max(min(dist(p, q) for q in model.vertices) for p in reference.vertices), 1e-8)
            self.assertLess(max(min(dist(p, q) for q in reference.vertices) for p in model.vertices), 1e-8)

    @unittest.skipUnless(sys.platform == "win32" and CELLGEN.is_file(), "optional C++ cell generator unavailable")
    def test_complex_edge_repair_cases_match_cpp_source_ear_mode(self):
        for case in (0, 11):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                source = Path(directory)/"case.swc"
                source.write_bytes((ROOT/"tests"/"fixtures"/f"source_ear_branch_{case}.swc").read_bytes())
                subprocess.run([str(CELLGEN), "-i", "case", "-force-meshing", "-no-mesh-pp",
                                "-no-meshing-flush", "-meshing-soma-refs", "2",
                                "-meshing-cansurf-angularsegments", "8", "-ana-nthreads", "1"],
                               cwd=directory, capture_output=True, text=True, check=True, timeout=30)
                reference = read_obj(Path(directory)/"case.obj")
                model = assemble_redblue_neuron(read_swc(source), angular_segments=8, soma_refinements=2,
                                                 random_generator="windows-c", source_ear_clipping=True)
                self.assertEqual((len(reference.vertices), len(reference.faces)),
                                 (len(model.vertices), len(model.faces)))
                self.assertLess(max(min(dist(p, q) for q in model.vertices) for p in reference.vertices), 1e-8)
                self.assertLess(max(min(dist(p, q) for q in reference.vertices) for p in model.vertices), 1e-8)

    @unittest.skipUnless(sys.platform == "win32" and CELLGEN.is_file(), "optional C++ cell generator unavailable")
    def test_final_inner_iteration_retry_matches_cpp(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/"case.swc"
            source.write_bytes((ROOT/"tests"/"fixtures"/"source_retry_limit.swc").read_bytes())
            subprocess.run([str(CELLGEN), "-i", "case", "-force-meshing", "-no-mesh-pp",
                            "-no-meshing-flush", "-meshing-soma-refs", "2",
                            "-meshing-cansurf-angularsegments", "8", "-ana-nthreads", "1"],
                           cwd=directory, capture_output=True, text=True, check=True, timeout=30)
            reference = read_obj(Path(directory)/"case.obj")
            settings = dict(angular_segments=8, soma_refinements=2, random_generator="windows-c",
                            source_ear_clipping=True, quad_sqrt_mode="windows-x87", render_trig_mode="windows-x87")
            model = assemble_redblue_neuron(read_swc(source), source_retry_semantics=True, **settings)
            self.assertEqual((len(reference.vertices), len(reference.faces)), (len(model.vertices), len(model.faces)))
            self.assertLess(max(min(dist(p, q) for q in model.vertices) for p in reference.vertices), 1e-8)
            self.assertLess(max(min(dist(p, q) for q in reference.vertices) for p in model.vertices), 1e-8)
            default = assemble_redblue_neuron(read_swc(source), **settings)
            self.assertNotEqual(len(default.faces), len(reference.faces))

    @unittest.skipUnless(sys.platform == "win32" and CELLGEN.is_file(), "optional C++ cell generator unavailable")
    def test_source_ear_cleanup_matches_cpp_but_gate_blocks_unclean_case(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/"case.swc"
            source.write_bytes((ROOT/"tests"/"fixtures"/"source_ear_collinear.swc").read_bytes())
            subprocess.run([str(CELLGEN), "-i", "case", "-force-meshing", "-no-mesh-pp",
                            "-no-meshing-flush", "-meshing-soma-refs", "2",
                            "-meshing-cansurf-angularsegments", "8", "-ana-nthreads", "1"],
                           cwd=directory, capture_output=True, text=True, check=True, timeout=30)
            reference = read_obj(Path(directory)/"case.obj")
            tree = read_swc(source)
            settings = dict(angular_segments=8, soma_refinements=2, random_generator="windows-c",
                            source_ear_clipping=True, quad_sqrt_mode="windows-x87", render_trig_mode="windows-x87",
                            source_retry_semantics=True)
            model = assemble_redblue_neuron(tree, **settings)
            self.assertEqual((len(reference.vertices), len(reference.faces)), (len(model.vertices), len(model.faces)))
            self.assertLess(max(min(dist(p, q) for q in model.vertices) for p in reference.vertices), 1e-8)
            for mesh in (model, reference):
                hits, _ = find_self_intersections(mesh, include_adjacent=True, max_hits=1)
                self.assertTrue(hits)
            with self.assertRaises(PartialAnalysisError):
                assemble_redblue_checked(tree, analysis_job_mode="source", analysis_gate="source", **settings)
            target = Path(directory)/"checked.obj"
            command = [sys.executable, "-m", "anamorph", "generate-redblue", str(source), str(target),
                       "--force", "--check-output", "--angular-segments", "8", "--soma-refinements", "2",
                       "--random-generator", "windows-c", "--source-ear-clipping", "--quad-sqrt-mode", "windows-x87",
                       "--render-trig-mode", "windows-x87", "--source-retry-semantics"]
            checked = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(checked.returncode, 0)
            self.assertFalse(target.exists())

    @unittest.skipUnless(sys.platform == "win32" and CELLGEN.is_file(), "optional C++ cell generator unavailable")
    def test_many_branches_streamed_matches_cpp_source_ear_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/"case.swc"
            source.write_bytes((ROOT/"examples"/"many_branches.swc").read_bytes())
            subprocess.run([str(CELLGEN), "-i", "case", "-force-meshing", "-no-cellnet-pc", "-no-mesh-pp",
                            "-no-meshing-flush", "-meshing-soma-refs", "2",
                            "-meshing-cansurf-angularsegments", "6", "-ana-nthreads", "1"],
                           cwd=directory, capture_output=True, text=True, check=True, timeout=90)
            reference = read_obj(Path(directory)/"case.obj")
            target = Path(directory)/"python.obj"
            report = assemble_redblue_to_obj(read_swc(source), target, angular_segments=6, soma_refinements=2,
                                             preconditioning=False, random_generator="windows-c",
                                             source_delaunay=True, source_render_vector=True,
                                             source_retry_semantics=True, arc_length_mode="source-x87",
                                             quad_sqrt_mode="windows-x87", render_trig_mode="windows-x87",
                                             flush_face_limit=1000, max_outer_attempts=30)
            model = read_obj(target)
            self.assertGreater(report["flushes"], 0)
            self.assertEqual((len(reference.vertices), len(reference.faces)), (len(model.vertices), len(model.faces)))
            tolerance = 1e-8
            def point_map(first, second):
                bins = {}
                for index, p in enumerate(second):
                    key = tuple(floor(x/tolerance) for x in p)
                    bins.setdefault(key, []).append((index, p))
                mapping = []
                for point in first:
                    key = tuple(floor(x/tolerance) for x in point)
                    candidates = (candidate for offset in product((-1, 0, 1), repeat=3)
                                  for candidate in bins.get(tuple(key[i]+offset[i] for i in range(3)), ()))
                    match = next((index for index, candidate in candidates if dist(point, candidate) < tolerance), None)
                    if match is None:
                        return None
                    mapping.append(match)
                return mapping
            mapping = point_map(reference.vertices, model.vertices)
            self.assertIsNotNone(mapping)
            self.assertIsNotNone(point_map(model.vertices, reference.vertices))
            self.assertEqual(len(set(mapping)), len(mapping))
            def oriented(face):
                a, b, c = face
                return min((a, b, c), (b, c, a), (c, a, b))
            reference_faces = Counter(oriented(tuple(mapping[v] for v in face)) for face in reference.faces)
            model_faces = Counter(oriented(face) for face in model.faces)
            self.assertEqual(reference_faces, model_faces)


if __name__ == "__main__":
    unittest.main()
