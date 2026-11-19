import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import unittest

from anamorph import taubin_smooth
from anamorph.canal_mesh import ico_sphere
from anamorph.mesh import Mesh, read_obj


ROOT = Path(__file__).resolve().parent.parent


class TaubinSmoothingTests(unittest.TestCase):
    def test_smooths_radial_irregularities_without_changing_connectivity(self):
        sphere = ico_sphere((0., 0., 0.), 1., 2)
        irregular = Mesh(tuple(tuple(x * (1.12 if i % 2 else .88) for x in p)
                               for i, p in enumerate(sphere.vertices)), sphere.faces)
        result = taubin_smooth(irregular, iterations=5)
        radial_spread = lambda mesh: statistics.pstdev(math.dist(p, (0., 0., 0.)) for p in mesh.vertices)
        self.assertLess(radial_spread(result), radial_spread(irregular))
        self.assertEqual(result.faces, irregular.faces)
        self.assertEqual(len(result.vertices), len(irregular.vertices))
        self.assertNotEqual(result.vertices, irregular.vertices)

    def test_translation_invariance_and_isolated_vertex(self):
        sphere = ico_sphere((0., 0., 0.), 1., 1)
        mesh = Mesh(sphere.vertices + ((8., 8., 8.),), sphere.faces)
        shift = (2., -3., 4.)
        translated = Mesh(tuple(tuple(x+d for x, d in zip(p, shift)) for p in mesh.vertices), mesh.faces)
        original_result = taubin_smooth(mesh, iterations=2)
        shifted_result = taubin_smooth(translated, iterations=2)
        self.assertEqual(original_result.vertices[-1], mesh.vertices[-1])
        for original, shifted in zip(original_result.vertices, shifted_result.vertices):
            for a, b, d in zip(original, shifted, shift):
                self.assertAlmostEqual(a+d, b)
        self.assertEqual(taubin_smooth(mesh, iterations=0).vertices, mesh.vertices)

    def test_rejects_invalid_parameters(self):
        mesh = ico_sphere((0., 0., 0.), 1., 0)
        for options in ({"lambda_factor": 0.}, {"lambda_factor": float("nan")},
                        {"mu_factor": 0.}, {"mu_factor": -1.1},
                        {"iterations": -1}, {"iterations": 1.5}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                taubin_smooth(mesh, **options)

    def test_postprocess_cli_taubin_only_keeps_input(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory)/"raw.obj", Path(directory)/"smooth.obj"
            ico_sphere((0., 0., 0.), 1., 1).write_obj(source)
            original = source.read_bytes()
            result = subprocess.run([sys.executable, "-m", "anamorph", "postprocess", str(source), str(output),
                                     "--taubin-iterations", "3", "--taubin-lambda", ".4", "--taubin-mu", "-.5"],
                                    cwd=ROOT, capture_output=True, text=True, timeout=60, check=True)
            info = json.loads(result.stdout)
            self.assertEqual(info["postprocess_settings"]["taubin_smoothing"],
                             {"iterations": 3, "lambda_factor": .4, "mu_factor": -.5})
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(read_obj(source).faces, read_obj(output).faces)
            self.assertNotEqual(read_obj(source).vertices, read_obj(output).vertices)

    def test_workflow_selects_taubin_and_records_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/"result"
            result = subprocess.run([sys.executable, "-m", "anamorph", "workflow", str(ROOT/"examples"/"no_soma.swc"),
                                     "--output-dir", str(target), "--smoothing", "taubin", "--taubin-iterations", "3",
                                     "--skip-intersection-check"], cwd=ROOT, capture_output=True, text=True, timeout=60, check=True)
            info = json.loads(result.stdout)
            report = json.loads((target/"report.json").read_text(encoding="utf-8"))
            settings = report["settings"]["remeshing"]
            self.assertEqual(settings["taubin_smoothing"], {"lambda_factor": .5, "mu_factor": -.53, "iterations": 3})
            self.assertNotIn("hc_smoothing", settings)
            self.assertIn("Taubin smoothing", (target/"report.md").read_text(encoding="utf-8"))
            self.assertTrue(Path(info["raw_mesh"]).exists())
            self.assertTrue(Path(info["remeshed_mesh"]).exists())

    def test_remesh_cli_selects_taubin_after_collapse(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory)/"raw.obj", Path(directory)/"processed.obj"
            ico_sphere((0., 0., 0.), 1., 1).write_obj(source)
            result = subprocess.run([sys.executable, "-m", "anamorph", "remesh", str(source), str(output),
                                     "--smoothing", "taubin", "--taubin-iterations", "2"],
                                    cwd=ROOT, capture_output=True, text=True, timeout=60, check=True)
            info = json.loads(result.stdout)
            self.assertEqual(info["postprocess_settings"]["taubin_smoothing"]["iterations"], 2)
            self.assertIn("edge_collapse", info["postprocess_settings"])
            self.assertNotIn("hc_smoothing", info["postprocess_settings"])
            self.assertTrue(output.is_file())

    def test_taubin_can_follow_isotropic_remeshing(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory)/"raw.obj", Path(directory)/"processed.obj"
            ico_sphere((0., 0., 0.), 1., 1).write_obj(source)
            result = subprocess.run([sys.executable, "-m", "anamorph", "remesh", str(source), str(output),
                                     "--isotropic", "--target-edge-length", ".5", "--iterations", "1",
                                     "--smoothing", "taubin", "--taubin-iterations", "2"], cwd=ROOT,
                                    capture_output=True, text=True, timeout=60, check=True)
            settings = json.loads(result.stdout)["postprocess_settings"]
            self.assertEqual(settings["iterations"], 1)
            self.assertEqual(settings["taubin_smoothing"]["iterations"], 2)

    def test_workflow_records_taubin_after_isotropic_remeshing(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/"result"
            subprocess.run([sys.executable, "-m", "anamorph", "workflow", str(ROOT/"examples"/"no_soma.swc"),
                            "--output-dir", str(target), "--isotropic", "--target-edge-length", ".15",
                            "--iterations", "1", "--smoothing", "taubin", "--taubin-iterations", "2",
                            "--skip-intersection-check"], cwd=ROOT, capture_output=True, text=True, timeout=60, check=True)
            settings = json.loads((target/"report.json").read_text(encoding="utf-8"))["settings"]["remeshing"]
            self.assertEqual(settings["method"], "isotropic")
            self.assertEqual(settings["taubin_smoothing"]["iterations"], 2)
            self.assertIn("Taubin smoothing", (target/"report.md").read_text(encoding="utf-8"))

    def test_incompatible_options_fail_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/"result"
            for extra in (["--smoothing", "hc", "--taubin-iterations", "3"],
                          ["--smoothing", "none", "--taubin-mu", "-.5"]):
                result = subprocess.run([sys.executable, "-m", "anamorph", "workflow", str(ROOT/"examples"/"no_soma.swc"),
                                         "--output-dir", str(target), *extra], cwd=ROOT, capture_output=True, text=True, timeout=30)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
