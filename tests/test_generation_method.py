import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from anamorph.mesh import read_obj


ROOT = Path(__file__).resolve().parent.parent


class GenerationMethodTests(unittest.TestCase):
    def run_command(self, *args):
        return subprocess.run([sys.executable, "-m", "anamorph", *map(str, args)], cwd=ROOT,
                              text=True, capture_output=True, timeout=60)

    def test_generate_defaults_to_analytic_for_soma_and_soma_free_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            for filename in ("branched.swc", "no_soma.swc"):
                with self.subTest(filename=filename):
                    source = ROOT/"examples"/filename
                    output = Path(directory)/(filename+".obj")
                    reference = Path(directory)/(filename+"-alias.obj")
                    result = self.run_command("generate", source, output, "--angular-segments", "8",
                                              "--soma-refinements", "2", "--check-output")
                    self.assertEqual(result.returncode, 0, result.stderr)
                    info = json.loads(result.stdout)
                    self.assertEqual(info["method"], "analytic")
                    self.assertFalse(info["analysis_performed"])
                    self.assertEqual(info["output_check"]["intersection_candidates"], 0)
                    alias = self.run_command("generate-redblue", source, reference, "--angular-segments", "8",
                                             "--soma-refinements", "2")
                    self.assertEqual(alias.returncode, 0, alias.stderr)
                    self.assertEqual(json.loads(alias.stdout)["method"], "analytic")
                    self.assertEqual(read_obj(output), read_obj(reference))

    def test_streamed_generate_reports_analytic_method(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"streamed.obj"
            result = self.run_command("generate", ROOT/"examples"/"no_soma.swc", output,
                                      "--flush-face-limit", "1")
            self.assertEqual(result.returncode, 0, result.stderr)
            info = json.loads(result.stdout)
            self.assertEqual(info["method"], "analytic")
            self.assertGreater(len(read_obj(output).faces), 0)

    def test_voxel_requires_an_explicit_command(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/"point.swc"
            source.write_text("1 3 0 0 0 .2 -1\n", encoding="utf-8")
            output = Path(directory)/"mesh.obj"
            analytic = self.run_command("generate", source, output)
            self.assertNotEqual(analytic.returncode, 0)
            self.assertIn("no segment to render", analytic.stderr)
            self.assertFalse(output.exists())
            voxel = self.run_command("generate-voxel", source, output, "--spacing", ".1")
            self.assertEqual(voxel.returncode, 0, voxel.stderr)
            self.assertEqual(json.loads(voxel.stdout)["method"], "voxel")
            self.assertGreater(len(read_obj(output).faces), 0)

    def test_voxel_switches_do_not_implicitly_change_method(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"mesh.obj"
            result = self.run_command("generate", ROOT/"examples"/"branched.swc", output, "--spacing", ".2")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("use generate-voxel explicitly", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
