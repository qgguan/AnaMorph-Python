import json
import math
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from anamorph import create_morphview, read_morphview, read_swc, write_morphview
from anamorph.bernstein import ClippingConvergenceError

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / "reference-build" / "morphview_probe.exe"


class MorphViewTests(unittest.TestCase):
    def test_roundtrip_and_bounding_boxes(self):
        model = create_morphview(read_swc(ROOT / "examples" / "branched.swc"))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model.amv"
            write_morphview(model, target)
            self.assertEqual(read_morphview(target), model)
            for path in model.paths:
                self.assertTrue(all(math.isfinite(x) for x in (*path.box_min, *path.box_size)))
                for segment in path.segments:
                    for i in range(3):
                        self.assertLessEqual(path.box_min[i], segment.box_min[i])
                        self.assertGreaterEqual(path.box_min[i] + path.box_size[i],
                                                segment.box_min[i] + segment.box_size[i] - 1e-12)

    def test_flags_and_incomplete_analysis(self):
        tree = read_swc(ROOT / "tests" / "fixtures" / "branch_curvature_collision.swc")
        self.assertEqual(create_morphview(tree).summary()["flag_counts"].get("lsi"), 2)
        with self.assertRaisesRegex(ValueError, "analysis jobs"):
            create_morphview(tree, max_jobs=1)
        with patch("anamorph.source_analysis.source_regularity_candidates",
                   side_effect=ClippingConvergenceError("injected limit")):
            with self.assertRaisesRegex(ValueError, "analysis jobs"):
                create_morphview(tree)

    def test_suppressed_global_failure_cannot_be_exported_as_clear(self):
        tree = read_swc(ROOT / "examples" / "branched.swc")
        with self.assertRaisesRegex(ValueError, "unresolved global self-intersection"):
            create_morphview(tree, bivariate_tolerance=2.)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model.amv"
            target.write_bytes(b"existing")
            run = subprocess.run([sys.executable, "-m", "anamorph", "export-amv",
                                  str(ROOT / "examples" / "branched.swc"), str(target),
                                  "--bivariate-tolerance", "2"], cwd=ROOT, text=True, capture_output=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("unresolved global self-intersection", run.stderr)
            self.assertEqual(target.read_bytes(), b"existing")

    def test_invalid_and_windows_text_mode_files(self):
        model = create_morphview(read_swc(ROOT / "examples" / "branched.swc"))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model.amv"
            write_morphview(model, target)
            data = target.read_bytes()
            self.assertIn(b"\n", data)
            for invalid in (data[:0], data[:10], data[:-1], data + b"x", data[:36] + b"\x02" + data[37:]):
                target.write_bytes(invalid)
                with self.assertRaises(ValueError):
                    read_morphview(target)
            target.write_bytes(data.replace(b"\n", b"\r\n"))
            self.assertEqual(read_morphview(target, windows_text_mode=True), model)

    def test_invalid_write_does_not_replace_existing_file(self):
        model = create_morphview(read_swc(ROOT / "examples" / "branched.swc"))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model.amv"
            target.write_bytes(b"existing")
            with self.assertRaises(ValueError):
                write_morphview(replace(model, soma_radius=-1), target)
            self.assertEqual(target.read_bytes(), b"existing")
            self.assertEqual(list(Path(directory).glob(".anamorph-*.amv")), [])

    def test_cli_export_inspect_and_failed_export(self):
        source = ROOT / "examples" / "branched.swc"
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model.amv"
            command = [sys.executable, "-m", "anamorph"]
            exported = subprocess.run(command + ["export-amv", str(source), str(target)], cwd=ROOT,
                                      text=True, capture_output=True, check=True)
            report = json.loads(exported.stdout)
            self.assertEqual(report["segments"], 5)
            self.assertTrue(report["first_soma_only"])
            self.assertFalse(report["certifies_consistency"])
            inspected = subprocess.run(command + ["inspect-amv", str(target)], cwd=ROOT,
                                       text=True, capture_output=True, check=True)
            self.assertEqual(json.loads(inspected.stdout)["flag_counts"], report["flag_counts"])
            target.write_bytes(target.read_bytes().replace(b"\n", b"\r\n"))
            inspected = subprocess.run(command + ["inspect-amv", str(target), "--windows-text-mode"], cwd=ROOT,
                                       text=True, capture_output=True, check=True)
            self.assertEqual(json.loads(inspected.stdout)["segments"], report["segments"])
            target.write_bytes(b"existing")
            failed = subprocess.run(command + ["export-amv", str(source), str(target), "--max-jobs", "1"],
                                    cwd=ROOT, text=True, capture_output=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual(target.read_bytes(), b"existing")

    def test_isolated_root_is_not_reported_as_analyzed(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "isolated.swc"
            source.write_text("1 1 0 0 0 1 -1\n2 3 1 0 0 .2 1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no segment"):
                create_morphview(read_swc(source), preconditioning=False, preserve_surface_roots=True)

    @unittest.skipUnless(sys.platform == "win32" and PROBE.is_file(), "C++ MorphView probe unavailable")
    def test_cpp_reference_writer(self):
        for name in ("examples/branched.swc", "examples/mixed_soma_roots.swc",
                     "tests/fixtures/branch_curvature_collision.swc"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                source = ROOT / name
                target = Path(directory) / "reference.amv"
                subprocess.run([str(PROBE), str(source), str(target), "1"], cwd=directory,
                               capture_output=True, check=True, timeout=90)
                reference = read_morphview(target, windows_text_mode=True)
                model = create_morphview(read_swc(source))
                self.assertEqual(reference.summary(), model.summary())
                self.assertEqual(set(reference.vertices), set(model.vertices))
                for expected, actual in zip(reference.paths, model.paths):
                    self.assertEqual(len(expected.segments), len(actual.segments))
                    for x, y in zip(expected.parameters + expected.radii, actual.parameters + actual.radii):
                        self.assertAlmostEqual(x, y, places=9)
                    self.assertTrue(all(math.isinf(x) for x in expected.box_min + expected.box_size))
                    for a, b in zip(expected.segments, actual.segments):
                        self.assertEqual(a.flags, b.flags)
                        for x, y in zip(sum(a.coefficients, ()), sum(b.coefficients, ())):
                            self.assertAlmostEqual(x, y, places=9)
                        for x, y in zip(a.box_min + a.box_size, b.box_min + b.box_size):
                            self.assertAlmostEqual(x, y, places=9)

    @unittest.skipUnless(sys.platform == "win32" and PROBE.is_file(), "C++ MorphView probe unavailable")
    def test_cpp_positive_soma_and_neurite_flags(self):
        cases = (
            ("1 1 -3 0 0 .3 -1\n2 3 -2 0 0 .1 1\n3 3 0 0 0 .1 2\n"
             "4 3 2 0 0 .1 3\n5 1 -1 .2 0 .3 -1\n", {"ic_sons": 1}),
            ("1 1 -3 -3 0 .2 -1\n2 3 -2 0 0 .1 1\n3 3 2 0 0 .1 2\n"
             "4 3 0 -2 0 .1 1\n5 3 0 2 0 .1 4\n", {"icrn_nsns": 2}),
        )
        for text, expected in cases:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                source, target = Path(directory) / "source.swc", Path(directory) / "reference.amv"
                source.write_text(text, encoding="utf-8")
                subprocess.run([str(PROBE), str(source), str(target), "0"], cwd=directory,
                               capture_output=True, check=True, timeout=90)
                reference = read_morphview(target, windows_text_mode=True)
                model = create_morphview(read_swc(source), preconditioning=False)
                self.assertEqual(reference.summary()["flag_counts"], expected)
                self.assertEqual(reference.summary(), model.summary())
                self.assertEqual([[s.flags for s in p.segments] for p in reference.paths],
                                 [[s.flags for s in p.segments] for p in model.paths])


if __name__ == "__main__":
    unittest.main()
