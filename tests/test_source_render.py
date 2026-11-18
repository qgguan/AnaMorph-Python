import math
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from anamorph import read_swc
from anamorph.analytic_mesher import _path_curves, _render_vector, _render_vector_score
from anamorph.bernstein import BernsteinPolynomial, ClippingConvergenceError
from anamorph.source_random import WindowsCRandom
from anamorph.geometry import partition_paths
from anamorph.model import prepare_morphology
from anamorph.source_render import source_render_polynomials, source_render_vector_score
from anamorph.source_spline import SourceCanalSegment
from anamorph.source_univariate import source_bezier_clip_roots

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / "reference-build" / "render_vector_probe.exe"
FRAME_PROBE = ROOT / "reference-build" / "frame_choice_probe.exe"


class SourceRenderVectorTests(unittest.TestCase):
    @staticmethod
    def small_curve():
        scale = 1e-4
        power = ((0., -.125*scale, .25*scale, 0.), (0., 0., 0., 0.), (0., scale, 0., 0.))
        controls = tuple(BernsteinPolynomial.from_power(component).coefficients for component in power)
        return SourceCanalSegment(power, .01, .01, controls)

    def test_source_tolerance_can_miss_a_parallel_tangent(self):
        segment = self.small_curve()
        axis = (0., 0., 1.)
        self.assertGreater(source_render_vector_score(segment, axis), .01)
        self.assertEqual(_render_vector_score(segment, axis), 0.)
        self.assertEqual(_render_vector([segment], rng=WindowsCRandom(0), source_style=True), axis)
        self.assertNotEqual(_render_vector([segment], rng=WindowsCRandom(0)), axis)
        self.assertEqual(segment.tangent(.25)[0], 0.)

    def test_zero_speed_and_nonfinite_polynomial_are_unresolved(self):
        zero = ((0., 0., 0., 0.),) * 3
        constant = SourceCanalSegment(zero, .1, .1, zero)
        with self.assertRaisesRegex(ClippingConvergenceError, "squared speed"):
            source_render_vector_score(constant, (0., 0., 1.))
        huge = SourceCanalSegment(zero, .1, .1, ((0., 1e200, 2e200, 3e200), zero[1], zero[2]))
        with self.assertRaisesRegex(ClippingConvergenceError, "polynomial"):
            source_render_polynomials(huge, (0., 0., 1.))

    @unittest.skipUnless(sys.platform == "win32" and PROBE.is_file(), "C++ render-vector probe unavailable")
    def test_source_polynomials_roots_and_score_match_cpp(self):
        rng = random.Random(4394511)
        zero = ((0., 0., 0., 0.),) * 3
        cases = [(self.small_curve(), (0., 0., 1.))]
        for _ in range(40):
            controls = tuple(tuple(rng.uniform(-3., 3.) for _ in range(4)) for _ in range(3))
            raw_axis = tuple(rng.uniform(-1., 1.) for _ in range(3))
            length = math.hypot(*raw_axis)
            cases.append((SourceCanalSegment(zero, .1, .1, controls), tuple(x/length for x in raw_axis)))
        for case, (segment, axis) in enumerate(cases):
            with self.subTest(case=case):
                data = " ".join(format(value, ".17g") for component in (*segment.source_controls, axis)
                                for value in component) + "\n"
                run = subprocess.run([str(PROBE)], input=data, capture_output=True, text=True,
                                     check=True, timeout=10)
                output = {parts[0]: parts[1:] for line in run.stdout.splitlines()
                          if (parts := line.split()) and parts[0] in ("P", "Q", "NUM", "ROOTS", "SCORE")}
                p, q, numerator = source_render_polynomials(segment, axis)
                for label, polynomial in zip(("P", "Q", "NUM"), (p, q, numerator)):
                    self.assertEqual(polynomial.coefficients, tuple(map(float, output[label])))
                roots = source_bezier_clip_roots(numerator)
                self.assertEqual([value for root in roots for value in (root.left, root.right)],
                                 list(map(float, output["ROOTS"][1:])))
                self.assertEqual(source_render_vector_score(segment, axis), float(output["SCORE"][0]))

    @unittest.skipUnless(sys.platform == "win32" and FRAME_PROBE.is_file(), "C++ frame-choice probe unavailable")
    def test_path_frames_and_random_consumption_match_cpp(self):
        with tempfile.TemporaryDirectory() as directory:
            vertical = Path(directory) / "vertical.swc"
            vertical.write_text("1 1 0 0 0 .5 -1\n2 3 0 0 .7 .1 1\n3 3 0 0 1.3 .1 2\n"
                                "4 3 0 0 2 .1 3\n5 3 0 .02 3 .1 4\n6 3 .1 0 2.8 .1 4\n",
                                encoding="utf-8")
            for source in (ROOT / "examples" / "branched.swc", vertical):
                for parameter in ("chord-length", "centripetal", "uniform"):
                    with self.subTest(source=source.name, parameter=parameter):
                        run = subprocess.run([str(FRAME_PROBE), str(source), "0", parameter], cwd=directory,
                                             capture_output=True, text=True, check=True, timeout=30)
                        lines = [line.split() for line in run.stdout.splitlines()
                                 if line.startswith(("FRAME ", "NEXT "))]
                        expected = [tuple(map(float, parts[-3:])) for parts in lines if parts[0] == "FRAME"]
                        tree = prepare_morphology(read_swc(source), preconditioning=False,
                                                  preserve_surface_roots=False).morphology
                        rng = WindowsCRandom(0)
                        actual = []
                        for path in partition_paths(tree):
                            _, _, root, segments = _path_curves(tree, path, parameter, source_curve_evaluation=True)
                            actual.append(_render_vector(segments[1:] if root else segments,
                                                         rng=rng, source_style=True))
                        self.assertEqual(actual, expected)
                        self.assertEqual(rng.rand(), int(lines[-1][1]))


if __name__ == "__main__":
    unittest.main()
