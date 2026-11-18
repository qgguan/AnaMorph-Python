import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from anamorph.analytic_intersections import CanalSegment, regularity_candidates
from anamorph.analytic_network import analyze_analytic_partial
from anamorph.bernstein import BernsteinPolynomial
from anamorph.source_bivariate import (SourceBivariateFailure, source_global_self_intersection_candidates,
                                       source_neurite_pair_candidates)
from anamorph.source_canal_polynomials import (source_local_self_intersection_candidates,
                                               source_regularity_candidates, source_soma_neurite_candidates)
from anamorph.swc import Morphology, Node

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / "reference-build" / "solver_candidates_probe.exe"


def _input(segment):
    values = [value for component in segment.power for value in component]
    return " ".join(format(value, ".17g") for value in (*values, segment.radius0, segment.radius1))


def _run(directory, mode, data):
    run = subprocess.run([str(PROBE), mode, "details"], input=data+"\n", cwd=directory,
                         capture_output=True, text=True, check=True, timeout=15)
    rows = [line.split() for line in run.stdout.splitlines()]
    result, = (parts for parts in rows if parts and parts[0] == "RESULT")
    candidates = [tuple(map(float, parts[1:])) for parts in rows if parts and parts[0] in ("ONE", "TWO")]
    return bool(int(result[1])), candidates


class SourceNumericalLimitationTests(unittest.TestCase):
    def test_source_regularity_can_miss_an_exact_zero_speed(self):
        # x(t) = (t - 1/2)^2 has exactly zero speed at 1/2. The source's
        # rounded squared-speed polynomial and clipping return no candidate.
        curve = CanalSegment(((.25, -1., 1., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)), .1, .1)
        self.assertEqual(source_regularity_candidates(curve), [])
        self.assertTrue(any(root.certified and root.left <= .5 <= root.right
                            for root in regularity_candidates(curve)))

        tree = Morphology([Node(1, 1, -3., 0., 0., .1, -1), Node(2, 3, 0., 0., 0., .1, 1),
                           Node(3, 3, 1., 0., 0., .1, 2)])
        controls = tuple(BernsteinPolynomial.from_power(component).coefficients for component in curve.power)
        with patch("anamorph.analytic_network.CanalSegment.from_spline", return_value=curve), \
                patch("anamorph.analytic_network.source_spline_controls", return_value=((0., 1.), (controls,))):
            report = analyze_analytic_partial(tree, job_mode="source", source_job_limit=1)
        self.assertTrue(any(item.kind == "regularity" for item in report.findings))
        self.assertIs(report.source_jobs_clean, False)


@unittest.skipUnless(sys.platform == "win32" and PROBE.is_file(), "C++ candidate probe unavailable")
class SourceCandidateReferenceTests(unittest.TestCase):
    def test_cpp_regularity_can_miss_an_exact_zero_speed(self):
        curve = CanalSegment(((.25, -1., 1., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)), .1, .1)
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(_run(directory, "regularity", _input(curve)), (False, []))

    def test_singular_pair_fallback_has_no_witness(self):
        first = CanalSegment(((0., 1., 0., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)), .0001, .0001)
        second = CanalSegment(((0., -1., 0., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)), .0001, .0001)
        with tempfile.TemporaryDirectory() as directory:
            for mode in ("pair", "adjacent"):
                with self.subTest(mode=mode):
                    data = _input(first)+" "+_input(second)+(" 0" if mode == "adjacent" else "")
                    self.assertEqual(_run(directory, mode, data), (True, []))
                    with self.assertRaises(SourceBivariateFailure):
                        source_neurite_pair_candidates(first, second, adjacent=mode == "adjacent")
            self.assertEqual(_run(directory, "adjacent", _input(first)+" "+_input(second)+" 1"), (False, []))
            self.assertEqual(source_neurite_pair_candidates(first, second, adjacent=True,
                                                            first_end_second_start=True), [])

    def test_univariate_candidates_match_cpp(self):
        rng, positives = random.Random(829313), 0
        with tempfile.TemporaryDirectory() as directory:
            for case in range(30):
                power = tuple(tuple(rng.uniform(-2., 2.) for _ in range(4)) for _ in range(3))
                segment = CanalSegment(power, rng.uniform(.05, .9), rng.uniform(.05, .9))
                center = tuple(rng.uniform(-1., 1.) for _ in range(3))
                radius, root = rng.uniform(.05, .9), bool(rng.randrange(2))
                for mode in ("regularity", "local", "soma"):
                    with self.subTest(case=case, mode=mode):
                        data = _input(segment)
                        if mode == "soma":
                            data += " " + " ".join(format(x, ".17g") for x in (*center, radius, int(root)))
                        found, expected = _run(directory, mode, data)
                        if mode == "regularity":
                            hits = source_regularity_candidates(segment)
                        elif mode == "local":
                            hits = source_local_self_intersection_candidates(segment)
                        else:
                            hits = source_soma_neurite_candidates(segment, center, radius, is_root_segment=root)
                        actual = [(hit.interval.left, hit.interval.right, hit.value) for hit in hits]
                        self.assertEqual((bool(hits), actual), (found, expected))
                        positives += found
        self.assertGreater(positives, 0)

    def test_bivariate_candidates_match_cpp(self):
        rng, positives = random.Random(820459), 0
        with tempfile.TemporaryDirectory() as directory:
            for case in range(25):
                def segment():
                    power = tuple(tuple(rng.uniform(-2., 2.) for _ in range(4)) for _ in range(3))
                    return CanalSegment(power, rng.uniform(.05, .9), rng.uniform(.05, .9))
                first, second = segment(), segment()
                for mode in ("global", "pair", "adjacent"):
                    with self.subTest(case=case, mode=mode):
                        end_start = mode == "adjacent" and bool(rng.randrange(2))
                        data = _input(first) + (" " + _input(second) if mode != "global" else "")
                        if mode == "adjacent":
                            data += " " + str(int(end_start))
                        found, expected = _run(directory, mode, data)
                        try:
                            hits = (source_global_self_intersection_candidates(first) if mode == "global" else
                                    source_neurite_pair_candidates(first, second, adjacent=mode == "adjacent",
                                                                   first_end_second_start=end_start))
                        except SourceBivariateFailure:
                            actual, decision = [], True
                        else:
                            actual = [(hit.rectangle.x0, hit.rectangle.x1, hit.rectangle.y0, hit.rectangle.y1,
                                       hit.distance_squared) for hit in hits]
                            decision = bool(hits)
                        self.assertEqual((decision, actual), (found, expected))
                        positives += found
        self.assertGreater(positives, 0)


if __name__ == "__main__":
    unittest.main()
