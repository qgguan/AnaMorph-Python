import math
import random
import subprocess
import sys
import unittest
from pathlib import Path

from anamorph.source_polygon import SourcePolygonError, source_ear_triangulate
from anamorph.source_random import WindowsCRandom

PROBE = Path(__file__).resolve().parent.parent / "reference-build" / "source_ear_probe.exe"


class SourcePolygonTests(unittest.TestCase):
    def test_triangle_does_not_consume_random_values(self):
        rng = WindowsCRandom(0)
        self.assertEqual(source_ear_triangulate(((0., 0.), (1., 0.), (0., 1.)), rng), ((0, 1, 2),))
        self.assertEqual(rng.rand(), 38)

    def test_input_validation(self):
        for points in ((), ((0., 0.),), ((0., 0.), (1., 0.), (float("inf"), 1.))):
            with self.subTest(points=points), self.assertRaises(ValueError):
                source_ear_triangulate(points, WindowsCRandom(0))

    @unittest.skipUnless(sys.platform == "win32" and PROBE.is_file(), "optional C++ ear probe unavailable")
    def test_sampled_polygon_ears_and_random_state_match_cpp(self):
        rng = random.Random(44094)
        successes = failures = 0
        for case in range(40):
            n = rng.randrange(3, 16)
            angles = sorted(rng.uniform(0, 2*math.pi) for _ in range(n))
            points = [(rng.uniform(.2, 2)*math.cos(a), rng.uniform(.2, 2)*math.sin(a)) for a in angles]
            seed = rng.randrange(0, 2**32)
            data = f"{seed} {n}\n"+"\n".join(f"{x:.17g} {y:.17g}" for x, y in points)+"\n"
            run = subprocess.run([str(PROBE)], input=data, text=True, capture_output=True, check=True, timeout=15)
            lines = run.stdout.splitlines()
            status = int(lines[0].split()[1])
            expected = tuple(tuple(map(int, line.split()[1:])) for line in lines if line.startswith("TRI "))
            next_value = int(lines[-1].split()[1])
            generator = WindowsCRandom(seed)
            with self.subTest(case=case, status=status):
                if status == 9:  # Aux::Geometry::IntersectionTestResults::SUCCESS
                    self.assertEqual(source_ear_triangulate(points, generator), expected)
                    successes += 1
                else:
                    self.assertIn(status, (7, 8))
                    with self.assertRaises(SourcePolygonError):
                        source_ear_triangulate(points, generator)
                    failures += 1
                self.assertEqual(generator.rand(), next_value)
        self.assertGreater(successes, 0)
        self.assertGreater(failures, 0)


if __name__ == "__main__":
    unittest.main()
