import math
import random
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from anamorph.redblue import RedBlueTriangulationError, SourceDelaunayLimitWarning, _triangulate_polygon
from anamorph.source_polygon import SourceDelaunayResult, SourcePolygonError, source_delaunay2d

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / "reference-build" / "source_delaunay_probe.exe"


class SourceDelaunayTests(unittest.TestCase):
    @staticmethod
    def cycling_input():
        points = {58: (-.23899649657628019, .8645012719141556),
                  41: (-.54877691051290745, .32794585578619939),
                  73: (-.26548993196811943, .32794585578619939),
                  72: (-.097353007303886172, .61916755193303208)}
        return points, [(58, 41, 73), (58, 73, 72)]

    def test_source_cleanup_can_select_a_different_diagonal(self):
        points = [(100000000., 100000000., 0.), (100000001.06576696, 100000000., 0.),
                  (100000001.06576696, 100000001.06576636, 0.),
                  (100000000., 100000001.065767, 0.)]
        robust = _triangulate_polygon([0, 1, 2, 3], points, (0., 0., 1.), basis=(1., 0., 0.))
        source = _triangulate_polygon([0, 1, 2, 3], points, (0., 0., 1.), basis=(1., 0., 0.),
                                      source_delaunay=True)
        def internal_edge(triangles):
            edges = [tuple(sorted((a, b))) for face in triangles for a, b in zip(face, face[1:]+face[:1])]
            return next(edge for edge in edges if edges.count(edge) == 2)
        self.assertEqual(internal_edge(robust), (0, 2))
        self.assertEqual(internal_edge(source), (1, 3))

    def test_unrepresentable_and_nonconvergent_cleanup_are_rejected(self):
        points = {0: (0., 0.), 1: (1e200, 0.), 2: (1e200, 1e200), 3: (0., 1e200)}
        with self.assertRaisesRegex(SourcePolygonError, "determinant"):
            source_delaunay2d(points, [(0, 1, 2), (0, 2, 3)])
        vertices = [(0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.)]
        with patch("anamorph.redblue.source_delaunay2d", return_value=SourceDelaunayResult((), 1, False)):
            with self.assertRaisesRegex(RedBlueTriangulationError, "did not converge"):
                _triangulate_polygon([0, 1, 2, 3], vertices, (0., 0., 1.), source_delaunay=True)
        points, triangles = self.cycling_input()
        result = source_delaunay2d(points, triangles)
        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 2*len(points)**2)

    def test_bounded_cleanup_accepts_only_a_valid_patch(self):
        vertices = [(0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.)]
        valid = SourceDelaunayResult(((0, 1, 2), (0, 2, 3)), 32, False)
        with patch("anamorph.redblue.source_delaunay2d", return_value=valid):
            with self.assertWarns(SourceDelaunayLimitWarning):
                triangles = _triangulate_polygon([0, 1, 2, 3], vertices, (0., 0., 1.),
                                                 source_delaunay=True, source_delaunay_allow_limit=True)
            self.assertEqual(triangles, list(valid.triangles))
        invalid = SourceDelaunayResult(((0, 1, 2), (0, 1, 2)), 32, False)
        with patch("anamorph.redblue.source_delaunay2d", return_value=invalid):
            with self.assertRaisesRegex(RedBlueTriangulationError, "boundary"):
                _triangulate_polygon([0, 1, 2, 3], vertices, (0., 0., 1.),
                                     source_delaunay=True, source_delaunay_allow_limit=True)

    @unittest.skipUnless(sys.platform == "win32" and PROBE.is_file(), "C++ Delaunay probe unavailable")
    def test_cpp_source_pass_also_hits_iteration_limit(self):
        points, triangles = self.cycling_input()
        data = f"{len(points)} {len(triangles)}\n"
        data += "".join(f"{i} {p[0]:.17g} {p[1]:.17g}\n" for i, p in points.items())
        data += "".join(f"{a} {b} {c}\n" for a, b, c in triangles)
        run = subprocess.run([str(PROBE)], input=data, capture_output=True, text=True, check=True, timeout=10)
        self.assertIn("maximum iteration limit", run.stdout)
        expected = tuple(tuple(map(int, line.split()[1:])) for line in run.stdout.splitlines()
                         if line.startswith("TRI "))
        self.assertEqual(source_delaunay2d(points, triangles).triangles, expected)

    @unittest.skipUnless(sys.platform == "win32" and PROBE.is_file(), "C++ Delaunay probe unavailable")
    def test_translated_source_diagonal_matches_cpp(self):
        points = {0: (100000000., 100000000.), 1: (100000001.06576696, 100000000.),
                  2: (100000001.06576696, 100000001.06576636), 3: (100000000., 100000001.065767)}
        triangles = [(0, 1, 2), (0, 2, 3)]
        data = "4 2\n" + "".join(f"{i} {p[0]:.17g} {p[1]:.17g}\n" for i, p in points.items())
        data += "0 1 2\n0 2 3\n"
        run = subprocess.run([str(PROBE)], input=data, capture_output=True, text=True, check=True, timeout=10)
        expected = tuple(tuple(map(int, line.split()[1:])) for line in run.stdout.splitlines()
                         if line.startswith("TRI "))
        self.assertEqual(source_delaunay2d(points, triangles).triangles, expected)
        self.assertEqual(expected, ((1, 2, 3), (1, 3, 0)))

    @unittest.skipUnless(sys.platform == "win32" and PROBE.is_file(), "C++ Delaunay probe unavailable")
    def test_source_flip_order_matches_cpp(self):
        rng = random.Random(80391)
        for case in range(80):
            count = rng.randrange(4, 12)
            angles = sorted(rng.uniform(0., 2*math.pi) for _ in range(count))
            points = {i: (radius*math.cos(angle), radius*math.sin(angle))
                      for i, angle in enumerate(angles) for radius in (rng.uniform(.6, 1.4),)}
            triangles = [(0, i, i+1) for i in range(1, count-1)]
            # Source vertex IDs need not follow polygon order after mesh edits.
            ids = list(range(17, 17+count))
            rng.shuffle(ids)
            points = {ids[i]: point for i, point in points.items()}
            triangles = [tuple(ids[i] for i in triangle) for triangle in triangles]
            data = f"{count} {len(triangles)}\n"
            data += "".join(f"{i} {point[0]:.17g} {point[1]:.17g}\n" for i, point in points.items())
            data += "".join(f"{a} {b} {c}\n" for a, b, c in triangles)
            with self.subTest(case=case):
                run = subprocess.run([str(PROBE)], input=data, capture_output=True, text=True,
                                     check=True, timeout=10)
                expected = tuple(tuple(map(int, line.split()[1:])) for line in run.stdout.splitlines()
                                 if line.startswith("TRI "))
                actual = source_delaunay2d(points, triangles)
                self.assertEqual(actual.triangles, expected)


if __name__ == "__main__":
    unittest.main()
