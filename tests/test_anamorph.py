import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from anamorph import Mesh, read_obj, read_swc
from anamorph.quality import mesh_quality


class AnaMorphTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)

    def write(self, name, contents):
        path = self.path / name
        path.write_text(contents, encoding="utf-8")
        return path

    def test_swc_tree_and_violation(self):
        path = self.write("cell.swc", "# cell\n1 1 0 0 0 2 -1\n2 3 2 0 0 1 1\n3 3 3 0 0 1 2\n4 3 8 0 0 1 3\n")
        tree = read_swc(path)
        self.assertEqual(tree.roots, (1,))
        self.assertEqual(tree.primary_distance_violations(), [(2, 3)])
        self.assertEqual(tree.children[3], (4,))

    def test_swc_validation(self):
        for text in ("1 1 0 0 0 1 -1\n1 2 0 0 0 1 1\n",
                     "1 1 0 0 0 1 -1\n2 3 0 0 0 1 3\n3 3 0 0 0 1 2\n",
                     "1 1 0 0 0 nan -1\n", "1 1 0 0 0 1 2\n"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                read_swc(self.write("bad.swc", text))

    def test_tetrahedron_statistics(self):
        path = self.write("tetra.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n")
        stats = read_obj(path).statistics()
        self.assertEqual((stats.vertices, stats.edges, stats.faces, stats.euler_characteristic), (4, 6, 4, 2))
        self.assertAlmostEqual(stats.volume, 1 / 6)
        self.assertAlmostEqual(stats.area, 1.5 + math.sqrt(3) / 2)
        self.assertEqual((stats.boundary_edges, stats.nonmanifold_edges), (0, 0))
        self.assertEqual(stats.obtuse_triangles, 0)
        source = read_obj(path)
        moved = Mesh(tuple(tuple(value+1e9 for value in vertex) for vertex in source.vertices), source.faces)
        moved_stats = moved.statistics()
        self.assertAlmostEqual(moved_stats.volume, stats.volume)
        self.assertAlmostEqual(moved_stats.area, stats.area)
        components = Mesh(source.vertices+moved.vertices,
                          source.faces+tuple(tuple(index+len(source.vertices) for index in face)
                                             for face in moved.faces))
        self.assertAlmostEqual(components.statistics().volume, 2*stats.volume)

    def test_negative_obj_indices_and_equilateral_triangle(self):
        path = self.write("triangle.obj", f"v 0 0 0\nv 1 0 0\nv 0.5 {math.sqrt(3)/2} 0\nf -3 -2 -1\n")
        stats = read_obj(path).statistics()
        self.assertAlmostEqual(stats.aspect_ratio_mean, 1)
        self.assertTrue(math.isnan(stats.aspect_ratio_stddev))
        self.assertEqual(stats.boundary_edges, 3)

    def test_disconnected_oriented_shells_preserve_cavity_volume(self):
        outer = ((0., 0., 0.), (2., 0., 0.), (0., 2., 0.), (0., 0., 2.))
        faces = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))
        inner = tuple(tuple(.2+.25*x for x in point) for point in outer)
        reversed_inner = tuple((a+4, c+4, b+4) for a, b, c in faces)
        mesh = Mesh(outer+inner, faces+reversed_inner)
        self.assertAlmostEqual(mesh.statistics().volume, (2**3-.5**3)/6)
        from fractions import Fraction
        r = math.nextafter(1., 0.)
        outer = ((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
        inner = tuple(tuple(r*x for x in point) for point in outer)
        shell = Mesh(outer+inner, faces+reversed_inner)
        expected = float((1-Fraction(r)**3)/6)
        self.assertEqual(shell.statistics().volume, expected)

    def test_statistics_reject_nonfinite_vertices(self):
        mesh = Mesh(((0., 0., 0.), (1., 0., 0.), (0., float("nan"), 0.)), ((0, 1, 2),))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            mesh.statistics()

    def test_mesh_statistics_and_quality_scale_without_intermediate_overflow(self):
        faces = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))
        for scale in (1e-100, 1., 1e100):
            with self.subTest(scale=scale):
                mesh = Mesh(((0., 0., 0.), (scale, 0., 0.), (0., scale, 0.), (0., 0., scale)), faces)
                stats, quality = mesh.statistics(), mesh_quality(mesh)
                self.assertAlmostEqual(stats.area/(scale*scale), (3+math.sqrt(3))/2, places=14)
                self.assertAlmostEqual(stats.volume/(scale*scale*scale), 1/6, places=14)
                self.assertAlmostEqual(stats.aspect_ratio_max, (1+math.sqrt(2))/2, places=14)
                self.assertEqual(stats.obtuse_triangles, 0)
                self.assertAlmostEqual(quality["triangle_areas"]["min"]/(scale*scale), .5, places=14)
        for scale, message in ((1e-110, "underflows"), (1e110, "exceeds")):
            mesh = Mesh(((0., 0., 0.), (scale, 0., 0.), (0., scale, 0.), (0., 0., scale)), faces)
            with self.subTest(scale=scale), self.assertRaisesRegex(ValueError, f"mesh volume {message}"):
                mesh.statistics()
        for scale in (1e-150, 1e150):
            mesh = Mesh(((0., 0., 0.), (scale, 0., 0.), (0., scale, 0.), (0., 0., scale)), faces)
            self.assertIsNone(mesh.statistics(include_volume=False).volume)
            self.assertAlmostEqual(mesh_quality(mesh)["triangle_areas"]["min"]/(scale*scale), .5, places=14)
        elongated = Mesh(((0., 0., 0.), (1e150, 0., 0.), (0., 1e-150, 0.), (0., 0., 1.)), faces)
        self.assertAlmostEqual(elongated.statistics().volume, 1/6, places=14)
        step = 2**-52
        thin = Mesh(((0., 0., 0.), (1., 1., 1.), (1., 1.+step, 1.), (1., 1., 1.+step)), faces)
        self.assertAlmostEqual(thin.statistics().volume/(step*step/6), 1., places=14)

    def test_redblue_output_check_does_not_require_representable_volume(self):
        source = self.write("large_soma.swc", "1 1 0 0 0 1e150 -1\n")
        output = self.path/"large_soma.obj"
        run = subprocess.run([sys.executable, "-m", "anamorph", "generate-redblue", str(source), str(output),
                              "--force", "--check-output", "--soma-refinements", "1"],
                             capture_output=True, text=True, check=True)
        report = json.loads(run.stdout)
        self.assertEqual(report["output_check"]["intersection_candidates"], 0)
        self.assertEqual(len(read_obj(output).faces), 80)

    def test_bad_obj_and_cli(self):
        path = self.write("bad.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 4\n")
        with self.assertRaises(ValueError):
            read_obj(path)
        result = subprocess.run([sys.executable, "-m", "anamorph", "meshstat", str(path)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("error:", result.stderr)
        swc = self.write("one.swc", "1 1 0 0 0 1 -1\n")
        result = subprocess.run([sys.executable, "-m", "anamorph", "inspect", str(swc)], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)["nodes"], 1)
        help_result = subprocess.run([sys.executable, "-m", "anamorph", "--help"],
                                     capture_output=True, text=True, check=True)
        self.assertIn("Based on AnaMorph", help_result.stderr)

    def test_meshquality_reports_unique_edge_and_triangle_sizes(self):
        path = self.write("rectangle.obj", "v 0 0 0\nv 2 0 0\nv 0 1 0\nv 2 1 0\nf 1 2 3\nf 2 4 3\n")
        run = subprocess.run([sys.executable, "-m", "anamorph", "meshquality", str(path)],
                             capture_output=True, text=True, check=True)
        result = json.loads(run.stdout)
        self.assertAlmostEqual(result["edge_lengths"]["min"], 1.)
        self.assertAlmostEqual(result["edge_lengths"]["median"], 2.)
        self.assertAlmostEqual(result["edge_lengths"]["max"], math.sqrt(5))
        self.assertAlmostEqual(result["triangle_areas"]["min"], 1.)
        self.assertAlmostEqual(result["triangle_areas"]["max"], 1.)
        self.assertFalse(result["certifies_quasi_uniformity"])


if __name__ == "__main__":
    unittest.main()
