import unittest
import random
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
from anamorph.mesh import Mesh
from anamorph import mesh_intersections as intersection_module
from anamorph.mesh_intersections import find_self_intersections
from anamorph.canal_mesh import ico_sphere


class MeshIntersectionTests(unittest.TestCase):
    def test_degenerate_triangle_is_not_reported_as_clean(self):
        mesh = Mesh(((0., 0., 0.), (1., 0., 0.), (2., 0., 0.)), ((0, 1, 2),))
        with self.assertRaisesRegex(ValueError, "degenerate"):
            find_self_intersections(mesh, include_adjacent=True)
        nonfinite = Mesh(((0., 0., 0.), (1., 0., 0.), (0., float("nan"), 0.)), ((0, 1, 2),))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            find_self_intersections(nonfinite, include_adjacent=True)

    def test_transverse_intersection(self):
        mesh = Mesh(((0.,0.,0.), (2.,0.,0.), (0.,2.,0.),
                     (.5,.5,-1.), (.5,.5,1.), (1.5,.5,0.)), ((0,1,2), (3,4,5)))
        hits, truncated = find_self_intersections(mesh)
        self.assertEqual(len(hits), 1)
        self.assertFalse(hits[0].coplanar)
        self.assertFalse(truncated)

    def test_nonadjacent_transverse_intersection_scales_with_mesh(self):
        vertices = ((0., 0., 0.), (2., 0., 0.), (0., 2., 0.),
                    (.5, .5, -1.), (.5, .5, 1.), (1.5, .5, 0.))
        for scale in (1e-200, 1e-150, 1e-70, 1., 1e70, 1e150, 1e200):
            with self.subTest(scale=scale):
                mesh = Mesh(tuple(tuple(scale*x for x in point) for point in vertices), ((0, 1, 2), (3, 4, 5)))
                hits, truncated = find_self_intersections(mesh)
                self.assertEqual(len(hits), 1)
                self.assertFalse(hits[0].coplanar)
                self.assertFalse(truncated)

    def test_coplanar_overlap_and_separation(self):
        vertices = ((0.,0.,0.), (2.,0.,0.), (0.,2.,0.),
                    (.2,.2,0.), (.5,.2,0.), (.2,.5,0.),
                    (3.,3.,0.), (4.,3.,0.), (3.,4.,0.))
        mesh = Mesh(vertices, ((0,1,2), (3,4,5), (6,7,8)))
        hits, truncated = find_self_intersections(mesh)
        self.assertEqual(len(hits), 1)
        self.assertTrue(hits[0].coplanar)
        self.assertFalse(truncated)

    def test_coplanar_diagnostic_is_independent_of_coordinate_translation(self):
        first = ((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
        separated = ((.6, .6, 0.), (1.6, .6, 0.), (.6, 1.6, 0.))
        overlapping = ((.2, .2, 0.), (.7, .2, 0.), (.2, .7, 0.))
        for shift in (0., 1e6, 1e9):
            def translated(points):
                return tuple(tuple(x+shift for x in point) for point in points)
            for other, expected in ((separated, 0), (overlapping, 1)):
                mesh = Mesh(translated(first+other), ((0, 1, 2), (3, 4, 5)))
                hits, truncated = find_self_intersections(mesh)
                self.assertEqual(len(hits), expected, (shift, other))
                self.assertFalse(truncated)

    def test_coplanar_diagnostic_scales_with_local_geometry(self):
        first = ((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
        separated = ((.6, .6, 0.), (1.6, .6, 0.), (.6, 1.6, 0.))
        overlapping = ((.2, .2, 0.), (.7, .2, 0.), (.2, .7, 0.))
        for scale in (1e-6, 1e-3, 1., 1e6):
            for other, expected in ((separated, 0), (overlapping, 1)):
                points = tuple(tuple(scale*x for x in point) for point in first+other)
                hits, _ = find_self_intersections(Mesh(points, ((0, 1, 2), (3, 4, 5))))
                self.assertEqual(len(hits), expected, (scale, other))

    def test_nearly_coplanar_distinct_triangles_use_exact_plane_signs(self):
        first = ((0., 0., 0.), (2., 0., 0.), (0., 2., 0.))
        above = ((.2, .2, 1e-12), (1., .2, 1e-12), (.2, 1., 1e-12))
        crossing = ((.2, .2, 1e-12), (1., .2, -1e-12), (.2, 1., 1e-12))
        self.assertEqual(find_self_intersections(Mesh(first+above, ((0, 1, 2), (3, 4, 5)))), ([], False))
        hits, _ = find_self_intersections(Mesh(first+crossing, ((0, 1, 2), (3, 4, 5))))
        self.assertEqual(len(hits), 1)
        self.assertFalse(hits[0].coplanar)

    def test_shallow_triangle_crossing_is_not_discarded_as_parallel(self):
        from math import sqrt
        tilt = .9e-9/sqrt(2)
        first = ((0., 0., 0.), (1., 1., 0.), (0., 1., 0.))
        crossing = ((0., 0., -.1*tilt), (1., 1., 1.9*tilt), (0., 1., .9*tilt))
        separated = tuple((x, y, z+2*tilt) for x, y, z in crossing)
        for scale in (1e-150, 1., 1e150):
            for other, expected in ((crossing, True), (separated, False)):
                with self.subTest(scale=scale, intersects=expected):
                    vertices = tuple(tuple(scale*x for x in point) for point in first+other)
                    hits, truncated = find_self_intersections(Mesh(vertices, ((0, 1, 2), (3, 4, 5))))
                    self.assertEqual(bool(hits), expected)
                    if hits:
                        self.assertFalse(hits[0].coplanar)
                    self.assertFalse(truncated)

    def test_thin_triangle_crossing_uses_exact_near_parallel_edges(self):
        vertices = ((0., 0., 0.), (2., 0., 0.), (0., 2., 0.),
                    (.5, .5, -1e-11), (1.5, .5, 1e-11), (1.2, .500000000001, 1e-11))
        for scale in (1e-150, 1., 1e150):
            with self.subTest(scale=scale):
                points = tuple(tuple(scale*x for x in point) for point in vertices)
                mesh = Mesh(points, ((0, 1, 2), (3, 4, 5)))
                hits, truncated = find_self_intersections(mesh)
                self.assertEqual(len(hits), 1)
                self.assertFalse(hits[0].coplanar)
                self.assertFalse(truncated)
        rng = random.Random(29617)
        for _ in range(30):
            x0, x1, y = rng.uniform(.1, .4), rng.uniform(.8, 1.1), rng.uniform(.2, .5)
            points = ((0., 0., 0.), (2., 0., 0.), (0., 2., 0.),
                      (x0, y, -1e-11), (x1, y, 1e-11),
                      ((x0+x1)/2, y+1e-12, 1e-11))
            for scale in (1e-100, 1., 1e100):
                mesh = Mesh(tuple(tuple(scale*x for x in point) for point in points), ((0, 1, 2), (3, 4, 5)))
                hits, truncated = find_self_intersections(mesh)
                self.assertEqual(len(hits), 1)
                self.assertFalse(truncated)

    def test_shared_vertices_are_excluded_and_cap_is_reported(self):
        vertices = ((0.,0.,0.), (2.,0.,0.), (0.,2.,0.),
                    (.2,.2,-1.), (.2,.2,1.), (1.,.2,0.),
                    (.4,.4,-1.), (.4,.4,1.), (1.,.4,0.))
        mesh = Mesh(vertices, ((0,1,2), (0,2,1), (3,4,5), (6,7,8)))
        hits, truncated = find_self_intersections(mesh, max_hits=1)
        self.assertEqual(len(hits), 1)
        self.assertTrue(truncated)
        self.assertNotEqual({hits[0].first_face, hits[0].second_face}, {0, 1})

    def test_shared_vertex_can_hide_a_transverse_or_coplanar_intersection(self):
        first = ((0., 0., 0.), (2., 0., 0.), (0., 2., 0.))
        crossing = Mesh(first+((1., 1., -1.), (1., 1., 1.)), ((0, 1, 2), (0, 3, 4)))
        self.assertEqual(find_self_intersections(crossing), ([], False))
        hits, truncated = find_self_intersections(crossing, include_adjacent=True)
        self.assertEqual((len(hits), hits[0].coplanar, truncated), (1, False, False))
        separated = Mesh(first+((-1., -1., -1.), (-1., -1., 1.)), ((0, 1, 2), (0, 3, 4)))
        self.assertEqual(find_self_intersections(separated, include_adjacent=True), ([], False))
        overlap = Mesh(first+((1., .1, 0.), (.1, 1., 0.)), ((0, 1, 2), (0, 3, 4)))
        hits, _ = find_self_intersections(overlap, include_adjacent=True)
        self.assertEqual((len(hits), hits[0].coplanar), (1, True))

    def test_shared_vertex_overlap_diagnostic_scales_with_mesh(self):
        first = ((0., 0., 0.), (2., 0., 0.), (0., 2., 0.))
        crossing = ((1., 1., -1.), (1., 1., 1.))
        separated = ((-1., -1., -1.), (-1., -1., 1.))
        overlap = ((1., .1, 0.), (.1, 1., 0.))
        for scale in (1e-200, 1e-150, 1e-70, 1e-30, 1e-12, 1e-9, 1., 1e6, 1e150, 1e200):
            for other, expected, coplanar in ((crossing, True, False),
                                               (separated, False, False),
                                               (overlap, True, True)):
                with self.subTest(scale=scale, kind=other):
                    points = tuple(tuple(scale*x for x in point) for point in first+other)
                    hits, truncated = find_self_intersections(Mesh(points, ((0, 1, 2), (0, 3, 4))),
                                                               include_adjacent=True)
                    self.assertEqual(bool(hits), expected)
                    if hits:
                        self.assertEqual(hits[0].coplanar, coplanar)
                    self.assertFalse(truncated)

    def test_meshcheck_cli_reports_small_shared_vertex_crossing(self):
        scale = 1e-12
        vertices = ((0., 0., 0.), (2., 0., 0.), (0., 2., 0.), (1., 1., -1.), (1., 1., 1.))
        mesh = Mesh(tuple(tuple(scale*x for x in point) for point in vertices), ((0, 1, 2), (0, 3, 4)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"crossing.obj"
            mesh.write_obj(path)
            result = subprocess.run([sys.executable, "-m", "anamorph", "meshcheck", str(path),
                                     "--include-adjacent"], text=True, capture_output=True, check=True)
        report = json.loads(result.stdout)
        self.assertEqual(len(report["intersections"]), 1)
        self.assertFalse(report["truncated"])

    def test_shared_edge_overlap_diagnostic_scales_with_mesh(self):
        vertices = ((0., 0., 0.), (2., 0., 0.), (0., 1., 0.), (1., 1., 0.), (1., -1., 0.))
        for scale in (1e-200, 1e-100, 1., 1e100, 1e200):
            points = tuple(tuple(scale*x for x in point) for point in vertices)
            for last, expected in ((3, True), (4, False)):
                with self.subTest(scale=scale, overlapping=expected):
                    mesh = Mesh(points, ((0, 1, 2), (1, 0, last)))
                    hits, truncated = find_self_intersections(mesh, include_adjacent=True)
                    self.assertEqual(bool(hits), expected)
                    self.assertFalse(truncated)

    def test_self_intersection_broad_phase_scales_with_geometry(self):
        sphere = ico_sphere((0., 0., 0.), 1., 1)
        baseline = None
        for scale in (1e-150, 1., 1e150):
            mesh = Mesh(tuple(tuple(scale*x for x in point) for point in sphere.vertices), sphere.faces)
            with patch.object(intersection_module, "_triangles_intersect",
                              wraps=intersection_module._triangles_intersect) as separate, \
                 patch.object(intersection_module, "_shared_vertex_intersection",
                              wraps=intersection_module._shared_vertex_intersection) as vertex, \
                 patch.object(intersection_module, "_shared_edge_intersection",
                              wraps=intersection_module._shared_edge_intersection) as edge:
                self.assertEqual(find_self_intersections(mesh, include_adjacent=True), ([], False))
            counts = (separate.call_count, vertex.call_count, edge.call_count)
            if baseline is None:
                baseline = counts
            else:
                self.assertEqual(counts, baseline)
            self.assertLess(sum(counts), len(mesh.faces)*(len(mesh.faces)-1)//2)

    def test_nearly_coplanar_shared_vertex_uses_plane_signs(self):
        first = ((0., 0., 0.), (2., 0., 0.), (0., 2., 0.))
        separated = Mesh(first+((1., 1., 1e-12), (1., .2, 1e-12)), ((0, 1, 2), (0, 3, 4)))
        self.assertEqual(find_self_intersections(separated, include_adjacent=True), ([], False))
        crossing = Mesh(first+((1., 1., 1e-12), (1., .2, -1e-12)), ((0, 1, 2), (0, 3, 4)))
        hits, _ = find_self_intersections(crossing, include_adjacent=True)
        self.assertEqual((len(hits), hits[0].coplanar), (1, False))
        # A pair produced by the C++ example has vertices near the opposite
        # plane, but its exact plane slices meet only at the shared vertex.
        vertices = ((3.217684852, .17916726652, .143523296),
                    (3.2823836013, .16100214944, .11634219415),
                    (3.2838472375, .16499551028, .11233196734),
                    (3.296190666, .07925242527, .17057499009),
                    (3.2952400346, .061526132574, .18484554167))
        observed = Mesh(vertices, ((1, 2, 0), (3, 0, 4)))
        self.assertEqual(find_self_intersections(observed, include_adjacent=True), ([], False))

    def test_shared_edge_overlap_and_well_formed_sphere(self):
        vertices = ((0., 0., 0.), (2., 0., 0.), (0., 1., 0.), (1., 1., 0.), (1., -1., 0.))
        overlap = Mesh(vertices, ((0, 1, 2), (1, 0, 3)))
        hits, _ = find_self_intersections(overlap, include_adjacent=True)
        self.assertEqual((len(hits), hits[0].coplanar), (1, True))
        adjacent = Mesh(vertices, ((0, 1, 2), (1, 0, 4)))
        self.assertEqual(find_self_intersections(adjacent, include_adjacent=True), ([], False))
        sphere = ico_sphere((0., 0., 0.), 1., 1)
        self.assertEqual(find_self_intersections(sphere, include_adjacent=True), ([], False))

    def test_duplicate_faces_are_detected_when_including_adjacent(self):
        mesh = Mesh(((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)), ((0, 1, 2), (0, 2, 1)))
        hits, _ = find_self_intersections(mesh, include_adjacent=True)
        self.assertEqual((len(hits), hits[0].coplanar), (1, True))


if __name__ == "__main__":
    unittest.main()
