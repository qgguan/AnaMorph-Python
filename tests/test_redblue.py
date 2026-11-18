import random
import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from fractions import Fraction
from math import cos, dist, fsum, pi, sin, sqrt
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from anamorph.mesh import Mesh, cross, dot, read_obj
from anamorph.canal_mesh import ico_sphere
from anamorph.mesh_intersections import find_self_intersections
from anamorph.source_random import WindowsCRandom
from anamorph.redblue import (RedBlueComplexEdges, RedBlueDisjoint, RedBlueError, RedBlueMultipleLoops, RedBlueTriangulationError, SourceDelaunayLimitWarning,
                              RedBlueNumericalEdgeCase, RedBlueAffectedCircleTrivial, _edge_face_hit, intersection_face_circles,
                              zip_intersection_circles, redblue_union, redblue_union_with_retries,
                              redblue_intersection, redblue_difference, redblue_boolean_with_retries,
                              _face_bvh, _query_face_bvh, _boxes_overlap, _point_inside_surface,
                              _projected_segment_disjoint, _edge_faces, _RedBlueCache, _triangulate_polygon,
                              _triangle_unit_normal, _generate_intersection_tuples,
                              cyclically_order_tuples,
                              _split_edge_at_parameters, _split_face_at_point)


def refined_tetrahedron():
    vertices = [(0., 0., 0.), (2., 0., 0.), (0., 2., 0.), (0., 0., 2.)]
    faces = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
    for _ in range(2):
        edges = {}
        new_faces = []
        def midpoint(a, b):
            edge = tuple(sorted((a, b)))
            if edge not in edges:
                edges[edge] = len(vertices)
                vertices.append(tuple((vertices[a][j]+vertices[b][j])/2 for j in range(3)))
            return edges[edge]
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))
        faces = new_faces
    return Mesh(tuple(vertices), tuple(faces))


class RedBlueTests(unittest.TestCase):
    def test_source_polygon_modes_work_for_all_boolean_operations(self):
        red = refined_tetrahedron()
        dx, dy, dz, scale = .08794858485146739, -.25131913562220887, .47920496341231167, 1.1169174342387775
        blue = Mesh(tuple((x*scale+dx, y*scale+dy, z*scale+dz) for x, y, z in red.vertices), red.faces)
        wrappers = {"union": redblue_union, "intersection": redblue_intersection, "difference": redblue_difference}
        volumes = {}
        for operation, wrapper in wrappers.items():
            with self.subTest(operation=operation):
                settings = dict(source_delaunay=True, random_seed=1, random_generator="windows-c")
                expected = redblue_boolean_with_retries(red, blue, operation=operation, **settings)
                actual = wrapper(red, blue, **settings)
                self.assertEqual(actual, expected)
                if operation == "union":
                    self.assertEqual(redblue_union_with_retries(red, blue, **settings), expected)
                stats = actual.statistics()
                self.assertEqual((stats.boundary_edges, stats.nonmanifold_edges), (0, 0))
                signed_volume = fsum(dot(actual.vertices[a], cross(actual.vertices[b], actual.vertices[c]))/6
                                     for a, b, c in actual.faces)
                self.assertGreater(signed_volume, 0.)
                volumes[operation] = stats.volume
        self.assertAlmostEqual(volumes["union"]+volumes["intersection"],
                               red.statistics().volume+blue.statistics().volume)
        self.assertAlmostEqual(volumes["difference"]+volumes["intersection"], red.statistics().volume)
        for invalid in ({"source_delaunay": 1}, {"source_ear_clipping": "yes"},
                        {"source_delaunay_allow_limit": 1},
                        {"source_delaunay_allow_limit": True, "source_delaunay": 1},
                        {"source_delaunay": True, "random_seed": 1.5},
                        {"source_delaunay": True, "random_generator": "invalid"}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                redblue_boolean_with_retries(red, blue, **invalid)

    def test_source_random_stream_persists_across_repairs(self):
        mesh = refined_tetrahedron()
        calls = []
        def operation(red, blue, **options):
            rng = options["_source_ear_rng"]
            calls.append((rng, rng.rand(), options["_source_delaunay"]))
            if len(calls) == 1:
                raise RedBlueAffectedCircleTrivial("injected split", red=True, face=0, position=(.1, .1, .1))
            return red
        with patch("anamorph.redblue.redblue_boolean", side_effect=operation), \
                patch("anamorph.redblue._split_face_at_point", return_value=mesh):
            actual = redblue_boolean_with_retries(mesh, mesh, source_delaunay=True,
                                                 random_seed=1, random_generator="windows-c")
        self.assertIs(actual, mesh)
        self.assertIs(calls[0][0], calls[1][0])
        expected = WindowsCRandom(1)
        self.assertEqual([item[1] for item in calls], [expected.rand(), expected.rand()])
        self.assertTrue(all(item[2] for item in calls))

    def test_source_boolean_modes_preserve_seeded_tetrahedron_volume(self):
        vertices = ((0., 0., 0.), (2., 0., 0.), (0., 2., 0.), (0., 0., 2.))
        faces = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))
        red = Mesh(vertices, faces)
        rng = random.Random(43937)
        for case in range(20):
            offset = tuple(rng.uniform(-.7, .7) for _ in range(3))
            scale = rng.uniform(.8, 1.2)
            blue = Mesh(tuple(tuple(x*scale+offset[i] for i, x in enumerate(point)) for point in vertices), faces)
            volumes = {}
            for operation in ("union", "intersection", "difference"):
                with self.subTest(case=case, operation=operation):
                    mesh = redblue_boolean_with_retries(red, blue, operation=operation,
                                                         source_delaunay=True, random_generator="windows-c")
                    stats = mesh.statistics()
                    self.assertEqual((stats.boundary_edges, stats.nonmanifold_edges), (0, 0))
                    self.assertEqual(find_self_intersections(mesh, include_adjacent=True, max_hits=1)[0], [])
                    volumes[operation] = stats.volume
            self.assertAlmostEqual(volumes["union"]+volumes["intersection"],
                                   red.statistics().volume+blue.statistics().volume)
            self.assertAlmostEqual(volumes["difference"]+volumes["intersection"], red.statistics().volume)

    def test_nonconvergent_source_cleanup_fails_without_hiding_it(self):
        red = refined_tetrahedron()
        offset = (-.3004177194797668, -.20031414355086757, -.2868115892937734)
        scale = .8761913254934592
        blue = Mesh(tuple(tuple(x*scale+offset[i] for i, x in enumerate(point)) for point in red.vertices), red.faces)
        with self.assertRaisesRegex(RedBlueTriangulationError, "did not converge"):
            redblue_boolean_with_retries(red, blue, source_delaunay=True, random_seed=1,
                                         random_generator="windows-c")
        result = redblue_boolean_with_retries(red, blue)
        self.assertEqual((result.statistics().boundary_edges, result.statistics().nonmanifold_edges), (0, 0))

    def test_source_iteration_limit_can_be_kept_explicitly(self):
        red = refined_tetrahedron()
        offset = (-.3004177194797668, -.20031414355086757, -.2868115892937734)
        blue = Mesh(tuple(tuple(x*.8761913254934592+offset[i] for i, x in enumerate(point))
                          for point in red.vertices), red.faces)
        settings = dict(source_delaunay_allow_limit=True, random_seed=1, random_generator="windows-c")
        with self.assertWarns(SourceDelaunayLimitWarning):
            result = redblue_boolean_with_retries(red, blue, **settings)
        self.assertEqual((len(result.vertices), len(result.faces)), (82, 160))
        self.assertEqual((result.statistics().boundary_edges, result.statistics().nonmanifold_edges), (0, 0))
        self.assertEqual(find_self_intersections(result, include_adjacent=True)[0], [])
        with self.assertWarns(SourceDelaunayLimitWarning):
            self.assertEqual(redblue_union_with_retries(red, blue, **settings), result)
        root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as directory:
            red_path, blue_path, output = (Path(directory)/name for name in ("red.obj", "blue.obj", "out.obj"))
            red.write_obj(red_path)
            blue.write_obj(blue_path)
            run = subprocess.run([sys.executable, "-m", "anamorph", "mesh-boolean", str(red_path), str(blue_path),
                                  str(output), "--source-delaunay-allow-limit", "--random-seed", "1",
                                  "--random-generator", "windows-c"], cwd=root, capture_output=True,
                                 text=True, check=True, timeout=30)
            self.assertIn("SourceDelaunayLimitWarning", run.stderr)
            self.assertEqual(json.loads(run.stdout)["faces"], 160)
            self.assertEqual(read_obj(output).faces, result.faces)

    def test_cli_source_delaunay_boolean_modes(self):
        red = refined_tetrahedron()
        dx, dy, dz, scale = .08794858485146739, -.25131913562220887, .47920496341231167, 1.1169174342387775
        blue = Mesh(tuple((x*scale+dx, y*scale+dy, z*scale+dz) for x, y, z in red.vertices), red.faces)
        root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as directory:
            red_path, blue_path, output = (Path(directory)/name for name in ("red.obj", "blue.obj", "result.obj"))
            red.write_obj(red_path)
            blue.write_obj(blue_path)
            for operation in ("union", "intersection", "difference"):
                for flag, keyword in (("--source-ear-clipping", "source_ear_clipping"),
                                      ("--source-delaunay", "source_delaunay")):
                    with self.subTest(operation=operation, flag=flag):
                        run = subprocess.run([sys.executable, "-m", "anamorph", "mesh-boolean", str(red_path),
                                              str(blue_path), str(output), "--operation", operation,
                                              flag, "--random-generator", "windows-c", "--random-seed", "1"],
                                             cwd=root, capture_output=True, text=True, check=True, timeout=30)
                        report = json.loads(run.stdout)
                        actual = read_obj(output)
                        expected = redblue_boolean_with_retries(red, blue, operation=operation, random_seed=1,
                                                                 random_generator="windows-c", **{keyword: True})
                        self.assertEqual(report["operation"], operation)
                        self.assertEqual(actual.faces, expected.faces)
                        self.assertLess(max(dist(a, b) for a, b in zip(actual.vertices, expected.vertices)), 1e-12)

    def test_cut_polygon_preserves_thin_valid_boundaries(self):
        # The old fixed 1e-14 projected-area tolerance rejected all of these
        # despite their distinct, representable vertices and positive area.
        polygons = (((0., 0.), (1., 0.), (1., 1e-16)),
                    ((0., 0.), (3., 0.), (3., 2e-16), (1.5, 1e-16), (0., 3e-16)))
        def twice_area(points):
            return sum((Fraction(a[0])*Fraction(b[1])-Fraction(b[0])*Fraction(a[1])
                        for a, b in zip(points, points[1:]+points[:1])), Fraction(0))
        for scale in (1e-120, 1., 1e120):
            for polygon in polygons:
                for reverse in (False, True):
                    points = tuple((scale*x, scale*y, 0.) for x, y in polygon)
                    if reverse:
                        points = tuple(reversed(points))
                    with self.subTest(scale=scale, polygon=polygon, reverse=reverse):
                        triangles = _triangulate_polygon(list(range(len(points))), points, (0., 0., 1.),
                                                         basis=(1., 0., 0.))
                        self.assertEqual(len(triangles), len(points)-2)
                        areas = [twice_area(tuple(points[i][:2] for i in face)) for face in triangles]
                        self.assertTrue(all(area > 0 for area in areas))
                        self.assertEqual(sum(areas), abs(twice_area(tuple(p[:2] for p in points))))
                        incidence = Counter(tuple(sorted((a, b))) for face in triangles
                                            for a, b in zip(face, face[1:]+face[:1]))
                        boundary = {tuple(sorted((i, (i+1)%len(points)))) for i in range(len(points))}
                        self.assertEqual({edge for edge, count in incidence.items() if count == 1}, boundary)
                        self.assertTrue(all(count in (1, 2) for count in incidence.values()))

    def test_cut_polygon_flips_nearly_cocircular_nondelaunay_diagonal(self):
        # Ear clipping initially chooses (1, 3). Raising vertex 3 moves
        # vertex 2 just inside the other triangle's circumcircle; the old
        # fixed determinant threshold incorrectly kept that diagonal.
        for scale in (1e-120, 1., 1e120):
            for delta, diagonal in ((-1e-15, {1, 3}), (0., {1, 3}), (1e-15, {0, 2})):
                polygon = tuple((scale*x, scale*y, 0.) for x, y in
                                ((0., 0.), (1., 0.), (1., 1.), (0., 1.+delta)))
                with self.subTest(scale=scale, delta=delta):
                    triangles = _triangulate_polygon([0, 1, 2, 3], polygon, (0., 0., 1.),
                                                     basis=(1., 0., 0.))
                    self.assertEqual(set(triangles[0]) & set(triangles[1]), diagonal)

    def test_cut_polygon_rejects_self_crossings_and_overlapping_edges(self):
        crossing = ((0., 2., 0.), (3., 0., 0.), (3., 3., 0.), (1., 2., 0.), (4., 3., 0.))
        backtracking = ((0., 0., 0.), (2., 0., 0.), (1., 0., 0.), (2., 2., 0.), (0., 2., 0.))
        nonplanar = ((0., 0., 0.), (2., 0., 0.), (2., 2., 1.), (0., 2., 0.))
        concave = ((0., 0., 0.), (3., 0., 0.), (3., 3., 0.), (1.5, 1., 0.), (0., 3., 0.))
        for scale in (1e-120, 1., 1e120):
            def scaled(points):
                return tuple(tuple(scale*x for x in point) for point in points)
            for invalid in (crossing, backtracking, nonplanar):
                with self.subTest(scale=scale, invalid=invalid), self.assertRaises(RedBlueTriangulationError):
                    _triangulate_polygon(list(range(len(invalid))), scaled(invalid), (0., 0., 1.))
            triangles = _triangulate_polygon(list(range(len(concave))), scaled(concave), (0., 0., 1.))
            self.assertEqual(len(triangles), len(concave)-2)

    def test_cut_polygon_delaunay_is_invariant_under_rigid_rotation(self):
        u = (1/sqrt(2), -1/sqrt(2), 0.)
        v = (1/sqrt(6), 1/sqrt(6), -2/sqrt(6))
        normal = (1/sqrt(3),)*3
        # Dropping a coordinate used to select diagonal (0, 2) after this
        # rotation, even though the planar polygon selects diagonal (1, 3).
        polygon = [(0., 0., 0.), (.7647458432741085, 0., 0.),
                   (.5181514489679929, 1.1205498235699354, 0.), (0., .9133812323456821, 0.)]
        def rotated(points):
            return [tuple(x*u[k]+y*v[k] for k in range(3)) for x, y, _ in points]
        def triangles(points, n, basis):
            return {frozenset(triangle) for triangle in
                    _triangulate_polygon(list(range(len(points))), points, n, basis=basis)}
        self.assertEqual(triangles(polygon, (0., 0., 1.), (1., 0., 0.)),
                         triangles(rotated(polygon), normal, u))
        self.assertEqual(triangles(polygon, (0., 0., 1.), (1., 0., 0.)),
                         triangles(polygon, (0., 0., 1.), (1., 0., 1.)))
        rng = random.Random(30928)
        for case in range(40):
            count = rng.randrange(4, 11)
            points = []
            for i in range(count):
                angle = 2*pi*(i+rng.uniform(-.15, .15))/count
                radius = rng.uniform(.7, 1.3)
                points.append((radius*cos(angle), radius*sin(angle), 0.))
            with self.subTest(case=case):
                self.assertEqual(triangles(points, (0., 0., 1.), (1., 0., 0.)),
                                 triangles(rotated(points), normal, u))

    def test_edge_incidence_keeps_face_ids_and_orientation(self):
        mesh = Mesh(((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.)),
                    ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)))
        self.assertEqual(_edge_faces(mesh), {(0, 2): [0, 2], (1, 2): [0, 3], (0, 1): [0, 1],
                                             (1, 3): [1, 3], (0, 3): [1, 2], (2, 3): [2, 3]})
        invalid = Mesh(mesh.vertices[:-1]+((0., 0., float("nan")),), mesh.faces)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            _edge_faces(invalid)

    def test_surface_incidence_rejects_vertex_pinches_and_duplicate_faces(self):
        # Both closed tetrahedra are edge-manifold, but their common vertex
        # has two disconnected face fans and cannot define one surface.
        vertices = ((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.),
                    (-1., 0., 0.), (0., -1., 0.), (0., 0., -1.))
        faces = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3),
                 (0, 5, 4), (0, 4, 6), (0, 6, 5), (4, 5, 6))
        pinched = Mesh(vertices, faces)
        with self.assertRaisesRegex(ValueError, "vertex-manifold"):
            _edge_faces(pinched)
        with self.assertRaisesRegex(ValueError, "vertex-manifold"):
            redblue_union(pinched, refined_tetrahedron())
        # Disconnected open fans also create an invalid boundary junction.
        open_pinched = Mesh(vertices, ((0, 2, 1), (0, 5, 4)))
        with self.assertRaisesRegex(ValueError, "vertex-manifold"):
            _edge_faces(open_pinched, allow_boundary=True)
        # Two opposite copies of one triangle have valid edge counts and
        # edge orientations but do not form a valid triangulated surface.
        duplicate = Mesh(vertices[:3], ((0, 1, 2), (0, 2, 1)))
        with self.assertRaisesRegex(ValueError, "distinct triangular faces"):
            _edge_faces(duplicate)
        # A single boundary loop around a removed face remains supported.
        open_tetrahedron = Mesh(vertices[:4], faces[:3])
        self.assertEqual(sum(len(adjacent) == 1 for adjacent in
                             _edge_faces(open_tetrahedron, allow_boundary=True).values()), 3)

    def test_acceleration_cache_does_not_reuse_mutable_mesh_data(self):
        tetra = refined_tetrahedron()
        mesh = Mesh(list(tetra.vertices), list(tetra.faces))
        cache = _RedBlueCache()
        first = cache.prepare(mesh, red=True)
        mesh.vertices[0] = (-.1, 0., 0.)
        second = cache.prepare(mesh, red=True)
        self.assertIsNot(first, second)

    def test_acceleration_cache_invalidates_when_operand_changes(self):
        red = refined_tetrahedron()
        blue = Mesh(tuple((x+.1, y-.25, z+.48) for x, y, z in red.vertices), red.faces)
        cache = _RedBlueCache()
        first = cache.prepare(red, red=True)
        self.assertIs(first, cache.prepare(red, red=True))
        self.assertIsNot(first, cache.prepare(blue, red=True))
        self.assertIsNot(cache.prepare(blue, red=True), cache.prepare(blue, red=False))

    def test_rounded_split_repairs_raise_retryable_numerical_errors(self):
        base = 1e15
        mesh = Mesh(((base, 0., 0.), (base+.25, 0., 0.), (base, 1., 0.), (base, 0., 1.)),
                    ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)))
        _edge_faces(mesh)
        with self.assertRaisesRegex(RedBlueNumericalEdgeCase, "not distinct"):
            _split_edge_at_parameters(mesh, (0, 1), [.1])
        with self.assertRaisesRegex(RedBlueNumericalEdgeCase, "degenerate or reversed"):
            _split_face_at_point(mesh, 0, mesh.vertices[0])
        with self.assertRaisesRegex(RedBlueNumericalEdgeCase, "degenerate or reversed"):
            _split_face_at_point(mesh, 0, (base+.25, .5, 0.))
        self.assertTrue(_edge_faces(_split_edge_at_parameters(mesh, (0, 1), [.5])))
        self.assertTrue(_edge_faces(_split_face_at_point(mesh, 0, (base+.125, .25, 0.))))

    def test_trivial_circle_split_position_avoids_coordinate_sum_overflow(self):
        mesh = refined_tetrahedron()
        expected = (1e308, 1e-300, -1e308)
        hits = [SimpleNamespace(position=expected, opposite_face=0) for _ in range(3)]
        with patch("anamorph.redblue._generate_intersection_tuples", side_effect=[([], []), (hits, [])]):
            with self.assertRaises(RedBlueAffectedCircleTrivial) as caught:
                intersection_face_circles(mesh, mesh)
        self.assertTrue(caught.exception.red)
        self.assertEqual(caught.exception.position, expected)

    def test_segment_triangle_intersection_and_edge_cases(self):
        triangle = ((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
        hit = _edge_face_hit((.2, .3, 1.), (.2, .3, -1.), triangle, 1e-10)
        self.assertAlmostEqual(hit[0], .5)
        self.assertAlmostEqual(hit[1], .2)
        self.assertAlmostEqual(hit[2], .3)
        self.assertEqual(hit[3], (.2, .3, 0.))
        self.assertIsNone(_edge_face_hit((2., 2., 1.), (2., 2., -1.), triangle, 1e-10))
        # Bounding boxes overlap, but a separating line proves that this
        # coplanar segment misses the triangle.
        self.assertIsNone(_edge_face_hit((-.1, .95, 0.), (.1, 1.15, 0.), triangle, 1e-10))
        with self.assertRaises(RedBlueNumericalEdgeCase):
            _edge_face_hit((-.1, .1, 0.), (.2, .1, 0.), triangle, 1e-10)
        with self.assertRaises(RedBlueNumericalEdgeCase):
            _edge_face_hit((0., .3, 1.), (0., .3, -1.), triangle, 1e-10)

    def test_edge_face_predicate_survives_extreme_uniform_scales(self):
        for scale in (1e-200, 1e-150, 1e-60, 1., 1e60, 1e150, 1e200):
            with self.subTest(scale=scale):
                triangle = ((0., 0., 0.), (scale, 0., 0.), (0., scale, 0.))
                start, end = (.2*scale, .3*scale, scale), (.2*scale, .3*scale, -scale)
                hit = _edge_face_hit(start, end, triangle, 1e-10)
                self.assertIsNotNone(hit)
                for actual, expected in zip(hit[:3], (.5, .2, .3)):
                    self.assertAlmostEqual(actual, expected, places=12)
                self.assertIsNone(_edge_face_hit((.8*scale, .8*scale, scale),
                                                 (.8*scale, .8*scale, -scale), triangle, 1e-10))
                with self.assertRaises(RedBlueNumericalEdgeCase):
                    _edge_face_hit((0., .3*scale, scale), (0., .3*scale, -scale), triangle, 1e-10)
                tetra = Mesh(((0., 0., 0.), (scale, 0., 0.), (0., scale, 0.), (0., 0., scale)),
                             ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)))
                self.assertEqual(len(_edge_faces(tetra)), 6)

    def test_edge_face_exact_fallback_finds_crossing_of_thin_triangle(self):
        # The former float-only barycentric calculation returned None here,
        # even though exact input-float arithmetic gives (s,t,lambda) =
        # (1/4, 1/2, 3/8) strictly inside the segment and triangle.
        length = 2.**52
        triangle = ((0., 0., 0.), (length, length, 0.), (length, length+1., -4.))
        start = (.75*length, .75*length+.5, -.5)
        end = (.75*length, .75*length+.5, -4.5)
        normal = tuple(v/sqrt(33.) for v in (-4., 4., 1.))
        for scale in (2.**-400, 1., 2.**400):
            with self.subTest(scale=scale):
                scaled = lambda p: tuple(scale*x for x in p)
                tri = tuple(scaled(p) for p in triangle)
                hit = _edge_face_hit(scaled(start), scaled(end), tri, 1e-10)
                self.assertEqual(hit[:3], (.375, .25, .5))
                self.assertEqual(hit[3], scaled((.75*length, .75*length+.5, -2.)))
                for got, want in zip(hit[4], normal):
                    self.assertAlmostEqual(got, want, places=14)
                for got, want in zip(_triangle_unit_normal(*tri), normal):
                    self.assertAlmostEqual(got, want, places=14)
                outside_start = (.75*length, .75*length+1., -.5)
                outside_end = (.75*length, .75*length+1., -4.5)
                self.assertIsNone(_edge_face_hit(scaled(outside_start), scaled(outside_end), tri, 1e-10))
                boundary_start = (.75*length, .75*length, 1.)
                boundary_end = (.75*length, .75*length, -1.)
                with self.assertRaises(RedBlueNumericalEdgeCase):
                    _edge_face_hit(scaled(boundary_start), scaled(boundary_end), tri, 1e-10)
        thin = ((0., 0., 0.), (1., 0., 0.), (1., 1e-200, 0.))
        self.assertEqual(_triangle_unit_normal(*thin), (0., 0., 1.))
        hit = _edge_face_hit((.75, .5e-200, 1.), (.75, .5e-200, -1.), thin, 1e-10)
        self.assertEqual(hit[:3], (.5, .25, .5))
        # An exact parallel plane offset or separating projection can reject
        # the segment, whereas coplanar overlapping geometry stays ambiguous.
        self.assertIsNone(_edge_face_hit((2*length, 2*length, 0.),
                                          (2*length+16, 2*length+16, 0.), triangle, 1e-10))
        with self.assertRaises(RedBlueNumericalEdgeCase):
            _edge_face_hit((.75*length, .75*length+.5, -2.),
                           (.75*length+16, .75*length+16+.5, -2.), triangle, 1e-10)
        with self.assertRaisesRegex(ValueError, "degenerate"):
            _triangle_unit_normal((0., 0., 0.), (1., 1., 1.), (2., 2., 2.))

    def test_thin_edge_face_results_match_independent_exact_test(self):
        from anamorph.mesh_intersections import _exact_segment_intersects_triangle
        rng = random.Random(491734)
        tested = positives = 0
        for _ in range(1000):
            length = float(2**rng.randint(43, 52))
            delta = float(2**rng.randint(0, 6))
            slope = rng.choice((0., .5, 1., 2., -1.))
            dz = float(rng.randint(-4, 4))*delta
            triangle = ((0., 0., 0.), (length, length, slope*length),
                        (length, length+delta, slope*length+dz))
            s, t = (rng.choice((-.25, .125, .25, .375, .5, .75, 1.)) for _ in range(2))
            point = tuple(s*triangle[1][i]+t*triangle[2][i] for i in range(3))
            height = delta*rng.choice((1., 2., 4.))
            start = (point[0], point[1], point[2]+height)
            end = (point[0], point[1], point[2]-height)
            expected = _exact_segment_intersects_triangle(start, end, triangle)
            try:
                actual = _edge_face_hit(start, end, triangle, 1e-10) is not None
            except RedBlueNumericalEdgeCase:
                # Contacts at a boundary remain intentionally unresolved.
                continue
            self.assertEqual(actual, expected, (triangle, start, end))
            tested += 1
            positives += actual
        self.assertGreater(tested, 400)
        self.assertGreater(positives, 100)

    def test_long_edge_crossing_preserves_parameters_and_tuple_orientation(self):
        triangle = ((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
        start = (-2251799813685248., .25, 4503599627370496.)
        end = (3377699720527873.5, .25, -6755399441055744.)
        # The previous float-only computation rejected this strict interior
        # crossing because it computed s=0.78125 instead of the exact 0.6.
        for scale in (2.**-400, 1., 2.**400):
            with self.subTest(scale=scale):
                scaled = lambda p: tuple(scale*x for x in p)
                hit = _edge_face_hit(scaled(start), scaled(end), tuple(map(scaled, triangle)), 1e-10)
                self.assertEqual(hit[:3], (.4, .6, .25))
                self.assertEqual(hit[3], scaled((.6, .25, 0.)))

        # This almost-tangent edge starts on the negative side of the face.
        # Rounded dot products previously classified its outside endpoint as
        # the start instead of the end, reversing the red-blue tuple.
        face_vertices = ((0., 0., 0.), (1., -1., 0.), (1., 0., -1.))
        start = (-2251799813685248., 2251799813685247., .75)
        end = (2251799813685249., -2251799813685247.8, -1.)
        for scale in (2.**-400, 1., 2.**400):
            with self.subTest(orientation_scale=scale):
                scaled = lambda p: tuple(scale*x for x in p)
                face_mesh = Mesh(tuple(map(scaled, face_vertices)), ((0, 1, 2),))
                edge_mesh = Mesh((scaled(start), scaled(end)), ())
                hits, complex_edges = _generate_intersection_tuples(
                    edge_mesh, {(0, 1): [0, 1]}, _face_bvh(face_mesh), red=True, eps=1e-10)
                self.assertEqual(complex_edges, [])
                self.assertEqual(len(hits), 1)
                self.assertEqual((hits[0].edge_out, hits[0].edge_in), (1, 0))
                self.assertEqual(hits[0].position, scaled((.5, -.375, -.125)))

    def test_long_edge_intersections_match_exact_plane_projection(self):
        rng = random.Random(90481)
        triangle = ((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
        checked = positives = 0
        for _ in range(600):
            magnitude = float(2**rng.randint(20, 52))
            a, b = rng.choice(((1, 1), (1, 2), (2, 1), (3, 1), (1, 3), (2, 3), (3, 2)))
            target = rng.choice((-.5, -.25, .0625, .125, .25, .375, .5, .625, .75, .875, 1.25))
            y = rng.choice((.0625, .125, .25, .375, .5, .625, .75))
            start = (-magnitude, y, a*magnitude)
            end = (magnitude*b/a+target*(a+b)/a, y, -b*magnitude)
            parameter = Fraction(start[2])/(Fraction(start[2])-Fraction(end[2]))
            x = Fraction(start[0])+parameter*(Fraction(end[0])-Fraction(start[0]))
            truth = 0 < x and x+Fraction(y) < 1
            if not truth and min(abs(x), abs(x+Fraction(y)-1)) < Fraction(1, 10**9):
                continue
            try:
                hit = _edge_face_hit(start, end, triangle, 1e-10)
            except RedBlueNumericalEdgeCase:
                continue
            self.assertEqual(hit is not None, truth, (start, end))
            if hit is not None:
                self.assertAlmostEqual(hit[1], float(x), delta=1e-10)
                positives += 1
            checked += 1
        self.assertGreater(checked, 400)
        self.assertGreater(positives, 150)

    def test_unrepresentable_intersection_points_raise_instead_of_cutting(self):
        # Exact segment/face parameters are strictly interior, but rounding
        # their point to doubles moves it outside this thin triangle.
        length = 2.**52
        thin = ((0., 0., 0.), (length, length, 0.), (length, length+1., -1.))
        start = (.75*length-1., .75*length+1.5, 3.75)
        end = (.75*length+1., .75*length-1.5, -8.25)
        from anamorph.mesh_intersections import _exact_segment_intersects_triangle
        self.assertTrue(_exact_segment_intersects_triangle(start, end, thin))
        with self.assertRaisesRegex(RedBlueNumericalEdgeCase, "outside the face interior"):
            _edge_face_hit(start, end, thin, 1e-10)

        # Here the rounded point stays inside the face but no longer lies
        # close to the crossing edge. A representable crossing on the same
        # translated face remains accepted.
        translated = ((length, length, 0.), (length+4., length, 0.), (length, length+4., 0.))
        start = (length+1., length+1., 1.)
        end = (length+2., length+1., -2.)
        self.assertTrue(_exact_segment_intersects_triangle(start, end, translated))
        with self.assertRaisesRegex(RedBlueNumericalEdgeCase, "too far from the crossing edge"):
            _edge_face_hit(start, end, translated, 1e-10)
        hit = _edge_face_hit(start, (length+1., length+1., -1.), translated, 1e-10)
        self.assertEqual(hit[3], (length+1., length+1., 0.))

    def test_scaled_edge_face_decisions_match_exact_plane_crossings(self):
        from fractions import Fraction
        rng = random.Random(55501)
        for scale in (1e-150, 1e-60, 1., 1e60, 1e150):
            triangle = ((0., 0., 0.), (scale, 0., 0.), (0., scale, 0.))
            for case in range(20):
                start = (rng.uniform(-.5, 1.5)*scale, rng.uniform(-.5, 1.5)*scale,
                         rng.uniform(.1, 2.)*scale)
                end = (rng.uniform(-.5, 1.5)*scale, rng.uniform(-.5, 1.5)*scale,
                       -rng.uniform(.1, 2.)*scale)
                size = Fraction(scale)
                parameter = Fraction(start[2])/(Fraction(start[2])-Fraction(end[2]))
                x = Fraction(start[0])+parameter*(Fraction(end[0])-Fraction(start[0]))
                y = Fraction(start[1])+parameter*(Fraction(end[1])-Fraction(start[1]))
                if min(abs(x), abs(y), abs(size-x-y)) < size/1000000:
                    continue
                expected = x > 0 and y > 0 and x+y < size
                with self.subTest(scale=scale, case=case):
                    self.assertEqual(_edge_face_hit(start, end, triangle, 1e-10) is not None, expected)

    def test_unrepresentable_edge_face_tolerance_is_not_reported_disjoint(self):
        triangle = ((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
        with self.assertRaisesRegex(RedBlueNumericalEdgeCase, "underflows"):
            _edge_face_hit((.2, .3, 1e-320), (.2, .3, -1e-320), triangle, 1e-10)

    def test_component_winding_classifies_and_rejects_touching_points(self):
        mesh = refined_tetrahedron()
        self.assertTrue(_point_inside_surface((.2, .3, .4), mesh, 1e-10))
        self.assertFalse(_point_inside_surface((4., 4., 4.), mesh, 1e-10))
        with self.assertRaises(RedBlueNumericalEdgeCase):
            _point_inside_surface((.3, .4, 0.), mesh, 1e-10)

    def test_component_winding_is_scale_invariant(self):
        mesh = refined_tetrahedron()
        rng = random.Random(19027)
        points = [tuple(rng.uniform(-.5, 2.5) for _ in range(3)) for _ in range(35)]
        for scale in (1e-200, 1e-150, 1., 1e150, 1e200):
            scaled = Mesh(tuple(tuple(value*scale for value in point) for point in mesh.vertices), mesh.faces)
            with self.subTest(scale=scale):
                self.assertTrue(_point_inside_surface((.2*scale, .3*scale, .4*scale), scaled, 1e-10))
                self.assertFalse(_point_inside_surface((4*scale, 4*scale, 4*scale), scaled, 1e-10))
                with self.assertRaises(RedBlueNumericalEdgeCase):
                    _point_inside_surface((.3*scale, .4*scale, 0.), scaled, 1e-10)
                for point in points:
                    expected = min(point) > 0 and sum(point) < 2
                    self.assertEqual(_point_inside_surface(tuple(scale*x for x in point), scaled, 1e-10), expected)

    def test_projected_separation_never_discards_exact_integer_intersections(self):
        def orient(a, b, c):
            return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
        def on(a, b, p):
            return min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])
        def intersects(a, b, c, d):
            o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
            return ((o1*o2 < 0 and o3*o4 < 0) or (o1 == 0 and on(a, b, c)) or
                    (o2 == 0 and on(a, b, d)) or (o3 == 0 and on(c, d, a)) or (o4 == 0 and on(c, d, b)))
        def contains(point, triangle):
            signs = [orient(triangle[i], triangle[(i+1)%3], point) for i in range(3)]
            return all(s >= 0 for s in signs) or all(s <= 0 for s in signs)
        rng = random.Random(197)
        for _ in range(500):
            a, b, c, start, end = [tuple(rng.randrange(-8, 9) for _ in range(2)) for _ in range(5)]
            if orient(a, b, c) == 0 or start == end:
                continue
            triangle = (a, b, c)
            contact = (contains(start, triangle) or contains(end, triangle) or
                       any(intersects(start, end, triangle[i], triangle[(i+1)%3]) for i in range(3)))
            xyz = lambda point: (float(point[0]), float(point[1]), 0.)
            disjoint = _projected_segment_disjoint(xyz(start), xyz(end), tuple(map(xyz, triangle)),
                                                   (0., 0., 1.), 1e-10)
            if contact:
                self.assertFalse(disjoint, (triangle, start, end))

    def test_bvh_candidate_pairs_match_brute_force(self):
        mesh = refined_tetrahedron()
        tree = _face_bvh(mesh, leaf_size=4)
        rng = random.Random(41)
        for _ in range(100):
            start = tuple(rng.uniform(-1., 3.) for _ in range(3))
            end = tuple(rng.uniform(-1., 3.) for _ in range(3))
            actual = {fid for fid, _ in _query_face_bvh(tree, start, end, 1e-10)}
            expected = {fid for fid, face in enumerate(mesh.faces)
                        if _boxes_overlap((start, end), tuple(mesh.vertices[i] for i in face), 1e-10)}
            self.assertEqual(actual, expected)

    def test_bvh_candidates_scale_with_geometry_and_keep_detected_hits(self):
        source = refined_tetrahedron()
        baseline = None
        rng = random.Random(74198)
        for scale in (1e-60, 1., 1e60):
            mesh = Mesh(tuple(tuple(scale*value for value in point) for point in source.vertices), source.faces)
            tree = _face_bvh(mesh, leaf_size=4)
            edges = _edge_faces(mesh)
            edge_candidates = [{fid for fid, _ in _query_face_bvh(tree, mesh.vertices[a], mesh.vertices[b], 1e-10)}
                               for a, b in edges]
            if baseline is None:
                baseline = edge_candidates
            else:
                self.assertEqual(edge_candidates, baseline)
            self.assertLess(sum(map(len, edge_candidates)), len(edges)*len(mesh.faces)//2)
            for _ in range(30):
                start = tuple(rng.uniform(-1., 3.)*scale for _ in range(3))
                end = tuple(rng.uniform(-1., 3.)*scale for _ in range(3))
                candidates = {fid for fid, _ in _query_face_bvh(tree, start, end, 1e-10)}
                for fid, face in enumerate(mesh.faces):
                    triangle = tuple(mesh.vertices[i] for i in face)
                    try:
                        hit = _edge_face_hit(start, end, triangle, 1e-10)
                    except RedBlueNumericalEdgeCase:
                        continue
                    if hit is not None:
                        self.assertIn(fid, candidates)

    def test_bvh_retains_near_coplanar_narrow_phase_ambiguity(self):
        for scale in (1e-150, 1e-60, 1., 1e60, 1e150):
            with self.subTest(scale=scale):
                triangle = ((0., 0., 0.), (scale, scale, 0.), (0., scale, 0.))
                start = (.2*scale, .4*scale, 1.2e-10*scale)
                end = (.4*scale, .6*scale, 1.2e-10*scale)
                with self.assertRaises(RedBlueNumericalEdgeCase):
                    _edge_face_hit(start, end, triangle, 1e-10)
                tree = _face_bvh(Mesh(triangle, ((0, 1, 2),)))
                self.assertEqual([fid for fid, _ in _query_face_bvh(tree, start, end, 1e-10)], [0])
        rng = random.Random(14033)
        for _ in range(100):
            triangle = tuple((rng.uniform(-1, 1), rng.uniform(-1, 1), 0.) for _ in range(3))
            if abs((triangle[1][0]-triangle[0][0])*(triangle[2][1]-triangle[0][1]) -
                   (triangle[1][1]-triangle[0][1])*(triangle[2][0]-triangle[0][0])) < .01:
                continue
            height = .9e-10*max(dist(triangle[0], triangle[1]), dist(triangle[0], triangle[2]))
            def point(weights):
                return (sum(weights[i]*triangle[i][0] for i in range(3)),
                        sum(weights[i]*triangle[i][1] for i in range(3)), height)
            start, end = point((.2, .3, .5)), point((.4, .4, .2))
            with self.assertRaises(RedBlueNumericalEdgeCase):
                _edge_face_hit(start, end, triangle, 1e-10)
            tree = _face_bvh(Mesh(triangle, ((0, 1, 2),)))
            self.assertEqual([fid for fid, _ in _query_face_bvh(tree, start, end, 1e-10)], [0])

    def test_ordered_red_and_blue_intersection_circles(self):
        red = refined_tetrahedron()
        dx, dy, dz, scale = .08794858485146739, -.25131913562220887, .47920496341231167, 1.1169174342387775
        blue = Mesh(tuple((x*scale+dx, y*scale+dy, z*scale+dz) for x, y, z in red.vertices), red.faces)
        red_circle, blue_circle = intersection_face_circles(red, blue)
        self.assertGreater(len(red_circle), 2)
        self.assertGreater(len(blue_circle), 2)
        for circle in (red_circle, blue_circle):
            for current, following in zip(circle, circle[1:]+circle[:1]):
                self.assertEqual(current.forward_face, following.backward_face)
                self.assertTrue(0 < current.edge_parameter < 1)
                self.assertTrue(0 < current.barycentric_s < 1)
                self.assertTrue(0 < current.barycentric_t < 1-current.barycentric_s)
        polygon = zip_intersection_circles(red_circle, blue_circle)
        self.assertEqual(len(polygon), len(red_circle)+len(blue_circle))
        for current, following in zip(polygon, polygon[1:]+polygon[:1]):
            if current.red and following.red:
                self.assertEqual(current.opposite_face, following.opposite_face)
                self.assertEqual(current.forward_face, following.backward_face)
            elif not current.red and not following.red:
                self.assertEqual(current.opposite_face, following.opposite_face)
                self.assertEqual(current.forward_face, following.backward_face)
            elif current.red:
                self.assertEqual(current.opposite_face, following.backward_face)
                self.assertEqual(current.forward_face, following.opposite_face)
            else:
                self.assertEqual(current.opposite_face, following.backward_face)
                self.assertEqual(current.forward_face, following.opposite_face)

        result = redblue_union(red, blue)
        stats = result.statistics()
        self.assertEqual((stats.boundary_edges, stats.nonmanifold_edges, stats.euler_characteristic), (0, 0, 2))
        # Both tetrahedra are axis-aligned; their intersection is another
        # tetrahedron whose lower bounds and sum bound are known analytically.
        intersection_side = min(2., 2*scale+dx+dy+dz)-max(0., dx)-max(0., dy)-max(0., dz)
        expected_volume = (2**3 + (2*scale)**3 - max(0., intersection_side)**3)/6
        self.assertAlmostEqual(stats.volume, expected_volume, places=9)

    def test_circle_order_is_stable_for_shuffled_intersection_tuples(self):
        red = refined_tetrahedron()
        dx, dy, dz, scale = .08794858485146739, -.25131913562220887, .47920496341231167, 1.1169174342387775
        blue = Mesh(tuple((x*scale+dx, y*scale+dy, z*scale+dz) for x, y, z in red.vertices), red.faces)
        red_circle, blue_circle = intersection_face_circles(red, blue)
        rng = random.Random(31082)
        for mesh, circle, is_red in ((red, red_circle, True), (blue, blue_circle, False)):
            for _ in range(100):
                shuffled = list(circle)
                rng.shuffle(shuffled)
                start = circle.index(shuffled[-1])
                expected = circle[start:]+circle[:start]
                self.assertEqual(cyclically_order_tuples(mesh, shuffled, red=is_red), expected)

    def test_union_is_scale_invariant_for_transverse_tetrahedra(self):
        red = refined_tetrahedron()
        dx, dy, dz, factor = .08794858485146739, -.25131913562220887, .47920496341231167, 1.1169174342387775
        blue = Mesh(tuple((factor*x+dx, factor*y+dy, factor*z+dz) for x, y, z in red.vertices), red.faces)
        expected = redblue_union(red, blue).statistics().volume
        for scale in (1e-60, 1e-40, 1e-3, 1e-6, 1e-9, 1e-15, 1e-18, 1e40, 1e60):
            def scaled(mesh):
                return Mesh(tuple(tuple(scale*x for x in point) for point in mesh.vertices), mesh.faces)
            with self.subTest(scale=scale):
                result = redblue_union(scaled(red), scaled(blue)).statistics()
                self.assertEqual((result.boundary_edges, result.nonmanifold_edges), (0, 0))
                self.assertAlmostEqual(result.volume/scale**3, expected, places=8)
        def translated(mesh):
            return Mesh(tuple(tuple(x+1e9 for x in point) for point in mesh.vertices), mesh.faces)
        moved = redblue_union(translated(red), translated(blue)).statistics()
        self.assertEqual((moved.boundary_edges, moved.nonmanifold_edges), (0, 0))
        self.assertAlmostEqual(moved.volume, expected, places=5)

    def test_boolean_stitching_preserves_connectivity_under_rotation(self):
        red = refined_tetrahedron()
        dx, dy, dz, factor = .08794858485146739, -.25131913562220887, .47920496341231167, 1.1169174342387775
        blue = Mesh(tuple((factor*x+dx, factor*y+dy, factor*z+dz) for x, y, z in red.vertices), red.faces)
        u = (1/sqrt(2), -1/sqrt(2), 0.)
        v = (1/sqrt(6), 1/sqrt(6), -2/sqrt(6))
        n = (1/sqrt(3),)*3
        def rotate_point(p):
            return tuple(p[0]*u[k]+p[1]*v[k]+p[2]*n[k] for k in range(3))
        def rotate_mesh(mesh):
            return Mesh(tuple(map(rotate_point, mesh.vertices)), mesh.faces)
        def canonical(face):
            return min(face, face[1:]+face[:1], face[2:]+face[:2])
        for operation in ("union", "intersection", "difference"):
            with self.subTest(operation=operation):
                original = redblue_boolean_with_retries(red, blue, operation=operation, max_attempts=20)
                rotated = redblue_boolean_with_retries(rotate_mesh(red), rotate_mesh(blue),
                                                       operation=operation, max_attempts=20)
                reference_vertices = tuple(map(rotate_point, original.vertices))
                mapping = [min(range(len(reference_vertices)), key=lambda i: dist(point, reference_vertices[i]))
                           for point in rotated.vertices]
                self.assertEqual(len(set(mapping)), len(reference_vertices))
                self.assertLess(max(dist(point, reference_vertices[index])
                                    for point, index in zip(rotated.vertices, mapping)), 1e-12)
                original_faces = {canonical(face) for face in original.faces}
                rotated_faces = {canonical(tuple(mapping[i] for i in face)) for face in rotated.faces}
                # Cocircular points do not determine a unique Delaunay
                # diagonal; rotation can change the rounding of that tie.
                # Different triangles must still cover the same planar patch
                # with exactly the same oriented boundary.
                old_faces, new_faces = original_faces-rotated_faces, rotated_faces-original_faces
                def on_plane(face, anchor, unit_normal):
                    return all(abs(dot(tuple(original.vertices[i][k]-anchor[k] for k in range(3)),
                                       unit_normal)) < 1e-10 for i in face)
                def boundary(faces):
                    directed = Counter((a, b) for face in faces for a, b in zip(face, face[1:]+face[:1]))
                    return directed-Counter({(b, a): count for (a, b), count in directed.items()})
                while old_faces:
                    face = next(iter(old_faces))
                    a, b, c = (original.vertices[i] for i in face)
                    ab = tuple(b[k]-a[k] for k in range(3))
                    ac = tuple(c[k]-a[k] for k in range(3))
                    normal = cross(ab, ac)
                    length = sqrt(dot(normal, normal))
                    unit_normal = tuple(x/length for x in normal)
                    old_patch = {f for f in old_faces if on_plane(f, a, unit_normal)}
                    new_patch = {f for f in new_faces if on_plane(f, a, unit_normal)}
                    self.assertEqual(boundary(old_patch), boundary(new_patch))
                    old_faces -= old_patch
                    new_faces -= new_patch
                self.assertFalse(new_faces)
                self.assertEqual((rotated.statistics().boundary_edges, rotated.statistics().nonmanifold_edges), (0, 0))
                self.assertEqual(find_self_intersections(rotated, include_adjacent=True), ([], False))

    def test_disjoint_surfaces_are_reported(self):
        red = refined_tetrahedron()
        blue = Mesh(tuple((x+10, y, z) for x, y, z in red.vertices), red.faces)
        with self.assertRaises(RedBlueDisjoint):
            intersection_face_circles(red, blue)

    def test_multiple_intersection_loops_are_reported(self):
        def pair(first, second):
            offset = len(first.vertices)
            return Mesh(first.vertices+second.vertices,
                        first.faces+tuple(tuple(offset+i for i in face) for face in second.faces))
        red = pair(ico_sphere((0., 0., 0.), 1., 2), ico_sphere((4., 0., 0.), 1., 2))
        blue = pair(ico_sphere((.67, .17, .11), .9, 2), ico_sphere((4.67, .17, .11), .9, 2))
        with self.assertRaises(RedBlueMultipleLoops):
            redblue_boolean_with_retries(red, blue, max_attempts=20, max_growth_factor=4)

    def test_inconsistent_input_orientation_is_rejected(self):
        red = refined_tetrahedron()
        a, b, c = red.faces[0]
        flipped = Mesh(red.vertices, ((a, c, b),) + red.faces[1:])
        blue = Mesh(tuple((x+10, y, z) for x, y, z in red.vertices), red.faces)
        with self.assertRaisesRegex(ValueError, "consistently oriented"):
            redblue_union(flipped, blue)

    def test_retry_splits_complex_edges_and_preserves_union_volume(self):
        red = refined_tetrahedron()
        dx, dy, dz, scale = -.5545983942840167, -.13790844459386586, .41659162898066593, .8913537773337944
        blue = Mesh(tuple((x*scale+dx, y*scale+dy, z*scale+dz) for x, y, z in red.vertices), red.faces)
        with self.assertRaises(RedBlueComplexEdges):
            redblue_union(red, blue)
        result = redblue_union_with_retries(red, blue, max_attempts=16, max_growth_factor=4)
        # Compare a genuinely retried case with reuse disabled.
        from anamorph.redblue import _PreparedMesh
        with patch.object(_RedBlueCache, 'prepare', lambda self, mesh, *, red: _PreparedMesh(mesh)):
            uncached = redblue_union_with_retries(red, blue, max_attempts=16, max_growth_factor=4)
        self.assertEqual(result, uncached)
        stats = result.statistics()
        self.assertEqual((stats.boundary_edges, stats.nonmanifold_edges, stats.euler_characteristic), (0, 0, 2))
        side = max(0., min(2., 2*scale+dx+dy+dz)-max(0., dx)-max(0., dy)-max(0., dz))
        self.assertAlmostEqual(stats.volume, (8+(2*scale)**3-side**3)/6, places=9)
        self.assertEqual(len(red.faces), 64)
        with self.assertRaises(RedBlueError):
            redblue_union_with_retries(red, blue, max_attempts=1)

    def test_varied_transverse_unions_against_exact_volume(self):
        red = refined_tetrahedron()
        rng = random.Random(721)
        for _ in range(6):
            dx, dy, dz = (rng.uniform(-.8, .8) for _ in range(3))
            scale = rng.uniform(.65, 1.35)
            blue = Mesh(tuple((x*scale+dx, y*scale+dy, z*scale+dz) for x, y, z in red.vertices), red.faces)
            stats = redblue_union_with_retries(red, blue, max_attempts=12, max_growth_factor=3).statistics()
            side = max(0., min(2., 2*scale+dx+dy+dz)-max(0., dx)-max(0., dy)-max(0., dz))
            self.assertAlmostEqual(stats.volume, (8+(2*scale)**3-side**3)/6, places=8)
            self.assertEqual((stats.boundary_edges, stats.nonmanifold_edges, stats.euler_characteristic), (0, 0, 2))

    def test_independently_rotated_tetrahedron_booleans_preserve_volume(self):
        red = refined_tetrahedron()
        cases = (
            (-3.074509339619913, 1.716126613675347, .5421606766598202,
             -.6122816178583257, .13431556520334897, .7690537868848408),
            (.958243463919251, .4009824047687798, -.009066183124029958,
             -.19567052569976706, .7663275050897929, 1.3910497339588797),
            (.2910687317128078, 1.2094022154886659, -.27153779269420264,
             -.8710077328273209, -.3398917546612852, .666966960304535),
        )
        for ax, az, dx, dy, dz, scale in cases:
            ca, sa, cb, sb = cos(ax), sin(ax), cos(az), sin(az)
            def rotate(point):
                x, y, z = (scale*value for value in point)
                y, z = ca*y-sa*z, sa*y+ca*z
                return cb*x-sb*y+dx, sb*x+cb*y+dy, z+dz
            blue = Mesh(tuple(map(rotate, red.vertices)), red.faces)
            with self.subTest(angles=(ax, az)):
                union, intersection, difference = (redblue_boolean_with_retries(
                    red, blue, operation=operation, max_attempts=20, max_growth_factor=5).statistics()
                    for operation in ("union", "intersection", "difference"))
                reverse_union = redblue_union_with_retries(blue, red, max_attempts=20,
                                                           max_growth_factor=5).statistics()
                for stats in (union, intersection, difference, reverse_union):
                    self.assertEqual((stats.boundary_edges, stats.nonmanifold_edges), (0, 0))
                self.assertAlmostEqual(union.volume, reverse_union.volume, places=9)
                self.assertAlmostEqual(union.volume+intersection.volume,
                                       red.statistics().volume+blue.statistics().volume, places=9)
                self.assertAlmostEqual(intersection.volume+difference.volume, red.statistics().volume, places=9)

    def test_intersection_and_subtraction_against_exact_tetrahedron_volumes(self):
        red = refined_tetrahedron()
        rng = random.Random(721)
        for _ in range(6):
            dx, dy, dz = (rng.uniform(-.8, .8) for _ in range(3))
            scale = rng.uniform(.65, 1.35)
            blue = Mesh(tuple((x*scale+dx, y*scale+dy, z*scale+dz) for x, y, z in red.vertices), red.faces)
            side = max(0., min(2., 2*scale+dx+dy+dz)-max(0., dx)-max(0., dy)-max(0., dz))
            common = side**3/6
            for operation, expected in (("intersection", common), ("difference", 8/6-common)):
                result = redblue_boolean_with_retries(red, blue, operation=operation, max_attempts=12,
                                                       max_growth_factor=3, return_vertex_maps=True)
                mesh, stats = result.mesh, result.mesh.statistics()
                self.assertEqual((stats.boundary_edges, stats.nonmanifold_edges, stats.euler_characteristic), (0, 0, 2))
                self.assertAlmostEqual(stats.volume, expected, places=8)
                signed = sum(dot(mesh.vertices[a], cross(mesh.vertices[b], mesh.vertices[c]))/6
                             for a, b, c in mesh.faces)
                self.assertAlmostEqual(signed, expected, places=8)
                for original, mapping in ((red, result.red_vertex_map), (blue, result.blue_vertex_map)):
                    self.assertEqual(len(mapping), len(original.vertices))
                    for i, target in enumerate(mapping):
                        if target is not None:
                            self.assertEqual(original.vertices[i], mesh.vertices[target])

    def test_intersection_and_subtraction_direct_wrappers(self):
        red = refined_tetrahedron()
        dx, dy, dz, scale = .08794858485146739, -.25131913562220887, .47920496341231167, 1.1169174342387775
        blue = Mesh(tuple((x*scale+dx, y*scale+dy, z*scale+dz) for x, y, z in red.vertices), red.faces)
        common = redblue_intersection(red, blue).statistics().volume
        self.assertAlmostEqual(redblue_difference(red, blue).statistics().volume + common, 8/6, places=9)
        self.assertAlmostEqual(redblue_difference(blue, red).statistics().volume + common, (2*scale)**3/6, places=9)

    def test_curved_operands_preserve_volume_identities(self):
        red = ico_sphere((0., 0., 0.), 1., 2)
        rng = random.Random(914)
        for _ in range(3):
            center = (rng.uniform(.35, 1.1), rng.uniform(-.35, .35), rng.uniform(-.35, .35))
            blue = ico_sphere(center, rng.uniform(.65, 1.15), 2)
            results = {op: redblue_boolean_with_retries(red, blue, operation=op, max_attempts=20,
                                                        max_growth_factor=4)
                       for op in ("union", "intersection", "difference")}
            stats = {op: mesh.statistics() for op, mesh in results.items()}
            self.assertAlmostEqual(stats["intersection"].volume + stats["difference"].volume,
                                   red.statistics().volume, places=9)
            self.assertAlmostEqual(stats["intersection"].volume + stats["union"].volume,
                                   red.statistics().volume + blue.statistics().volume, places=9)
            for mesh in results.values():
                s = mesh.statistics()
                self.assertEqual((s.boundary_edges, s.nonmanifold_edges, s.euler_characteristic), (0, 0, 2))
                self.assertEqual(find_self_intersections(mesh, include_adjacent=True), ([], False))

    def test_untouched_components_are_classified_for_all_operations(self):
        red = refined_tetrahedron()
        dx, dy, dz, scale = .08794858485146739, -.25131913562220887, .47920496341231167, 1.1169174342387775
        blue = Mesh(tuple((x*scale+dx, y*scale+dy, z*scale+dz) for x, y, z in red.vertices), red.faces)
        side = min(2., 2*scale+dx+dy+dz)-max(0., dx)-max(0., dy)-max(0., dz)
        common, small_volume = side**3/6, (2*.05)**3/6
        red_volume, blue_volume = 8/6, (2*scale)**3/6
        # Test an untouched extra component on either operand, both outside
        # and fully inside the other operand. Its small tetrahedron lies away
        # from the main intersection and has an analytically known volume.
        for extra_red, inside, shift in ((True, False, (10., 0., 0.)),
                                         (True, True, (.2, .1, 2.05)),
                                         (False, False, (10., 0., 0.)),
                                         (False, True, (.2, .05, .05))):
            extra = tuple(tuple(shift[j]+.05*p[j] for j in range(3)) for p in red.vertices)
            original = red if extra_red else blue
            n = len(original.vertices)
            expanded = Mesh(original.vertices+extra, original.faces+tuple(tuple(n+i for i in f) for f in red.faces))
            first, second = (expanded, blue) if extra_red else (red, expanded)
            expected = {"union": red_volume+blue_volume-common+(0 if inside else small_volume),
                        "intersection": common+(small_volume if inside else 0),
                        "difference": red_volume-common +
                        (small_volume if extra_red and not inside else -small_volume if not extra_red and inside else 0)}
            for operation, volume in expected.items():
                result = redblue_boolean_with_retries(first, second, operation=operation, max_attempts=20,
                                                       max_growth_factor=4, return_vertex_maps=True)
                mesh, stats = result.mesh, result.mesh.statistics()
                self.assertEqual((stats.boundary_edges, stats.nonmanifold_edges), (0, 0))
                signed = sum(dot(mesh.vertices[a], cross(mesh.vertices[b], mesh.vertices[c]))/6
                             for a, b, c in mesh.faces)
                self.assertAlmostEqual(signed, volume, places=9)
                mapping = result.red_vertex_map if extra_red else result.blue_vertex_map
                should_keep = ((operation == "union" and not inside) or
                               (operation == "intersection" and inside) or
                               (operation == "difference" and ((extra_red and not inside) or (not extra_red and inside))))
                self.assertEqual(any(vertex is not None for vertex in mapping[n:]), should_keep)

    def test_boolean_disconnected_component_is_scale_invariant(self):
        red = refined_tetrahedron()
        dx, dy, dz, factor = .08794858485146739, -.25131913562220887, .47920496341231167, 1.1169174342387775
        blue = Mesh(tuple((x*factor+dx, y*factor+dy, z*factor+dz) for x, y, z in red.vertices), red.faces)
        n = len(red.vertices)
        extra = tuple((10+.05*x, .05*y, .05*z) for x, y, z in red.vertices)
        expanded = Mesh(red.vertices+extra, red.faces+tuple(tuple(n+i for i in face) for face in red.faces))
        baseline = {}
        for scale in (1e-100, 1., 1e100):
            def scaled(mesh):
                return Mesh(tuple(tuple(scale*x for x in point) for point in mesh.vertices), mesh.faces)
            for operation in ("union", "intersection", "difference"):
                with self.subTest(scale=scale, operation=operation):
                    result = redblue_boolean_with_retries(scaled(expanded), scaled(blue), operation=operation,
                                                         max_attempts=20, max_growth_factor=4, return_vertex_maps=True)
                    retained = any(value is not None for value in result.red_vertex_map[n:])
                    self.assertEqual(retained, operation != "intersection")
                    self.assertEqual(len(result.mesh.faces), baseline.setdefault(operation, len(result.mesh.faces)))
                    self.assertTrue(_edge_faces(result.mesh))

    def test_boolean_command_writes_difference_without_overwriting_input(self):
        red = refined_tetrahedron()
        dx, dy, dz, scale = .08794858485146739, -.25131913562220887, .47920496341231167, 1.1169174342387775
        blue = Mesh(tuple((x*scale+dx, y*scale+dy, z*scale+dz) for x, y, z in red.vertices), red.faces)
        with tempfile.TemporaryDirectory() as tmp:
            red_file, blue_file, output = (Path(tmp)/name for name in ("red.obj", "blue.obj", "result.obj"))
            red.write_obj(red_file)
            blue.write_obj(blue_file)
            args = [sys.executable, "-m", "anamorph", "mesh-boolean", str(red_file), str(blue_file)]
            result = subprocess.run(args+[str(output), "--operation", "difference"], text=True,
                                    capture_output=True, check=True)
            self.assertEqual(json.loads(result.stdout)["operation"], "difference")
            self.assertAlmostEqual(read_obj(output).statistics().volume,
                                   redblue_difference(red, blue).statistics().volume, places=9)
            before = red_file.read_bytes()
            rejected = subprocess.run(args+[str(red_file)], text=True, capture_output=True)
            self.assertEqual(rejected.returncode, 1)
            self.assertEqual(red_file.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
