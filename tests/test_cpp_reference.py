"""Optional parity checks against the locally built, Windows-adapted C++ source."""
import subprocess
import shutil
import random
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from math import cos, dist, fsum, pi, sin, sqrt, ulp

from anamorph.analytic_intersections import (CanalSegment, regularity_candidates,
    local_self_intersection_candidates, global_self_intersection_candidates,
    soma_neurite_candidates, neurite_neurite_candidates)
from anamorph.canal_mesh import CanalPathMesh, append_terminal_hemisphere, ico_sphere, _triangle_edge_ratio
from anamorph.mesh import Mesh, cross, dot, read_obj, sub
from anamorph.smoothing import hc_smooth, laplacian_smooth
from anamorph.collapse import (greedy_edge_collapse, _SourcePriorityQueue,
                              _source_face_neighborhood, _source_neighborhood_average)
from anamorph.analytic_mesher import assemble_redblue_neuron, render_individual_surfaces
from anamorph.analytic_network import analyze_analytic_partial
from anamorph.swc import read_swc
from anamorph.model import prepare_morphology
from anamorph.redblue import redblue_boolean, _triangulate_polygon
from anamorph.mesh_intersections import find_self_intersections

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / "reference-build" / "solver_probe.exe"
INITIAL_FIT_PROBE = ROOT / "reference-build" / "initial_fit_probe.exe"
RECTANGLE_PROBE = ROOT / "reference-build" / "rectangle_probe.exe"
BIVARIATE_ROOTS_PROBE = ROOT / "reference-build" / "bivariate_roots_probe.exe"
BIVARIATE7_ROOTS_PROBE = ROOT / "reference-build" / "bivariate7_roots_probe.exe"
GLOBAL_SYSTEM_PROBE = ROOT / "reference-build" / "global_system_probe.exe"
PAIR_EDGES_PROBE = ROOT / "reference-build" / "pair_edges_probe.exe"
UNIVARIATE_ROOTS_PROBE = ROOT / "reference-build" / "univariate_roots_probe.exe"
CANAL_UNIVARIATE_PROBE = ROOT / "reference-build" / "canal_univariate_probe.exe"
CELLGEN = ROOT / "reference-build" / "source" / "bin" / "am_cellgen.exe"
HEMISPHERE_PROBE = ROOT / "reference-build" / "hemisphere_probe.exe"
HEMISPHERE_FRAME_PROBE = ROOT / "reference-build" / "hemisphere_frame_probe.exe"
QUEUE_PROBE = ROOT / "reference-build" / "queue_probe.exe"
QUEUE_CHANGE_PROBE = ROOT / "reference-build" / "queue_change_probe.exe"
POLYGON_PROBE = ROOT / "reference-build" / "polygon_probe.exe"
REDBLUE_PROBE = ROOT / "reference-build" / "redblue_probe.exe"
QUAD_PROBE = ROOT / "reference-build" / "quad_probe.exe"
LAPLACIAN_PROBE = ROOT / "reference-build" / "laplacian_probe.exe"
COLLAPSE_NEIGHBORHOOD_PROBE = ROOT / "reference-build" / "collapse_neighborhood_probe.exe"
SOMA_CENTROID_PROBE = ROOT / "reference-build" / "soma_centroid_probe.exe"


def line(start, end, radius=.1):
    return CanalSegment(tuple((a, b-a, 0., 0.) for a, b in zip(start, end)), radius, radius)


def encode(curve):
    return " ".join(map(repr, [value for axis in curve.power for value in axis] +
                        [curve.radius0, curve.radius1]))


def refined_tetrahedron():
    vertices = [(0., 0., 0.), (2., 0., 0.), (0., 2., 0.), (0., 0., 2.)]
    faces = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
    for _ in range(2):
        cache, refined = {}, []
        def midpoint(a, b):
            edge = tuple(sorted((a, b)))
            if edge not in cache:
                cache[edge] = len(vertices)
                vertices.append(tuple((vertices[a][k]+vertices[b][k])/2 for k in range(3)))
            return cache[edge]
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            refined.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))
        faces = refined
    return Mesh(tuple(vertices), tuple(faces))


@unittest.skipUnless(sys.platform == "win32" and PROBE.is_file(),
                     "optional Windows C++ reference probe is not available")
class CppReferenceTests(unittest.TestCase):
    def test_reversing_tangent_exposes_source_pair_miss(self):
        from anamorph.analytic_intersections import CanalSegment, neurite_neurite_candidates
        from anamorph.source_bivariate import source_neurite_pair_candidates
        first = CanalSegment(((0., 1., 0., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)), .125, .125)
        second = CanalSegment(((2., -6., 9., 0.), (.25, 0., 0., 0.), (0., 0., 0., 0.)), .125, .125)
        values = [value for segment in (first, second)
                  for value in (*[value for axis in segment.power for value in axis],
                                segment.radius0, segment.radius1)]
        run = subprocess.run([str(PROBE), "pair"], input=" ".join(map(repr, values)), text=True,
                             capture_output=True, check=True, timeout=10)
        self.assertIn("RESULT 0 0", run.stdout)
        self.assertEqual(source_neurite_pair_candidates(first, second), [])
        hit, = neurite_neurite_candidates(first, second, max_rectangles=1, max_intervals=1)
        self.assertEqual(hit.exact_parameters, ((1, 1), (1, 3)))
        self.assertEqual(hit.distance_squared, .0625)

    @unittest.skipUnless(SOMA_CENTROID_PROBE.is_file(), "optional C++ soma-centroid probe is not built")
    def test_source_soma_centroid_matches_cpp_section_order(self):
        from anamorph.roots import normalize_soma_sections
        rng = random.Random(671034)
        different_stable = 0
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"soma.swc"
            for case in range(40):
                rows = []
                for node in range(1, rng.randrange(5, 14)):
                    parent = -1 if node == 1 else rng.randrange(1, node)
                    xyz = tuple(rng.uniform(-100., 100.) for _ in range(3))
                    rows.append(f"{node} 1 {xyz[0]:.17g} {xyz[1]:.17g} {xyz[2]:.17g} "
                                f"{rng.uniform(.1, 2.):.17g} {parent}")
                path.write_text("\n".join(rows)+"\n", encoding="utf-8")
                run = subprocess.run([str(SOMA_CENTROID_PROBE), str(path)], cwd=tmp,
                                     capture_output=True, text=True, timeout=10, check=True)
                line = next(line for line in run.stdout.splitlines() if line.startswith("SOMA "))
                expected = tuple(map(float, line.split()[1:4]))
                tree = read_swc(path)
                source = normalize_soma_sections(tree, source_soma_centroid=True).nodes[1].position
                stable = normalize_soma_sections(tree).nodes[1].position
                with self.subTest(case=case):
                    self.assertEqual(source, expected)
                different_stable += stable != expected
            self.assertGreater(different_stable, 0)
            path.write_text("1 1 1e16 0 0 1 -1\n2 1 1 1 0 1 1\n3 1 -1e16 2 0 1 2\n", encoding="utf-8")
            run = subprocess.run([str(SOMA_CENTROID_PROBE), str(path)], cwd=tmp,
                                 capture_output=True, text=True, timeout=10, check=True)
            self.assertIn("SOMA 0 1 0 1", run.stdout)
            self.assertEqual(normalize_soma_sections(read_swc(path), source_soma_centroid=True).nodes[1].position,
                             (0., 1., 0.))

    @unittest.skipUnless(CELLGEN.is_file(), "optional C++ cell generator is not built")
    def test_source_soma_centroid_reproduces_individual_surface(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"case.swc"
            path.write_text("1 1 1e16 0 0 1 -1\n2 1 1 1 0 1 1\n3 1 -1e16 2 0 1 2\n"
                            "4 3 2 1 0 .15 3\n5 3 3 1 0 .15 4\n", encoding="utf-8")
            run = subprocess.run([str(CELLGEN), "-i", "case", "-force-meshing", "-no-mesh-pp",
                                  "-no-meshing-flush", "-no-cellnet-pc", "-meshing-cansurf-angularsegments", "8",
                                  "-ana-nthreads", "1", "-meshing-individual-surfaces"], cwd=tmp,
                                 capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stderr+run.stdout)
            reference = read_obj(Path(tmp)/"case_individual_modelling_surfaces.obj")
            settings = dict(preconditioning=False, preserve_surface_roots=False, angular_segments=8,
                            soma_refinements=3, random_generator="windows-c", source_render_vector=True,
                            source_frame=True, arc_length_mode="source-x87", quad_sqrt_mode="windows-x87",
                            render_trig_mode="windows-x87")
            source = render_individual_surfaces(read_swc(path), source_soma_centroid=True, **settings)
            stable = render_individual_surfaces(read_swc(path), **settings)
            self.assertEqual((len(reference.vertices), len(reference.faces)), (len(source.vertices), len(source.faces)))
            self.assertEqual(len(reference.faces), 1824)
            self.assertNotEqual(len(stable.faces), len(reference.faces))
            self.assertLess(max(min(dist(a, b) for b in source.vertices) for a in reference.vertices), 1e-8)
            self.assertLess(max(min(dist(a, b) for b in reference.vertices) for a in source.vertices), 1e-8)

    @unittest.skipUnless(QUAD_PROBE.is_file(), "optional C++ quad probe is not built")
    def test_quad_diagonal_choice_near_ties_against_cpp(self):
        rng = random.Random(100091)
        matches = 0
        for case in range(100):
            t, s, height = rng.uniform(0, 6.3), rng.uniform(0, 6.3), rng.uniform(.05, 2)
            x, y = cos(t), sin(t)
            u = (x, y, 0.)
            v = (-y*cos(s)*height, x*cos(s)*height, sin(s)*height)
            a = tuple(rng.uniform(-5, 5) for _ in range(3))
            points = (a, tuple(a[i]+u[i] for i in range(3)),
                      tuple(a[i]+u[i]+v[i] for i in range(3)), tuple(a[i]+v[i] for i in range(3)))
            data = "\n".join(" ".join(map(repr, point)) for point in points)+"\n"
            run = subprocess.run([str(QUAD_PROBE)], input=data, capture_output=True, text=True,
                                 check=True, timeout=10)
            faces = [set(map(int, line.split())) for line in run.stdout.splitlines()]
            self.assertEqual(len(faces), 2)
            expected = faces[0] & faces[1]
            def quality(a, b, c):
                return _triangle_edge_ratio(points, a, b, c, "quad")
            first = quality(0, 1, 2)+quality(0, 2, 3)
            second = quality(0, 1, 3)+quality(1, 2, 3)
            actual = {0, 2} if first < second else {1, 3}
            with self.subTest(case=case):
                if actual == expected:
                    matches += 1
                else:
                    # Different Windows runtimes can round sqrt differently;
                    # an unresolved numerical tie is not source parity.
                    self.assertLessEqual(abs(first-second), 4*ulp(max(first, second)))
                if case == 1:  # Former math.dist calculation chose the opposite diagonal.
                    self.assertEqual(actual, expected)
                def x87_quality(a, b, c):
                    return _triangle_edge_ratio(points, a, b, c, "quad", quad_sqrt_mode="windows-x87")
                x87_choice = (x87_quality(0, 1, 2)+x87_quality(0, 2, 3) <
                              x87_quality(0, 1, 3)+x87_quality(1, 2, 3))
                self.assertEqual({0, 2} if x87_choice else {1, 3}, expected)
        self.assertGreaterEqual(matches, 98)

    @unittest.skipUnless(PAIR_EDGES_PROBE.is_file(), "optional pair-edge C++ probe is not built")
    def test_source_pair_edges_and_candidate_decisions_match_cpp(self):
        from anamorph.source_bivariate import (source_neurite_pair_candidates, source_neurite_pair_edge_polynomials,
                                               SourceBivariateFailure)
        rng = random.Random(411047)
        for mode in ("pair", "adjacent"):
            for index in range(12):
                with self.subTest(mode=mode, index=index):
                    a = tuple(tuple(rng.uniform(-2., 2.) for _ in range(4)) for _ in range(3))
                    b = tuple(tuple(a[k][0] if mode == "adjacent" and j == 0 else rng.uniform(-2., 2.)
                                    for j in range(4)) for k in range(3))
                    r1, r2 = rng.uniform(.02, .7), rng.uniform(.02, .7)
                    first, second = CanalSegment(a, r1, r1), CanalSegment(b, r2, r2)
                    data = encode(first)+" "+encode(second)
                    probe = subprocess.run([str(PAIR_EDGES_PROBE)], input=data, text=True, capture_output=True,
                                           check=True, timeout=5)
                    expected_edges = [tuple(map(float, line.split()[1:])) for line in probe.stdout.splitlines()]
                    self.assertEqual([edge.coefficients for edge in source_neurite_pair_edge_polynomials(first, second)],
                                     expected_edges)
                    try:
                        hits = source_neurite_pair_candidates(first, second, adjacent=mode == "adjacent")
                        actual = (bool(hits), len(hits))
                    except SourceBivariateFailure:
                        actual = (True, 0)
                    result = subprocess.run([str(PROBE), mode], input=data+(" 0" if mode == "adjacent" else ""),
                                            text=True, capture_output=True, check=True, timeout=5)
                    result_line = next(line for line in result.stdout.splitlines() if line.startswith("RESULT "))
                    _, found, count = result_line.split()
                    self.assertEqual(actual, (bool(int(found)), int(count)))
        first = line((-1., 0., 0.), (0., 0., 0.), .1)
        second = line((.1+.2, -1., 0.), (.1+.2, 1., 0.), .2)
        hits = source_neurite_pair_candidates(first, second)
        self.assertTrue(any(hit.source == "edge" and hit.t == 1. and hit.u == .5 for hit in hits))
        result = subprocess.run([str(PROBE), "pair"], input=encode(first)+" "+encode(second), text=True,
                                capture_output=True, check=True, timeout=5)
        self.assertIn("RESULT 1 1", result.stdout)
        self.assertEqual(source_neurite_pair_candidates(first, second, adjacent=True, first_end_second_start=True), [])
        result = subprocess.run([str(PROBE), "adjacent"], input=encode(first)+" "+encode(second)+" 1", text=True,
                                capture_output=True, check=True, timeout=5)
        self.assertIn("RESULT 0 0", result.stdout)

    @unittest.skipUnless(BIVARIATE7_ROOTS_PROBE.is_file(), "optional degree-seven C++ root probe is not built")
    def test_source_degree_seven_bivariate_iterations_match_cpp(self):
        from anamorph.bernstein import BiBernsteinPolynomial
        from anamorph.source_bivariate import source_bilinear_clip_roots, SourceBivariateFailure
        rng = random.Random(42813)
        for index in range(12):
            with self.subTest(index=index):
                p = BiBernsteinPolynomial(tuple(tuple(rng.uniform(-1., 1.) for _ in range(8)) for _ in range(8)))
                q = BiBernsteinPolynomial(tuple(tuple(rng.uniform(-1., 1.) for _ in range(8)) for _ in range(8)))
                data = " ".join(map(repr, (*sum(p.coefficients, ()), *sum(q.coefficients, ()), 1e-4)))
                result = subprocess.run([str(BIVARIATE7_ROOTS_PROBE)], input=data, text=True,
                                        capture_output=True, check=True, timeout=5)
                try:
                    actual = source_bilinear_clip_roots(p, q)
                except SourceBivariateFailure:
                    self.assertIn("ERROR ", result.stdout)
                    continue
                self.assertIn("DONE ", result.stdout)
                expected = [tuple(map(float, line.split()[1:])) for line in result.stdout.splitlines()
                            if line.startswith("ROOT ")]
                self.assertEqual(len(actual), len(expected))
                for root, values in zip(actual, expected):
                    for got, want in zip((root.x0, root.x1, root.y0, root.y1), values):
                        self.assertAlmostEqual(got, want, delta=1e-9)

    @unittest.skipUnless(GLOBAL_SYSTEM_PROBE.is_file(), "optional global-system C++ probe is not built")
    def test_source_global_system_and_candidates_match_cpp(self):
        from anamorph.bernstein import BernsteinPolynomial
        from anamorph.source_bivariate import (source_global_stationary_bernstein,
                                               source_global_self_intersection_candidates)
        from anamorph.source_canal_polynomials import source_soma_stationary_polynomial
        rng = random.Random(182799)
        segments = [CanalSegment(tuple(tuple(rng.uniform(-2., 2.) for _ in range(4)) for _ in range(3)),
                                 .1, .2) for _ in range(12)]
        segments.append(CanalSegment(((0., 1., -1., 0.), (0., -1., 3., -2.), (0., 0., 0., 0.)), .05, .05))
        with tempfile.TemporaryDirectory() as directory:
            for segment in segments:
                with self.subTest(segment=segment):
                    data = encode(segment)
                    probe = subprocess.run([str(GLOBAL_SYSTEM_PROBE)], input=data, text=True, capture_output=True,
                                           check=True, timeout=5)
                    expected = [tuple(map(float, line.split()[1:])) for line in probe.stdout.splitlines()]
                    p, q = source_global_stationary_bernstein(segment)
                    components = tuple(BernsteinPolynomial.from_power(axis) for axis in segment.power)
                    centers = [tuple(component.evaluate(t) for component in components) for t in (0., 1.)]
                    actual = [sum(p.coefficients, ()), sum(q.coefficients, ()),
                              *(source_soma_stationary_polynomial(segment, center).coefficients for center in centers)]
                    self.assertEqual(actual, expected)
                    source_hits = source_global_self_intersection_candidates(segment)
                    result = subprocess.run([str(PROBE), "global"], input=data, text=True, capture_output=True,
                                            check=True, timeout=5, cwd=directory)
                    line = next(line for line in result.stdout.splitlines() if line.startswith("RESULT "))
                    _, found, count = line.split()
                    self.assertEqual((bool(int(found)), int(count)), (bool(source_hits), len(source_hits)))
            # A bivariate exception alone is not a GSI positive: the C++
            # function resets its result after catching that exception.
            segment = segments[-1]
            suppressed = []
            self.assertEqual(source_global_self_intersection_candidates(segment, bivariate_tolerance=2.,
                                                                         _suppressed_failures=suppressed), [])
            self.assertEqual(len(suppressed), 1)
            self.assertIn("smaller than tolerance", str(suppressed[0]))
            result = subprocess.run([str(PROBE), "global"], input=encode(segment)+" 2", text=True,
                                    capture_output=True, check=True, timeout=10, cwd=directory)
            self.assertIn("RESULT 0 0", result.stdout)
            self.assertEqual(len(list(Path(directory).glob("gsi_exception_*.plot"))), 2)

    @unittest.skipUnless(CANAL_UNIVARIATE_PROBE.is_file(), "optional C++ canal polynomial probe is not built")
    def test_source_canal_univariate_polynomials_and_candidates_match_cpp(self):
        from anamorph.source_canal_polynomials import (source_regularity_polynomial,
            source_soma_stationary_polynomial, source_local_self_intersection_polynomial,
            source_regularity_candidates, source_soma_neurite_candidates, source_local_self_intersection_candidates)
        rng = random.Random(19282)
        examples = []
        for _ in range(18):
            power = tuple(tuple(rng.uniform(-2., 2.) for _ in range(4)) for _ in range(3))
            segment = CanalSegment(power, rng.uniform(.05, .8), rng.uniform(.05, .8))
            center = tuple(rng.uniform(-2., 2.) for _ in range(3))
            examples.append((segment, center, rng.uniform(.05, 1.5), False))
        examples.extend((
            (CanalSegment(((0., -1., 1., 0.), (0., -2., 2., 0.), (0., 0., 0., 0.)), .2, .2),
             (0., 0., 0.), .2, False),
            (CanalSegment(((-1., 2., 0., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)), 1e-9, 1e-9),
             (0., 1e-6, 0.), 1e-9, True),
        ))
        for segment, center, radius, is_root in examples:
            with self.subTest(segment=segment, center=center):
                data = encode(segment)
                probe = subprocess.run([str(CANAL_UNIVARIATE_PROBE)],
                                       input=data+" "+" ".join(map(repr, center)), text=True,
                                       capture_output=True, check=True, timeout=5)
                expected = {parts[0]: tuple(map(float, parts[1:])) for line in probe.stdout.splitlines()
                            if (parts := line.split())}
                self.assertEqual(expected["REG"], source_regularity_polynomial(segment).coefficients)
                self.assertEqual(expected["SONS"], source_soma_stationary_polynomial(segment, center).coefficients)
                self.assertEqual(expected["LSI"], source_local_self_intersection_polynomial(segment).coefficients)
                checks = (("regularity", "", source_regularity_candidates(segment)),
                          ("local", "", source_local_self_intersection_candidates(segment)),
                          ("soma", " "+" ".join(map(repr, (*center, radius, int(is_root)))),
                           source_soma_neurite_candidates(segment, center, radius, is_root_segment=is_root)))
                for mode, extra, hits in checks:
                    result = subprocess.run([str(PROBE), mode], input=data+extra, text=True,
                                            capture_output=True, check=True, timeout=5)
                    line = next(line for line in result.stdout.splitlines() if line.startswith("RESULT "))
                    _, found, count = line.split()
                    self.assertEqual((bool(int(found)), int(count)), (bool(hits), len(hits)))

    @unittest.skipUnless(UNIVARIATE_ROOTS_PROBE.is_file(), "optional C++ univariate probe is not built")
    def test_source_univariate_iterations_match_cpp(self):
        from anamorph.bernstein import BernsteinPolynomial
        from anamorph.source_univariate import source_bezier_clip_roots
        rng = random.Random(12793)
        examples = [(1, (-1., 1.), 0., 1.), (3, (-.125, .125, -.125, .125), 0., 1.)]
        for degree in (1, 2, 3, 4, 5, 6, 10, 12):
            for index in range(8):
                coefficients = tuple(rng.uniform(-2, 2) for _ in range(degree+1))
                left, right = (0., 1.) if index % 2 == 0 else (.125, .875)
                examples.append((degree, coefficients, left, right))
        for degree, coefficients, left, right in examples:
            with self.subTest(degree=degree, coefficients=coefficients, domain=(left, right)):
                data = " ".join(map(repr, (degree, *coefficients, left, right, 1e-5, 1e-11)))
                result = subprocess.run([str(UNIVARIATE_ROOTS_PROBE)], input=data, text=True,
                                        capture_output=True, check=True, timeout=5)
                self.assertIn("DONE ", result.stdout)
                expected = [tuple(map(float, line.split()[1:])) for line in result.stdout.splitlines()
                            if line.startswith("ROOT ")]
                actual = source_bezier_clip_roots(BernsteinPolynomial(coefficients), left=left, right=right,
                                                  tolerance=1e-5)
                self.assertEqual(len(actual), len(expected))
                for root, (lo, hi) in zip(actual, expected):
                    self.assertAlmostEqual(root.left, lo, delta=1e-10)
                    self.assertAlmostEqual(root.right, hi, delta=1e-10)

    @unittest.skipUnless(RECTANGLE_PROBE.is_file(), "optional C++ rectangle probe is not built")
    def test_source_rectangle_step_and_four_way_split_match_cpp(self):
        from anamorph.bernstein import BiBernsteinPolynomial, ClippingConvergenceError, RootRectangle
        from anamorph.source_bivariate import source_clip_rectangle, source_split_xy
        rng = random.Random(518380)
        box = RootRectangle(.13, .78, .19, .91)
        for case in range(60):
            values = [rng.uniform(-3, 3) for _ in range(6)]
            noise = .03 if case % 2 == 0 else 0.
            def poly(a, b, c):
                return BiBernsteinPolynomial(tuple(tuple(c+a*i/5+b*j/5+rng.uniform(-noise, noise)
                                                          for j in range(6)) for i in range(6)))
            p, q = poly(*values[:3]), poly(*values[3:])
            xf, yf = ((False, False), (True, False), (False, True))[case % 3]
            data = " ".join(repr(value) for polynomial in (p, q)
                            for row in polynomial.coefficients for value in row)
            data += f" {box.x0} {box.x1} {box.y0} {box.y1} {int(xf)} {int(yf)} 1e-11 1e-11"
            with self.subTest(case=case, frozen=(xf, yf)):
                run = subprocess.run([str(RECTANGLE_PROBE)], input=data, text=True, capture_output=True,
                                     timeout=10, cwd=ROOT)
                self.assertEqual(run.returncode, 0, run.stderr)
                result = next(line for line in run.stdout.splitlines() if line.startswith(("RESULT ", "ERROR ")))
                if result.startswith("ERROR "):
                    with self.assertRaises(ClippingConvergenceError):
                        source_clip_rectangle(p, q, box, x_frozen=xf, y_frozen=yf)
                else:
                    actual = source_clip_rectangle(p, q, box, x_frozen=xf, y_frozen=yf)
                    if result == "RESULT 0":
                        self.assertIsNone(actual)
                    else:
                        self.assertIsNotNone(actual)
                        expected = tuple(map(float, result.split()[2:]))
                        self.assertEqual((actual.x0, actual.x1, actual.y0, actual.y1), expected)
        for case in range(6):
            polynomial = BiBernsteinPolynomial(tuple(tuple(rng.uniform(-1, 1) for _ in range(6))
                                                        for _ in range(6)))
            data = " ".join(repr(value) for row in polynomial.coefficients for value in row)
            run = subprocess.run([str(RECTANGLE_PROBE), "split"], input=data, text=True,
                                 capture_output=True, timeout=10, cwd=ROOT)
            self.assertEqual(run.returncode, 0, run.stderr)
            children = source_split_xy(polynomial)
            rows = [line.split() for line in run.stdout.splitlines() if line.startswith("SPLIT ")]
            self.assertEqual(len(rows), 36)
            for _, i, j, *values in rows:
                self.assertEqual(tuple(child.coefficients[int(i)][int(j)] for child in children),
                                 tuple(map(float, values)))
        zero = BiBernsteinPolynomial(tuple((0.,)*6 for _ in range(6)))
        linear = BiBernsteinPolynomial.from_power(((0., 1.), (1., 0.))).elevate(5, 5)
        doubled = BiBernsteinPolynomial.from_power(((0., 2.), (2., 0.))).elevate(5, 5)
        for p, q, xf, yf in ((linear, doubled, False, False), (zero, zero, True, False),
                             (linear, doubled, True, True)):
            data = " ".join(repr(value) for polynomial in (p, q)
                            for row in polynomial.coefficients for value in row)
            data += f" {box.x0} {box.x1} {box.y0} {box.y1} {int(xf)} {int(yf)} 1e-11 1e-11"
            with self.subTest(singular=(xf, yf)):
                run = subprocess.run([str(RECTANGLE_PROBE)], input=data, text=True, capture_output=True,
                                     timeout=10, cwd=ROOT)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertTrue(any(line.startswith("ERROR ") for line in run.stdout.splitlines()))
                with self.assertRaises(ClippingConvergenceError):
                    source_clip_rectangle(p, q, box, x_frozen=xf, y_frozen=yf)

    @unittest.skipUnless(BIVARIATE_ROOTS_PROBE.is_file(), "optional C++ bivariate roots probe is not built")
    def test_source_bivariate_iterations_match_cpp(self):
        import json
        from anamorph.bernstein import BiBernsteinPolynomial, ClippingConvergenceError, RootRectangle
        from anamorph.source_bivariate import source_bilinear_clip_roots
        rng = random.Random(43092)
        cases = [(BiBernsteinPolynomial(tuple(tuple(rng.uniform(-1, 1) for _ in range(6)) for _ in range(6))),
                  BiBernsteinPolynomial(tuple(tuple(rng.uniform(-1, 1) for _ in range(6)) for _ in range(6))))
                 for _ in range(8)]
        fixture = json.loads((ROOT/"tests"/"fixtures"/"source_split_missed_root.json").read_text())
        cases.append(tuple(BiBernsteinPolynomial(tuple(tuple(value/fixture["denominator"] for value in row)
                                                        for row in matrix))
                           for matrix in fixture["coefficients"]))
        cases.append((BiBernsteinPolynomial.from_power(((-.25,), (0.,), (1.,))).elevate(5, 5),
                      BiBernsteinPolynomial.from_power(((-.5, 1.),)).elevate(5, 5)))
        for case, (p, q) in enumerate(cases):
            coefficients = " ".join(repr(value) for polynomial in (p, q)
                                    for row in polynomial.coefficients for value in row)
            for blocked in (False, True):
                blacklist = (RootRectangle(0., .2, 0., .2),) if blocked else ()
                data = coefficients+(" 0.01 1 0 .2 0 .2" if blocked else " 0.01 0")
                with self.subTest(case=case, blocked=blocked):
                    run = subprocess.run([str(BIVARIATE_ROOTS_PROBE)], input=data, text=True,
                                         capture_output=True, timeout=10, cwd=ROOT)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    if any(line.startswith("ERROR ") for line in run.stdout.splitlines()):
                        with self.assertRaises(ClippingConvergenceError):
                            source_bilinear_clip_roots(p, q, tolerance=.01, blacklist=blacklist)
                    else:
                        expected = [tuple(map(float, line.split()[1:])) for line in run.stdout.splitlines()
                                    if line.startswith("ROOT ")]
                        actual = source_bilinear_clip_roots(p, q, tolerance=.01, blacklist=blacklist)
                        self.assertEqual([(r.x0, r.x1, r.y0, r.y1) for r in actual], expected)

    @unittest.skipUnless(QUEUE_CHANGE_PROBE.is_file(), "optional C++ queue change-key probe is not built")
    def test_queue_key_changes_match_cpp_with_reused_tie_ids(self):
        rng = random.Random(77291)
        queue, active, operations, expected = _SourcePriorityQueue(), set(), [], []
        for _ in range(500):
            value, choice = rng.randrange(25), rng.random()
            if choice < .30 and value not in active:
                priority = rng.choice((-2., -1., 0., 1., 2.))
                operations.append(f"insert {value} {priority}")
                queue.insert(value, priority)
                active.add(value)
            elif choice < .70:
                priority = rng.choice((-2., -1., 0., 1., 2.))
                operations.append(f"change {value} {priority}")
                expected.append(f"CHANGE {int(queue.change_key(value, priority))}")
            elif choice < .85:
                operations.append(f"discard {value}")
                queue.discard(value)
                active.discard(value)
            elif active:
                operations.append("pop")
                _, value = queue.pop()
                active.remove(value)
                expected.append(f"POP {value}")
        while active:
            operations.append("pop")
            _, value = queue.pop()
            active.remove(value)
            expected.append(f"POP {value}")
        run = subprocess.run([str(QUEUE_CHANGE_PROBE)], input="\n".join(operations)+"\n", text=True,
                             capture_output=True, timeout=15, cwd=ROOT)
        self.assertEqual(run.returncode, 0, run.stderr)
        actual = [line for line in run.stdout.splitlines() if line.startswith(("CHANGE ", "POP "))]
        self.assertEqual(actual, expected)

    @unittest.skipUnless(POLYGON_PROBE.is_file(), "optional C++ polygon probe is not built")
    def test_thin_triangle_matches_cpp_polygon_triangulation(self):
        data = "3 0 0 1 1 0 0\n0 0 0\n1 0 0\n1 1e-16 0\n"
        run = subprocess.run([str(POLYGON_PROBE)], input=data, text=True, capture_output=True,
                             timeout=15, cwd=ROOT)
        self.assertEqual(run.returncode, 0, run.stderr+run.stdout)
        reference = [tuple(map(int, line.split()[1:])) for line in run.stdout.splitlines()
                     if line.startswith("TRI ")]
        points = ((0., 0., 0.), (1., 0., 0.), (1., 1e-16, 0.))
        self.assertEqual(_triangulate_polygon([0, 1, 2], points, (0., 0., 1.), basis=(1., 0., 0.)),
                         reference)

    @unittest.skipUnless(POLYGON_PROBE.is_file(), "optional C++ polygon probe is not built")
    def test_cut_polygon_triangulation_matches_cpp_on_tilted_planes(self):
        rng = random.Random(30928)
        u = (1/sqrt(2), -1/sqrt(2), 0.)
        v = (1/sqrt(6), 1/sqrt(6), -2/sqrt(6))
        normal = (1/sqrt(3),)*3
        for case in range(20):
            count = rng.randrange(4, 11)
            points = []
            for i in range(count):
                angle = 2*pi*(i+rng.uniform(-.15, .15))/count
                radius = rng.uniform(.7, 1.3)
                points.append(tuple(radius*cos(angle)*u[k]+radius*sin(angle)*v[k] for k in range(3)))
            data = " ".join(map(repr, [count, *normal, *u, *(value for point in points for value in point)]))
            with self.subTest(case=case):
                run = subprocess.run([str(POLYGON_PROBE)], input=data, text=True, capture_output=True,
                                     timeout=15, cwd=ROOT)
                self.assertEqual(run.returncode, 0, run.stderr+run.stdout)
                reference = [tuple(map(int, line.split()[1:])) for line in run.stdout.splitlines()
                             if line.startswith("TRI ")]
                actual = _triangulate_polygon(list(range(count)), points, normal, basis=u)
                def canonical(triangle):
                    return min(triangle, triangle[1:]+triangle[:1], triangle[2:]+triangle[:2])
                self.assertEqual({canonical(triangle) for triangle in actual},
                                 {canonical(triangle) for triangle in reference})

    @unittest.skipUnless(QUEUE_PROBE.is_file(), "optional C++ priority queue probe is not built")
    def test_collapse_priority_queue_matches_cpp_ties_and_id_reuse(self):
        rng = random.Random(44917)
        queue = _SourcePriorityQueue()
        active, operations, expected = set(), [], []
        for _ in range(500):
            choice = rng.random()
            face = rng.randrange(25)
            if choice < .62:
                priority = rng.choice((-2., -1., 0., 1.))
                if face in active:
                    operations.append(f"discard {face}")
                operations.append(f"insert {face} {priority}")
                queue.insert(face, priority)
                active.add(face)
            elif choice < .80:
                operations.append(f"discard {face}")
                queue.discard(face)
                active.discard(face)
            elif active:
                operations.append("pop")
                _, popped = queue.pop()
                active.remove(popped)
                expected.append(popped)
        while active:
            operations.append("pop")
            _, popped = queue.pop()
            active.remove(popped)
            expected.append(popped)
        run = subprocess.run([str(QUEUE_PROBE)], input="\n".join(operations)+"\n", text=True,
                             capture_output=True, timeout=15, cwd=ROOT)
        self.assertEqual(run.returncode, 0, run.stderr)
        actual = [int(line.split()[1]) for line in run.stdout.splitlines() if line.startswith("POP ")]
        self.assertEqual(actual, expected)

    def probe(self, mode, first, extra=""):
        result = subprocess.run([str(PROBE), mode], input=encode(first)+" "+extra,
                                text=True, capture_output=True, timeout=10,
                                cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        line = next((line for line in result.stdout.splitlines() if line.startswith("RESULT ")), None)
        self.assertIsNotNone(line, result.stdout)
        return bool(int(line.split()[1]))

    def test_nonlinear_parallel_pair_distinguishes_source_fallback(self):
        first = CanalSegment(((0., 0., 1., 0.), (0., 0., 2., 0.), (0., 0., 0., 0.)), .125, .125)
        with tempfile.TemporaryDirectory() as directory:
            for height, expected in ((.125, True), (.25, True), (.5, False)):
                with self.subTest(height=height):
                    second = CanalSegment(((.125, 0., 0., 1.), (.25, 0., 0., 2.),
                                           (height, 0., 0., 0.)), .125, .125)
                    self.assertEqual(bool(neurite_neurite_candidates(first, second, max_rectangles=1)), expected)
                    run = subprocess.run([str(PROBE), "pair"], input=encode(first)+" "+encode(second),
                                         cwd=directory, text=True, capture_output=True, timeout=10, check=True)
                    # The checked-in source defaults to positive when its
                    # bivariate solve fails, without a geometric witness.
                    self.assertIn("RESULT 1 0", run.stdout)

    def test_source_bivariate_singular_fallback_has_no_witness(self):
        from anamorph.analytic_intersections import _source_pair_singular_risk
        from anamorph.source_bivariate import source_initial_singular_fallback
        first = line((0., 0., 0.), (1., 0., 0.), .0001)
        cases = (
            ("adjacent", line((0., 0., 0.), (-1., 0., 0.), .0001), True, True),
            ("adjacent", line((0., 0., 0.), (-1., 1e-6, 0.), .0001), True, True),
            ("adjacent", CanalSegment(((0., -1., -1e-12, 0.), (0., 0., 0., 0.),
                                       (0., 0., 0., 0.)), .0001, .0001), True, True),
            ("adjacent", CanalSegment(((0., -1., -1e-10, 0.), (0., 0., 0., 0.),
                                       (0., 0., 0., 0.)), .0001, .0001), True, False),
            ("adjacent", line((0., 0., 0.), (0., 1., 0.), .0001), False, False),
            ("pair", line((0., .0003, 0.), (1., .0003, 0.), .0001), True, True),
        )
        for mode, second, fallback, initial in cases:
            with self.subTest(mode=mode, second=second):
                data = encode(first)+" "+encode(second)+(" 0" if mode == "adjacent" else "")
                run = subprocess.run([str(PROBE), mode], input=data, text=True, capture_output=True,
                                     timeout=10, cwd=ROOT)
                self.assertEqual(run.returncode, 0, run.stderr)
                result = next(line for line in run.stdout.splitlines() if line.startswith("RESULT "))
                self.assertEqual(result, "RESULT 1 0" if fallback else "RESULT 0 0")
                self.assertEqual(_source_pair_singular_risk(first, second), fallback)
                self.assertEqual(source_initial_singular_fallback(first, second), initial)
        orthogonal = line((0., 0., 0.), (0., 1., 0.), .0001)
        run = subprocess.run([str(PROBE), "adjacent"], input=encode(first)+" "+encode(orthogonal)+" 0 1.1",
                             text=True, capture_output=True, timeout=10, cwd=ROOT)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("RESULT 1 0", run.stdout)
        self.assertTrue(source_initial_singular_fallback(first, orthogonal, tolerance=1.1))

    @unittest.skipUnless(INITIAL_FIT_PROBE.is_file(), "optional C++ initial-fit probe is not built")
    def test_initial_cubic_fit_matches_cpp(self):
        from anamorph.analytic_intersections import _source_initial_linear_fit
        from anamorph.source_bivariate import source_initial_fit, source_stationary_bernstein
        rng = random.Random(570241)
        for case in range(30):
            scale = rng.choice((.01, 1., 100.))
            curves = [CanalSegment(tuple(tuple(rng.uniform(-2, 2)*scale for _ in range(4))
                                         for _ in range(3)), .1, .1) for _ in range(2)]
            with self.subTest(case=case):
                run = subprocess.run([str(INITIAL_FIT_PROBE)], input=" ".join(map(encode, curves)),
                                     text=True, capture_output=True, timeout=10, cwd=ROOT)
                self.assertEqual(run.returncode, 0, run.stderr)
                line = next(line for line in run.stdout.splitlines() if line.startswith("FIT "))
                reference = tuple(map(float, line.split()[1:]))
                actual = tuple(map(float, _source_initial_linear_fit(*curves)))
                for first, second in zip(actual, reference):
                    self.assertLessEqual(abs(first-second), 2e-11*max(1., abs(first), abs(second)))
                self.assertEqual(source_initial_fit(*curves), reference)
                raw = subprocess.run([str(INITIAL_FIT_PROBE), "coefficients"],
                                     input=" ".join(map(encode, curves)), text=True,
                                     capture_output=True, timeout=10, cwd=ROOT)
                self.assertEqual(raw.returncode, 0, raw.stderr)
                p, q = source_stationary_bernstein(*curves)
                rows = [line.split() for line in raw.stdout.splitlines() if line.startswith("COEFF ")]
                self.assertEqual(len(rows), 36)
                for _, i, j, expected_p, expected_q in rows:
                    self.assertEqual(p.coefficients[int(i)][int(j)], float(expected_p))
                    self.assertEqual(q.coefficients[int(i)][int(j)], float(expected_q))

    def test_analytic_check_results_match_on_representative_curves(self):
        straight = line((-1., 0., 0.), (1., 0., 0.))
        irregular = CanalSegment(((.04, -.4, 1., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)), .1, .1)
        tight = CanalSegment(((0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 0.)), 1., 1.)
        loop = CanalSegment(((0., 1., -1., 0.), (0., -1., 3., -2.), (0., 0., 0., 0.)), .05, .05)
        near = line((0., -1., .15), (0., 1., .15))
        far = line((0., -1., .3), (0., 1., .3))
        cases = (("regularity", straight, "", lambda: regularity_candidates(straight)),
                 ("regularity", irregular, "", lambda: regularity_candidates(irregular)),
                 ("local", straight, "", lambda: local_self_intersection_candidates(straight)),
                 ("local", tight, "", lambda: local_self_intersection_candidates(tight)),
                 ("global", straight, "", lambda: global_self_intersection_candidates(straight)),
                 ("global", loop, "", lambda: global_self_intersection_candidates(loop)),
                 ("soma", straight, "0 .2 0 .15 0", lambda: soma_neurite_candidates(straight, (0., .2, 0.), .15)),
                 ("soma", straight, "0 .4 0 .15 0", lambda: soma_neurite_candidates(straight, (0., .4, 0.), .15)),
                 ("pair", straight, encode(near), lambda: neurite_neurite_candidates(straight, near)),
                 ("pair", straight, encode(far), lambda: neurite_neurite_candidates(straight, far)))
        for mode, curve, extra, python_check in cases:
            with self.subTest(mode=mode, curve=curve, extra=extra):
                self.assertEqual(self.probe(mode, curve, extra), bool(python_check()))

    def test_seeded_local_and_global_curve_checks_match_cpp(self):
        rng = random.Random(672)
        positive = {"local": 0, "global": 0}
        for index in range(24):
            power = tuple(tuple(rng.uniform(-1, 1) for _ in range(4)) for _ in range(3))
            radius = rng.uniform(.05, 1.)
            curve = CanalSegment(power, radius, radius)
            for mode, check in (("local", local_self_intersection_candidates),
                                ("global", global_self_intersection_candidates)):
                with self.subTest(index=index, mode=mode):
                    found = bool(check(curve))
                    positive[mode] += found
                    self.assertEqual(self.probe(mode, curve), found)
        self.assertGreater(positive["local"], 0)
        self.assertGreater(positive["global"], 0)

    def test_regularization_avoids_source_absolute_epsilon_miss(self):
        scale = 1e-3
        curve = CanalSegment(((.04*scale, -.4*scale, scale, 0.),
                              (0., 0., 0., 0.), (0., 0., 0., 0.)), .1, .1)
        self.assertFalse(self.probe("regularity", curve))
        self.assertTrue(regularity_candidates(curve))

    def test_python_detects_proximity_missed_by_this_cpp_solver(self):
        first = CanalSegment(((-.34793157453065215, .769380229684256, -.5836968689785138, -.8876240682599044),
                              (.5125548270898603, .11786673329157393, -.0060744889277624114, .3940825681847342),
                              (.7110593400176666, -.6290370472897477, -.7881708463948436, .3324191404859931)),
                             .0778792706334942, .20553415251406576)
        second = CanalSegment(((-.6813238979187228, .9465693704716893, -.7674282342423369, .8480206665732368),
                               (.14764963819142052, .9888164551661183, .011297815978747572, .5796215374594855),
                               (-.27104355646043765, -.9512796119110336, .7838889217669949, -.07048867808018144)),
                              .06845839730924852, .43442057200436657)
        hits = neurite_neurite_candidates(first, second)
        self.assertTrue(any(hit.source == "interior" for hit in hits))
        # Independently check a nearby sampled pair, without relying on the
        # stationary-point solver: the maximum-radius pipes overlap there.
        self.assertLess(dist(first.point(.7933333333333333), second.point(.4)),
                        first.max_radius + second.max_radius)
        self.assertFalse(self.probe("pair", first, encode(second)))

    def test_source_root_soma_exemption_can_hide_other_soma_at_start(self):
        segment = line((1., 0., 0.), (2., 0., 0.), .05)
        self.assertFalse(self.probe("soma", segment, "1 .2 0 .2 1"))
        self.assertTrue(self.probe("soma", segment, "1 .2 0 .2 0"))
        self.assertTrue(soma_neurite_candidates(segment, (1., .2, 0.), .2, is_root_segment=False))

    @unittest.skipUnless(CELLGEN.is_file(), "optional C++ cell generator is not built")
    def test_individual_surfaces_match_cpp_for_path_and_parametrization_options(self):
        options = product(("max-chordal-depth", "min-angle", "simple-neurite-paths"),
                          ("chord-length", "centripetal", "uniform"))
        for strategy, parametrization in options:
            with self.subTest(strategy=strategy, parametrization=parametrization), tempfile.TemporaryDirectory() as tmp:
                swc = Path(tmp) / "case.swc"
                shutil.copyfile(ROOT / "examples" / "branched.swc", swc)
                run = subprocess.run([str(CELLGEN), "-i", "case", "-force-meshing", "-no-mesh-pp",
                                      "-no-meshing-flush", "-meshing-cansurf-angularsegments", "8",
                                      "-ana-nthreads", "1", "-meshing-individual-surfaces",
                                      "-cellnet-partition-strategy", strategy, "90", "10",
                                      "-cellnet-parametrization-strategy", parametrization],
                                     cwd=tmp, text=True, capture_output=True, timeout=30)
                self.assertEqual(run.returncode, 0, run.stderr + run.stdout)
                reference = read_obj(Path(tmp) / "case_individual_modelling_surfaces.obj")
                actual = render_individual_surfaces(read_swc(swc), angular_segments=8, soma_refinements=3,
                                                    strategy=strategy, parametrization=parametrization,
                                                    arc_length_mode="source")
                self.assertEqual(len(reference.faces), len(actual.faces))
                # Order can vary, so compare the point sets in both directions.
                self.assertLess(max(min(dist(p, q) for q in actual.vertices) for p in reference.vertices), 1e-6)
                self.assertLess(max(min(dist(p, q) for q in reference.vertices) for p in actual.vertices), 1e-6)

    @unittest.skipUnless(CELLGEN.is_file(), "optional C++ cell generator is not built")
    def test_multisection_somas_and_inside_roots_match_cpp(self):
        cases = {
            "mixed_root_order": "1 1 0 0 0 1 -1\n2 3 1.5 0 0 .1 1\n3 3 2.5 0 0 .1 2\n"
                                "4 3 -.2 0 0 .1 1\n5 3 -1.4 0 0 .1 4\n6 3 -2.5 0 0 .1 5\n",
            "branched_soma": "1 1 .1 -.1 0 .9 -1\n2 1 -.2 .3 0 1.2 1\n3 1 .3 .2 0 .8 1\n"
                              "4 1 0 -.4 0 1 2\n5 3 1.7 .2 0 .12 3\n6 3 2.8 .2 0 .11 5\n"
                              "7 3 -1.8 -.4 0 .13 4\n8 3 -2.9 -.4 0 .12 7\n",
            "inside_chain": "1 1 0 0 0 1 -1\n2 3 .2 0 0 .3 1\n3 3 .6 0 0 .25 2\n"
                            "4 3 1.4 0 0 .2 3\n5 3 2.5 0 0 .15 4\n",
            "two_cells": "1 1 0 0 0 1 -1\n2 1 0 .2 0 1.1 1\n3 3 1.5 0 0 .1 2\n4 3 2.5 0 0 .1 3\n"
                         "5 1 8 0 0 1 -1\n6 1 8 -.2 0 .9 5\n7 3 9.5 0 0 .1 6\n8 3 10.5 0 0 .1 7\n",
        }
        for name, swc_text in cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as tmp:
                swc = Path(tmp)/"case.swc"
                swc.write_text(swc_text, encoding="utf-8")
                run = subprocess.run([str(CELLGEN), "-i", "case", "-force-meshing", "-no-mesh-pp",
                                      "-no-meshing-flush", "-meshing-cansurf-angularsegments", "8",
                                      "-ana-nthreads", "1", "-meshing-individual-surfaces"],
                                     cwd=tmp, text=True, capture_output=True, timeout=30)
                self.assertEqual(run.returncode, 0, run.stderr+run.stdout)
                reference = read_obj(Path(tmp)/"case_individual_modelling_surfaces.obj")
                actual = render_individual_surfaces(read_swc(swc), angular_segments=8, soma_refinements=3)
                self.assertEqual((len(reference.vertices), len(reference.faces)),
                                 (len(actual.vertices), len(actual.faces)))
                self.assertLess(max(min(dist(p, q) for q in actual.vertices) for p in reference.vertices), 1e-6)
                self.assertLess(max(min(dist(p, q) for p in reference.vertices) for q in actual.vertices), 1e-6)
                if name in ("mixed_root_order", "two_cells"):
                    # A three-refinement icosphere has 642 vertices; the next
                    # vertex belongs to the first rendered neurite path.
                    self.assertLess(dist(reference.vertices[642], actual.vertices[642]), 1e-9)

    @unittest.skipUnless(HEMISPHERE_PROBE.is_file(), "optional C++ hemisphere probe is not built")
    def test_terminal_hemisphere_matches_cpp_geometry(self):
        for n, offset in product((3, 5, 6, 7, 8, 9, 12), (0., 1.)):
            with self.subTest(angular_segments=n, ring_offset=offset), tempfile.TemporaryDirectory() as tmp:
                phi = .137
                ring = tuple((0., cos(2*pi*j/n+phi+offset*pi/n), sin(2*pi*j/n+phi+offset*pi/n))
                             for j in range(n))
                cap = n
                mesh = Mesh(ring+((0., 0., 0.),), tuple((cap, j, (j+1)%n) for j in range(n)))
                actual = append_terminal_hemisphere(CanalPathMesh(mesh, -1, tuple(range(n)), cap, False),
                                                     start=(0., 0., 0.), direction=(1., 0., 0.), radius=1.,
                                                     angular_segments=n, phi0=phi)
                run = subprocess.run([str(HEMISPHERE_PROBE), str(n), str(phi), str(offset*pi/n)],
                                     cwd=tmp, text=True, capture_output=True, timeout=15)
                self.assertEqual(run.returncode, 0, run.stderr + run.stdout)
                reference = read_obj(Path(tmp) / "hemisphere.obj")
                self.assertEqual((len(reference.vertices), len(reference.faces)),
                                 (len(actual.vertices), len(actual.faces)))
                self.assertLess(max(min(dist(p, q) for q in actual.vertices) for p in reference.vertices), 1e-9)
                self.assertAlmostEqual(reference.statistics().area, actual.statistics().area, places=8)

    @unittest.skipUnless(HEMISPHERE_FRAME_PROBE.is_file(), "optional C++ hemisphere frame probe is not built")
    def test_source_terminal_frames_match_cpp_oblique_caps(self):
        from anamorph.source_float import x87_sincos, x87_sqrt
        from anamorph.canal_mesh import _source_render_frame
        rng = random.Random(95847)
        def source_normalize(vector):
            length = x87_sqrt(vector[0]**2+vector[1]**2+vector[2]**2)
            return tuple(value/length for value in vector)
        with tempfile.TemporaryDirectory() as tmp:
            for case in range(80):
                n = rng.choice((3, 4, 5, 6, 7, 8, 9, 12, 16))
                phi = rng.uniform(-1., 1.)
                offset = rng.choice((0., pi/n))
                start = tuple(rng.uniform(-2, 2) for _ in range(3))
                direction = tuple(rng.uniform(-2, 2) for _ in range(3))
                render = tuple(rng.uniform(-2, 2) for _ in range(3))
                radius = rng.uniform(.1, 3.)
                # The probe constructs the incoming ring using one direction
                # normalization, like a preceding C++ rendering operation.
                axis = source_normalize(direction)
                y = source_normalize(cross(render, axis))
                z = cross(axis, y)
                ring = []
                for j in range(n):
                    sine, cosine = x87_sincos((2*pi*j)/n+phi+offset)
                    ring.append(tuple((start[k]+y[k]*(radius*cosine))+z[k]*(radius*sine) for k in range(3)))
                mesh = Mesh(tuple(ring)+(start,), tuple((n, j, (j+1)%n) for j in range(n)))
                actual = append_terminal_hemisphere(CanalPathMesh(mesh, -1, tuple(range(n)), n, False),
                                                     start=start, direction=direction, radius=radius,
                                                     angular_segments=n, render_vector=render, phi0=phi,
                                                     source_frame=True, quad_sqrt_mode="windows-x87",
                                                     render_trig_mode="windows-x87")
                args = [n, phi, offset, *start, *direction, *render, radius]
                run = subprocess.run([str(HEMISPHERE_FRAME_PROBE), *(str(value) for value in args)],
                                     cwd=tmp, text=True, capture_output=True, timeout=10)
                with self.subTest(case=case):
                    self.assertEqual(run.returncode, 0, run.stderr+run.stdout)
                    source_frame = tuple(float(value) for value in run.stdout.split("FRAME ", 1)[1].splitlines()[0].split())
                    python_frame = tuple(value for vector in _source_render_frame(direction, render, sqrt_mode="windows-x87")
                                         for value in vector)
                    self.assertEqual(source_frame, python_frame)
                    reference = read_obj(Path(tmp)/"hemisphere_frame.obj")
                    self.assertEqual((len(reference.vertices), len(reference.faces)),
                                     (len(actual.vertices), len(actual.faces)))
                    mapping = [min(range(len(actual.vertices)), key=lambda i: dist(point, actual.vertices[i]))
                               for point in reference.vertices]
                    self.assertLess(max(dist(point, actual.vertices[mapping[i]])
                                        for i, point in enumerate(reference.vertices)), 1e-9)
                    expected = {tuple(sorted(mapping[i] for i in face)) for face in reference.faces}
                    observed = {tuple(sorted(face)) for face in actual.faces}
                    self.assertEqual(expected, observed)

    @unittest.skipUnless(CELLGEN.is_file(), "optional C++ cell generator is not built")
    def test_individual_surfaces_match_cpp_after_preconditioning(self):
        rng = random.Random(71)
        x, y = 1., 0.
        lines = ["1 1 0 0 0 1 -1"]
        for i in range(2, 9):
            x += rng.uniform(.2, 1.5)
            y += rng.uniform(-.2, .2)
            radius = rng.uniform(.13, .3)
            lines.append(f"{i} 3 {x:.6f} {y:.6f} 0 {radius:.6f} {i-1}")
        with tempfile.TemporaryDirectory() as tmp:
            swc = Path(tmp) / "case.swc"
            swc.write_text("\n".join(lines)+"\n")
            tree = read_swc(swc)
            self.assertGreater(prepare_morphology(tree).collapsed_segments, 0)
            run = subprocess.run([str(CELLGEN), "-i", "case", "-force-meshing", "-no-mesh-pp",
                                  "-no-meshing-flush", "-meshing-cansurf-angularsegments", "8",
                                  "-ana-nthreads", "1", "-meshing-individual-surfaces"],
                                 cwd=tmp, text=True, capture_output=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stderr + run.stdout)
            reference = read_obj(Path(tmp) / "case_individual_modelling_surfaces.obj")
            actual = render_individual_surfaces(tree, angular_segments=8)
            self.assertEqual(len(reference.faces), len(actual.faces))
            self.assertLess(max(min(dist(p, q) for q in actual.vertices) for p in reference.vertices), 1e-6)
            self.assertLess(max(min(dist(p, q) for q in reference.vertices) for p in actual.vertices), 1e-6)

    @unittest.skipUnless(CELLGEN.is_file(), "optional C++ cell generator is not built")
    def test_individual_surfaces_match_cpp_after_branch_tie(self):
        with tempfile.TemporaryDirectory() as tmp:
            swc = Path(tmp) / "case.swc"
            shutil.copyfile(ROOT / "tests" / "fixtures" / "branch_tie.swc", swc)
            run = subprocess.run([str(CELLGEN), "-i", "case", "-force-meshing", "-no-mesh-pp",
                                  "-no-meshing-flush", "-meshing-cansurf-angularsegments", "8",
                                  "-ana-nthreads", "1", "-meshing-individual-surfaces"],
                                 cwd=tmp, text=True, capture_output=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stderr + run.stdout)
            reference = read_obj(Path(tmp) / "case_individual_modelling_surfaces.obj")
            actual = render_individual_surfaces(read_swc(swc), angular_segments=8)
            self.assertEqual(len(reference.faces), len(actual.faces))
            self.assertLess(max(min(dist(p, q) for q in actual.vertices) for p in reference.vertices), 1e-6)
            self.assertLess(max(min(dist(p, q) for q in reference.vertices) for p in actual.vertices), 1e-6)

    @unittest.skipUnless(CELLGEN.is_file(), "optional C++ cell generator is not built")
    def test_individual_surfaces_match_cpp_for_multisample_soma(self):
        lines = ("1 1 0 0 0 1 -1", "2 1 .5 .1 0 .5 1", "3 1 -.25 0 0 .75 1",
                 "4 3 1.8 0 0 .2 2", "5 3 3 0 0 .18 4", "6 3 4 .2 0 .15 5",
                 "7 3 -1.8 0 0 .2 3", "8 3 -3 .2 0 .2 7")
        with tempfile.TemporaryDirectory() as tmp:
            swc = Path(tmp) / "case.swc"
            swc.write_text("\n".join(lines)+"\n")
            run = subprocess.run([str(CELLGEN), "-i", "case", "-force-meshing", "-no-mesh-pp",
                                  "-no-meshing-flush", "-meshing-cansurf-angularsegments", "8",
                                  "-ana-nthreads", "1", "-meshing-individual-surfaces"],
                                 cwd=tmp, text=True, capture_output=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stderr + run.stdout)
            reference = read_obj(Path(tmp) / "case_individual_modelling_surfaces.obj")
            actual = render_individual_surfaces(read_swc(swc), angular_segments=8)
            self.assertEqual(len(reference.faces), len(actual.faces))
            self.assertLess(max(min(dist(p, q) for q in actual.vertices) for p in reference.vertices), 1e-6)
            self.assertLess(max(min(dist(p, q) for q in reference.vertices) for p in actual.vertices), 1e-6)

    @unittest.skipUnless(CELLGEN.is_file(), "optional C++ cell generator is not built")
    def test_individual_vertical_path_matches_cpp_surface_totals(self):
        lines = ("1 1 0 0 0 1 -1", "2 3 0 0 1.3 .18 1",
                 "3 3 0 0 2.4 .16 2", "4 3 0 0 3.4 .14 3")
        with tempfile.TemporaryDirectory() as tmp:
            swc = Path(tmp) / "case.swc"
            swc.write_text("\n".join(lines)+"\n")
            run = subprocess.run([str(CELLGEN), "-i", "case", "-force-meshing", "-no-mesh-pp",
                                  "-no-meshing-flush", "-meshing-cansurf-angularsegments", "8",
                                  "-ana-nthreads", "1", "-meshing-individual-surfaces"],
                                 cwd=tmp, text=True, capture_output=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stderr + run.stdout)
            reference = read_obj(Path(tmp) / "case_individual_modelling_surfaces.obj").statistics()
            actual = render_individual_surfaces(read_swc(swc), angular_segments=8).statistics()
            # Both choose a nonparallel render vector, but random choices can
            # rotate the rings relative to one another around this straight path.
            self.assertEqual(reference.faces, actual.faces)
            self.assertAlmostEqual(reference.area, actual.area, places=8)
            self.assertAlmostEqual(reference.volume, actual.volume, places=8)

    @unittest.skipUnless(CELLGEN.is_file(), "optional C++ cell generator is not built")
    def test_individual_surfaces_match_cpp_after_edge_splits(self):
        lines = ("1 1 0 0 0 1 -1", "2 3 1.2 0 0 .13 1",
                 "3 3 5.2 .2 0 .25 2", "4 3 10 .3 0 .18 3")
        with tempfile.TemporaryDirectory() as tmp:
            swc = Path(tmp) / "case.swc"
            swc.write_text("\n".join(lines)+"\n")
            tree = read_swc(swc)
            self.assertGreater(prepare_morphology(tree).split_segments, 0)
            run = subprocess.run([str(CELLGEN), "-i", "case", "-force-meshing", "-no-mesh-pp",
                                  "-no-meshing-flush", "-meshing-cansurf-angularsegments", "8",
                                  "-ana-nthreads", "1", "-meshing-individual-surfaces"],
                                 cwd=tmp, text=True, capture_output=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stderr + run.stdout)
            reference = read_obj(Path(tmp) / "case_individual_modelling_surfaces.obj")
            actual = render_individual_surfaces(tree, angular_segments=8)
            self.assertEqual(len(reference.faces), len(actual.faces))
            self.assertLess(max(min(dist(p, q) for q in actual.vertices) for p in reference.vertices), 1e-6)
            self.assertLess(max(min(dist(p, q) for q in reference.vertices) for p in actual.vertices), 1e-6)

    @unittest.skipUnless(REDBLUE_PROBE.is_file(), "optional C++ red-blue probe is not built")
    def test_boolean_signed_volumes_and_areas_match_cpp(self):
        red = refined_tetrahedron()
        dx, dy, dz, scale = .08794858485146739, -.25131913562220887, .47920496341231167, 1.1169174342387775
        blue = Mesh(tuple((x*scale+dx, y*scale+dy, z*scale+dz) for x, y, z in red.vertices), red.faces)
        for operation in ("union", "intersection", "difference"):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)
                red.write_obj(path / "red.obj")
                blue.write_obj(path / "blue.obj")
                run = subprocess.run([str(REDBLUE_PROBE), operation], cwd=tmp, text=True,
                                     capture_output=True, timeout=20)
                self.assertEqual(run.returncode, 0, run.stderr + run.stdout)
                reference = read_obj(path / "result.obj")
                # Compute these independently because the C++ OBJ can contain
                # collinear triangles, which statistics() deliberately rejects.
                def area(mesh):
                    return fsum(sqrt(dot(v, v))/2 for a, b, c in mesh.faces
                                for v in [cross(sub(mesh.vertices[b], mesh.vertices[a]),
                                                sub(mesh.vertices[c], mesh.vertices[a]))])
                def signed_volume(mesh):
                    return fsum(dot(mesh.vertices[a], cross(mesh.vertices[b], mesh.vertices[c]))/6
                                for a, b, c in mesh.faces)
                for source_delaunay in (False, True):
                    with self.subTest(source_delaunay=source_delaunay):
                        actual = redblue_boolean(red, blue, operation=operation, source_delaunay=source_delaunay,
                                                 random_seed=1, random_generator="windows-c")
                        self.assertEqual((len(reference.vertices), len(reference.faces)),
                                         (len(actual.vertices), len(actual.faces)))
                        self.assertAlmostEqual(area(reference), area(actual), places=8)
                        self.assertAlmostEqual(signed_volume(reference), signed_volume(actual), places=8)
                        self.assertGreater(actual.statistics().volume, 0.)

    @unittest.skipUnless(REDBLUE_PROBE.is_file(), "optional C++ red-blue probe is not built")
    def test_bounded_delaunay_union_matches_cpp_geometry(self):
        red = refined_tetrahedron()
        offset = (-.3004177194797668, -.20031414355086757, -.2868115892937734)
        blue = Mesh(tuple(tuple(x*.8761913254934592+offset[i] for i, x in enumerate(point))
                          for point in red.vertices), red.faces)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            red.write_obj(path/"red.obj")
            blue.write_obj(path/"blue.obj")
            run = subprocess.run([str(REDBLUE_PROBE), "union"], cwd=tmp, text=True,
                                 capture_output=True, timeout=20)
            self.assertEqual(run.returncode, 0, run.stderr+run.stdout)
            reference = read_obj(path/"result.obj")
        from anamorph.redblue import SourceDelaunayLimitWarning
        with self.assertWarns(SourceDelaunayLimitWarning):
            actual = redblue_boolean(red, blue, source_delaunay_allow_limit=True,
                                     random_seed=1, random_generator="windows-c")
        def area(mesh):
            return fsum(sqrt(dot(v, v))/2 for a, b, c in mesh.faces
                        for v in [cross(sub(mesh.vertices[b], mesh.vertices[a]),
                                        sub(mesh.vertices[c], mesh.vertices[a]))])
        def signed_volume(mesh):
            return fsum(dot(mesh.vertices[a], cross(mesh.vertices[b], mesh.vertices[c]))/6
                        for a, b, c in mesh.faces)
        self.assertEqual((len(reference.vertices), len(reference.faces)), (len(actual.vertices), len(actual.faces)))
        self.assertAlmostEqual(area(reference), area(actual), places=8)
        self.assertAlmostEqual(signed_volume(reference), signed_volume(actual), places=8)

    @unittest.skipUnless(CELLGEN.is_file(), "optional C++ cell generator is not built")
    def test_network_reports_crossing_missed_in_cpp_example(self):
        lines = ("1 1 -3 -3 0 .2 -1", "2 3 -2 0 0 .1 1", "3 3 2 0 0 .1 2",
                 "4 3 0 -2 0 .1 1", "5 3 0 2 0 .1 4")
        with tempfile.TemporaryDirectory() as tmp:
            swc = Path(tmp) / "case.swc"
            swc.write_text("\n".join(lines)+"\n")
            run = subprocess.run([str(CELLGEN), "-i", "case", "-no-meshing", "-no-mesh-pp",
                                  "-no-cellnet-pc", "-ana-nthreads", "1"],
                                 cwd=tmp, text=True, capture_output=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stderr + run.stdout)
            self.assertEqual(run.stdout.count("intra-cell inter-neurite:"), 1)
            # The source prints the first segment's midpoint; this is the
            # near-soma pair, not the later crossing at the origin.
            self.assertIn("at [-2.468377, -1.405132, 0.000000]", run.stdout)
            report = analyze_analytic_partial(read_swc(swc), prepare=True, preconditioning=False)
            intersections = [f for f in report.findings if f.classification == "intra_cell_inter_neurite"]
            self.assertEqual(len(intersections), 2)
            crossing = next(f for f in intersections if set(f.segments) == {(2, 3), (4, 5)})
            self.assertTrue(any(c.source == "interior" and c.value < 1e-8 for c in crossing.candidates))

    @unittest.skipUnless(CELLGEN.is_file(), "optional C++ cell generator is not built")
    def test_redblue_assembly_is_close_to_cpp_on_branching_example(self):
        with tempfile.TemporaryDirectory() as tmp:
            swc = Path(tmp) / "case.swc"
            shutil.copyfile(ROOT / "examples" / "branched.swc", swc)
            run = subprocess.run([str(CELLGEN), "-i", "case", "-force-meshing", "-no-mesh-pp",
                                  "-no-meshing-flush", "-meshing-cansurf-angularsegments", "8",
                                  "-meshing-soma-refs", "2", "-ana-nthreads", "1"],
                                 cwd=tmp, text=True, capture_output=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stderr + run.stdout)
            reference = read_obj(Path(tmp) / "case.obj").statistics()
            actual_mesh = assemble_redblue_neuron(read_swc(swc), angular_segments=8, soma_refinements=2)
            actual = actual_mesh.statistics()
            # Source and Python choose different seam angles and may refine
            # different faces, so compare geometric totals, not vertex IDs.
            self.assertLess(abs(actual.area/reference.area-1), .001)
            self.assertLess(abs(actual.volume/reference.volume-1), .001)
            self.assertEqual((actual.boundary_edges, actual.nonmanifold_edges), (0, 0))
            hits, truncated = find_self_intersections(actual_mesh, max_hits=1, include_adjacent=True)
            self.assertFalse(hits)
            self.assertFalse(truncated)

    @unittest.skipUnless(CELLGEN.is_file(), "optional C++ cell generator is not built")
    def test_hc_smoothing_matches_cpp_on_the_same_mesh(self):
        sphere = ico_sphere((.1, -.2, .3), 1., 1)
        # Break perfect symmetry so the comparison covers different local
        # neighborhoods and displacements, rather than a single radius.
        mesh = Mesh(tuple((x*(1+.002*i), y, z) for i, (x, y, z) in enumerate(sphere.vertices)), sphere.faces)
        expected = hc_smooth(mesh)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "case.obj"
            mesh.write_obj(path)
            run = subprocess.run([str(CELLGEN), "-i", "case", "-no-analysis", "-no-mesh-pp-gec"],
                                 cwd=tmp, text=True, capture_output=True, timeout=15)
            self.assertEqual(run.returncode, 0, run.stderr)
            actual = read_obj(Path(tmp) / "case_post_processed.obj")
        self.assertEqual(len(actual.vertices), len(expected.vertices))
        self.assertEqual(actual.faces, expected.faces)
        self.assertLess(max(dist(a, b) for a, b in zip(actual.vertices, expected.vertices)), 2e-9)

    @unittest.skipUnless(LAPLACIAN_PROBE.is_file(), "optional C++ Laplacian probe is not built")
    def test_laplacian_smoothing_matches_cpp_on_the_same_mesh(self):
        sphere = ico_sphere((.1, -.2, .3), 1., 1)
        mesh = Mesh(tuple((x*(1+.002*i), y, z) for i, (x, y, z) in enumerate(sphere.vertices)), sphere.faces)
        expected = laplacian_smooth(mesh, smoothing_factor=.35, iterations=3)
        with tempfile.TemporaryDirectory() as tmp:
            mesh.write_obj(Path(tmp) / "mesh.obj")
            run = subprocess.run([str(LAPLACIAN_PROBE), ".35", "3"], cwd=tmp,
                                 text=True, capture_output=True, timeout=15)
            self.assertEqual(run.returncode, 0, run.stderr + run.stdout)
            actual = read_obj(Path(tmp) / "result.obj")
        self.assertEqual(actual.faces, expected.faces)
        self.assertLess(max(dist(a, b) for a, b in zip(actual.vertices, expected.vertices)), 2e-9)

    @unittest.skipUnless(COLLAPSE_NEIGHBORHOOD_PROBE.is_file(), "optional C++ collapse-neighborhood probe is not built")
    def test_collapse_neighborhood_order_and_area_match_cpp(self):
        rng = random.Random(814200)
        sphere = ico_sphere((0., 0., 0.), 1., 2)
        mesh = Mesh(tuple(tuple(value*rng.uniform(.7, 1.3) for value in vertex)
                          for vertex in sphere.vertices), sphere.faces)
        edge_faces = defaultdict(set)
        for fid, face in enumerate(mesh.faces):
            for a, b in zip(face, face[1:]+face[:1]):
                edge_faces[tuple(sorted((a, b)))].add(fid)
        with tempfile.TemporaryDirectory() as tmp:
            mesh.write_obj(Path(tmp)/"case.obj")
            for depth in (0, 1, 3, 5):
                with self.subTest(depth=depth):
                    run = subprocess.run([str(COLLAPSE_NEIGHBORHOOD_PROBE), str(depth), "1.5"],
                                         cwd=tmp, capture_output=True, text=True, timeout=30, check=True)
                    metrics, rows = {}, []
                    for line in run.stdout.splitlines():
                        fields = line.split()
                        if fields and fields[0] == "METRIC":
                            metrics[int(fields[1])] = (float(fields[2]), float(fields[3]))
                        elif fields and fields[0] == "NBHD":
                            rows.append((int(fields[1]), float(fields[2]), tuple(map(int, fields[4:]))))
                    self.assertEqual(len(rows), len(mesh.faces))
                    for face_id, average, expected in rows:
                        actual = _source_face_neighborhood(face_id, mesh.faces, edge_faces, depth)
                        self.assertEqual(tuple(actual), expected)
                        # Supply the reference metrics to isolate ordering and
                        # floating-point summation from triangle-metric rounding.
                        self.assertEqual(_source_neighborhood_average(face_id, actual, metrics, 1.5), average)

    @unittest.skipUnless(CELLGEN.is_file(), "optional C++ cell generator is not built")
    def test_collapse_neighborhood_order_resolves_threshold_case(self):
        rng = random.Random(814200)
        sphere = ico_sphere((0., 0., 0.), 1., 2)
        mesh = Mesh(tuple(tuple(value*rng.uniform(.7, 1.3) for value in vertex)
                          for vertex in sphere.vertices), sphere.faces)
        ceiling = 0.5565047593904789
        actual = greedy_edge_collapse(mesh, area_floor=2., area_ceiling=ceiling, depth=3, safety="source")
        with tempfile.TemporaryDirectory() as tmp:
            mesh.write_obj(Path(tmp)/"case.obj")
            run = subprocess.run([str(CELLGEN), "-i", "case", "-no-analysis", "-no-mesh-pp-hc",
                                  "-mesh-pp-gec", "1.5", "2", repr(ceiling), "3"],
                                 cwd=tmp, text=True, capture_output=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stderr+run.stdout)
            reference = read_obj(Path(tmp)/"case_post_processed.obj")
        self.assertEqual((len(reference.vertices), len(reference.faces)), (115, 226))
        self.assertEqual((len(actual.vertices), len(actual.faces)), (115, 226))
        mapping = [min(range(len(actual.vertices)), key=lambda i: dist(vertex, actual.vertices[i]))
                   for vertex in reference.vertices]
        self.assertEqual(len(set(mapping)), len(mapping))
        self.assertLess(max(dist(vertex, actual.vertices[mapping[i]])
                            for i, vertex in enumerate(reference.vertices)), 2e-9)
        def canonical(face):
            return min(face, face[1:]+face[:1], face[2:]+face[:2])
        self.assertEqual(Counter(canonical(tuple(mapping[i] for i in face)) for face in reference.faces),
                         Counter(canonical(face) for face in actual.faces))

    @unittest.skipUnless(CELLGEN.is_file(), "optional C++ cell generator is not built")
    def test_edge_collapse_matches_cpp_on_small_sliver(self):
        sphere = ico_sphere((0., 0., 0.), 1., 1)
        a, b, c = sphere.faces[0]
        point = tuple(.98*sphere.vertices[a][i]+.01*(sphere.vertices[b][i]+sphere.vertices[c][i])
                      for i in range(3))
        n = len(sphere.vertices)
        mesh = Mesh(sphere.vertices+(point,), ((a, b, n), (b, c, n), (c, a, n))+sphere.faces[1:])
        expected = greedy_edge_collapse(mesh)
        with tempfile.TemporaryDirectory() as tmp:
            mesh.write_obj(Path(tmp) / "case.obj")
            run = subprocess.run([str(CELLGEN), "-i", "case", "-no-analysis", "-no-mesh-pp-hc"],
                                 cwd=tmp, text=True, capture_output=True, timeout=15)
            self.assertEqual(run.returncode, 0, run.stderr)
            actual = read_obj(Path(tmp) / "case_post_processed.obj")
        self.assertEqual(len(actual.faces), len(expected.faces))
        self.assertLess(len(actual.faces), len(mesh.faces))
        # Vertex IDs can be reused in a different order after collapse.
        self.assertLess(max(min(dist(a, b) for b in expected.vertices) for a in actual.vertices), 2e-9)
        self.assertAlmostEqual(actual.statistics().volume, expected.statistics().volume, places=9)

    @unittest.skipUnless(CELLGEN.is_file(), "optional C++ cell generator is not built")
    def test_edge_collapse_matches_cpp_on_multiple_symmetric_slivers(self):
        sphere = ico_sphere((0., 0., 0.), 1., 1)
        for count in (4, 10, 25, 80):
            with self.subTest(subdivided_faces=count), tempfile.TemporaryDirectory() as tmp:
                vertices, faces = list(sphere.vertices), []
                for index, (a, b, c) in enumerate(sphere.faces):
                    if index < count:
                        point = tuple(.98*vertices[a][i]+.01*(vertices[b][i]+vertices[c][i]) for i in range(3))
                        new = len(vertices)
                        vertices.append(point)
                        faces.extend(((a, b, new), (b, c, new), (c, a, new)))
                    else:
                        faces.append((a, b, c))
                mesh = Mesh(tuple(vertices), tuple(faces))
                expected = greedy_edge_collapse(mesh, safety="source")
                mesh.write_obj(Path(tmp)/"case.obj")
                run = subprocess.run([str(CELLGEN), "-i", "case", "-no-analysis", "-no-mesh-pp-hc"],
                                     cwd=tmp, text=True, capture_output=True, timeout=15)
                self.assertEqual(run.returncode, 0, run.stderr)
                actual = read_obj(Path(tmp)/"case_post_processed.obj")
                self.assertEqual((len(actual.vertices), len(actual.faces)),
                                 (len(expected.vertices), len(expected.faces)))
                self.assertLess(max(min(dist(a, b) for b in expected.vertices) for a in actual.vertices), 2e-9)
                self.assertLess(max(min(dist(a, b) for b in actual.vertices) for a in expected.vertices), 2e-9)
                mapping = [min(range(len(expected.vertices)), key=lambda i: dist(vertex, expected.vertices[i]))
                           for vertex in actual.vertices]
                self.assertEqual(len(set(mapping)), len(mapping))
                def canonical(face):
                    return min(face, face[1:]+face[:1], face[2:]+face[:2])
                self.assertEqual({canonical(face) for face in expected.faces},
                                 {canonical(tuple(mapping[i] for i in face)) for face in actual.faces})
                self.assertAlmostEqual(actual.statistics().area, expected.statistics().area, places=8)

    @unittest.skipUnless(CELLGEN.is_file(), "optional C++ cell generator is not built")
    def test_source_collapse_mode_reproduces_coincident_faces_on_tetrahedron(self):
        mesh = Mesh(((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.)),
                    ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)))
        expected = greedy_edge_collapse(mesh, alpha=1., area_floor=2., area_ceiling=2., depth=2, safety="source")
        with tempfile.TemporaryDirectory() as tmp:
            mesh.write_obj(Path(tmp)/"case.obj")
            run = subprocess.run([str(CELLGEN), "-i", "case", "-no-analysis", "-no-mesh-pp-hc",
                                  "-mesh-pp-gec", "1", "2", "2", "2"],
                                 cwd=tmp, text=True, capture_output=True, timeout=15)
            self.assertEqual(run.returncode, 0, run.stderr)
            actual = read_obj(Path(tmp)/"case_post_processed.obj")
        self.assertEqual((len(actual.vertices), len(actual.faces)), (3, 2))
        self.assertEqual(set(actual.faces[0]), set(actual.faces[1]))
        self.assertLess(max(min(dist(a, b) for b in expected.vertices) for a in actual.vertices), 2e-9)

    @unittest.skipUnless(CELLGEN.is_file(), "optional C++ cell generator is not built")
    def test_source_collapse_matches_cpp_on_seeded_irregular_slivers(self):
        rng = random.Random(19422)
        sphere = ico_sphere((0., 0., 0.), 1., 1)
        for trial in range(5):
            with self.subTest(trial=trial), tempfile.TemporaryDirectory() as tmp:
                selected = set(rng.sample(range(len(sphere.faces)), rng.randint(5, 40)))
                vertices, faces = list(sphere.vertices), []
                for index, (a, b, c) in enumerate(sphere.faces):
                    if index in selected:
                        weight = rng.choice((.005, .01, .02, .04))
                        point = tuple((1-2*weight)*vertices[a][k]+weight*(vertices[b][k]+vertices[c][k])
                                      for k in range(3))
                        new = len(vertices)
                        vertices.append(point)
                        faces.extend(((a, b, new), (b, c, new), (c, a, new)))
                    else:
                        faces.append((a, b, c))
                mesh = Mesh(tuple(vertices), tuple(faces))
                expected = greedy_edge_collapse(mesh, safety="source")
                mesh.write_obj(Path(tmp)/"case.obj")
                run = subprocess.run([str(CELLGEN), "-i", "case", "-no-analysis", "-no-mesh-pp-hc"],
                                     cwd=tmp, text=True, capture_output=True, timeout=15)
                self.assertEqual(run.returncode, 0, run.stderr)
                actual = read_obj(Path(tmp)/"case_post_processed.obj")
                self.assertEqual((len(actual.vertices), len(actual.faces)),
                                 (len(expected.vertices), len(expected.faces)))
                mapping = [min(range(len(expected.vertices)), key=lambda i: dist(vertex, expected.vertices[i]))
                           for vertex in actual.vertices]
                self.assertEqual(len(set(mapping)), len(mapping))
                self.assertLess(max(dist(actual.vertices[i], expected.vertices[j]) for i, j in enumerate(mapping)), 2e-9)
                def canonical(face):
                    return min(face, face[1:]+face[:1], face[2:]+face[:2])
                self.assertEqual({canonical(face) for face in expected.faces},
                                 {canonical(tuple(mapping[i] for i in face)) for face in actual.faces})


if __name__ == "__main__":
    unittest.main()
