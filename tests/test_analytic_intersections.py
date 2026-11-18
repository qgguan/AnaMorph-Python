import unittest
import json
import subprocess
import sys
import random
from fractions import Fraction
from math import comb, inf, ldexp
from pathlib import Path
from unittest.mock import patch

from anamorph.analytic_intersections import (CanalSegment, CurveCandidate, IntersectionSolveError, ScalarCandidate, _relative_displacement, local_self_intersection_candidates,
                                            neurite_neurite_candidates, regularity_candidates,
                                            soma_neurite_candidates, global_self_intersection_candidates,
                                            adjacent_neurite_candidates)
from anamorph.bernstein import BernsteinPolynomial, BiBernsteinPolynomial, ClippingConvergenceError, RootInterval
from anamorph.polynomial import multiply
from anamorph.analytic_network import (_BoxIndex, _bounding_box, _overlap, _overlapping_box_pairs, _soma_box,
                                       _verified_corner_candidate, analyze_analytic_partial)
from anamorph.swc import Morphology, Node, read_swc


def line(start, end, radius=.1):
    return CanalSegment(tuple((a, b-a, 0., 0.) for a, b in zip(start, end)), radius, radius)


class AnalyticIntersectionTests(unittest.TestCase):
    def test_network_analysis_handles_extreme_uniform_coordinate_scales(self):
        tree = read_swc(Path(__file__).resolve().parent.parent/"examples"/"branched.swc")
        for scale in (1e-200, 1., 1e200):
            with self.subTest(scale=scale):
                scaled = Morphology([Node(node.id, node.kind, node.x*scale, node.y*scale, node.z*scale,
                                          node.radius*scale, node.parent) for node in tree.nodes.values()])
                report = analyze_analytic_partial(scaled, max_rectangles=300)
                self.assertEqual((report.segment_count, report.findings, report.unresolved), (5, (), ()))

    def test_polynomial_normalization_rejects_lost_nonzero_coefficients(self):
        from anamorph.analytic_intersections import _unit_scale_polynomial
        for polynomial in (BernsteinPolynomial((1e308, -1e-300)),
                           BiBernsteinPolynomial(((1e308, -1e-300), (1., -1.)))):
            with self.subTest(polynomial=polynomial), self.assertRaisesRegex(ClippingConvergenceError, "scale ratio"):
                _unit_scale_polynomial(polynomial)

    def test_soma_box_index_matches_brute_force_and_prunes_distant_boxes(self):
        rng = random.Random(92641)
        boxes = []
        for _ in range(300):
            center = [rng.uniform(-30., 30.) for _ in range(3)]
            radius = rng.uniform(.01, 4.)
            boxes.append(tuple((c-radius, c+radius) for c in center))
        boxes.extend((((-float('inf'), 0.), (-1., 1.), (-1., 1.)),
                      ((3., float('inf')), (-1., 1.), (-1., 1.)),
                      ((1., 1.), (1., 1.), (1., 1.))))
        index = _BoxIndex(boxes)
        queries = [((1., 1.), (1., 1.), (1., 1.)),
                   ((-float('inf'), float('inf')), (-1., 1.), (-1., 1.))]
        for _ in range(500):
            center = [rng.uniform(-35., 35.) for _ in range(3)]
            radius = rng.uniform(0., 8.)
            queries.append(tuple((c-radius, c+radius) for c in center))
        for box in queries:
            self.assertEqual(index.query(box), [i for i, candidate in enumerate(boxes)
                                                 if _overlap(candidate, box)])
        self.assertEqual(_BoxIndex([]).query(queries[0]), [])
        sparse = [((float(i*10), float(i*10+1)), (0., 1.), (0., 1.)) for i in range(512)]
        sparse_index = _BoxIndex(sparse)
        with patch("anamorph.analytic_network._overlap", wraps=_overlap) as overlap:
            self.assertEqual(sparse_index.query(((2000., 2001.), (1., 2.), (1., 2.))), [200])
            self.assertLess(overlap.call_count, 40)

    def test_soma_index_preserves_network_findings(self):
        tree = Morphology([Node(1, 1, -3, 0, 0, .5, -1), Node(2, 3, 0, 0, 0, .1, 1),
                           Node(3, 3, 2, 0, 0, .1, 2), Node(4, 1, 1, .15, 0, .1, -1),
                           Node(5, 1, 50, 0, 0, .1, -1)])
        class BruteForceIndex:
            def __init__(self, boxes):
                self.boxes = boxes

            def query(self, box):
                return [i for i, candidate in enumerate(self.boxes) if _overlap(candidate, box)]

        expected = analyze_analytic_partial(tree).as_dict()
        self.assertTrue(any(f["kind"] == "soma_neurite_candidate" and f["somas"] == (4,)
                            for f in expected["findings"]))
        with patch("anamorph.analytic_network._BoxIndex", BruteForceIndex):
            self.assertEqual(analyze_analytic_partial(tree).as_dict(), expected)

    def test_pair_limit_preserves_actual_interior_root_and_network_finding(self):
        power = multiply([-.2, 1.], multiply([-.7, 1.], [-.7, 1.]))
        first = line((0., 0., 0.), (1., 0., 0.), 1e-5)
        second = CanalSegment(((0., 1., 0., 0.), tuple(power), (0., 0., 0., 0.)), 1e-5, 1e-5)
        with self.assertRaises(IntersectionSolveError) as raised:
            neurite_neurite_candidates(first, second, max_rectangles=23)
        self.assertTrue(any(hit.source == "interior" and abs(hit.t-.2) < 1e-4 and abs(hit.u-.2) < 1e-4
                            for hit in raised.exception.candidates))
        self.assertTrue(any("rectangle limit" in failure for failure in raised.exception.failures))
        tree = Morphology([Node(1, 1, -3, 0, 0, .1, -1), Node(2, 3, 0, 0, 0, 1e-5, 1),
                           Node(3, 3, 1, 0, 0, 1e-5, 2), Node(4, 1, -3, 1, 0, .1, -1),
                           Node(5, 3, 0, power[0], 0, 1e-5, 4),
                           Node(6, 3, 1, sum(power), 0, 1e-5, 5)])
        with patch("anamorph.analytic_network.CanalSegment.from_spline", side_effect=(first, second)):
            report = analyze_analytic_partial(tree, max_rectangles=23)
        finding = next(f for f in report.findings if f.kind == "neurite_neurite_candidate")
        self.assertTrue(any(c.source == "interior" and abs(c.parameters[0]-.2) < 1e-4
                            for c in finding.candidates))
        self.assertTrue(any("rectangle limit" in message for message in report.unresolved))

    def test_global_limit_preserves_actual_interior_root(self):
        loop = CanalSegment(((0., -1., 1., 0.), (-.08, .66, -1.5, 1.),
                             (0., 0., 0., 0.)), 1e-8, 1e-8)
        with self.assertRaises(IntersectionSolveError) as raised:
            global_self_intersection_candidates(loop, max_rectangles=94)
        self.assertTrue(any(hit.source == "interior" and abs(hit.t-.2) < 1e-6 and abs(hit.u-.8) < 1e-6
                            for hit in raised.exception.candidates))
        self.assertTrue(any("rectangle limit" in failure for failure in raised.exception.failures))

    def test_univariate_limit_preserves_partial_regularity_lsi_and_soma_results(self):
        derivative = multiply([-.2, 1.], [-.7, 1.])
        irregular = CanalSegment(((0., derivative[0], derivative[1]/2, derivative[2]/3),
                                  (0., 0., 0., 0.), (0., 0., 0., 0.)), .01, .01)
        with self.assertRaises(IntersectionSolveError) as reg_error:
            regularity_candidates(irregular, tolerance=1e-8, max_intervals=34)
        self.assertTrue(any(abs(root.midpoint-.7) < 2e-6 for root in reg_error.exception.candidates))
        tree = Morphology([Node(1, 1, -3, 0, 0, .1, -1), Node(2, 3, 0, 0, 0, .01, 1),
                           Node(3, 3, 1, 0, 0, .01, 2)])
        with patch("anamorph.analytic_network.CanalSegment.from_spline", return_value=irregular):
            report = analyze_analytic_partial(tree, univariate_tolerance=1e-8, max_intervals=34, max_rectangles=100)
        reg = next(finding for finding in report.findings if finding.kind == "regularity")
        self.assertTrue(any(abs(candidate.parameters[0]-.7) < 2e-6 for candidate in reg.candidates))
        self.assertTrue(any("regularity roots" in message for message in report.unresolved))
        tight = CanalSegment(((0., .6515463227265368, -.41392866402616635, -1.605147009650167),
                              (0., .2650348134208267, .06796280051741599, 1.1803563948397557),
                              (0., -.9944088154835407, .22070900700414153, .24741423331226509)),
                             1.1873073516013635, 1.8819851508607668)
        with self.assertRaises(IntersectionSolveError) as lsi_error:
            local_self_intersection_candidates(tight, tolerance=1e-8, max_intervals=11, with_bounds=True)
        self.assertTrue(any(hit.source == "root_interval" and abs(hit.parameter-.619043) < 1e-5
                            for hit in lsi_error.exception.candidates))
        power = multiply([-.2, 1.], multiply([-.7, 1.], [-.7, 1.]))
        curve = CanalSegment((tuple(power), (0., 0., 0., 0.), (0., 0., 0., 0.)), 1e-4, 1e-4)
        with self.assertRaises(IntersectionSolveError) as soma_error:
            soma_neurite_candidates(curve, (0., 0., 0.), 1e-4, tolerance=1e-8,
                                    max_intervals=19, with_bounds=True)
        self.assertTrue(any(hit.source == "root_interval" and abs(hit.parameter-.2) < 1e-6
                            for hit in soma_error.exception.candidates))

    def test_curve_bounding_box_does_not_shrink_past_rounded_endpoint(self):
        power = ((9468013402.120762, -6929321858.934339, -696411075.9220085, -8503788217.286934),
                 (-3948748771.5028505, -7109407552.210476, -2891016099.5906305, 3150098711.104439),
                 (8690355867.345402, 3984605774.911831, -9309676771.804413, 3479144427.2132893))
        segment = CanalSegment(power, 1e-200, 1e-200)
        box = _bounding_box(segment)
        for t in (0., .5, .9, 1.):
            self.assertTrue(all(low <= value <= high for value, (low, high) in zip(segment.point(t), box)))

    def test_curve_bounds_enclose_exact_bezier_hull_across_float_range(self):
        rng = random.Random(72291)
        curves = [CanalSegment(((1.4e308, 1e308, -1e308, -1e308),
                                (-1.4e308, -1e308, 1e308, 1e308),
                                (0., 0., 0., 0.)), 1e-300, 1e-300)]
        for _ in range(200):
            exponent = rng.randrange(-1050, 1000)
            power = tuple(tuple(ldexp(rng.uniform(-1., 1.), exponent) for _ in range(4))
                          for _ in range(3))
            radius = ldexp(rng.uniform(.1, 1.), exponent)
            curves.append(CanalSegment(power, radius, radius))
        for segment in curves:
            for power, (lower, upper) in zip(segment.power, _bounding_box(segment)):
                controls = [sum((Fraction(power[j])*Fraction(comb(i, j), comb(3, j))
                                 for j in range(i+1)), Fraction(0)) for i in range(4)]
                exact_lower = min(controls)-Fraction(segment.max_radius)
                exact_upper = max(controls)+Fraction(segment.max_radius)
                self.assertTrue(lower == -inf or Fraction(lower) <= exact_lower)
                self.assertTrue(upper == inf or Fraction(upper) >= exact_upper)

    def test_soma_box_rounds_outward_at_large_coordinates(self):
        from fractions import Fraction
        soma = Node(1, 1, 1e16, 0., 0., 1., -1)
        lower, upper = _soma_box(soma)[0]
        center, radius = Fraction(soma.x), Fraction(soma.radius)
        self.assertLessEqual(Fraction(lower), center-radius)
        self.assertGreaterEqual(Fraction(upper), center+radius)

    def test_line_pipe_candidates_match_closed_form_segment_distance(self):
        rng = random.Random(851)
        def sub(a, b):
            return tuple(x-y for x, y in zip(a, b))
        def dot(a, b):
            return sum(x*y for x, y in zip(a, b))
        def segment_distance_squared(a, b, c, d):
            u, v, w = sub(b, a), sub(d, c), sub(a, c)
            aa, bb, cc, dd, ee = dot(u, u), dot(u, v), dot(v, v), dot(u, w), dot(v, w)
            candidates = [(t, max(0., min(1., (ee+bb*t)/cc))) for t in (0., 1.)]
            candidates += [(max(0., min(1., (bb*s-dd)/aa)), s) for s in (0., 1.)]
            determinant = aa*cc-bb*bb
            if determinant > 1e-15:
                t, s = (bb*ee-cc*dd)/determinant, (aa*ee-bb*dd)/determinant
                if 0 <= t <= 1 and 0 <= s <= 1:
                    candidates.append((t, s))
            return min(sum((w[i]+t*u[i]-s*v[i])**2 for i in range(3)) for t, s in candidates)
        for index in range(100):
            a, b, c, d = [tuple(rng.uniform(-2, 2) for _ in range(3)) for _ in range(4)]
            r1, r2 = rng.uniform(.05, .5), rng.uniform(.05, .5)
            expected = segment_distance_squared(a, b, c, d) <= (r1+r2)**2
            with self.subTest(index=index):
                self.assertEqual(bool(neurite_neurite_candidates(line(a, b, r1), line(c, d, r2))), expected)

    def test_cubic_pairs_with_sampled_proximity_witness_are_reported(self):
        rng = random.Random(2211)
        witnesses = 0
        for index in range(100):
            curves = []
            for _ in range(2):
                power = tuple(tuple(rng.uniform(-1., 1.) for _ in range(4)) for _ in range(3))
                radius = rng.uniform(.05, .35)
                curves.append(CanalSegment(power, radius, radius))
            first, second = curves
            threshold = (first.max_radius+second.max_radius)**2
            first_samples = [first.point(i/30) for i in range(31)]
            second_samples = [second.point(i/30) for i in range(31)]
            witness = any(sum((x-y)**2 for x, y in zip(a, b)) < threshold
                          for a in first_samples for b in second_samples)
            if witness:
                witnesses += 1
                with self.subTest(index=index):
                    self.assertTrue(neurite_neurite_candidates(first, second))
        self.assertGreater(witnesses, 0)

    def test_thin_curved_intersections_are_not_lost_at_root_midpoints(self):
        rng = random.Random(190233)
        for index in range(20):
            first_power = tuple(tuple(rng.uniform(-1., 1.) for _ in range(4)) for _ in range(3))
            second_power = tuple(tuple(rng.uniform(-1., 1.) for _ in range(4)) for _ in range(3))
            first = CanalSegment(first_power, 1e-8, 1e-8)
            t, u = rng.uniform(.2, .8), rng.uniform(.2, .8)
            point = first.point(t)
            second = CanalSegment(tuple((point[j]-sum(second_power[j][k]*u**k for k in range(1, 4)),
                                         *second_power[j][1:]) for j in range(3)), 1e-8, 1e-8)
            witness = sum((a-b)**2 for a, b in zip(first.point(t), second.point(u)))
            with self.subTest(index=index):
                self.assertLess(witness, (2e-8)**2)
                hits = neurite_neurite_candidates(first, second)
                self.assertTrue(any(hit.distance_squared <= (2e-8)**2 for hit in hits))
                for hit in hits:
                    self.assertTrue(hit.parameter_bounds[0][0] <= hit.t <= hit.parameter_bounds[0][1])
                    self.assertTrue(hit.parameter_bounds[1][0] <= hit.u <= hit.parameter_bounds[1][1])

    def test_thin_curved_intersections_survive_large_translation(self):
        from fractions import Fraction
        rng = random.Random(83414)
        for index in range(25):
            first_power = [[rng.randrange(-16, 17)/16 for _ in range(4)] for _ in range(3)]
            second_power = [[rng.randrange(-16, 17)/16 for _ in range(4)] for _ in range(3)]
            for axis in range(3):
                first_power[axis][0] += -1e9 if axis == 1 else 1e9
            first = CanalSegment(tuple(map(tuple, first_power)), 1e-12, 1e-12)
            t, u = rng.randrange(7, 26)/32, rng.randrange(7, 26)/32
            point = first.point(t)
            for axis in range(3):
                second_power[axis][0] = point[axis]-sum(second_power[axis][k]*u**k for k in range(1, 4))
            second = CanalSegment(tuple(map(tuple, second_power)), 1e-12, 1e-12)
            def exact(component, parameter):
                parameter, value = Fraction(parameter), Fraction(0)
                for coefficient in reversed(component):
                    value = value*parameter+Fraction(coefficient)
                return value
            witness = sum((exact(a, t)-exact(b, u))**2 for a, b in zip(first.power, second.power))
            with self.subTest(index=index):
                self.assertLess(witness, (2*Fraction(first.radius0))**2)
                hits = neurite_neurite_candidates(first, second)
                self.assertTrue(any(hit.distance_squared < (2e-12)**2 for hit in hits))

    def test_translated_intersection_keeps_witness_if_other_roots_are_ambiguous(self):
        first = CanalSegment(((1e9, -.6875, -.5625, 0.), (-1000000000.5625, -.5, .125, .8125),
                              (1000000000.25, -.4375, 1., .875)), 1e-12, 1e-12)
        second = CanalSegment(((999999999.2294312, .125, .125, .5625),
                               (-1000000000.8639851, .9375, -.5625, -.875),
                               (1000000000.893116, -.75, .25, -.4375)), 1e-12, 1e-12)
        try:
            hits = neurite_neurite_candidates(first, second)
        except IntersectionSolveError as error:
            self.assertTrue(error.failures)
            hits = error.candidates
        self.assertTrue(any(hit.distance_squared < (2e-12)**2 for hit in hits))

    def test_unrepresentable_relative_displacement_is_unresolved(self):
        first = CanalSegment(((1e308, 1e308, 0., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)), 1., 1.)
        second = CanalSegment(((0., 0., 0., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)), 1., 1.)
        with self.assertRaisesRegex(ClippingConvergenceError, "displacement exceeds float range"):
            _relative_displacement(first, second, 1., 0.)

    def test_thin_boundary_adjacent_and_global_witnesses_are_refined(self):
        t = .3141592653589793
        first = CanalSegment(((0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 0.)), 1e-10, 1e-10)
        point = first.point(t)
        second = CanalSegment(((point[0], 0., 0., 0.), (point[1], 0., 0., 0.),
                               (0., 1., 0., 0.)), 1e-10, 1e-10)
        self.assertTrue(any(hit.source == "edge" and abs(hit.t-t) < 1e-8 and hit.u == 0.
                            for hit in neurite_neurite_candidates(first, second)))
        alpha = .32123456789
        shared = line((0., 0., 0.), (2., 0., 0.), 1e-10)
        returning = CanalSegment(((0., 2., 0., 0.), (0., -alpha, 1., 0.),
                                  (0., 0., 0., 0.)), 1e-10, 1e-10)
        self.assertTrue(any(abs(hit.t-alpha) < 1e-8 and abs(hit.u-alpha) < 1e-8
                            for hit in adjacent_neurite_candidates(shared, returning, first_end_second_start=False)))
        loop = CanalSegment(((0., -1., 1., 0.), (-.08, .66, -1.5, 1.),
                             (0., 0., 0., 0.)), 1e-8, 1e-8)
        self.assertTrue(any(abs(hit.t-.2) < 1e-6 and abs(hit.u-.8) < 1e-6
                            for hit in global_self_intersection_candidates(loop)))

    def test_ambiguous_candidate_rectangle_is_not_reported_as_clear(self):
        first = CanalSegment(((0., 0., 10., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)), .1, .1)
        second = CanalSegment(((5., 0., 10., 0.), (.05, 0., 0., 0.), (0., 0., 0., 0.)), .1, .1)
        broad_candidate = (0., 0., "interior", ((0., 1.), (0., 1.)))
        with patch("anamorph.analytic_intersections._distance_candidates", return_value=([broad_candidate], [], None)), \
                patch("anamorph.analytic_intersections._parallel_nonlinear_line_candidates", return_value=None):
            with self.assertRaisesRegex(IntersectionSolveError, "distance remains ambiguous"):
                neurite_neurite_candidates(first, second)

    def test_regular_line_and_local_curvature(self):
        straight = line((0., 0., 0.), (1., 0., 0.))
        self.assertEqual(regularity_candidates(straight), [])
        self.assertEqual(local_self_intersection_candidates(straight), [])
        # gamma(t)=(t,t²,0) has curvature 2 at its starting point.
        tight = CanalSegment(((0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 0.)), 1., 1.)
        self.assertTrue(local_self_intersection_candidates(tight))

    def test_irregular_centerline(self):
        # gamma(t)=((t-.5)^2,0,0), so gamma' vanishes at t=.5.
        curve = CanalSegment(((.25, -1., 1., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)), .1, .1)
        roots = regularity_candidates(curve)
        self.assertTrue(any(r.left <= .5 <= r.right for r in roots), roots)

    def test_analytic_roots_remain_visible_when_geometry_is_scaled(self):
        for scale in (1., 1e-3, 1e-6, 1e-9):
            # gamma' vanishes at .2 at every geometric scale.
            curve = CanalSegment(((.04*scale, -.4*scale, scale, 0.),
                                  (0., 0., 0., 0.), (0., 0., 0., 0.)), scale*.1, scale*.1)
            roots = regularity_candidates(curve)
            self.assertTrue(any(r.left <= .2 <= r.right for r in roots), (scale, roots))
            first = line((-scale, 0., 0.), (scale, 0., 0.), scale*.1)
            second = line((0., -scale, 0.), (0., scale, 0.), scale*.1)
            candidates = neurite_neurite_candidates(first, second)
            self.assertTrue(any(abs(c.t-.5) < 1e-4 and abs(c.u-.5) < 1e-4 for c in candidates),
                            (scale, candidates))

    def test_constructed_zero_speed_points_across_coordinate_scales(self):
        import random
        rng = random.Random(771)
        for _ in range(50):
            root, scale = rng.uniform(.05, .95), 10**rng.uniform(-3, 1)
            components = []
            for _ in range(3):
                a, b = rng.uniform(.2, 2.), rng.uniform(-.5, .5)
                # Its derivative is scale*(t-root)*(a+b*t).
                components.append((rng.uniform(-1, 1)*scale, -root*a*scale,
                                   .5*(a-root*b)*scale, b*scale/3))
            segment = CanalSegment(tuple(components), .1, .1)
            hits = regularity_candidates(segment)
            self.assertTrue(any(hit.left <= root <= hit.right for hit in hits), (root, scale, hits))

    def test_regularity_polynomial_is_scaled_before_products(self):
        for scale in (1e-200, 1e-160, 1., 1e160, 1e200):
            with self.subTest(scale=scale):
                curve = CanalSegment(((.25*scale, -scale, scale, 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)),
                                     .1*scale, .1*scale)
                hits = regularity_candidates(curve)
                self.assertTrue(any(hit.left <= .5 <= hit.right for hit in hits))
                self.assertEqual(regularity_candidates(line((0., 0., 0.), (scale, 0., 0.), .1*scale)), [])

    def test_extreme_regularity_candidates_survive_network_reporting(self):
        for scale in (1e-200, 1e200):
            with self.subTest(scale=scale):
                curve = CanalSegment(((0., -scale, 1.5*scale, 0.), (0., 0., 0., 0.),
                                      (0., 0., 0., 0.)), .1*scale, .1*scale)
                tree = Morphology([Node(1, 1, -scale, 0., 0., scale, -1),
                                   Node(2, 3, 0., 0., 0., .1*scale, 1),
                                   Node(3, 3, .5*scale, 0., 0., .1*scale, 2)])
                intervals = regularity_candidates(curve)
                self.assertTrue(intervals)
                for partial in (False, True):
                    with self.subTest(partial=partial), \
                            patch("anamorph.analytic_network.CanalSegment.from_spline", return_value=curve):
                        if partial:
                            with patch("anamorph.analytic_network.regularity_candidates",
                                       side_effect=IntersectionSolveError(("interrupted",), intervals)):
                                report = analyze_analytic_partial(tree, max_rectangles=10)
                        else:
                            report = analyze_analytic_partial(tree, max_rectangles=10)
                    finding = next(item for item in report.findings if item.kind == "regularity")
                    candidate = finding.candidates[0]
                    parameter = Fraction(candidate.parameters[0])
                    velocity = Fraction(curve.power[0][1])+2*Fraction(curve.power[0][2])*parameter
                    self.assertNotEqual(velocity, 0)
                    self.assertIsNone(candidate.value)
                    self.assertEqual(Fraction(*candidate.exact_value), velocity*velocity)
                    flags = {status.edge: status for status in report.segment_flags}
                    self.assertTrue(flags[(2, 3)].reg)
                    self.assertEqual(any("regularity" in item for item in report.unresolved), partial)
                    data = json.loads(json.dumps(report.as_dict()))
                    item = next(item for item in data["findings"] if item["kind"] == "regularity")
                    self.assertEqual(Fraction(*item["candidates"][0]["exact_value"]), velocity*velocity)

    def test_local_curvature_does_not_silently_underflow_or_overflow(self):
        for scale in (1e-50, 1., 1e50):
            with self.subTest(scale=scale):
                tight = CanalSegment(((0., scale, 0., 0.), (0., 0., scale, 0.), (0., 0., 0., 0.)), scale, scale)
                hits = local_self_intersection_candidates(tight, with_bounds=True)
                self.assertTrue(any(hit.parameter == 0. and hit.value > 0. for hit in hits))
                self.assertEqual(local_self_intersection_candidates(line((0., 0., 0.), (scale, 0., 0.), scale)), [])
        for scale in (1e-160, 1e-60, 1e80, 1e160):
            with self.subTest(scale=scale):
                tight = CanalSegment(((0., scale, 0., 0.), (0., 0., scale, 0.), (0., 0., 0., 0.)), scale, scale)
                hits = local_self_intersection_candidates(tight, with_bounds=True)
                endpoint = next(hit for hit in hits if hit.parameter == 0.)
                self.assertIsNone(endpoint.value)
                self.assertEqual(Fraction(*endpoint.exact_value), 3*Fraction(scale)**6)
                plain = dict(local_self_intersection_candidates(tight))
                self.assertEqual(plain[0.], 3*Fraction(scale)**6)
                self.assertIsInstance(plain[0.], Fraction)
                self.assertEqual(local_self_intersection_candidates(line((0., 0., 0.), (scale, 0., 0.), scale)), [])

    def test_extreme_local_curvature_values_survive_network_reporting(self):
        for scale in (1e-160, 1e160):
            with self.subTest(scale=scale):
                tight = CanalSegment(((0., scale, 0., 0.), (0., 0., scale, 0.),
                                      (0., 0., 0., 0.)), scale, scale)
                tree = Morphology([Node(1, 1, -scale, 0., 0., scale, -1),
                                   Node(2, 3, 0., 0., 0., scale, 1),
                                   Node(3, 3, scale, scale, 0., scale, 2)])
                with patch("anamorph.analytic_network.CanalSegment.from_spline", return_value=tight):
                    report = analyze_analytic_partial(tree)
                finding = next(item for item in report.findings if item.kind == "local_self_intersection")
                endpoint = next(item for item in finding.candidates if item.parameters == (0.,))
                self.assertIsNone(endpoint.value)
                self.assertEqual(Fraction(*endpoint.exact_value), 3*Fraction(scale)**6)
                self.assertFalse(any("local_self_intersection" in item for item in report.unresolved))
                decoded = json.loads(json.dumps(report.as_dict()))
                item = next(item for item in decoded["findings"] if item["kind"] == "local_self_intersection")
                self.assertEqual(Fraction(*item["candidates"][0]["exact_value"]), 3*Fraction(scale)**6)

        # If normalization itself fails, the independent endpoint fallback
        # must retain the exact positive value alongside the unresolved job.
        tiny, huge = 1e-200, 1e200
        ill_scaled = CanalSegment(((0., tiny, 0., huge), (0., 0., tiny, 0.), (0., 0., 0., 0.)), huge, huge)
        tree = Morphology([Node(1, 1, -huge, 0., 0., huge, -1), Node(2, 3, 0., 0., 0., huge, 1),
                           Node(3, 3, huge, tiny, 0., huge, 2)])
        with patch("anamorph.analytic_network.CanalSegment.from_spline", return_value=ill_scaled):
            report = analyze_analytic_partial(tree)
        finding = next(item for item in report.findings if item.kind == "local_self_intersection")
        endpoint = next(item for item in finding.candidates if item.parameters == (0.,))
        expected = 4*Fraction(huge)**2*Fraction(tiny)**4-Fraction(tiny)**6
        self.assertGreater(expected, 0)
        self.assertIsNone(endpoint.value)
        self.assertEqual(Fraction(*endpoint.exact_value), expected)
        self.assertTrue(any("local_self_intersection" in item for item in report.unresolved))

    def test_extreme_internal_scale_ratios_are_unresolved(self):
        curve = CanalSegment(((0., 1e200, 0., 0.), (0., 1e-200, 0., 0.), (0., 0., 0., 0.)), 1., 1.)
        with self.assertRaisesRegex(ClippingConvergenceError, "coefficient scale ratio"):
            regularity_candidates(curve)

    def test_exact_regularity_roots_survive_failed_normalization(self):
        tiny = 2.**-700
        examples = (
            ((-1., 0., 1.), ("sqrt_third",)),
            ((.5625, -1.5, 1.), (Fraction(1, 4), Fraction(3, 4))),
            ((.75, -1.5, 1.), (Fraction(1, 2),)),
            ((-1., 1.5, 0.), (Fraction(1, 3),)),
            ((1., 0., 1.), ()),
        )
        for coefficients, expected in examples:
            component = (0., *coefficients)
            small = (0., *(tiny*value for value in coefficients))
            curve = CanalSegment((component, small, (0., 0., 0., 0.)), .1, .1)
            with self.subTest(coefficients=coefficients):
                with self.assertRaisesRegex(IntersectionSolveError, "coefficient scale ratio") as raised:
                    regularity_candidates(curve, tolerance=1e-9)
                roots = raised.exception.candidates
                self.assertEqual(len(roots), len(expected))
                self.assertTrue(all(root.certified for root in roots))
                for root, value in zip(roots, expected):
                    if value == "sqrt_third":
                        self.assertLessEqual(3*Fraction(root.left)**2, 1)
                        self.assertGreaterEqual(3*Fraction(root.right)**2, 1)
                    else:
                        self.assertLessEqual(Fraction(root.left), value)
                        self.assertGreaterEqual(Fraction(root.right), value)

        # A completed but incorrect numerical search cannot hide an exact
        # common root of the represented derivative coefficients.
        curve = CanalSegment(((0., -1., 0., 1.), (0., 0., 0., 0.),
                              (0., 0., 0., 0.)), .1, .1)
        with patch("anamorph.analytic_intersections._curve_roots", return_value=[]):
            root, = regularity_candidates(curve, tolerance=1e-9)
        self.assertTrue(root.certified)
        self.assertLessEqual(3*Fraction(root.left)**2, 1)
        self.assertGreaterEqual(3*Fraction(root.right)**2, 1)

        tree = Morphology([Node(1, 1, -2., 0., 0., 1., -1), Node(2, 3, 0., 0., 0., .1, 1),
                           Node(3, 3, 1., 0., 0., .1, 2)])
        scaled_curve = CanalSegment(((0., -1., 0., 1.), (0., -tiny, 0., tiny),
                                     (0., 0., 0., 0.)), .1, .1)
        with patch("anamorph.analytic_network.CanalSegment.from_spline", return_value=scaled_curve):
            report = analyze_analytic_partial(tree, max_rectangles=1)
        finding = next(item for item in report.findings if item.kind == "regularity")
        self.assertTrue(any(hit.source == "exact_root_interval" for hit in finding.candidates))
        self.assertTrue(any("regularity normalization" in message for message in report.unresolved))
        self.assertTrue(next(status for status in report.segment_flags if status.edge == (2, 3)).reg)

    def test_local_curvature_matches_parabola_formula_across_scales(self):
        rng = random.Random(85031)
        for scale in (1e-50, 1., 1e50):
            for index in range(30):
                curvature, radius = rng.uniform(.2, 2.), rng.uniform(.05, 2.)
                if abs(2*radius*curvature-1) < .01:
                    continue
                # For gamma(t)=(t,k*t^2), curvature decreases from 2*k.
                curve = CanalSegment(((0., scale, 0., 0.), (0., 0., scale*curvature, 0.),
                                      (0., 0., 0., 0.)), radius*scale, radius*scale)
                with self.subTest(scale=scale, index=index):
                    self.assertEqual(bool(local_self_intersection_candidates(curve)), 2*radius*curvature > 1)

    def test_soma_candidate_interior_and_miss(self):
        curve = line((-1., 0., 0.), (1., 0., 0.))
        hits = soma_neurite_candidates(curve, (0., .2, 0.), .15)
        self.assertTrue(any(abs(t-.5) < 1e-5 for t, _ in hits), hits)
        bounded = soma_neurite_candidates(curve, (0., .2, 0.), .15, with_bounds=True)
        self.assertTrue(any(hit.source == "root_interval" and
                            hit.parameter_bounds[0] <= .5 <= hit.parameter_bounds[1] for hit in bounded))
        self.assertEqual(soma_neurite_candidates(curve, (0., .4, 0.), .15), [])

    def test_soma_small_pipe_requires_an_actual_distance_witness(self):
        for scale in (1., 1e-50, 1e-160):
            with self.subTest(scale=scale):
                radius = 1e-9*scale
                curve = CanalSegment(((-scale, 2*scale, 0., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)),
                                     radius, radius)
                # The old epsilon accepted this offset although it is 500
                # times the combined soma and pipe radii.
                self.assertEqual(soma_neurite_candidates(curve, (0., 1e-6*scale, 0.), radius), [])
                hits = soma_neurite_candidates(curve, (0., 0., 0.), radius, with_bounds=True)
                self.assertTrue(any(hit.parameter == .5 and hit.value == 0. for hit in hits))

    def test_soma_stationary_point_is_refined_to_a_verified_witness(self):
        curve = CanalSegment(((0., 0., 1., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)), 1e-10, 1e-10)
        hits = soma_neurite_candidates(curve, (.123456789, 0., 0.), 1e-10, with_bounds=True)
        self.assertTrue(any(abs(hit.parameter**2-.123456789) < 2e-10 and hit.value <= (2e-10)**2
                            for hit in hits))
        scale = 2.**-530
        tiny = CanalSegment(((0., 0., scale, 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)), .01*scale, .01*scale)
        hits = soma_neurite_candidates(tiny, (.25*scale, 0., 0.), .01*scale, with_bounds=True)
        self.assertTrue(any(hit.parameter == .5 and hit.value == 0. for hit in hits))

    def test_soma_line_decisions_match_exact_projection(self):
        from fractions import Fraction
        rng = random.Random(72351)
        for scale in (1., 1e-50):
            radius = scale*1e-9
            curve = CanalSegment(((0., scale, 0., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)), radius, radius)
            for index in range(40):
                x, y = scale*rng.uniform(.05, .95), radius*rng.uniform(0., 4.)
                expected = Fraction(y)**2 <= (2*Fraction(radius))**2
                with self.subTest(scale=scale, index=index):
                    self.assertEqual(bool(soma_neurite_candidates(curve, (x, y, 0.), radius)), expected)

    def test_soma_unresolved_root_search_retains_verified_endpoint(self):
        curve = line((0., 0., 0.), (1., 0., 0.), .1)
        with patch("anamorph.analytic_intersections._curve_roots", side_effect=ClippingConvergenceError("forced")):
            with self.assertRaises(IntersectionSolveError) as raised:
                soma_neurite_candidates(curve, (0., .1, 0.), .1, with_bounds=True)
        self.assertTrue(any(hit.parameter == 0. and hit.source == "endpoint" for hit in raised.exception.candidates))

    def test_soma_refinement_does_not_reintroduce_exempt_root_junction(self):
        curve = line((0., 0., 0.), (1., 0., 0.), .001)
        with patch("anamorph.analytic_intersections._curve_roots", return_value=[RootInterval(0., .2)]):
            with self.assertRaises(IntersectionSolveError) as raised:
                soma_neurite_candidates(curve, (.005, 0., 0.), .001, is_root_segment=True,
                                        tolerance=.01, with_bounds=True)
        self.assertFalse(raised.exception.candidates)

    def test_nonneighbor_curves_cross_and_are_separated(self):
        first = line((-1., 0., 0.), (1., 0., 0.))
        near = line((0., -1., .15), (0., 1., .15))
        far = line((0., -1., .3), (0., 1., .3))
        hits = neurite_neurite_candidates(first, near)
        self.assertTrue(any(abs(c.t-.5) < 1e-4 and abs(c.u-.5) < 1e-4 for c in hits), hits)
        self.assertEqual(neurite_neurite_candidates(first, far), [])

    def test_constructed_cubic_crossings_survive_changes_of_scale(self):
        import random
        rng = random.Random(112)
        t, u = .37, .61
        def polynomial_value(coefficients, x):
            return sum(c*x**i for i, c in enumerate(coefficients))
        for _ in range(12):
            first = [(0., 2., rng.uniform(-.3, .3), rng.uniform(-.2, .2)),
                     (0., rng.uniform(-.2, .2), rng.uniform(-.2, .2), rng.uniform(-.1, .1)),
                     (0., rng.uniform(-.1, .1), rng.uniform(-.1, .1), rng.uniform(-.1, .1))]
            second = [(0., rng.uniform(-.2, .2), rng.uniform(-.2, .2), rng.uniform(-.1, .1)),
                      (0., 2., rng.uniform(-.3, .3), rng.uniform(-.2, .2)),
                      (0., rng.uniform(-.1, .1), rng.uniform(-.1, .1), rng.uniform(-.1, .1))]
            # Adjust the second curve so that the two curves cross exactly at
            # (t,u); the remaining coefficients vary the geometry.
            second = [(polynomial_value(a, t)-polynomial_value(b, u), *b[1:])
                      for a, b in zip(first, second)]
            for scale in (1., 1e-6):
                curves = [CanalSegment(tuple(tuple(value*scale for value in component) for component in data),
                                       .01*scale, .01*scale) for data in (first, second)]
                hits = neurite_neurite_candidates(*curves)
                self.assertTrue(any(abs(hit.t-t) < .001 and abs(hit.u-u) < .001 for hit in hits), hits)

    def test_nonneighbor_closest_point_on_boundary(self):
        first = line((0., 0., 0.), (1., 0., 0.))
        second = line((1.15, -1., 0.), (1.15, 1., 0.))
        candidates = neurite_neurite_candidates(first, second)
        self.assertTrue(any(c.t == 1 and abs(c.u-.5) < 1e-4 and c.source == "edge"
                            for c in candidates), candidates)

    def test_parallel_affine_pairs_resolve_overlap_and_separation(self):
        first = line((0., 0., 0.), (1., 0., 0.))
        parallel = line((0., 1., 0.), (1., 1., 0.))
        self.assertEqual(neurite_neurite_candidates(first, parallel, max_rectangles=50), [])
        close = line((.5, .05, 0.), (1.5, .05, 0.))
        candidates = neurite_neurite_candidates(first, close, max_rectangles=50)
        self.assertEqual(candidates[0].source, "linear_minimum")
        self.assertAlmostEqual(candidates[0].distance_squared, .05**2)

    def test_affine_exact_witness_parameters_survive_rounding_and_reporting(self):
        from fractions import Fraction
        from anamorph.analytic_intersections import _exact_distance_at
        first = line((0., 0., 0.), (3., 0., 0.), 1e-30)
        second = line((1., -1., 0.), (1., 2., 0.), 1e-30)
        hits = neurite_neurite_candidates(first, second, max_rectangles=1)
        self.assertEqual(len(hits), 1)
        hit = hits[0]
        self.assertEqual(hit.exact_parameters, ((1, 3), (1, 3)))
        self.assertEqual(hit.distance_squared, 0.)
        self.assertGreater(_exact_distance_at(first, second, hit.t, hit.u),
                           (Fraction(first.max_radius)+Fraction(second.max_radius))**2)
        self.assertEqual(_exact_distance_at(first, second,
                                           *(Fraction(n, d) for n, d in hit.exact_parameters)), 0)
        tree = Morphology([Node(1, 1, -5, 0, 0, .1, -1), Node(2, 3, 0, 0, 0, 1e-30, 1),
                           Node(3, 3, 3, 0, 0, 1e-30, 2), Node(4, 1, 1, -5, 0, .1, -1),
                           Node(5, 3, 1, -1, 0, 1e-30, 4), Node(6, 3, 1, 2, 0, 1e-30, 5)])
        report = analyze_analytic_partial(tree, max_rectangles=1)
        finding = next(f for f in report.findings if f.kind == "neurite_neurite_candidate")
        witness = finding.candidates[0]
        self.assertEqual(witness.exact_parameters, ((1, 3), (1, 3)))
        serialized = report.as_dict()
        item = next(f for f in serialized["findings"] if f["kind"] == "neurite_neurite_candidate")
        self.assertEqual(item["candidates"][0]["exact_parameters"], ((1, 3), (1, 3)))
        decoded = json.loads(json.dumps(serialized))
        item = next(f for f in decoded["findings"] if f["kind"] == "neurite_neurite_candidate")
        self.assertEqual(item["candidates"][0]["exact_parameters"], [[1, 3], [1, 3]])

    def test_exact_affine_minimum_backstops_a_silently_missing_clipped_root(self):
        first = line((0., 0., 0.), (3., 0., 0.), 1e-30)
        second = line((1., -1., 0.), (1., 2., 0.), 1e-30)
        # Model a numerical interior search that returns no root and no error;
        # all edge and corner distances are larger than the radii.
        with patch("anamorph.analytic_intersections.bilinear_clip_roots", return_value=[]):
            hits = neurite_neurite_candidates(first, second)
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0].source, "linear_minimum")
            self.assertEqual(hits[0].exact_parameters, ((1, 3), (1, 3)))
            self.assertEqual(hits[0].distance_squared, 0.)
            # The network report must preserve the finding and both segment
            # flags even if the clipping routine misses the interior root.
            tree = Morphology([Node(1, 1, -5, 0, 0, .1, -1), Node(2, 3, 0, 0, 0, 1e-30, 1),
                               Node(3, 3, 3, 0, 0, 1e-30, 2), Node(4, 1, 1, -5, 0, .1, -1),
                               Node(5, 3, 1, -1, 0, 1e-30, 4), Node(6, 3, 1, 2, 0, 1e-30, 5)])
            report = analyze_analytic_partial(tree)
        finding = next(f for f in report.findings if f.kind == "neurite_neurite_candidate")
        self.assertEqual(finding.candidates[0].exact_parameters, ((1, 3), (1, 3)))
        flags = {status.edge: status for status in report.segment_flags}
        self.assertTrue(flags[(2, 3)].rc_nsns and flags[(5, 6)].rc_nsns)
        with patch("anamorph.analytic_intersections.bilinear_clip_roots", return_value=[]), \
                patch("anamorph.analytic_intersections._curve_roots",
                      side_effect=ClippingConvergenceError("injected edge failure")):
            with self.assertRaises(IntersectionSolveError) as raised:
                neurite_neurite_candidates(first, second)
        self.assertTrue(any(hit.exact_parameters == ((1, 3), (1, 3)) for hit in raised.exception.candidates))
        self.assertTrue(any("edge" in error for error in raised.exception.failures))


    def test_perpendicular_translated_cubics_resolve_nonisolated_pairs(self):
        from fractions import Fraction
        for scale in (1e-200, 1., 1e200):
            radius = .125*scale
            first = CanalSegment(((0., scale, 0., 0.), (0., 0., scale, .25*scale),
                                  (0., 0., 0., 0.)), radius, radius)
            for offset_factor in (0., .125, .25, .375):
                offset = offset_factor*scale
                second = CanalSegment(((0., scale, 0., 0.), (0., 0., scale, .25*scale),
                                       (offset, 0., 0., 0.)), radius, radius)
                with self.subTest(scale=scale, offset=offset_factor):
                    hits = neurite_neurite_candidates(first, second, max_rectangles=1)
                    expected = Fraction(offset)**2 <= (2*Fraction(radius))**2
                    self.assertEqual(bool(hits), expected)
                    if hits:
                        hit, = hits
                        self.assertEqual((hit.t, hit.u, hit.source), (.5, .5, "translated_minimum"))
                        if hit.exact_distance_squared is not None:
                            self.assertEqual(Fraction(*hit.exact_distance_squared), Fraction(offset)**2)
                        else:
                            self.assertEqual(hit.distance_squared, float(Fraction(offset)**2))

        first = CanalSegment(((0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 0.)), .125, .125)
        tangential = CanalSegment(((.1, 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 0.)), .125, .125)
        different = CanalSegment(((0., 1., .01, 0.), (0., 0., 1., 0.), (.125, 0., 0., 0.)), .125, .125)
        for other in (tangential, different):
            with self.subTest(other=other), \
                    patch("anamorph.analytic_intersections.bilinear_clip_roots",
                          side_effect=ClippingConvergenceError("injected unresolved system")):
                with self.assertRaises(IntersectionSolveError):
                    neurite_neurite_candidates(first, other)

        translated = CanalSegment(((0., 1., 0., 0.), (0., 0., 1., 0.), (.125, 0., 0., 0.)), .125, .125)
        tree = Morphology([Node(1, 1, -2, 0, 0, 1, -1), Node(2, 3, 0, 0, 0, .125, 1),
                           Node(3, 3, 1, 1, 0, .125, 2), Node(4, 1, -2, 0, .125, 1, -1),
                           Node(5, 3, 0, 0, .125, .125, 4), Node(6, 3, 1, 1, .125, .125, 5)])
        with patch("anamorph.analytic_network.CanalSegment.from_spline", side_effect=(first, translated)):
            report = analyze_analytic_partial(tree)
        finding = next(f for f in report.findings if f.kind == "neurite_neurite_candidate")
        self.assertEqual(finding.classification, "inter_cell")
        self.assertEqual(finding.candidates[0].source, "translated_minimum")
        flags = {status.edge: status for status in report.segment_flags}
        self.assertTrue(all(flags[edge].rc_nsns for edge in finding.segments))
        self.assertFalse(any("segments (2, 3) and (5, 6)" in error for error in report.unresolved))

    def test_reversed_translated_curves_resolve_exactly(self):
        from fractions import Fraction
        for scale in (1e-200, 1., 1e200):
            radius = .125*scale
            first = CanalSegment(((0., scale, 0., 0.), (0., 0., scale, 0.),
                                  (0., 0., 0., 0.)), radius, radius)
            for offset_factor in (0., .125, .25, .375):
                offset = offset_factor*scale
                second = CanalSegment(((scale, -scale, 0., 0.), (scale, -2*scale, scale, 0.),
                                       (offset, 0., 0., 0.)), radius, radius)
                with self.subTest(scale=scale, offset=offset_factor):
                    hits = neurite_neurite_candidates(first, second, max_rectangles=1)
                    expected = Fraction(offset)**2 <= (2*Fraction(radius))**2
                    self.assertEqual(bool(hits), expected)
                    if hits:
                        hit, = hits
                        self.assertEqual((hit.t, hit.u, hit.source), (.5, .5, "translated_minimum"))
                        distance = (Fraction(*hit.exact_distance_squared) if hit.exact_distance_squared
                                    else Fraction(hit.distance_squared))
                        self.assertEqual(distance, Fraction(offset)**2)

        # The cubic reversal below is exact at binary power-of-two scales.
        for scale in (2.**-600, 1., 2.**600):
            first = CanalSegment(((0., scale, 0., 0.), (0., 0., scale, .25*scale),
                                  (0., 0., 0., 0.)), .125*scale, .125*scale)
            second = CanalSegment(((scale, -scale, 0., 0.),
                                   (1.25*scale, -2.75*scale, 1.75*scale, -.25*scale),
                                   (.125*scale, 0., 0., 0.)), .125*scale, .125*scale)
            with self.subTest(cubic_scale=scale):
                hit, = neurite_neurite_candidates(first, second, max_rectangles=1)
                self.assertEqual((hit.t, hit.u, hit.source), (.5, .5, "translated_minimum"))

        # Rounded floating arithmetic makes the reversed linear coefficient
        # look equal, but the exact represented coefficients differ by one.
        first = CanalSegment(((0., -1e16, .5, 0.), (0., 0., 1., 0.), (0., 0., 0., 0.)), .125, .125)
        rounded = CanalSegment(((0., 1e16, .5, 0.), (1., -2., 1., 0.), (.125, 0., 0., 0.)), .125, .125)
        self.assertEqual(first.power[0][1], -rounded.power[0][1]-2*rounded.power[0][2])
        with patch("anamorph.analytic_intersections._distance_candidates",
                   side_effect=ClippingConvergenceError("general solver reached")):
            with self.assertRaisesRegex(ClippingConvergenceError, "general solver reached"):
                neurite_neurite_candidates(first, rounded)

        first = CanalSegment(((0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 0.)), .125, .125)
        reversed_curve = CanalSegment(((1., -1., 0., 0.), (1., -2., 1., 0.),
                                       (.125, 0., 0., 0.)), .125, .125)
        tree = Morphology([Node(1, 1, -2, 0, 0, 1, -1), Node(2, 3, 0, 0, 0, .125, 1),
                           Node(3, 3, 1, 1, 0, .125, 2), Node(4, 1, 3, 2, .125, 1, -1),
                           Node(5, 3, 1, 1, .125, .125, 4), Node(6, 3, 0, 0, .125, .125, 5)])
        with patch("anamorph.analytic_network.CanalSegment.from_spline", side_effect=(first, reversed_curve)):
            report = analyze_analytic_partial(tree)
        finding = next(f for f in report.findings if f.kind == "neurite_neurite_candidate")
        self.assertEqual(finding.classification, "inter_cell")
        self.assertEqual(finding.candidates[0].source, "translated_minimum")
        flags = {status.edge: status for status in report.segment_flags}
        self.assertTrue(all(flags[edge].rc_nsns and not flags[edge].unresolved for edge in finding.segments))

    def test_shifted_translated_curves_resolve_on_shared_parameter_interval(self):
        from fractions import Fraction
        from anamorph.analytic_intersections import _exact_distance_at
        first = CanalSegment(((0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 0.)), .125, .125)
        for second, expected_u in (
            (CanalSegment(((.5, 1., 0., 0.), (.25, 1., 1., 0.), (.125, 0., 0., 0.)), .125, .125), .25),
            (CanalSegment(((1.5, -1., 0., 0.), (2.25, -3., 1., 0.), (.125, 0., 0., 0.)), .125, .125), .75),
        ):
            with self.subTest(expected_u=expected_u):
                hit, = neurite_neurite_candidates(first, second, max_rectangles=1)
                self.assertEqual((hit.t, hit.u, hit.source), (.75, expected_u, "translated_minimum"))
                self.assertEqual(_exact_distance_at(first, second, hit.t, hit.u), Fraction(1, 64))
        # Both curves are polynomially related by t=u+1/3, although their
        # coefficients are integers. The exact witness cannot be stored in
        # binary floating-point parameters.
        cubic = CanalSegment(((1., -6., 9., 0.), (-1., 9., -27., 27.), (0., 0., 0., 0.)), .125, .125)
        shifted = CanalSegment(((0., 0., 9., 0.), (0., 0., 0., 27.), (.125, 0., 0., 0.)), .125, .125)
        hit, = neurite_neurite_candidates(cubic, shifted, max_rectangles=1)
        self.assertEqual(hit.exact_parameters, ((2, 3), (1, 3)))
        self.assertIsNone(hit.parameter_bounds)
        self.assertEqual(_exact_distance_at(cubic, shifted, Fraction(2, 3), Fraction(1, 3)), Fraction(1, 64))

        # Without a shared parameter interval, or with a tangential offset,
        # this exact shortcut cannot make a distance decision.
        no_overlap = CanalSegment(((2., 1., 0., 0.), (4., 4., 1., 0.), (.125, 0., 0., 0.)), .125, .125)
        tangential = CanalSegment(((.625, 1., 0., 0.), (.25, 1., 1., 0.), (.125, 0., 0., 0.)), .125, .125)
        for other in (no_overlap, tangential):
            with self.subTest(other=other), patch("anamorph.analytic_intersections._distance_candidates",
                                                side_effect=ClippingConvergenceError("general solver reached")):
                with self.assertRaisesRegex(ClippingConvergenceError, "general solver reached"):
                    neurite_neurite_candidates(first, other)

        tree = Morphology([Node(1, 1, -2, 0, 0, 1, -1), Node(2, 3, 1, -1, 0, .125, 1),
                           Node(3, 3, 4, 8, 0, .125, 2), Node(4, 1, -2, 0, .125, 1, -1),
                           Node(5, 3, 0, 0, .125, .125, 4), Node(6, 3, 9, 27, .125, .125, 5)])
        with patch("anamorph.analytic_network.CanalSegment.from_spline", side_effect=(cubic, shifted)):
            report = analyze_analytic_partial(tree)
        finding = next(f for f in report.findings if f.kind == "neurite_neurite_candidate")
        self.assertEqual(finding.candidates[0].exact_parameters, ((2, 3), (1, 3)))
        self.assertEqual(finding.classification, "inter_cell")
        flags = {status.edge: status for status in report.segment_flags}
        self.assertTrue(all(flags[edge].rc_nsns for edge in finding.segments))

    def test_parallel_nonuniform_line_parameterization_has_exact_minimum(self):
        from anamorph.analytic_intersections import _exact_distance_at
        for scale in (2.**-600, 1., 2.**600):
            first = CanalSegment(((0., scale, 0., 0.), (0., 2*scale, 0., 0.),
                                  (0., 0., 0., 0.)), .125*scale, .125*scale)
            # Each second curve has direction (1,2,0) and a perpendicular
            # displacement of .25*scale; speed is nonlinear in u.
            for start, coefficient, expected_u in ((.125, .5, 0.), (-2., 4., .75),
                                                    (2., -4., .5)):
                second = CanalSegment(((start*scale, 0., coefficient*scale, 0.),
                                       (2*start*scale, 0., 2*coefficient*scale, 0.),
                                       (.25*scale, 0., 0., 0.)), .125*scale, .125*scale)
                for a, b, swapped in ((first, second, False), (second, first, True)):
                    with self.subTest(scale=scale, start=start, swapped=swapped):
                        hit, = neurite_neurite_candidates(a, b, max_rectangles=1)
                        self.assertEqual(hit.source, "parallel_line_minimum")
                        t, u = ((Fraction(n, d) for n, d in hit.exact_parameters)
                                if hit.exact_parameters else (Fraction(hit.t), Fraction(hit.u)))
                        self.assertEqual(t if swapped else u, Fraction(expected_u))
                        self.assertEqual(_exact_distance_at(a, b, t, u), Fraction(.25*scale)**2)

        line_segment = CanalSegment(((0., 1., 0., 0.), (0., 2., 0., 0.),
                                     (0., 0., 0., 0.)), .625, .625)
        disjoint = CanalSegment(((1.5, 0., .5, 0.), (3., 0., 1., 0.),
                                 (.25, 0., 0., 0.)), .625, .625)
        hit, = neurite_neurite_candidates(line_segment, disjoint, max_rectangles=1)
        self.assertEqual((hit.t, hit.u, hit.distance_squared), (1., 0., 1.3125))
        self.assertEqual(neurite_neurite_candidates(
            CanalSegment(line_segment.power, .125, .125),
            CanalSegment(disjoint.power, .125, .125), max_rectangles=1), [])

        # The exact result must propagate through the network classifier.
        first = CanalSegment(((0., 1., 0., 0.), (0., 2., 0., 0.),
                              (0., 0., 0., 0.)), .125, .125)
        second = CanalSegment(((.125, .25, .25, 0.), (.25, .5, .5, 0.),
                               (.25, 0., 0., 0.)), .125, .125)
        tree = Morphology([Node(1, 1, -2, 0, 0, 1, -1), Node(2, 3, 0, 0, 0, .125, 1),
                           Node(3, 3, 1, 2, 0, .125, 2), Node(4, 1, -2, 0, .25, 1, -1),
                           Node(5, 3, .125, .25, .25, .125, 4),
                           Node(6, 3, .625, 1.25, .25, .125, 5)])
        with patch("anamorph.analytic_network.CanalSegment.from_spline", side_effect=(first, second)):
            report = analyze_analytic_partial(tree, max_rectangles=1)
        finding = next(item for item in report.findings if item.kind == "neurite_neurite_candidate")
        self.assertEqual(finding.classification, "inter_cell")
        self.assertEqual(finding.candidates[0].source, "parallel_line_minimum")
        flags = {status.edge: status for status in report.segment_flags}
        self.assertTrue(all(flags[edge].rc_nsns for edge in finding.segments))

        # A nonparallel curve cannot use the parallel-line proof.
        unsupported = (
            CanalSegment(((0., 0., .5, 0.), (0., 0., 1.01, 0.), (.25, 0., 0., 0.)), .125, .125),
        )
        for second in unsupported:
            with self.subTest(second=second), patch("anamorph.analytic_intersections._distance_candidates",
                                                    side_effect=ClippingConvergenceError("general solver reached")):
                with self.assertRaisesRegex(ClippingConvergenceError, "general solver reached"):
                    neurite_neurite_candidates(line_segment, second)

    def test_reversing_parallel_curve_matches_exact_quadratic_range(self):
        from anamorph.analytic_intersections import _exact_distance_at
        rng = random.Random(77420)
        for scale in (2.**-600, 1., 2.**600):
            first = CanalSegment(((0., scale, 0., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)),
                                 .125*scale, .125*scale)
            cases = [(9, -6, 2, .25), (6, -6, 2, .125), (4, -4, 3, .125)]
            while len(cases) < 83:
                a, b, c = (rng.randint(-8, 8) for _ in range(3))
                if a and 0 < Fraction(-b, 2*a) < 1:
                    cases.append((a, b, c, rng.choice((0., .125, .25, .5))))
            for a, b, c, height in cases:
                vertex = Fraction(-b, 2*a)
                values = (Fraction(c), Fraction(a+b+c), Fraction(c)+b*vertex+a*vertex*vertex)
                low, high = min(values), max(values)
                gap = max(Fraction(0), -high, low-1)
                expected = gap*gap+Fraction(height)**2 <= Fraction(1, 4)**2
                second = CanalSegment(((c*scale, b*scale, a*scale, 0.),
                                       (height*scale, 0., 0., 0.), (0., 0., 0., 0.)),
                                      .125*scale, .125*scale)
                for left, right in ((first, second), (second, first)):
                    with self.subTest(scale=scale, a=a, b=b, c=c, height=height, swapped=left is second):
                        hits = neurite_neurite_candidates(left, right, max_rectangles=1, max_intervals=1)
                        self.assertEqual(bool(hits), expected)
                        if hits and hits[0].source != "parallel_line_tangency":
                            hit = hits[0]
                            t, u = ((Fraction(n, d) for n, d in hit.exact_parameters)
                                    if hit.exact_parameters else (Fraction(hit.t), Fraction(hit.u)))
                            self.assertLessEqual(_exact_distance_at(left, right, t, u), Fraction(.25*scale)**2)

    def test_reversing_parallel_tangency_retains_nondyadic_parameters(self):
        first = CanalSegment(((0., 1., 0., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)), .125, .125)
        second = CanalSegment(((2., -6., 9., 0.), (.25, 0., 0., 0.), (0., 0., 0., 0.)), .125, .125)
        for a, b, expected in ((first, second, ((1, 1), (1, 3))),
                               (second, first, ((1, 3), (1, 1)))):
            hit, = neurite_neurite_candidates(a, b, max_rectangles=1, max_intervals=1)
            self.assertEqual(hit.exact_parameters, expected)
            self.assertEqual(hit.distance_squared, .0625)
        tree = Morphology([Node(1, 1, -2, 0, 0, .5, -1), Node(2, 3, 0, 0, 0, .125, 1),
                           Node(3, 3, 1, 0, 0, .125, 2), Node(4, 1, -2, .25, 0, .5, -1),
                           Node(5, 3, 2, .25, 0, .125, 4), Node(6, 3, 5, .25, 0, .125, 5)])
        with patch("anamorph.analytic_network.CanalSegment.from_spline", side_effect=(first, second)):
            report = analyze_analytic_partial(tree, max_rectangles=1)
        finding = next(item for item in report.findings if item.kind == "neurite_neurite_candidate")
        self.assertEqual(finding.classification, "inter_cell")
        self.assertEqual(finding.candidates[0].exact_parameters, ((1, 1), (1, 3)))
        flags = {status.edge: status for status in report.segment_flags}
        self.assertTrue(all(flags[edge].rc_nsns for edge in finding.segments))

    def test_reversing_parallel_cubics_match_high_precision_range(self):
        from decimal import Decimal, localcontext
        rng = random.Random(650234)
        first = CanalSegment(((0., 1., 0., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)), .125, .125)
        checked = 0
        with localcontext() as context:
            context.prec = 80
            for _ in range(500):
                c, b, a, d = (rng.randint(-8, 8) for _ in range(4))
                discriminant = 4*a*a-12*d*b
                if not d or discriminant <= 0:
                    continue
                roots = [(-Decimal(2*a)+sign*Decimal(discriminant).sqrt())/Decimal(6*d)
                         for sign in (-1, 1)]
                interior = [root for root in roots if 0 < root < 1]
                if not interior:
                    continue
                values = [Decimal(c), Decimal(c+b+a+d)]
                values.extend(Decimal(c)+root*(Decimal(b)+root*(Decimal(a)+root*Decimal(d)))
                              for root in interior)
                low, high = min(values), max(values)
                gap = max(Decimal(0), -high, low-1)
                height = rng.choice((0., .125, .25, .5))
                difference = gap*gap+Decimal(height)**2-Decimal(".0625")
                if abs(difference) < Decimal("1e-40"):
                    continue
                second = CanalSegment(((float(c), float(b), float(a), float(d)),
                                       (height, 0., 0., 0.), (0., 0., 0., 0.)), .125, .125)
                for left, right in ((first, second), (second, first)):
                    with self.subTest(coefficients=(c, b, a, d), height=height, swapped=left is second):
                        hits = neurite_neurite_candidates(left, right, max_rectangles=1, max_intervals=1)
                        self.assertEqual(bool(hits), difference < 0)
                checked += 1
        self.assertGreater(checked, 150)

    def test_two_nonlinear_parallel_lines_resolve_witnesses_and_tangency(self):
        from anamorph.analytic_intersections import _exact_distance_at
        for scale in (2.**-600, 2.**-400, 1., 2.**400, 2.**600):
            first = CanalSegment(((0., 0., scale, 0.), (0., 0., 2*scale, 0.),
                                  (0., 0., 0., 0.)), .125*scale, .125*scale)
            def second(start, height):
                return CanalSegment(((start*scale, 0., 0., scale), (2*start*scale, 0., 0., 2*scale),
                                     (height*scale, 0., 0., 0.)), .125*scale, .125*scale)
            for a, b in ((first, second(.125, .125)), (second(.125, .125), first)):
                hit, = neurite_neurite_candidates(a, b, max_rectangles=1)
                self.assertEqual(hit.source, "parallel_line_witness")
                t, u = ((Fraction(n, d) for n, d in hit.exact_parameters)
                        if hit.exact_parameters else (Fraction(hit.t), Fraction(hit.u)))
                self.assertLessEqual(_exact_distance_at(a, b, t, u),
                                     (Fraction(a.max_radius)+Fraction(b.max_radius))**2)
            self.assertEqual(neurite_neurite_candidates(first, second(.125, .5), max_rectangles=1), [])
            self.assertEqual(neurite_neurite_candidates(first, second(2., .125), max_rectangles=1), [])
            for a, b, swapped in ((first, second(.125, .25), False),
                                  (second(.125, .25), first, True)):
                with self.subTest(scale=scale, swapped=swapped):
                    hit, = neurite_neurite_candidates(a, b, max_rectangles=1, bivariate_tolerance=1e-7)
                    self.assertEqual(hit.source, "parallel_line_tangency")
                    fixed, free = hit.parameter_bounds
                    self.assertEqual(fixed, (0., 0.) if swapped else (1., 1.))
                    degree, target = (2, Fraction(1, 8)) if swapped else (3, Fraction(7, 8))
                    self.assertLessEqual(Fraction(free[0])**degree, target)
                    self.assertGreaterEqual(Fraction(free[1])**degree, target)
                    self.assertLessEqual(free[1]-free[0], 1e-7)
                    expected = Fraction(.25*scale)**2
                    actual = (Fraction(*hit.exact_distance_squared) if hit.exact_distance_squared else
                              Fraction(hit.distance_squared))
                    self.assertEqual(actual, expected)

    def test_nonlinear_parallel_line_decisions_match_interval_distance(self):
        from anamorph.analytic_intersections import _exact_distance_at
        rng = random.Random(219501)
        for case in range(600):
            astart, bstart = Fraction(rng.randrange(-12, 13), 8), Fraction(rng.randrange(-12, 13), 8)
            a2 = Fraction(rng.choice((-1, 1))*rng.randrange(1, 17), 8)
            b3 = Fraction(rng.choice((-1, 1))*rng.randrange(1, 17), 8)
            height = Fraction(rng.randrange(0, 6), 8)
            first = CanalSegment(((float(astart), 0., float(a2), 0.),
                                  (float(2*astart), 0., float(2*a2), 0.),
                                  (0., 0., 0., 0.)), .125, .125)
            second = CanalSegment(((float(bstart), 0., 0., float(b3)),
                                   (float(2*bstart), 0., 0., float(2*b3)),
                                   (float(height), 0., 0., 0.)), .125, .125)
            alo, ahi = sorted((astart, astart+a2))
            blo, bhi = sorted((bstart, bstart+b3))
            gap = max(Fraction(0), alo-bhi, blo-ahi)
            minimum = height*height+5*gap*gap
            with self.subTest(case=case):
                hits = neurite_neurite_candidates(first, second, max_rectangles=1)
                self.assertEqual(bool(hits), minimum <= Fraction(1, 16))
                if hits and hits[0].source != "parallel_line_tangency":
                    hit = hits[0]
                    t, u = ((Fraction(n, d) for n, d in hit.exact_parameters)
                            if hit.exact_parameters else (Fraction(hit.t), Fraction(hit.u)))
                    self.assertLessEqual(_exact_distance_at(first, second, t, u), Fraction(1, 16))

    def test_nonlinear_line_shortcut_falls_back_for_curved_or_reversing_paths(self):
        first = CanalSegment(((0., 0., 1., 0.), (0., 0., 2., 0.), (0., 0., 0., 0.)), .125, .125)
        unsupported = (
            CanalSegment(((0., 2., -3., 0.), (0., 4., -6., 0.), (.125, 0., 0., 0.)), .125, .125),
            CanalSegment(((0., 0., 0., 1.), (0., 0., 0., 2.01), (.125, 0., 0., 0.)), .125, .125),
        )
        for second in unsupported:
            with self.subTest(second=second), patch("anamorph.analytic_intersections._distance_candidates",
                                                    side_effect=ClippingConvergenceError("general solver reached")):
                with self.assertRaisesRegex(ClippingConvergenceError, "general solver reached"):
                    neurite_neurite_candidates(first, second)

    def test_nonlinear_parallel_cubics_in_three_dimensions(self):
        from anamorph.analytic_intersections import _exact_distance_at
        rng = random.Random(715203)
        direction, perpendicular = (1, -2, 3), (2, 1, 0)
        for case in range(500):
            values = []
            for _ in range(2):
                start = Fraction(rng.randrange(-12, 13), 8)
                sign = rng.choice((-1, 1))
                coefficients = [Fraction(sign*rng.randrange(0, 7), 8) for _ in range(3)]
                if not any(coefficients):
                    coefficients[-1] = Fraction(sign, 8)
                values.append((start, coefficients))
            height = Fraction(rng.randrange(-5, 6), 8)
            segments = []
            for index, (start, coefficients) in enumerate(values):
                power = tuple(tuple(float(direction[axis]*value +
                                          (perpendicular[axis]*height if index and degree == 0 else 0))
                                    for degree, value in enumerate((start, *coefficients)))
                              for axis in range(3))
                segments.append(CanalSegment(power, .25, .25))
            intervals = [sorted((start, start+sum(coefficients))) for start, coefficients in values]
            gap = max(Fraction(0), intervals[0][0]-intervals[1][1], intervals[1][0]-intervals[0][1])
            expected = 5*height*height+14*gap*gap
            for first, second in (segments, tuple(reversed(segments))):
                with self.subTest(case=case, order=first is segments[0]):
                    hits = neurite_neurite_candidates(first, second, max_rectangles=1)
                    self.assertEqual(bool(hits), expected <= Fraction(1, 4))
                    if hits and hits[0].source != "parallel_line_tangency":
                        hit = hits[0]
                        t, u = ((Fraction(n, d) for n, d in hit.exact_parameters)
                                if hit.exact_parameters else (Fraction(hit.t), Fraction(hit.u)))
                        self.assertLessEqual(_exact_distance_at(first, second, t, u), Fraction(1, 4))
                    elif hits:
                        self.assertEqual(expected, Fraction(1, 4))
                        parameterizations = values if first is segments[0] else tuple(reversed(values))
                        projected = []
                        for (start, coefficients), bounds in zip(parameterizations, hits[0].parameter_bounds):
                            endpoint_values = []
                            for bound in bounds:
                                parameter = Fraction(bound)
                                endpoint_values.append(start + sum(coefficient*parameter**(i+1)
                                                                   for i, coefficient in enumerate(coefficients)))
                            projected.append(sorted(endpoint_values))
                        self.assertLessEqual(max(projected[0][0], projected[1][0]),
                                             min(projected[0][1], projected[1][1]))

    def test_nonlinear_parallel_tangency_propagates_through_network_report(self):
        first = CanalSegment(((0., 0., 1., 0.), (0., 0., 2., 0.), (0., 0., 0., 0.)), .125, .125)
        second = CanalSegment(((.125, 0., 0., 1.), (.25, 0., 0., 2.), (.25, 0., 0., 0.)), .125, .125)
        tree = Morphology([Node(1, 1, -2, 0, 0, 1, -1), Node(2, 3, 0, 0, 0, .125, 1),
                           Node(3, 3, 1, 2, 0, .125, 2), Node(4, 1, -2, 0, .25, 1, -1),
                           Node(5, 3, .125, .25, .25, .125, 4), Node(6, 3, 1.125, 2.25, .25, .125, 5)])
        with patch("anamorph.analytic_network.CanalSegment.from_spline", side_effect=(first, second)):
            report = analyze_analytic_partial(tree, max_rectangles=1)
        finding = next(item for item in report.findings if item.kind == "neurite_neurite_candidate")
        self.assertEqual(finding.classification, "inter_cell")
        self.assertEqual(finding.candidates[0].source, "parallel_line_tangency")
        self.assertIsNotNone(finding.candidates[0].parameter_bounds)
        self.assertFalse(any("segments (2, 3) and (5, 6)" in error for error in report.unresolved))

    def test_source_initial_cubic_fit_matches_exact_bivariate_integration(self):
        from anamorph.analytic_intersections import _source_initial_linear_fit
        rng = random.Random(721963)
        for case in range(40):
            curves = [CanalSegment(tuple(tuple(rng.randrange(-12, 13)/8 for _ in range(4))
                                         for _ in range(3)), .1, .1) for _ in range(2)]
            p, q = {}, {}
            for axis_a, axis_b in zip(curves[0].power, curves[1].power):
                a = tuple(map(Fraction, axis_a))
                b = tuple(map(Fraction, axis_b))
                delta = {(0, 0): a[0]-b[0]}
                delta.update({(i, 0): a[i] for i in range(1, 4)})
                delta.update({(0, i): -b[i] for i in range(1, 4)})
                for (i, j), value in delta.items():
                    for degree in range(1, 4):
                        p[i+degree-1, j] = p.get((i+degree-1, j), Fraction(0))+value*degree*a[degree]
                        q[i, j+degree-1] = q.get((i, j+degree-1), Fraction(0))+value*degree*b[degree]
            def slope(poly, axis):
                total = Fraction(0)
                for (i, j), coefficient in poly.items():
                    exponent, other = (i, j) if axis == 0 else (j, i)
                    total += coefficient*(Fraction(1, exponent+2)-Fraction(1, 2*(exponent+1)))/(other+1)
                return 12*total
            with self.subTest(case=case):
                self.assertEqual(_source_initial_linear_fit(*curves),
                                 (slope(p, 0), slope(p, 1), slope(q, 0), slope(q, 1)))

    def test_source_stationary_bernstein_matches_curve_distance_derivatives(self):
        from anamorph.source_bivariate import source_stationary_bernstein
        rng = random.Random(880741)
        for case in range(30):
            curves = [CanalSegment(tuple(tuple(rng.uniform(-2, 2) for _ in range(4))
                                         for _ in range(3)), .1, .1) for _ in range(2)]
            p, q = source_stationary_bernstein(*curves)
            for t, u in ((0., 0.), (.13, .71), (.52, .29), (1., 1.)):
                difference = tuple(x-y for x, y in zip(curves[0].point(t), curves[1].point(u)))
                def velocity(curve, parameter):
                    return tuple(sum(i*axis[i]*parameter**(i-1) for i in range(1, 4)) for axis in curve.power)
                expected_p = sum(x*y for x, y in zip(difference, velocity(curves[0], t)))
                expected_q = sum(x*y for x, y in zip(difference, velocity(curves[1], u)))
                with self.subTest(case=case, t=t, u=u):
                    self.assertAlmostEqual(p.evaluate(t, u), expected_p, delta=1e-11*max(1., abs(expected_p)))
                    self.assertAlmostEqual(q.evaluate(t, u), expected_q, delta=1e-11*max(1., abs(expected_q)))
        huge = CanalSegment(((0., 1e200, 0., 0.), (0., 0., 0., 0.),
                             (0., 0., 0., 0.)), .1, .1)
        with self.assertRaisesRegex(ClippingConvergenceError, "source-style stationary"):
            source_stationary_bernstein(huge, huge)

    def test_reparameterized_curves_resolve_exact_nonisolated_minimum(self):
        from anamorph.analytic_intersections import _exact_distance_at
        first = CanalSegment(((0., 3., 0., 0.), (0., 0., 9., 27.), (0., 0., 0., 0.)), .125, .125)
        for reverse in (False, True):
            for offset, overlaps in ((0., True), (.125, True), (.25, True), (.375, False)):
                second = (CanalSegment(((1., -1., 0., 0.), (2., -5., 4., -1.), (offset, 0., 0., 0.)), .125, .125)
                          if reverse else
                          CanalSegment(((0., 1., 0., 0.), (0., 0., 1., 1.), (offset, 0., 0., 0.)), .125, .125))
                with self.subTest(reverse=reverse, offset=offset):
                    hits = neurite_neurite_candidates(first, second, max_rectangles=1)
                    self.assertEqual(bool(hits), overlaps)
                    if hits:
                        hit, = hits
                        self.assertEqual(hit.source, "translated_minimum")
                        self.assertEqual(hit.exact_parameters, ((1, 6), (1, 2)))
                        self.assertIsNone(hit.parameter_bounds)
                        self.assertEqual(_exact_distance_at(first, second, Fraction(1, 6), Fraction(1, 2)),
                                         Fraction(offset)**2)
                        self.assertEqual(hit.distance_squared, offset**2)

        parabola = CanalSegment(((0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 0.)), .125, .125)
        variants = (
            (CanalSegment(((.25, .5, 0., 0.), (.0625, .25, .25, 0.), (.125, 0., 0., 0.)), .125, .125), None),
            (CanalSegment(((1.5, -2., 0., 0.), (2.25, -6., 4., 0.), (.125, 0., 0., 0.)), .125, .125), None),
            (CanalSegment(((0., 1.5, 0., 0.), (0., 0., 2.25, 0.), (.125, 0., 0., 0.)), .125, .125),
             ((1, 2), (1, 3))),
        )
        for second, exact_parameters in variants:
            with self.subTest(second=second):
                hit, = neurite_neurite_candidates(parabola, second, max_rectangles=1)
                self.assertEqual(hit.source, "translated_minimum")
                self.assertEqual(hit.exact_parameters, exact_parameters)
                t, u = ((Fraction(n, d) for n, d in exact_parameters) if exact_parameters
                        else (Fraction(hit.t), Fraction(hit.u)))
                self.assertEqual(_exact_distance_at(parabola, second, t, u), Fraction(1, 64))

        for scale in (2.**-600, 1., 2.**600):
            with self.subTest(scale=scale):
                scaled_first = CanalSegment(((0., scale, 0., 0.), (0., 0., scale, 0.),
                                              (0., 0., 0., 0.)), .125*scale, .125*scale)
                scaled_second = CanalSegment(((0., .5*scale, 0., 0.), (0., 0., .25*scale, 0.),
                                               (.125*scale, 0., 0., 0.)), .125*scale, .125*scale)
                hit, = neurite_neurite_candidates(scaled_first, scaled_second, max_rectangles=1)
                self.assertEqual((hit.t, hit.u, hit.source), (.25, .5, "translated_minimum"))
                distance = (Fraction(*hit.exact_distance_squared) if hit.exact_distance_squared
                            else Fraction(hit.distance_squared))
                self.assertEqual(distance, Fraction(.125*scale)**2)

        # Irrational speed, tangential displacement, and disjoint parameter
        # intervals must continue to the numerical solver.
        unsupported = (
            CanalSegment(((0., 1., 0., 0.), (0., 0., 2., 0.), (.125, 0., 0., 0.)), .125, .125),
            CanalSegment(((.125, 1.5, 0., 0.), (0., 0., 2.25, 0.), (.125, 0., 0., 0.)), .125, .125),
            CanalSegment(((2., .5, 0., 0.), (4., 2., .25, 0.), (.125, 0., 0., 0.)), .125, .125),
        )
        for second in unsupported:
            with self.subTest(second=second), patch("anamorph.analytic_intersections._distance_candidates",
                                                    side_effect=ClippingConvergenceError("general solver reached")):
                with self.assertRaisesRegex(ClippingConvergenceError, "general solver reached"):
                    neurite_neurite_candidates(parabola, second)

        second = CanalSegment(((0., 1., 0., 0.), (0., 0., 1., 1.), (.125, 0., 0., 0.)), .125, .125)
        tree = Morphology([Node(1, 1, -2, 0, 0, 1, -1), Node(2, 3, 0, 0, 0, .125, 1),
                           Node(3, 3, 3, 36, 0, .125, 2), Node(4, 1, -2, 0, .125, 1, -1),
                           Node(5, 3, 0, 0, .125, .125, 4), Node(6, 3, 1, 2, .125, .125, 5)])
        with patch("anamorph.analytic_network.CanalSegment.from_spline", side_effect=(first, second)):
            report = analyze_analytic_partial(tree, max_rectangles=1)
        finding = next(item for item in report.findings if item.kind == "neurite_neurite_candidate")
        self.assertEqual(finding.classification, "inter_cell")
        self.assertEqual(finding.candidates[0].exact_parameters, ((1, 6), (1, 2)))
        flags = {status.edge: status for status in report.segment_flags}
        self.assertTrue(all(flags[edge].rc_nsns for edge in finding.segments))

    def test_affine_fallback_keeps_simultaneous_edge_and_interior_failures(self):
        first = line((0., 0., 0.), (3., 0., 0.), 1e-30)
        crossing = line((1., -1., 0.), (1., 2., 0.), 1e-30)
        separated = line((1., -1., 1.), (1., 2., 1.), 1e-30)
        for second, positive in ((crossing, True), (separated, False)):
            with self.subTest(positive=positive), \
                    patch("anamorph.analytic_intersections.bilinear_clip_roots",
                          side_effect=ClippingConvergenceError("injected interior failure")), \
                    patch("anamorph.analytic_intersections._curve_roots",
                          side_effect=ClippingConvergenceError("injected edge failure")):
                with self.assertRaises(IntersectionSolveError) as raised:
                    neurite_neurite_candidates(first, second)
                self.assertTrue(any(item.startswith("interior:") for item in raised.exception.failures))
                self.assertTrue(any(item.startswith("edge ") for item in raised.exception.failures))
                self.assertEqual(bool(raised.exception.candidates), positive)
                if positive:
                    self.assertEqual(raised.exception.candidates[0].exact_parameters, ((1, 3), (1, 3)))
                    self.assertEqual(raised.exception.candidates[0].distance_squared, 0.)

        tree = Morphology([Node(1, 1, -5, 0, 0, .1, -1), Node(2, 3, 0, 0, 0, 1e-30, 1),
                           Node(3, 3, 3, 0, 0, 1e-30, 2), Node(4, 1, 1, -5, 0, .1, -1),
                           Node(5, 3, 1, -1, 0, 1e-30, 4), Node(6, 3, 1, 2, 0, 1e-30, 5)])
        failures = ["interior: injected failure", "edge t=0: injected failure"]
        with patch("anamorph.analytic_intersections._distance_candidates", return_value=([], failures, None)):
            report = analyze_analytic_partial(tree)
        finding = next(f for f in report.findings if f.kind == "neurite_neurite_candidate")
        self.assertEqual(finding.candidates[0].exact_parameters, ((1, 3), (1, 3)))
        self.assertTrue(any("edge t=0" in item for item in report.unresolved))
        flags = {status.edge: status for status in report.segment_flags}
        self.assertTrue(all(flags[edge].rc_nsns and flags[edge].unresolved for edge in finding.segments))

    def test_exact_parallel_affine_decisions_match_independent_projection(self):
        from fractions import Fraction
        rng = random.Random(98142)
        for index in range(250):
            direction = tuple(rng.randint(-4, 4) for _ in range(3))
            if direction == (0, 0, 0):
                direction = (1, 0, 0)
            multiple = rng.choice((-3, -2, -1, 1, 2, 3))
            first_start = tuple(rng.randint(-5, 5) for _ in range(3))
            second_start = tuple(first_start[i]+rng.randint(-3, 3) for i in range(3))
            first_end = tuple(first_start[i]+direction[i] for i in range(3))
            second_end = tuple(second_start[i]+multiple*direction[i] for i in range(3))
            delta = tuple(second_start[i]-first_start[i] for i in range(3))
            norm = sum(x*x for x in direction)
            projection = Fraction(sum(delta[i]*direction[i] for i in range(3)), norm)
            interval_a = (Fraction(0), Fraction(1))
            interval_b = sorted((projection, projection+multiple))
            gap = max(Fraction(0), max(interval_a[0], interval_b[0])-min(interval_a[1], interval_b[1]))
            perpendicular2 = Fraction(sum(x*x for x in delta))-projection*projection*norm
            expected_distance2 = perpendicular2+gap*gap*norm
            radius = rng.choice((.125, .25, .5, 1.))
            expected = expected_distance2 <= (Fraction(radius)*2)**2
            with self.subTest(index=index):
                hits = neurite_neurite_candidates(line(first_start, first_end, radius),
                                                   line(second_start, second_end, radius), max_rectangles=1)
                self.assertEqual(bool(hits), expected)
                if hits:
                    self.assertAlmostEqual(hits[0].distance_squared, float(expected_distance2), places=12)

    def test_nearly_parallel_affine_fallback_and_point_segments(self):
        from math import inf, nextafter
        first = line((0., 0., 0.), (10., 0., 0.), 1e-12)
        nearly_parallel = line((0., 1e-10, 0.), (10., -1e-10, 0.), 1e-12)
        with patch("anamorph.analytic_intersections.bilinear_clip_roots",
                   side_effect=ClippingConvergenceError("injected interior failure")):
            hits = neurite_neurite_candidates(first, nearly_parallel, max_rectangles=10)
        self.assertEqual(hits[0].source, "linear_minimum")
        self.assertAlmostEqual(hits[0].t, .5)
        self.assertAlmostEqual(hits[0].u, .5)
        point = line((3., .1, 0.), (3., .1, 0.), .05)
        axis = line((0., 0., 0.), (10., 0., 0.), .05)
        self.assertAlmostEqual(neurite_neurite_candidates(point, axis)[0].distance_squared, .1**2)
        self.assertFalse(neurite_neurite_candidates(line((3., .11, 0.), (3., .11, 0.), .05), axis))
        baseline = line((0., 0., 0.), (1., 0., 0.), .1)
        touching = line((0., .2, 0.), (1., .2, 0.), .1)
        separated = line((0., nextafter(.2, inf), 0.), (1., nextafter(.2, inf), 0.), .1)
        self.assertTrue(neurite_neurite_candidates(baseline, touching))
        self.assertFalse(neurite_neurite_candidates(baseline, separated))
        for scale in (1e-200, 1e155):
            with self.subTest(scale=scale):
                hit = neurite_neurite_candidates(line((0., 0., 0.), (0., 0., 0.), scale),
                                                 line((scale, 0., 0.), (scale, 0., 0.), scale))[0]
                self.assertIsNone(hit.distance_squared)
                self.assertEqual(Fraction(*hit.exact_distance_squared), Fraction(scale)**2)

    def test_affine_extreme_distance_is_preserved_in_network_json(self):
        for scale in (1e-200, 1e155):
            with self.subTest(scale=scale):
                tree = Morphology([Node(1, 1, -3, 0, 0, .1, -1), Node(2, 3, 0, 0, 0, scale, 1),
                                   Node(3, 3, 1, 0, 0, scale, 2), Node(4, 1, -3, scale, 0, .1, -1),
                                   Node(5, 3, 0, scale, 0, scale, 4), Node(6, 3, 1, scale, 0, scale, 5)])
                report = analyze_analytic_partial(tree, max_rectangles=100)
                finding = next(f for f in report.findings if f.kind == "neurite_neurite_candidate")
                witness = finding.candidates[0]
                self.assertIsNone(witness.value)
                self.assertEqual(Fraction(*witness.exact_value), Fraction(scale)**2)
                self.assertFalse(any("segments (2, 3) and (5, 6)" in error for error in report.unresolved))
                flags = {status.edge: status for status in report.segment_flags}
                self.assertTrue(flags[(2, 3)].rc_nsns and flags[(5, 6)].rc_nsns)
                decoded = json.loads(json.dumps(report.as_dict(), allow_nan=False))
                item = next(f for f in decoded["findings"] if f["kind"] == "neurite_neurite_candidate")
                self.assertIsNone(item["candidates"][0]["value"])
                self.assertEqual(Fraction(*item["candidates"][0]["exact_value"]), Fraction(scale)**2)

    def test_extreme_soma_and_corner_witnesses_remain_exact(self):
        from anamorph.analytic_intersections import _refine_distance_candidate
        for scale in (1e-200, 1e155):
            with self.subTest(scale=scale):
                first = line((0., 0., 0.), (1., 0., 0.), scale)
                second = line((0., scale, 0.), (1., scale, 0.), scale)
                threshold = (2*Fraction(scale))**2
                corner = _verified_corner_candidate(first, second, 0., 0., threshold)
                self.assertIsNone(corner.value)
                self.assertEqual(Fraction(*corner.exact_value), Fraction(scale)**2)
                refined, ambiguous = _refine_distance_candidate(first, second, 0., 0., "corner",
                                                                  ((0., 0.), (0., 0.)), threshold)
                self.assertFalse(ambiguous)
                self.assertIsNone(refined.distance_squared)
                self.assertEqual(Fraction(*refined.exact_distance_squared), Fraction(scale)**2)
                hits = soma_neurite_candidates(first, (0., scale, 0.), scale, with_bounds=True)
                endpoint = next(hit for hit in hits if hit.parameter == 0.)
                self.assertIsNone(endpoint.value)
                self.assertEqual(Fraction(*endpoint.exact_value), Fraction(scale)**2)
                bare = soma_neurite_candidates(first, (0., scale, 0.), scale)
                self.assertTrue(any(t == 0. and value == Fraction(scale)**2 for t, value in bare))

    def test_extreme_soma_endpoint_fallback_survives_unresolved_job(self):
        for scale in (1e-200, 1e155):
            with self.subTest(scale=scale):
                tree = Morphology([Node(1, 1, -3, 0, 0, .1, -1), Node(2, 3, 0, 0, 0, scale, 1),
                                   Node(3, 3, 1, 0, 0, scale, 2), Node(4, 1, 0, scale, 0, scale, -1)])
                with patch("anamorph.analytic_network.soma_neurite_candidates",
                           side_effect=ClippingConvergenceError("injected limit")):
                    report = analyze_analytic_partial(tree)
                finding = next(f for f in report.findings if f.kind == "soma_neurite_candidate" and f.somas == (4,))
                witness = next(c for c in finding.candidates if c.parameters == (0.,))
                self.assertIsNone(witness.value)
                self.assertEqual(Fraction(*witness.exact_value), Fraction(scale)**2)
                self.assertTrue(any("soma 4" in item and "injected limit" in item for item in report.unresolved))

    def test_affine_fallback_matches_independent_three_dimensional_distance(self):
        rng = random.Random(7712)
        def dot(a, b):
            return sum(x*y for x, y in zip(a, b))
        def subtract(a, b):
            return tuple(x-y for x, y in zip(a, b))
        def clamp(x):
            return max(0., min(1., x))
        with patch("anamorph.analytic_intersections.bilinear_clip_roots",
                   side_effect=ClippingConvergenceError("injected interior failure")):
            for index in range(100):
                a, b, c, d = [tuple(rng.uniform(-2., 2.) for _ in range(3)) for _ in range(4)]
                u, v, w = subtract(b, a), subtract(d, c), subtract(a, c)
                aa, bb, cc, dd, ee = dot(u, u), dot(u, v), dot(v, v), dot(u, w), dot(v, w)
                candidates = [(t, clamp((ee+bb*t)/cc)) for t in (0., 1.)]
                candidates += [(clamp((bb*s-dd)/aa), s) for s in (0., 1.)]
                determinant = aa*cc-bb*bb
                if determinant > 0:
                    t, s = (bb*ee-cc*dd)/determinant, (aa*ee-bb*dd)/determinant
                    if 0 <= t <= 1 and 0 <= s <= 1:
                        candidates.append((t, s))
                expected_distance2 = min(sum((w[i]+t*u[i]-s*v[i])**2 for i in range(3))
                                         for t, s in candidates)
                radius = rng.uniform(.05, .5)
                with self.subTest(index=index):
                    hits = neurite_neurite_candidates(line(a, b, radius), line(c, d, radius), max_rectangles=1)
                    self.assertEqual(bool(hits), expected_distance2 <= (2*radius)**2)
                    if hits:
                        self.assertAlmostEqual(hits[0].distance_squared, expected_distance2, places=11)

    def test_unresolved_pair_keeps_edge_witness_when_all_corners_are_clear(self):
        first = CanalSegment(((0., 0., 10., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)), .1, .1)
        second = CanalSegment(((5., 0., 10., 0.), (.05, 0., 0., 0.), (0., 0., 0., 0.)), .1, .1)
        with patch("anamorph.analytic_intersections._parallel_nonlinear_line_candidates", return_value=None):
            with self.assertRaises(IntersectionSolveError) as raised:
                neurite_neurite_candidates(first, second, max_rectangles=100)
        self.assertTrue(any(hit.source == "edge" and hit.distance_squared < .2**2
                            for hit in raised.exception.candidates))
        self.assertFalse(any(hit.source == "corner" for hit in raised.exception.candidates))
        tree = Morphology([Node(1, 1, -2, 0, 0, 1, -1), Node(2, 3, 0, 0, 0, .1, 1),
                           Node(3, 3, 10, 0, 0, .1, 2), Node(4, 1, 3, .05, 0, 1, -1),
                           Node(5, 3, 5, .05, 0, .1, 4), Node(6, 3, 15, .05, 0, .1, 5)])
        with patch("anamorph.analytic_network.CanalSegment.from_spline", side_effect=(first, second)), \
                patch("anamorph.analytic_intersections._parallel_nonlinear_line_candidates", return_value=None):
            report = analyze_analytic_partial(tree, max_rectangles=100)
        finding = next(f for f in report.findings if f.kind == "neurite_neurite_candidate")
        self.assertEqual(finding.classification, "inter_cell")
        self.assertTrue(any(hit.source == "edge" and hit.value < .2**2 for hit in finding.candidates))
        self.assertTrue(any("bivariate clipping" in error for error in report.unresolved))
        flags = {entry.edge: entry for entry in report.segment_flags}
        self.assertTrue(all(flags[edge].rc_nsns and flags[edge].unresolved for edge in finding.segments))

    def test_network_resolves_parallel_affine_nonisolated_system(self):
        tree = Morphology([Node(1, 1, -2, 0, 0, 1, -1), Node(2, 3, 0, 0, 0, .1, 1),
                           Node(3, 3, 10, 0, 0, .1, 2), Node(4, 1, 3, .05, 0, 1, -1),
                           Node(5, 3, 5, .05, 0, .1, 4), Node(6, 3, 15, .05, 0, .1, 5)])
        report = analyze_analytic_partial(tree, max_rectangles=100)
        finding = next(f for f in report.findings if f.kind == "neurite_neurite_candidate")
        self.assertEqual(finding.candidates[0].source, "linear_minimum")
        self.assertAlmostEqual(finding.candidates[0].value, .05**2)
        self.assertFalse(any("segments (2, 3) and (5, 6)" in error for error in report.unresolved))

    def test_unresolved_adjacent_and_global_checks_keep_edge_witnesses(self):
        first = line((0., 0., 0.), (10., 0., 0.), .1)
        second = line((0., 0., 0.), (5., 0., 0.), .1)
        with self.assertRaises(IntersectionSolveError) as adjacent:
            adjacent_neurite_candidates(first, second, first_end_second_start=False, max_rectangles=100)
        self.assertTrue(any(hit.source == "edge" and hit.t == .5 and hit.u == 1.
                            for hit in adjacent.exception.candidates))
        self.assertFalse(any(hit.t <= .01 and hit.u <= .01 for hit in adjacent.exception.candidates))
        tree = Morphology([Node(1, 1, -2, 0, 0, 1, -1), Node(2, 3, 0, 0, 0, .1, 1),
                           Node(3, 3, 10, 0, 0, .1, 2), Node(4, 3, 5, 0, 0, .1, 2)])
        report = analyze_analytic_partial(tree, max_rectangles=100)
        finding = next(f for f in report.findings if f.kind == "adjacent_neurite_candidate")
        self.assertTrue(any(hit.source == "edge" and hit.value == 0. for hit in finding.candidates))
        self.assertTrue(report.unresolved)
        returning = CanalSegment(((0., -.5, 1., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)), .01, .01)
        with self.assertRaises(IntersectionSolveError) as global_error:
            global_self_intersection_candidates(returning, max_rectangles=100)
        self.assertTrue(any(hit.source == "edge" and hit.t == 0. and abs(hit.u-.5) < 1e-4
                            for hit in global_error.exception.candidates))
        with patch("anamorph.analytic_network.CanalSegment.from_spline", return_value=returning):
            global_report = analyze_analytic_partial(tree, max_rectangles=100)
        self.assertTrue(any(f.kind == "global_self_intersection" and
                            any(c.source == "edge" and abs(c.parameters[1]-.5) < 1e-4 for c in f.candidates)
                            for f in global_report.findings))
        self.assertTrue(any("global_self_intersection" in item for item in global_report.unresolved))

    def test_unresolved_edge_solves_keep_independent_interior_witness(self):
        first = line((0., 0., 0.), (2., 0., 0.), .1)
        second = line((1., -1., 0.), (1., 1., 0.), .1)
        with patch("anamorph.analytic_intersections._curve_roots",
                   side_effect=ClippingConvergenceError("injected edge failure")):
            with self.assertRaises(IntersectionSolveError) as raised:
                neurite_neurite_candidates(first, second)
        self.assertEqual(len(raised.exception.failures), 4)
        self.assertTrue(any(hit.source == "interior" and hit.distance_squared < 1e-12
                            for hit in raised.exception.candidates))

    def test_global_self_intersection_factors_trivial_diagonal(self):
        straight = line((0., 0., 0.), (1., 0., 0.), .05)
        self.assertEqual(global_self_intersection_candidates(straight), [])
        # A regular cubic returning to its starting point has a nontrivial
        # intersection at (t,u) = (0,1).
        loop = CanalSegment(((0., 1., -1., 0.), (0., -1., 3., -2.), (0., 0., 0., 0.)), .05, .05)
        candidates = global_self_intersection_candidates(loop)
        self.assertTrue(any(c.t == 0 and c.u == 1 for c in candidates), candidates)

    def test_global_cubic_loop_is_not_silently_lost_at_extreme_scales(self):
        base = ((0., -1., 1., 0.), (-.08, .66, -1.5, 1.), (0., 0., 0., 0.))
        for scale in (1., 1e-50, 1e160):
            with self.subTest(scale=scale):
                curve = CanalSegment(tuple(tuple(scale*c for c in axis) for axis in base),
                                     1e-8*scale, 1e-8*scale)
                hits = global_self_intersection_candidates(curve)
                self.assertTrue(any(abs(hit.t-.2) < 1e-6 and abs(hit.u-.8) < 1e-6 for hit in hits))
        scale = 1e-160
        curve = CanalSegment(tuple(tuple(scale*c for c in axis) for axis in base), 1e-8*scale, 1e-8*scale)
        hits = global_self_intersection_candidates(curve)
        witness = next(hit for hit in hits if abs(hit.t-.2) < 1e-6 and abs(hit.u-.8) < 1e-6)
        self.assertIsNone(witness.distance_squared)
        self.assertLess(Fraction(*witness.exact_distance_squared), (2*Fraction(curve.max_radius))**2)

    def test_scaled_nonlinear_pair_can_refine_to_representable_crossing(self):
        for scale in (2.**-530, 1., 2.**530):
            with self.subTest(scale=scale):
                first = CanalSegment(((-.5625*scale, scale, .25*scale, 0.), (0., 0., 0., 0.),
                                      (0., 0., 0., 0.)), .01*scale, .01*scale)
                second = CanalSegment(((0., 0., 0., 0.), (-.5625*scale, scale, .25*scale, 0.),
                                       (0., 0., 0., 0.)), .01*scale, .01*scale)
                hits = neurite_neurite_candidates(first, second)
                self.assertTrue(any(abs(hit.t-.5) < 1e-5 and abs(hit.u-.5) < 1e-5 for hit in hits))
                if scale < 1e-100:
                    self.assertTrue(any(hit.distance_squared == 0. for hit in hits))

    def test_seeded_cubic_crossings_survive_extreme_scale_changes(self):
        rng = random.Random(57019)
        for index in range(15):
            curves = []
            for _ in range(2):
                axes = []
                for _ in range(2):
                    a, b, c = (rng.randint(-6, 6)/4 for _ in range(3))
                    axes.append((-(a/2+b/4+c/8), a, b, c))
                curves.append((*axes, (0., 0., 0., 0.)))
            for scale in (2.**-400, 1., 2.**400):
                with self.subTest(index=index, scale=scale):
                    first, second = (CanalSegment(tuple(tuple(scale*x for x in axis) for axis in curve),
                                                  1e-4*scale, 1e-4*scale) for curve in curves)
                    hits = neurite_neurite_candidates(first, second, max_rectangles=5000)
                    self.assertTrue(any(abs(hit.t-.5) < .001 and abs(hit.u-.5) < .001 for hit in hits))

    def test_extreme_system_ratio_preserves_exact_corner_witness(self):
        first = CanalSegment(((0., 1e200, -1e200, 0.), (0., 1e-200, -1e-200, 0.),
                              (0., 0., 0., 0.)), 1., 1.)
        second = CanalSegment(((0., 0., 1., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)), 1., 1.)
        with self.assertRaises(IntersectionSolveError) as pair_error:
            neurite_neurite_candidates(first, second)
        self.assertTrue(any(hit.t == hit.u == 0. and hit.distance_squared == 0.
                            for hit in pair_error.exception.candidates))
        with self.assertRaises(IntersectionSolveError) as global_error:
            global_self_intersection_candidates(first)
        self.assertTrue(any(hit.t == 0. and hit.u == 1. and hit.distance_squared == 0.
                            for hit in global_error.exception.candidates))

    def test_corner_fallback_does_not_use_underflowed_squared_distances(self):
        from fractions import Fraction
        first = line((0., 0., 0.), (1e-160, 0., 0.), 1e-162)
        second = line((0., 1e-160, 0.), (1e-160, 1e-160, 0.), 1e-162)
        threshold = (Fraction(first.max_radius)+Fraction(second.max_radius))**2
        self.assertIsNone(_verified_corner_candidate(first, second, 0., 0., threshold))

    def test_partial_network_report_flags_crossing_and_never_certifies_clean(self):
        tree = Morphology([Node(1, 1, -3, 0, 0, .5, -1), Node(2, 3, -2, 0, 0, .1, 1),
                           Node(3, 3, 2, 0, 0, .1, 2), Node(4, 1, 0, -3, 0, .5, -1),
                           Node(5, 3, 0, -2, 0, .1, 4), Node(6, 3, 0, 2, 0, .1, 5)])
        report = analyze_analytic_partial(tree)
        self.assertFalse(report.complete)
        self.assertEqual(report.tested_nonadjacent_pairs, 1)
        self.assertTrue(any(f.kind == "neurite_neurite_candidate" and f.classification == "inter_cell"
                            for f in report.findings))
        crossing = next(f for f in report.findings if f.kind == "neurite_neurite_candidate")
        self.assertTrue(any(abs(c.parameters[0]-.5) < .01 and abs(c.parameters[1]-.5) < .01
                            and c.value < 1e-8 for c in crossing.candidates))
        interior = next(c for c in crossing.candidates if c.source == "interior")
        self.assertTrue(all(left <= .5 <= right for left, right in interior.parameter_bounds))
        statuses = {status.edge: status for status in report.segment_flags}
        self.assertTrue(all(statuses[edge].rc_nsns for edge in crossing.segments))
        self.assertTrue(report.as_dict()["findings"])
        limited = analyze_analytic_partial(tree, max_pairs=0)
        self.assertTrue(any("limit" in error for error in limited.unresolved))
        self.assertTrue(limited.pair_limit_reached)
        self.assertEqual(set(limited.first_untested_pair), {(2, 3), (5, 6)})
        self.assertTrue(all(status.pair_limit_may_affect for status in limited.segment_flags))

    def test_pair_limit_marks_potentially_affected_segments_conservatively(self):
        tree = Morphology([Node(1, 1, -12, 0, 0, .2, -1), Node(2, 3, -10, 0, 0, .1, 1),
                           Node(3, 3, -9, 0, 0, .1, 2), Node(4, 1, -2, 0, 0, .2, -1),
                           Node(5, 3, 0, 0, 0, .1, 4), Node(6, 3, 1, 0, 0, .1, 5),
                           Node(7, 1, -2, .1, 0, .2, -1), Node(8, 3, 0, .1, 0, .1, 7),
                           Node(9, 3, 1, .1, 0, .1, 8), Node(10, 1, 8, 0, 0, .2, -1),
                           Node(11, 3, 10, 0, 0, .1, 10), Node(12, 3, 11, 0, 0, .1, 11)])
        limited = analyze_analytic_partial(tree, max_pairs=0)
        self.assertEqual(set(limited.first_untested_pair), {(5, 6), (8, 9)})
        flags = {status.edge: status.pair_limit_may_affect for status in limited.segment_flags}
        self.assertFalse(flags[(2, 3)])  # Its box ends before the cutoff.
        self.assertTrue(flags[(5, 6)] and flags[(8, 9)])
        self.assertTrue(flags[(11, 12)])  # Conservative: a later box may have no partner.
        full = analyze_analytic_partial(tree, max_pairs=10)
        self.assertFalse(full.pair_limit_reached)
        self.assertIsNone(full.first_untested_pair)
        self.assertTrue(all(not status.pair_limit_may_affect for status in full.segment_flags))

    def test_adjacent_segments_allow_junction_but_detect_later_crossing(self):
        first = line((0., 0., 0.), (2., 0., 0.), .05)
        other = line((0., 0., 0.), (0., 2., 0.), .05)
        self.assertEqual(adjacent_neurite_candidates(first, other, first_end_second_start=False), [])
        # gamma=(2t,0), delta=(2u,2u-4u²): they share their start and
        # cross again halfway through both segments.
        returning = CanalSegment(((0., 2., 0., 0.), (0., 2., -4., 0.), (0., 0., 0., 0.)), .05, .05)
        hits = adjacent_neurite_candidates(first, returning, first_end_second_start=False)
        self.assertTrue(any(abs(c.t-.5) < .01 and abs(c.u-.5) < .01 for c in hits), hits)
        continuation = line((2., 0., 0.), (3., 0., 0.), .05)
        self.assertEqual(adjacent_neurite_candidates(first, continuation, first_end_second_start=True), [])
        self.assertEqual(adjacent_neurite_candidates(first, continuation, first_end_second_start=True,
                                                     check_end_start=True), [])

    def test_optional_consecutive_check_finds_a_second_crossing(self):
        first = line((0., 0., 0.), (1., 0., 0.), .02)
        # The curves share first(1)=second(0), then cross at first(.4)=second(.6).
        second = CanalSegment(((1., -1., 0., 0.), (0., -.6, 1., 0.), (0., 0., 0., 0.)), .02, .02)
        self.assertEqual(adjacent_neurite_candidates(first, second, first_end_second_start=True), [])
        hits = adjacent_neurite_candidates(first, second, first_end_second_start=True, check_end_start=True)
        self.assertTrue(any(abs(hit.t-.4) < .001 and abs(hit.u-.6) < .001 and hit.distance_squared < 1e-10
                            for hit in hits), hits)

    def test_network_classifies_inter_neurite_candidates(self):
        tree = Morphology([Node(1,1,-3,-3,0,.2,-1), Node(2,3,-2,0,0,.1,1), Node(3,3,2,0,0,.1,2),
                           Node(4,3,0,-2,0,.1,1), Node(5,3,0,2,0,.1,4)])
        report = analyze_analytic_partial(tree)
        self.assertTrue(any(f.classification == "intra_cell_inter_neurite" for f in report.findings))

    def test_unresolved_pair_retains_independent_endpoint_witness(self):
        tree = Morphology([Node(1, 1, -1, 0, 0, .2, -1), Node(2, 3, 0, 0, 0, .1, 1),
                           Node(3, 3, 1, 0, 0, .1, 2), Node(4, 1, -1, .15, 0, .2, -1),
                           Node(5, 3, 0, .15, 0, .1, 4), Node(6, 3, 1, .15, 0, .1, 5)])
        first = CanalSegment(((0., 0., 1., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)), .1, .1)
        # Different speeds keep this outside the exact translated-curve case.
        second = CanalSegment(((0., 0., 1.01, 0.), (.15, 0., 0., 0.), (0., 0., 0., 0.)), .1, .1)
        with patch("anamorph.analytic_network.CanalSegment.from_spline", side_effect=(first, second)), \
                patch("anamorph.analytic_intersections._parallel_nonlinear_line_candidates", return_value=None):
            report = analyze_analytic_partial(tree, max_rectangles=300)
        hit = next(f for f in report.findings if f.kind == "neurite_neurite_candidate")
        self.assertTrue(any(c.source == "corner_fallback" and c.value < .2**2 for c in hit.candidates))
        self.assertTrue(any("bivariate clipping" in message for message in report.unresolved))
        flags = {entry.edge: entry for entry in report.segment_flags}
        self.assertTrue(flags[(2, 3)].rc_nsns)
        self.assertTrue(flags[(2, 3)].unresolved)

    def test_unresolved_adjacent_pair_retains_nonjunction_witness(self):
        tree = Morphology([Node(1, 1, -1, 0, 0, .2, -1), Node(2, 3, 0, 0, 0, .1, 1),
                           Node(3, 3, 1, 0, 0, .1, 2), Node(4, 3, 1, .05, 0, .1, 2)])
        with patch("anamorph.analytic_network.adjacent_neurite_candidates",
                   side_effect=ClippingConvergenceError("test limit")):
            report = analyze_analytic_partial(tree)
        hit = next(f for f in report.findings if f.kind == "adjacent_neurite_candidate")
        self.assertTrue(any(c.parameters == (1., 1.) and c.source == "corner_fallback" for c in hit.candidates))
        self.assertFalse(any(c.parameters == (0., 0.) for c in hit.candidates))
        self.assertTrue(any("test limit" in message for message in report.unresolved))

    def test_unresolved_consecutive_pair_excludes_end_start_junction(self):
        tree = Morphology([Node(1, 1, -1, 0, 0, .2, -1), Node(2, 3, 0, 0, 0, .1, 1),
                           Node(3, 3, 1, 0, 0, .1, 2), Node(4, 3, 0, .05, 0, .1, 3)])
        with patch("anamorph.analytic_network.adjacent_neurite_candidates",
                   side_effect=ClippingConvergenceError("test limit")):
            report = analyze_analytic_partial(tree, check_adjacent_end_start=True)
        hit = next(f for f in report.findings if f.kind == "adjacent_neurite_candidate")
        self.assertTrue(any(c.parameters == (0., 1.) for c in hit.candidates))
        self.assertFalse(any(c.parameters == (1., 0.) for c in hit.candidates))

    def test_unresolved_segment_jobs_retain_independent_endpoint_witnesses(self):
        tree = Morphology([Node(1, 1, -3, 0, 0, .2, -1), Node(2, 3, 0, 0, 0, 1., 1),
                           Node(3, 3, 1, 1, 0, 1., 2)])
        irregular = CanalSegment(((0., 0., 1., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)), .1, .1)
        tight = CanalSegment(((0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 0.)), 1., 1.)
        loop = CanalSegment(((0., .1, 0., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)), .2, .2)
        cases = (("regularity_candidates", "regularity", irregular),
                 ("local_self_intersection_candidates", "local_self_intersection", tight),
                 ("global_self_intersection_candidates", "global_self_intersection", loop))
        for function, kind, curve in cases:
            with self.subTest(kind=kind), \
                 patch("anamorph.analytic_network.CanalSegment.from_spline", return_value=curve), \
                 patch("anamorph.analytic_network."+function, side_effect=ClippingConvergenceError("test limit")):
                report = analyze_analytic_partial(tree)
            hit = next(f for f in report.findings if f.kind == kind)
            self.assertTrue(any("fallback" in c.source for c in hit.candidates))
            self.assertTrue(any(kind in message for message in report.unresolved))

    def test_unresolved_soma_job_retains_independent_endpoint_witness(self):
        tree = Morphology([Node(1, 1, 0, 0, 0, 1, -1), Node(2, 3, 1, 0, 0, .05, 1),
                           Node(3, 3, 2, 0, 0, .05, 2), Node(4, 1, 1, .2, 0, .2, -1)])
        with patch("anamorph.analytic_network.soma_neurite_candidates",
                   side_effect=ClippingConvergenceError("test limit")):
            report = analyze_analytic_partial(tree)
        hit = next(f for f in report.findings if f.kind == "soma_neurite_candidate" and f.somas == (4,))
        self.assertTrue(any(c.source == "endpoint_fallback" for c in hit.candidates))
        self.assertTrue(any("soma 4" in message for message in report.unresolved))

    def test_unresolved_soma_job_keeps_solver_witness(self):
        tree = Morphology([Node(1, 1, 0, 0, 0, 1, -1), Node(2, 3, 1, 0, 0, .05, 1),
                           Node(3, 3, 2, 0, 0, .05, 2), Node(4, 1, 1, .2, 0, .2, -1)])
        witness = ScalarCandidate(.4, .01, "root_interval", (.39, .41))
        with patch("anamorph.analytic_network.soma_neurite_candidates",
                   side_effect=IntersectionSolveError(("unresolved stationary root",), (witness,))):
            report = analyze_analytic_partial(tree)
        hit = next(f for f in report.findings if f.kind == "soma_neurite_candidate" and f.somas == (4,))
        self.assertTrue(any(c.parameters == (.4,) and c.source == "root_interval" for c in hit.candidates))
        self.assertTrue(any("unresolved stationary root" in message for message in report.unresolved))

    def test_root_junction_exemption_only_applies_to_its_own_soma(self):
        tree = Morphology([Node(1, 1, 0, 0, 0, 1, -1), Node(2, 3, 1, 0, 0, .05, 1),
                           Node(3, 3, 2, 0, 0, .05, 2), Node(4, 1, 1, .2, 0, .2, -1)])
        report = analyze_analytic_partial(tree)
        hits = [f for f in report.findings if f.kind == "soma_neurite_candidate" and
                f.classification == "inter_cell" and f.segments == ((2, 3),)]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].somas, (4,))
        self.assertTrue(any(c.parameters == (0.,) for c in hits[0].candidates))
        self.assertFalse(any(f.kind == "soma_neurite_candidate" and f.classification == "intra_cell"
                             for f in report.findings))

    def test_intercell_soma_spheres_are_checked(self):
        tree = Morphology([Node(1, 1, 0, 0, 0, 1, -1), Node(2, 1, .2, 0, 0, .2, 1),
                           Node(3, 1, 1.5, 0, 0, .6, -1), Node(4, 1, 10, 0, 0, 1, -1)])
        report = analyze_analytic_partial(tree)
        hits = [f for f in report.findings if f.kind == "soma_soma_intersection"]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].somas, (1, 3))
        self.assertAlmostEqual(hits[0].candidates[0].value, 2.25)
        self.assertEqual(hits[0].classification, "inter_cell")

    def test_soma_sphere_overlap_decision_survives_float_square_limits(self):
        for scale in (1e-200, 1., 1e200):
            for coordinate, expected in ((1., True), (1.5, False)):
                with self.subTest(scale=scale, coordinate=coordinate):
                    tree = Morphology([Node(1, 1, 0., 0., 0., scale, -1),
                                       Node(2, 1, coordinate*scale, coordinate*scale, 0., scale, -1)])
                    report = analyze_analytic_partial(tree)
                    hits = [finding for finding in report.findings if finding.kind == "soma_soma_intersection"]
                    self.assertEqual(bool(hits), expected)
                    if expected and scale != 1.:
                        witness = hits[0].candidates[0]
                        self.assertIsNone(witness.value)
                        self.assertEqual(Fraction(*witness.exact_value), 2*Fraction(scale)**2)
                    json.dumps(report.as_dict(), allow_nan=False)

    def test_soma_sphere_broad_phase_agrees_with_exact_comparison(self):
        from fractions import Fraction
        rng = random.Random(26831)
        for scale in (1e-200, 1., 1e200):
            for index in range(30):
                x, y = (rng.uniform(-2.1, 2.1)*scale for _ in range(2))
                tree = Morphology([Node(1, 1, 0., 0., 0., scale, -1), Node(2, 1, x, y, 0., scale, -1)])
                with self.subTest(scale=scale, index=index):
                    found = any(f.kind == "soma_soma_intersection" for f in analyze_analytic_partial(tree).findings)
                    self.assertEqual(found, Fraction(x)**2+Fraction(y)**2 <= (2*Fraction(scale))**2)

    def test_adjacent_junction_validation_uses_local_scale(self):
        for scale in (1e-18, 1., 1e12):
            with self.subTest(scale=scale):
                first = line((0., 0., 0.), (scale, 0., 0.), .1*scale)
                valid = line((scale, 0., 0.), (2*scale, 0., 0.), .1*scale)
                self.assertEqual(adjacent_neurite_candidates(first, valid, first_end_second_start=True), [])
                displaced = line((1.1*scale, 0., 0.), (2*scale, 0., 0.), .1*scale)
                with self.assertRaisesRegex(ValueError, "must share"):
                    adjacent_neurite_candidates(first, displaced, first_end_second_start=True)
        first = line((1e6, 0., 0.), (1e6+.01, 0., 0.), .001)
        valid = line((1e6+.01, 0., 0.), (1e6+.02, 0., 0.), .001)
        self.assertEqual(adjacent_neurite_candidates(first, valid, first_end_second_start=True), [])
        displaced = line((1e6+.0101, 0., 0.), (1e6+.02, 0., 0.), .001)
        with self.assertRaisesRegex(ValueError, "must share"):
            adjacent_neurite_candidates(first, displaced, first_end_second_start=True)

    def test_isolated_surface_root_is_reported_as_unresolved(self):
        tree = Morphology([Node(1, 1, 0, 0, 0, 1, -1), Node(2, 3, 1, 0, 0, .2, 1)])
        report = analyze_analytic_partial(tree, prepare=True)
        self.assertTrue(any("no segment to analyze" in item for item in report.unresolved))
        self.assertFalse(report.complete)

    def test_analysis_cli_accepts_source_geometry_and_tolerance_options(self):
        example = Path(__file__).resolve().parent.parent / "examples" / "branched.swc"
        command = [sys.executable, "-m", "anamorph", "analyze-analytic", str(example),
                   "--partition", "min-angle", "--parametrization", "uniform",
                   "--univariate-tolerance", "1e-6", "--bivariate-tolerance", "1e-4",
                   "--scale-radius", ".7", "--check-adjacent-end-start"]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        report = json.loads(result.stdout)
        self.assertEqual(report["segment_count"], 6)
        self.assertFalse(report["complete"])
        self.assertTrue(any(f["candidates"] for f in report["findings"]))
        invalid = subprocess.run(command+["--univariate-tolerance", "0"], capture_output=True, text=True)
        self.assertEqual(invalid.returncode, 1)
        self.assertIn("tolerances", invalid.stderr)

    def test_one_failed_curve_job_does_not_hide_other_curve_findings(self):
        tree = Morphology([Node(1, 1, -3, 0, 0, .2, -1),
                           Node(2, 3, 0, 0, 0, .1, 1),
                           Node(3, 3, 1, 0, 0, .1, 2)])
        with patch("anamorph.analytic_network.regularity_candidates",
                   side_effect=ClippingConvergenceError("unresolved regularity")), \
             patch("anamorph.analytic_network.local_self_intersection_candidates",
                   return_value=[ScalarCandidate(.5, 0., "root_interval", (.49, .51))]), \
             patch("anamorph.analytic_network.global_self_intersection_candidates",
                   return_value=[CurveCandidate(.1, .9, 0., "interior")]):
            report = analyze_analytic_partial(tree)
        self.assertTrue(any("regularity" in item for item in report.unresolved))
        self.assertTrue(any(f.kind == "local_self_intersection" for f in report.findings))
        self.assertTrue(any(f.kind == "global_self_intersection" for f in report.findings))
        status = report.segment_flags[0]
        self.assertFalse(status.reg)
        self.assertTrue(status.lsi and status.gsi)
        self.assertTrue(any("regularity" in item for item in status.unresolved))

    def test_unresolved_local_check_keeps_representable_witness(self):
        tree = Morphology([Node(1, 1, -3, 0, 0, .2, -1), Node(2, 3, 0, 0, 0, .1, 1),
                           Node(3, 3, 1, 0, 0, .1, 2)])
        witness = ScalarCandidate(.4, .01, "root_interval", (.39, .41))
        with patch("anamorph.analytic_network.local_self_intersection_candidates",
                   side_effect=IntersectionSolveError(("value exceeds float range",), (witness,))):
            report = analyze_analytic_partial(tree)
        hit = next(f for f in report.findings if f.kind == "local_self_intersection")
        self.assertTrue(any(c.parameters == (.4,) for c in hit.candidates))
        self.assertTrue(any("value exceeds float range" in message for message in report.unresolved))
        self.assertTrue(report.segment_flags[0].lsi)

    def test_sweep_candidates_equal_brute_force(self):
        import random
        rng = random.Random(44)
        boxes = []
        for _ in range(100):
            lows = [rng.uniform(-5, 5) for _ in range(3)]
            boxes.append(tuple((low, low+rng.uniform(.1, 3)) for low in lows))
        actual = set(_overlapping_box_pairs(boxes))
        expected = {(i, j) for i in range(len(boxes)) for j in range(i+1, len(boxes)) if _overlap(boxes[i], boxes[j])}
        self.assertEqual(actual, expected)
        ordered = list(_overlapping_box_pairs(boxes))
        for cutoff in range(len(ordered)):
            first, second = ordered[cutoff]
            cutoff_x = max(boxes[first][0][0], boxes[second][0][0])
            later_vertices = {vertex for pair in ordered[cutoff:] for vertex in pair}
            possible = {i for i, box in enumerate(boxes) if box[0][1] >= cutoff_x}
            self.assertTrue(later_vertices <= possible)


if __name__ == "__main__":
    unittest.main()
