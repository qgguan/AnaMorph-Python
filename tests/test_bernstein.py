import math
import json
import random
import unittest
from fractions import Fraction
from pathlib import Path

from anamorph.bernstein import (BernsteinPolynomial, BiBernsteinPolynomial, ClippingConvergenceError,
                                 RootRectangle, bezier_clip_roots, bilinear_clip_roots, _has_unit_interval_root,
                                 _shared_quadratic_with_affine_cofactors)
from anamorph.polynomial import evaluate, multiply


class BernsteinTests(unittest.TestCase):
    def test_source_univariate_trace_midpoint_and_degree_one_rules(self):
        from anamorph.source_univariate import source_bezier_clip_roots
        cubic = BernsteinPolynomial((-.125, .125, -.125, .125))
        root, = source_bezier_clip_roots(cubic, tolerance=.01)
        self.assertAlmostEqual(root.left, .49)
        self.assertAlmostEqual(root.right, .51)
        # The source degree-one hull contains only its first endpoint, so it
        # misses a simple root between opposite-sign coefficients.
        linear = BernsteinPolynomial((-1., 1.))
        self.assertEqual(source_bezier_clip_roots(linear), [])
        self.assertTrue(any(root.left <= .5 <= root.right for root in bezier_clip_roots(linear)))
        with self.assertRaisesRegex(ClippingConvergenceError, "degree-zero"):
            source_bezier_clip_roots(BernsteinPolynomial((0.,)))
        with self.assertRaises(ClippingConvergenceError) as caught:
            source_bezier_clip_roots(cubic, epsilon=0., max_intervals=1)
        self.assertEqual(caught.exception.processed, 1)

    def test_source_rectangle_clipping_and_bounded_trace(self):
        from anamorph.source_bivariate import (SourceBivariateFailure, source_bilinear_clip_roots,
                                               source_clip_rectangle)
        from anamorph.bernstein import RootRectangle
        p = BiBernsteinPolynomial.from_power(((-.2,), (1.,))).elevate(5, 5)
        q = BiBernsteinPolynomial.from_power(((-.7, 1.),)).elevate(5, 5)
        box = RootRectangle(0., 1., 0., 1.)
        clipped = source_clip_rectangle(p, q, box)
        self.assertLessEqual(clipped.x0, .2)
        self.assertGreaterEqual(clipped.x1, .2)
        self.assertLessEqual(clipped.y0, .7)
        self.assertGreaterEqual(clipped.y1, .7)
        roots = source_bilinear_clip_roots(p, q)
        self.assertTrue(any(root.x0 <= .2 <= root.x1 and root.y0 <= .7 <= root.y1 for root in roots))
        parallel = BiBernsteinPolynomial.from_power(((0., 2.), (2., 0.))).elevate(5, 5)
        other = BiBernsteinPolynomial.from_power(((0., 1.), (1., 0.))).elevate(5, 5)
        with self.assertRaises(SourceBivariateFailure):
            source_clip_rectangle(parallel, other, box)
        zeros = BiBernsteinPolynomial(tuple((0.,)*6 for _ in range(6)))
        with self.assertRaises(ClippingConvergenceError) as raised:
            source_bilinear_clip_roots(zeros, zeros, max_rectangles=3)
        self.assertNotIsInstance(raised.exception, SourceBivariateFailure)
        self.assertEqual(raised.exception.processed, 3)
        self.assertGreater(raised.exception.pending, 0)
        # In the source's y-frozen convergence branch, the emitted x bounds
        # are the old bounds, even though the new x interval has converged.
        quadratic = BiBernsteinPolynomial.from_power(((-.25,), (0.,), (1.,))).elevate(5, 5)
        horizontal = BiBernsteinPolynomial.from_power(((-.5, 1.),)).elevate(5, 5)
        root, = source_bilinear_clip_roots(quadratic, horizontal, tolerance=1e-4)
        self.assertLessEqual(root.x0, .5)
        self.assertGreaterEqual(root.x1, .5)
        self.assertGreater(root.x1-root.x0, 1e-4)

    def test_source_four_way_split_defect_and_missed_exact_root(self):
        from anamorph.source_bivariate import source_bilinear_clip_roots, source_split_xy
        from unittest.mock import patch
        x = BiBernsteinPolynomial.from_power(((0.,), (1.,))).elevate(5, 5)
        left_down, right_down, right_up, left_up = source_split_xy(x)
        self.assertEqual((left_down.evaluate(1., 0.), right_down.evaluate(0., 0.),
                          right_down.evaluate(1., 0.), right_up.evaluate(1., 0.), left_up.evaluate(1., 0.)),
                         (.5, 0., .5, .5, .5))
        self.assertEqual(x.restrict(.5, 1., 0., .5).evaluate(0., 0.), .5)
        data = json.loads((Path(__file__).parent/"fixtures"/"source_split_missed_root.json").read_text())
        denominator = data["denominator"]
        p, q = (BiBernsteinPolynomial(tuple(tuple(value/denominator for value in row) for row in matrix))
                for matrix in data["coefficients"])
        weights = [[math.comb(5, i)*3**i*math.comb(5, j)*3**j for j in range(6)] for i in range(6)]
        for poly in (p, q):
            self.assertEqual(sum(Fraction(poly.coefficients[i][j])*weights[i][j]
                                 for i in range(6) for j in range(6)), 0)
        with patch("anamorph.source_bivariate.source_split_xy", wraps=source_split_xy) as split:
            source = source_bilinear_clip_roots(p, q, tolerance=.005)
        self.assertGreater(split.call_count, 0)
        contains = lambda root: root.x0 <= .75 <= root.x1 and root.y0 <= .75 <= root.y1
        self.assertFalse(any(map(contains, source)))
        self.assertTrue(any(map(contains, bilinear_clip_roots(p, q, tolerance=.005))))
    def test_power_conversion_restriction_derivative_and_product(self):
        rng = random.Random(814)
        for degree in range(1, 9):
            power = [rng.uniform(-2, 2) for _ in range(degree+1)]
            p = BernsteinPolynomial.from_power(power)
            lower, upper = .17, .81
            piece = p.restrict(lower, upper)
            for t in (0, .13, .5, .89, 1):
                self.assertAlmostEqual(p.evaluate(t), evaluate(power, t), places=10)
                self.assertAlmostEqual(piece.evaluate(t), p.evaluate(lower+t*(upper-lower)), places=10)
                self.assertAlmostEqual(p.elevate(degree+2).evaluate(t), p.evaluate(t), places=10)
                self.assertAlmostEqual(p.multiply(p).evaluate(t), p.evaluate(t)**2, places=9)
                derivative = [i*power[i] for i in range(1, len(power))]
                self.assertAlmostEqual(p.derivative().evaluate(t), evaluate(derivative, t), places=9)

    def test_clipping_separates_roots_and_repeated_tangent(self):
        p = [1.0]
        for root in (.1, .3, .3, .6, .9):
            p = multiply(p, [-root, 1.0])
        intervals = bezier_clip_roots(BernsteinPolynomial.from_power(p), tolerance=1e-7, eps=1e-13)
        for root in (.1, .3, .6, .9):
            self.assertTrue(any(r.left-1e-8 <= root <= r.right+1e-8 for r in intervals), (root, intervals))
        self.assertEqual(len(intervals), 4)

    def test_zero_epsilon_univariate_clipping_keeps_dyadic_roots(self):
        rng = random.Random(316710)
        cases = [(-5, 7, -9, 11)]
        for _ in range(200):
            b, c, d = (rng.randrange(-100, 101) for _ in range(3))
            cases.append((-3*b-3*c-d, b, c, d))
        for coefficients in cases:
            for scale in (2.**-400, 1., 2.**400):
                with self.subTest(coefficients=coefficients, scale=scale):
                    polynomial = BernsteinPolynomial(tuple(scale*value for value in coefficients))
                    roots = bezier_clip_roots(polynomial, tolerance=1e-8, eps=0.)
                    self.assertTrue(any(root.left-1e-12 <= .5 <= root.right+1e-12 for root in roots), roots)
                    if coefficients == cases[0]:
                        self.assertTrue(any(root.left-1e-12 <= .3125 <= root.right+1e-12 for root in roots), roots)
        # Exact rational evaluation shows a sign change across this bracket;
        # the unscaled convex hull previously overflowed and lost the root.
        extreme = BernsteinPolynomial((-1.72e308, 2.5e307, 2.4e307, 2.5e307))
        roots = bezier_clip_roots(extreme, tolerance=1e-7, eps=0.)
        self.assertTrue(any(root.left <= .500001 and root.right >= .499999 for root in roots), roots)
        with self.assertRaisesRegex(ClippingConvergenceError, "coefficient scale ratio"):
            bezier_clip_roots(BernsteinPolynomial((1e308, -1e-300)), eps=0.)

    def test_boundaries_subdomain_and_no_root(self):
        p = BernsteinPolynomial.from_power([0., -.4, 1.])
        roots = bezier_clip_roots(p, tolerance=1e-8)
        self.assertTrue(any(r.left <= 0 <= r.right for r in roots))
        self.assertTrue(any(r.left <= .4 <= r.right for r in roots))
        roots = bezier_clip_roots(p, left=.2, right=.8, tolerance=1e-8)
        self.assertEqual(len(roots), 1)
        self.assertTrue(roots[0].left <= .4 <= roots[0].right)
        self.assertEqual(bezier_clip_roots(BernsteinPolynomial.from_power([1, 0, 1])), [])
        self.assertEqual(bezier_clip_roots(BernsteinPolynomial((0., 0.)))[0].right, 1)

    def test_epsilon_and_work_limit_are_explicit(self):
        p = BernsteinPolynomial((1e-13, 1e-13))
        self.assertEqual(len(bezier_clip_roots(p, eps=1e-11)), 1)
        self.assertEqual(bezier_clip_roots(p, eps=0), [])
        with self.assertRaises(ValueError):
            bezier_clip_roots(p, tolerance=0)
        with self.assertRaises(ClippingConvergenceError):
            bezier_clip_roots(BernsteinPolynomial((1., -1., 1.)), max_intervals=1)

    def test_univariate_limit_retains_completed_root_candidates(self):
        power = multiply([-.2, 1.], multiply([-.7, 1.], [-.7, 1.]))
        polynomial = BernsteinPolynomial.from_power(power)
        with self.assertRaises(ClippingConvergenceError) as raised:
            bezier_clip_roots(polynomial, tolerance=1e-8, max_intervals=9)
        error = raised.exception
        self.assertEqual(error.processed, 9)
        self.assertGreater(error.pending, 0)
        self.assertTrue(any(root.left <= .2 <= root.right for root in error.partial_roots))

    def test_bivariate_limit_retains_completed_root_rectangles(self):
        power = multiply([-.2, 1.], multiply([-.7, 1.], [-.7, 1.]))
        p = BiBernsteinPolynomial.from_power(tuple((coefficient,) for coefficient in power))
        q = BiBernsteinPolynomial.from_power(((-.3, 1.),))
        with self.assertRaises(ClippingConvergenceError) as raised:
            bilinear_clip_roots(p, q, tolerance=1e-6, max_rectangles=13)
        error = raised.exception
        self.assertEqual(error.processed, 13)
        self.assertGreater(error.pending, 0)
        self.assertTrue(any(root.x0 <= .2 <= root.x1 and root.y0 <= .3 <= root.y1
                            for root in error.partial_roots))

    def test_bivariate_power_conversion_and_restriction(self):
        # 1 + 2x + 3y + 4xy + 5x²y
        p = BiBernsteinPolynomial.from_power(((1., 3.), (2., 4.), (0., 5.)))
        piece = p.restrict(.1, .8, .2, .9)
        for x in (0., .3, 1.):
            for y in (0., .6, 1.):
                self.assertAlmostEqual(p.evaluate(x, y), 1+2*x+3*y+4*x*y+5*x*x*y)
                self.assertAlmostEqual(piece.evaluate(x, y), p.evaluate(.1+.7*x, .2+.7*y))

    def test_bivariate_linear_clipping_known_roots(self):
        # p=(x-0.2)(x-0.8); q=y-0.4
        p = BiBernsteinPolynomial.from_power(((.16,), (-1.,), (1.,)))
        q = BiBernsteinPolynomial.from_power(((-.4, 1.),))
        roots = bilinear_clip_roots(p, q, tolerance=1e-6)
        for x in (.2, .8):
            self.assertTrue(any(r.x0-1e-7 <= x <= r.x1+1e-7 and r.y0-1e-7 <= .4 <= r.y1+1e-7
                                for r in roots), (x, roots))
        self.assertTrue(all(r.x1-r.x0 <= 1e-6 and r.y1-r.y0 <= 1e-6 for r in roots))

    def test_bivariate_parallel_fits_and_no_solution(self):
        # x+y=0.5 and x+y=1.5 are parallel and cannot intersect.
        p = BiBernsteinPolynomial.from_power(((-.5, 1.), (1., 0.)))
        q = BiBernsteinPolynomial.from_power(((-1.5, 1.), (1., 0.)))
        self.assertEqual(bilinear_clip_roots(p, q), [])
        zero = BiBernsteinPolynomial(((0.,),))
        with self.assertRaises(ClippingConvergenceError):
            bilinear_clip_roots(p, zero)

    def test_dependent_bivariate_system_is_reported_unresolved(self):
        p = BiBernsteinPolynomial.from_power(((-.5,), (1.,)))
        for multiplier in (1., -2., 1e200, 1e-200):
            q = BiBernsteinPolynomial(tuple(tuple(multiplier*value for value in row)
                                            for row in p.coefficients))
            with self.subTest(multiplier=multiplier), self.assertRaisesRegex(
                    ClippingConvergenceError, "dependent polynomials"):
                bilinear_clip_roots(p, q, tolerance=.01, eps=0.)
        positive = BiBernsteinPolynomial(((1.,), (2.,)))
        self.assertEqual(bilinear_clip_roots(positive, positive), [])
        # A small but nonzero independent y term defines one isolated root.
        p = BiBernsteinPolynomial.from_power(((-.5, 0.), (1., 0.)))
        q = BiBernsteinPolynomial.from_power(((-.5000005, 1e-6), (1., 0.)))
        roots = bilinear_clip_roots(p, q, tolerance=1e-5, eps=1e-14)
        self.assertTrue(any(r.x0 <= .5 <= r.x1 and r.y0-1e-8 <= .5 <= r.y1+1e-8 for r in roots))

    def test_different_polynomials_with_shared_axis_root_are_unresolved(self):
        x = BiBernsteinPolynomial.from_power(((-.5, 0.), (1., 0.)))
        xy = BiBernsteinPolynomial.from_power(((0., -.5), (0., 1.)))
        y = BiBernsteinPolynomial.from_power(((-.5, 1.), (0., 0.)))
        yx = BiBernsteinPolynomial.from_power(((0., 0.), (-.5, 1.)))
        for first, second in ((x, xy), (y, yx)):
            with self.subTest(axis="x" if first is x else "y"), self.assertRaisesRegex(
                    ClippingConvergenceError, "shared axis root"):
                bilinear_clip_roots(first, second, tolerance=.01, eps=0.)
            # A fully excluded domain has no roots to search.
            self.assertEqual(bilinear_clip_roots(first, second, blacklist=(RootRectangle(0., 1., 0., 1.),)), [])

    def test_shared_diagonal_affine_factor_is_reported_unresolved(self):
        # x+y-1 divides x²-y²-x+y, so the shared roots include a diagonal.
        p = BiBernsteinPolynomial.from_power(((-1., 1., 0.), (1., 0., 0.), (0., 0., 0.)))
        q = BiBernsteinPolynomial.from_power(((0., 1., -1.), (-1., 0., 0.), (1., 0., 0.)))
        for first, second in ((p, q), (q, p)):
            with self.assertRaisesRegex(ClippingConvergenceError, "shared affine root line"):
                bilinear_clip_roots(first, second, tolerance=.02)
        # An added constant removes the common factor and all common roots.
        no_common_root = BiBernsteinPolynomial.from_power(((.125, 1., -1.), (-1., 0., 0.), (1., 0., 0.)))
        self.assertEqual(bilinear_clip_roots(p, no_common_root, tolerance=1e-5), [])
        # A shared line that touches the square only at one corner is not a
        # non-isolated solution in the search domain.
        corner = BiBernsteinPolynomial.from_power(((0., 1., 0.), (1., 0., 0.), (0., 0., 0.)))
        corner_times = BiBernsteinPolynomial.from_power(((0., 1., 0.), (1., 1., 0.), (1., 0., 0.)))
        roots = bilinear_clip_roots(corner, corner_times, tolerance=1e-5)
        self.assertTrue(any(r.x0 == 0 and r.y0 == 0 for r in roots))
        rng = random.Random(61084)
        for _ in range(30):
            b, c = rng.choice((1, 2, 3)), rng.choice((1, 2, 3))
            a = -rng.randrange(1, b+c)
            d, e, f = (rng.randrange(-3, 4) for _ in range(3))
            if not e and not f:
                e = 1
            first = [[0.]*3 for _ in range(3)]
            first[0][0], first[1][0], first[0][1] = a, b, c
            second = [[0.]*3 for _ in range(3)]
            for i, j, value in ((0, 0, a), (1, 0, b), (0, 1, c)):
                for k, l, factor in ((0, 0, d), (1, 0, e), (0, 1, f)):
                    second[i+k][j+l] += value*factor
            with self.assertRaises(ClippingConvergenceError):
                bilinear_clip_roots(BiBernsteinPolynomial.from_power(first),
                                    BiBernsteinPolynomial.from_power(second), tolerance=.02)

    def test_shared_nonlinear_sign_changing_factor_is_unresolved(self):
        # p=(x-.5)^2+(y-.5)^2-1/8; q=9*(x+y)*p.  Their common
        # roots contain a circle entirely inside the unit square.
        p = [[0.]*3 for _ in range(3)]
        p[0][0], p[1][0], p[0][1], p[2][0], p[0][2] = .375, -1., -1., 1., 1.
        q = [[0.]*4 for _ in range(4)]
        for i in range(3):
            for j in range(3):
                q[i+1][j] += 9*p[i][j]
                q[i][j+1] += 9*p[i][j]
        first = BiBernsteinPolynomial.from_power(p)
        second = BiBernsteinPolynomial.from_power(q)
        for a, b in ((first, second), (second, first)):
            with self.assertRaisesRegex(ClippingConvergenceError, "shared sign-changing factor"):
                bilinear_clip_roots(a, b, tolerance=.05)
        q[0][0] = .125
        shifted = BiBernsteinPolynomial.from_power(q)
        self.assertEqual(bilinear_clip_roots(first, shifted, tolerance=1e-5), [])
        # The only zero of this divisor is an isolated point, so divisibility
        # alone does not prove a curve of solutions.
        p[0][0] = .5
        q = [[0.]*4 for _ in range(4)]
        for i in range(3):
            for j in range(3):
                q[i+1][j] += 9*p[i][j]
                q[i][j+1] += 9*p[i][j]
        first = BiBernsteinPolynomial.from_power(p)
        second = BiBernsteinPolynomial.from_power(q)
        roots = bilinear_clip_roots(first, second, tolerance=1e-4)
        self.assertTrue(any(r.x0 <= .5 <= r.x1 and r.y0 <= .5 <= r.y1 for r in roots))

    def test_shared_nonlinear_factor_when_neither_polynomial_divides(self):
        h = [[0.]*3 for _ in range(3)]
        h[0][0], h[1][0], h[0][1], h[2][0], h[0][2] = .375, -1., -1., 1., 1.
        def multiply_factor(a, b):
            power = [[0.]*4 for _ in range(4)]
            for i in range(3):
                for j in range(3):
                    power[i+1][j] += 9*a*h[i][j]
                    power[i][j+1] += 9*b*h[i][j]
            return power
        p, q = multiply_factor(1, 1), multiply_factor(1, -1)
        first, second = BiBernsteinPolynomial.from_power(p), BiBernsteinPolynomial.from_power(q)
        with self.assertRaisesRegex(ClippingConvergenceError, "shared factor"):
            bilinear_clip_roots(first, second, tolerance=.05)
        # Moving one polynomial by a constant removes the shared curve.
        q[0][0] = 1.
        shifted = BiBernsteinPolynomial.from_power(q)
        self.assertEqual(bilinear_clip_roots(first, shifted, tolerance=1e-5), [])

    def test_small_shared_quadratic_curve_between_sampling_lines_is_unresolved(self):
        def make(center_x, center_y, radius, affine, shift=Fraction(0)):
            h = {(0, 0): center_x**2+center_y**2-radius**2,
                 (1, 0): -2*center_x, (0, 1): -2*center_y,
                 (2, 0): Fraction(1), (0, 2): Fraction(2)}
            # Ellipse uses 2*(y-center_y)^2.
            h[0, 0] += center_y**2
            h[0, 1] *= 2
            power = [[Fraction(0)]*4 for _ in range(4)]
            for (i, j), value in h.items():
                for (di, dj), coefficient in zip(((0, 0), (1, 0), (0, 1)), affine):
                    power[i+di][j+dj] += coefficient*value
            power[0][0] += shift
            return BiBernsteinPolynomial.from_power([[float(value) for value in row] for row in power])
        rng = random.Random(990431)
        for index in range(25):
            x, y = Fraction(rng.randrange(2, 15), 16), Fraction(rng.randrange(2, 15), 16)
            radius = Fraction(1, 4096)
            first = make(x, y, radius, (9, 9, 9))
            second = make(x, y, radius, (9, 9, -9))
            with self.subTest(index=index):
                self.assertTrue(_shared_quadratic_with_affine_cofactors(first, second))
                with self.assertRaisesRegex(ClippingConvergenceError, "quadratic root curve"):
                    bilinear_clip_roots(first, second, eps=0., tolerance=.01)
                shifted = make(x, y, radius, (9, 9, -9), Fraction(1, 8))
                self.assertFalse(_shared_quadratic_with_affine_cofactors(first, shifted))
        # A shared quadratic with a single real zero is not a root curve.
        isolated_first = make(Fraction(1, 2), Fraction(3, 4), Fraction(0), (9, 9, 9))
        isolated_second = make(Fraction(1, 2), Fraction(3, 4), Fraction(0), (9, 9, -9))
        self.assertFalse(_shared_quadratic_with_affine_cofactors(isolated_first, isolated_second))

        # AnaMorph elevates its intersection systems to degree 5 or 7 in
        # each axis. Build exactly representable elevated Bernstein inputs.
        from math import comb
        for degree, multiple in ((5, 100), (7, 11025)):
            def elevated(sign):
                x, y, radius = Fraction(1, 2), Fraction(3, 4), Fraction(1, 1024)
                h = {(0, 0): x*x+y*y-radius*radius, (1, 0): -2*x, (0, 1): -2*y,
                     (2, 0): Fraction(1), (0, 2): Fraction(1)}
                power = [[Fraction(0)]*(degree+1) for _ in range(degree+1)]
                for (i, j), value in h.items():
                    for (di, dj), factor in zip(((0, 0), (1, 0), (0, 1)), (multiple, multiple, sign*multiple)):
                        power[i+di][j+dj] += factor*value
                coefficients = [[sum((power[k][l]*Fraction(comb(i, k), comb(degree, k))*
                                      Fraction(comb(j, l), comb(degree, l))
                                      for k in range(i+1) for l in range(j+1)), Fraction(0))
                                 for j in range(degree+1)] for i in range(degree+1)]
                self.assertTrue(all(Fraction(float(value)) == value for row in coefficients for value in row))
                return BiBernsteinPolynomial(tuple(tuple(float(value) for value in row) for row in coefficients))
            with self.subTest(elevated_degree=degree):
                first, second = elevated(1), elevated(-1)
                self.assertTrue(_shared_quadratic_with_affine_cofactors(first, second))
                with self.assertRaisesRegex(ClippingConvergenceError, "quadratic root curve"):
                    bilinear_clip_roots(first, second, eps=0., tolerance=.01)

    def test_exact_unit_interval_root_count_handles_tangencies(self):
        def fractions(values):
            return [Fraction(value) for value in values]
        for polynomial, expected in (([1, 0, 1], False), ([-1, 1], True),
                                     ([0, 1], True), ([Fraction(1, 4), -1, 1], True),
                                     ([-2, 0, 1], False), ([-Fraction(1, 2), 0, 1], True)):
            with self.subTest(polynomial=polynomial):
                self.assertEqual(_has_unit_interval_root(fractions(polynomial)), expected)

    def test_shared_horizontal_factor_without_whole_polynomial_divisibility(self):
        # (y-.5)(x+y) and (y-.5)(x-y) share a horizontal line.
        p = BiBernsteinPolynomial.from_power(((0., -.5, 1.), (-.5, 1., 0.)))
        q = BiBernsteinPolynomial.from_power(((0., .5, -1.), (-.5, 1., 0.)))
        with self.assertRaisesRegex(ClippingConvergenceError, "shared factor"):
            bilinear_clip_roots(p, q, tolerance=.05)

    def test_bivariate_clipping_contains_constructed_roots(self):
        rng = random.Random(2026)
        for _ in range(30):
            x, y = rng.uniform(.05, .95), rng.uniform(.05, .95)
            a, b, c, d = (rng.uniform(-.7, .7) for _ in range(4))
            p = BiBernsteinPolynomial.from_power(((-x-a*y+c*x*x, a), (1-2*c*x, 0), (c, 0)))
            q = BiBernsteinPolynomial.from_power(((-y-b*x+d*y*y, 1-2*d*y, d), (b, 0, 0)))
            roots = bilinear_clip_roots(p, q, tolerance=1e-6)
            self.assertTrue(any(r.x0-1e-7 <= x <= r.x1+1e-7 and r.y0-1e-7 <= y <= r.y1+1e-7
                                for r in roots), (x, y, roots))

    def test_zero_epsilon_bilinear_clipping_keeps_exact_dyadic_roots(self):
        rng = random.Random(871230)
        cases = [(.140625, .296875, (1, 3, 4, -9, 8, 0))]
        while len(cases) < 151:
            x, y = rng.randrange(1, 64)/64, rng.randrange(1, 64)/64
            values = tuple(rng.randrange(-9, 10) for _ in range(6))
            if values[0]*values[4] != values[1]*values[3]:
                cases.append((x, y, values))
        def make(a, b, c, x, y, scale):
            return BiBernsteinPolynomial.from_power(
                ((scale*(c*x*y-a*x-b*y), scale*(b-c*x)),
                 (scale*(a-c*y), scale*c)))
        for x, y, (a, b, c, d, e, f) in cases:
            for scale in (2.**-400, 1., 2.**400):
                with self.subTest(x=x, y=y, scale=scale):
                    p, q = make(a, b, c, x, y, scale), make(d, e, f, x, y, scale)
                    roots = bilinear_clip_roots(p, q, tolerance=1e-7, eps=0., max_rectangles=3000)
                    self.assertTrue(any(r.x0-1e-12 <= x <= r.x1+1e-12 and
                                        r.y0-1e-12 <= y <= r.y1+1e-12 for r in roots), roots)
        # The inputs are finite, but the unscaled Legendre sums previously
        # overflowed and returned no roots at this magnitude.
        x, y = .140625, .296875
        p, q = make(1, 3, 4, x, y, 1e307), make(-9, 8, 0, x, y, 1e307)
        roots = bilinear_clip_roots(p, q, tolerance=1e-7, eps=0.)
        self.assertTrue(any(r.x0-1e-12 <= x <= r.x1+1e-12 and
                            r.y0-1e-12 <= y <= r.y1+1e-12 for r in roots), roots)

    def test_bivariate_unrepresentable_coefficient_ratio_is_unresolved(self):
        p = BiBernsteinPolynomial(((1e-300, -1e308), (1e308, 1e-300)))
        q = BiBernsteinPolynomial(((-1., 1.), (1., 0.)))
        with self.assertRaisesRegex(ClippingConvergenceError, "coefficient scale ratio"):
            bilinear_clip_roots(p, q, eps=0.)

    def test_bivariate_cubic_systems_contain_known_dyadic_roots(self):
        rng = random.Random(40741)
        for index in range(100):
            x, y = Fraction(rng.randrange(2, 15), 16), Fraction(rng.randrange(2, 15), 16)
            systems = []
            for _ in range(2):
                power = [[Fraction(rng.randrange(-16, 17), 16) for _ in range(4)] for _ in range(4)]
                power[0][0] = -sum((power[i][j]*x**i*y**j for i in range(4) for j in range(4) if i or j),
                                   Fraction(0))
                # All constructed coefficients and the selected root are
                # exactly representable as binary floats.
                systems.append(BiBernsteinPolynomial.from_power(tuple(tuple(float(value) for value in row)
                                                                     for row in power)))
            with self.subTest(index=index):
                roots = bilinear_clip_roots(*systems, tolerance=1e-6, max_rectangles=10000)
                self.assertTrue(any(r.x0-1e-8 <= float(x) <= r.x1+1e-8 and
                                    r.y0-1e-8 <= float(y) <= r.y1+1e-8 for r in roots), (x, y, roots))

    def test_bivariate_nearly_parallel_systems_retain_known_root(self):
        rng = random.Random(29055)
        for exponent in (3, 6, 9, 12, 15):
            for index in range(10):
                x, y = rng.randrange(2, 15)/16, rng.randrange(2, 15)/16
                slope, delta = rng.randrange(-16, 17)/16, 2.**-exponent
                first = BiBernsteinPolynomial.from_power(((-x-slope*y, slope), (1., 0.)))
                second = BiBernsteinPolynomial.from_power(((-x-(slope+delta)*y, slope+delta), (1., 0.)))
                with self.subTest(exponent=exponent, index=index):
                    roots = bilinear_clip_roots(first, second, tolerance=1e-5, max_rectangles=5000)
                    self.assertTrue(any(r.x0-1e-8 <= x <= r.x1+1e-8 and r.y0-1e-8 <= y <= r.y1+1e-8
                                        for r in roots), (x, y, roots))


if __name__ == "__main__":
    unittest.main()
