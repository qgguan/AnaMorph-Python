import unittest
from anamorph.polynomial import evaluate, multiply, roots_unit_interval


class PolynomialTests(unittest.TestCase):
    def test_roots_include_boundaries_and_double_root(self):
        p = multiply(multiply([0, 1], [-1, 1]), multiply([-.3, 1], [-.3, 1]))
        roots = roots_unit_interval(p)
        for expected in (0, .3, 1):
            self.assertTrue(any(abs(x-expected) < 1e-7 for x in roots), roots)
        self.assertTrue(all(abs(evaluate(p, x)) < 1e-8 for x in roots))

    def test_roots_empty_and_linear(self):
        self.assertEqual(roots_unit_interval([2.0]), [])
        self.assertEqual(roots_unit_interval([1, 0, 1]), [])
        self.assertAlmostEqual(roots_unit_interval([-.25, 1])[0], .25, places=7)


if __name__ == "__main__":
    unittest.main()
