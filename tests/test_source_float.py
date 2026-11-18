import math
import random
import struct
import subprocess
import sys
import unittest
from pathlib import Path

from anamorph.source_float import x87_sincos, x87_sqrt

PROBE = Path(__file__).resolve().parent.parent / "reference-build" / "sqrt_probe.exe"
TRIG_PROBE = PROBE.parent / "trig_probe.exe"


def bits(value):
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def from_bits(value):
    return struct.unpack("<d", struct.pack("<Q", value))[0]


class SourceFloatTests(unittest.TestCase):
    def test_render_trig_known_case_and_fallback(self):
        value = 5.487951789725289
        self.assertEqual(x87_sincos(value)[0], -0.7140271141857651)
        # Host libm may agree with the x87-style result on some platforms.
        self.assertEqual(bits(x87_sincos(0.)[0]), bits(0.))
        self.assertEqual(bits(x87_sincos(-0.)[0]), bits(-0.))
        self.assertEqual(x87_sincos(100.), (math.sin(100.), math.cos(100.)))
        with self.assertRaises(ValueError):
            x87_sincos(float("nan"))

    def test_double_rounding_and_input_validation(self):
        almost_one = math.nextafter(1., 0.)
        self.assertEqual(x87_sqrt(almost_one), 1.)
        self.assertNotEqual(x87_sqrt(almost_one), math.sqrt(almost_one))
        self.assertEqual(bits(x87_sqrt(-0.)), bits(-0.))
        for value in (-1., float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                x87_sqrt(value)

    @unittest.skipUnless(sys.platform == "win32" and PROBE.is_file(), "optional C++ square-root probe unavailable")
    def test_x87_sqrt_matches_cpp_reference(self):
        rng = random.Random(851550)
        values = [math.nextafter(1., 0.), 2.8059324805599006, from_bits(1), sys.float_info.max]
        for _ in range(2000):
            candidate = rng.getrandbits(63)
            if 0 < candidate < 0x7ff0000000000000:
                values.append(from_bits(candidate))
        input_text = "".join(f"{bits(value):x}\n" for value in values)
        run = subprocess.run([str(PROBE)], input=input_text, capture_output=True, text=True, check=True, timeout=30)
        expected = [int(line, 16) for line in run.stdout.splitlines()]
        self.assertEqual([bits(x87_sqrt(value)) for value in values], expected)

    @unittest.skipUnless(sys.platform == "win32" and TRIG_PROBE.is_file(), "optional C++ trigonometry probe unavailable")
    def test_render_trig_approximates_cpp_reference(self):
        rng = random.Random(33810)
        values = [5.487951789725289]+[rng.uniform(-8, 8) for _ in range(3000)]
        run = subprocess.run([str(TRIG_PROBE)], input="".join(f"{bits(x):x}\n" for x in values),
                             capture_output=True, text=True, check=True, timeout=30)
        expected = [tuple(int(v, 16) for v in line.split()) for line in run.stdout.splitlines()]
        mismatches = 0
        for value, pair in zip(values, expected):
            actual = tuple(map(bits, x87_sincos(value)))
            for observed, reference in zip(actual, pair):
                if observed != reference:
                    mismatches += 1
                    self.assertEqual(abs(observed-reference), 1)
        self.assertEqual(len(expected), len(values))
        # This is a high-precision approximation of the source libm, not a
        # bit-exact port; a few one-ulp differences are expected.
        self.assertLessEqual(mismatches, 5)


if __name__ == "__main__":
    unittest.main()
