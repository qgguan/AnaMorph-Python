import random
import subprocess
import sys
import unittest
from pathlib import Path

from anamorph.canal_mesh import _source_arc_length_function
from anamorph.source_spline import SourceCanalSegment

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / "reference-build" / "arc_radius_probe.exe"


class SourceArcLengthTests(unittest.TestCase):
    def test_unrepresentable_source_distances_fail_explicitly(self):
        zero = ((0., 0., 0., 0.),) * 3
        for scale in (1e-200, 1e200):
            segment = SourceCanalSegment(zero, .1, .2, ((0., scale, scale, scale), zero[1], zero[2]))
            with self.subTest(scale=scale), self.assertRaisesRegex(ValueError, "not representable"):
                _source_arc_length_function(segment, .01, windows_x87=True)
        segment = SourceCanalSegment(zero, .1, .2, zero)
        for step in (0., -1., float("inf")):
            with self.subTest(step=step), self.assertRaisesRegex(ValueError, "sampling settings"):
                _source_arc_length_function(segment, step, windows_x87=True)

    @unittest.skipUnless(sys.platform == "win32" and PROBE.is_file(), "C++ arc-length probe unavailable")
    def test_cpp_arc_and_radius_arithmetic(self):
        rng = random.Random(273521)
        zero = ((0., 0., 0., 0.),) * 3
        times = (.011, .037, .113, .4, .543, .9, 1.)
        for case in range(30):
            with self.subTest(case=case):
                controls = tuple(tuple(rng.uniform(-3., 3.) for _ in range(4)) for _ in range(3))
                radius0, radius1 = rng.uniform(.1, .8), rng.uniform(.1, .8)
                data = " ".join(format(value, ".17g") for component in controls for value in component)
                data += f" {radius0:.17g} {radius1:.17g}\n"
                data += " ".join(format(time, ".17g") for time in times) + "\n"
                run = subprocess.run([str(PROBE)], input=data, capture_output=True, text=True,
                                     check=True, timeout=10)
                lines = [parts for line in run.stdout.splitlines() if (parts := line.split())
                         and parts[0] in ("TOTAL", "PARAM")]
                self.assertEqual(len(lines), len(times) + 1)
                segment = SourceCanalSegment(zero, radius0, radius1, controls)
                arc = _source_arc_length_function(segment, .001, windows_x87=True)
                radius_arc = _source_arc_length_function(segment, .01, windows_x87=True)
                total = radius_arc(1.)
                actual = [(arc(1.), total)]
                for time in times:
                    partial = radius_arc(time)
                    ratio = min(1., max(0., partial/total))
                    actual.append((time, arc(time), partial, radius0+(radius1-radius0)*ratio))
                for parts, values in zip(lines, actual):
                    self.assertEqual(tuple(map(float, parts[1:])), values)


if __name__ == "__main__":
    unittest.main()
