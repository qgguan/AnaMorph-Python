import random
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from anamorph import create_morphview, read_morphview, read_swc
from anamorph.analytic_network import _bounding_box, analyze_analytic_partial
from anamorph.analytic_mesher import _render_vector
from anamorph.canal_mesh import _segment_tangent
from anamorph.source_random import WindowsCRandom
from anamorph.source_analysis import trace_source_analysis
from anamorph.source_spline import (SourceCanalSegment, exact_source_distance_squared,
                                    source_root_cylinder, source_spline_controls)
from anamorph.source_jobs import _source_segment_box

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "source_spline_adjacent.swc"
MORPHVIEW_PROBE = ROOT / "reference-build" / "morphview_probe.exe"
JOBS_PROBE = ROOT / "reference-build" / "network_jobs_probe.exe"
OPTIONS_PROBE = ROOT / "reference-build" / "morphview_options_probe.exe"
OPTIONS_JOBS_PROBE = ROOT / "reference-build" / "network_jobs_options_probe.exe"
CYLINDER_PROBE = ROOT / "reference-build" / "root_cylinder_probe.exe"


class SourceSplineTests(unittest.TestCase):
    def test_root_cylinder_retains_source_endpoint(self):
        cylinder = source_root_cylinder((1e16, 0., 0.), (.01, 1., 0.), .1)
        self.assertEqual(cylinder.point(1.), (.01, 1., 0.))
        # Reconstructing the end from start + rounded displacement loses .01.
        self.assertEqual(cylinder.power[0][0] + cylinder.power[0][1], 0.)

    @unittest.skipUnless(sys.platform == "win32" and CYLINDER_PROBE.is_file(), "C++ cylinder probe unavailable")
    def test_cpp_root_cylinder_controls_and_bounds(self):
        rng = random.Random(42219)
        for case in range(30):
            start = tuple(rng.uniform(-100., 100.) for _ in range(3))
            end = tuple(rng.uniform(-100., 100.) for _ in range(3))
            radius = rng.uniform(.01, 2.)
            data = " ".join(format(x, ".17g") for x in (*start, *end, radius)) + "\n"
            with self.subTest(case=case):
                run = subprocess.run([str(CYLINDER_PROBE)], input=data, capture_output=True, text=True,
                                     check=True, timeout=10)
                rows = [line.split() for line in run.stdout.splitlines() if line.startswith("AXIS ")]
                self.assertEqual(len(rows), 3)
                cylinder = source_root_cylinder(start, end, radius)
                box = _source_segment_box(cylinder)
                for axis, row in enumerate(rows):
                    self.assertEqual(tuple(map(float, row[2:6])), cylinder.source_controls[axis])
                    self.assertEqual(tuple(map(float, row[6:8])), box[axis])
    def test_exact_distance_uses_source_controls(self):
        zero = ((0., 0., 0., 0.),) * 3
        shifted = ((.25, .25, .25, .25), zero[1], zero[2])
        first = SourceCanalSegment(zero, .1, .1, zero)
        second = SourceCanalSegment(zero, .1, .1, shifted)
        self.assertEqual(exact_source_distance_squared(first, second, .5, .5), Fraction(1, 16))

    def test_source_controls_drive_render_points_tangents_and_bounds(self):
        # Deliberately different power coefficients make it clear that the
        # source path evaluates and bounds the actual Bézier representation.
        zero = ((0., 0., 0., 0.),) * 3
        controls = ((2., 3., 4., 5.), (1., 1., 1., 1.), zero[2])
        segment = SourceCanalSegment(zero, .1, .2, controls)
        self.assertEqual(segment.point(.5), (3.5, 1., 0.))
        self.assertEqual(_segment_tangent(segment, .5), (3., 0., 0.))
        bounds = _bounding_box(segment)
        self.assertLess(bounds[0][0], 1.8)
        self.assertGreater(bounds[0][1], 5.2)
        with self.assertRaisesRegex(ValueError, "positive radii"):
            SourceCanalSegment(zero, 0., .2, controls)
        with self.assertRaisesRegex(ValueError, "finite cubic coordinates"):
            SourceCanalSegment(zero, .1, .2, ((float("nan"),)*4, controls[1], controls[2]))

    def test_source_controls_drive_frame_selection(self):
        local = ((0., 1., 0., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.))
        controls = ((0., 0., 0., 0.), (0., 0., 0., 0.), (0., 1., 2., 3.))
        segment = SourceCanalSegment(local, .1, .1, controls)
        # The local power curve runs along x; the rendered source curve runs
        # along z and therefore cannot use z as a stable frame vector.
        self.assertNotEqual(_render_vector([segment], rng=WindowsCRandom(0)), (0., 0., 1.))

    def test_chord_scale_uses_source_accumulation_order(self):
        # This coefficient was produced by the Windows reference. Python
        # 3.12's compensated sum(chords) changes it by several ulps.
        points = ((-2.843475241575015, -2.6869504831500297, 0.), (-2., -1., 0.),
                  (-3.050821902368754, -1.5121901238419977, -1.6345546214933946),
                  (-1.5096656456140651, -2.102346754543399, -3.193064536040806))
        direction = (0.15652475842498514, 0.3130495168499703, 0.)
        _, controls = source_spline_controls(points, start_direction=direction)
        self.assertEqual(controls[1][0][1], -2.089850317124398)

    def test_straight_segment_and_invalid_length(self):
        parameters, controls = source_spline_controls(((0., 0., 0.), (3., 0., 0.)))
        self.assertEqual(parameters, (0., 1.))
        self.assertEqual(controls[0], ((0., 1., 2., 3.), (0., 0., 0., 0.), (0., 0., 0., 0.)))
        for points in (((0., 0., 0.), (0., 0., 0.)),
                       ((0., 0., 0.), (1e200, 0., 0.)),
                       ((0., 0., 0.), (1e-200, 0., 0.))):
            with self.subTest(points=points), self.assertRaisesRegex(ValueError, "representable"):
                source_spline_controls(points)

    def test_sensitive_adjacent_source_decision(self):
        tree = read_swc(FIXTURE)
        trace = trace_source_analysis(tree, preconditioning=False)
        job, = (item for item in trace.jobs if item.job.kind == "NSNS_ADJ" and
                item.job.segments == ((5, 6), (5, 8)))
        self.assertIs(job.result, True)
        self.assertTrue(job.candidates)
        report = analyze_analytic_partial(tree, prepare=True, preconditioning=False,
                                          preserve_surface_roots=False, job_mode="source")
        self.assertTrue(any(finding.kind == "source_bivariate_candidate" and
                            finding.segments == ((5, 6), (5, 8)) for finding in report.findings))

    @unittest.skipUnless(sys.platform == "win32" and MORPHVIEW_PROBE.is_file() and JOBS_PROBE.is_file(),
                         "C++ reference probes unavailable")
    def test_sensitive_branch_controls_and_trace_match_cpp(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "reference.amv"
            subprocess.run([str(MORPHVIEW_PROBE), str(FIXTURE), str(target), "0"], cwd=directory,
                           capture_output=True, check=True, timeout=90)
            reference = read_morphview(target, windows_text_mode=True)
            model = create_morphview(read_swc(FIXTURE), preconditioning=False)
            # These are the two paths starting at node 5, whose 1e-15
            # coefficient differences changed a source solver decision.
            for path_index in (2, 3):
                self.assertEqual(model.paths[path_index].segments[0].coefficients,
                                 reference.paths[path_index].segments[0].coefficients)
            run = subprocess.run([str(JOBS_PROBE), str(FIXTURE), "0", "1"], cwd=directory,
                                 capture_output=True, text=True, check=True, timeout=90)
            expected = [(int(parts[1]), bool(int(parts[-1]))) for line in run.stdout.splitlines()
                        if (parts := line.split()) and parts[0] == "JOB"]
            kinds = {"REG": 0, "LSI": 1, "GSI": 2, "SONS": 3, "NSNS_ADJ": 4, "NSNS_NONADJ": 5}
            trace = trace_source_analysis(read_swc(FIXTURE), preconditioning=False)
            self.assertEqual([(kinds[item.job.kind], item.result) for item in trace.jobs], expected)

    @unittest.skipUnless(sys.platform == "win32" and OPTIONS_PROBE.is_file(),
                         "C++ parametrization probe unavailable")
    def test_cpp_spline_options_and_preconditioning(self):
        for source in (ROOT / "examples" / "branched.swc", FIXTURE):
            for parameter in ("chord-length", "centripetal", "uniform"):
                for precondition in (False, True):
                    with self.subTest(source=source.name, parameter=parameter, precondition=precondition), \
                            tempfile.TemporaryDirectory() as directory:
                        target = Path(directory) / "reference.amv"
                        subprocess.run([str(OPTIONS_PROBE), str(source), str(target),
                                        str(int(precondition)), parameter], cwd=directory,
                                       capture_output=True, check=True, timeout=90)
                        expected = read_morphview(target, windows_text_mode=True)
                        actual = create_morphview(read_swc(source), parametrization=parameter,
                                                  preconditioning=precondition)
                        self.assertEqual(len(expected.paths), len(actual.paths))
                        for reference_path, path in zip(expected.paths, actual.paths):
                            self.assertEqual(reference_path.parameters, path.parameters)
                            self.assertEqual(len(reference_path.segments), len(path.segments))
                            for reference_segment, segment in zip(reference_path.segments, path.segments):
                                self.assertEqual(reference_segment.coefficients, segment.coefficients)
                                self.assertEqual(reference_segment.flags, segment.flags)

    @unittest.skipUnless(sys.platform == "win32" and OPTIONS_JOBS_PROBE.is_file(),
                         "C++ parametrization job probe unavailable")
    def test_cpp_source_job_decisions_for_spline_options(self):
        kinds = {"REG": 0, "LSI": 1, "GSI": 2, "SONS": 3, "NSNS_ADJ": 4, "NSNS_NONADJ": 5}
        for parameter in ("chord-length", "centripetal", "uniform"):
            for precondition in (False, True):
                with self.subTest(parameter=parameter, precondition=precondition), \
                        tempfile.TemporaryDirectory() as directory:
                    run = subprocess.run([str(OPTIONS_JOBS_PROBE), str(FIXTURE), str(int(precondition)),
                                          "1", parameter], cwd=directory, capture_output=True,
                                         text=True, check=True, timeout=90)
                    expected = [(int(parts[1]), bool(int(parts[-1]))) for line in run.stdout.splitlines()
                                if (parts := line.split()) and parts[0] == "JOB"]
                    trace = trace_source_analysis(read_swc(FIXTURE), parametrization=parameter,
                                                  preconditioning=precondition)
                    self.assertFalse(trace.truncated)
                    self.assertFalse(trace.unresolved)
                    self.assertEqual([(kinds[item.job.kind], item.result) for item in trace.jobs], expected)

    @unittest.skipUnless(sys.platform == "win32" and OPTIONS_PROBE.is_file(),
                         "C++ parametrization probe unavailable")
    def test_cpp_seeded_branch_controls(self):
        rng = random.Random(2049481)
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "case.swc", Path(directory) / "case.amv"
            for case in range(6):
                rows = ["1 1 -3 -3 0 .35 -1", "2 3 -2 -1 0 .17 1"]
                positions = {2: (-2., -1., 0.)}
                for node in range(3, rng.randrange(6, 11)):
                    parent = rng.randrange(2, node) if rng.random() < .3 else node - 1
                    position = tuple(positions[parent][i] + rng.uniform(-1.3, 1.3) for i in range(3))
                    positions[node] = position
                    rows.append(f"{node} 3 {position[0]:.17g} {position[1]:.17g} {position[2]:.17g} "
                                f"{rng.uniform(.07, .4):.17g} {parent}")
                source.write_text("\n".join(rows) + "\n", encoding="utf-8")
                for parameter in ("chord-length", "centripetal", "uniform"):
                    with self.subTest(case=case, parameter=parameter):
                        subprocess.run([str(OPTIONS_PROBE), str(source), str(target), "0", parameter],
                                       cwd=directory, capture_output=True, check=True, timeout=90)
                        reference = read_morphview(target, windows_text_mode=True)
                        actual = create_morphview(read_swc(source), preconditioning=False,
                                                  parametrization=parameter)
                        self.assertEqual(len(reference.paths), len(actual.paths))
                        for a, b in zip(reference.paths, actual.paths):
                            self.assertEqual(a.parameters, b.parameters)
                            self.assertEqual(len(a.segments), len(b.segments))
                            for x, y in zip(a.segments, b.segments):
                                self.assertEqual(x.coefficients, y.coefficients)
                                self.assertEqual(x.flags, y.flags)


if __name__ == "__main__":
    unittest.main()
