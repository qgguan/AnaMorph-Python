import math
import json
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from anamorph import Node, Morphology, read_obj, read_swc
from anamorph.geometry import Spline, _unit, order_paths_breadth_first, partition_paths
from anamorph.voxel import generate_voxel_mesh
from anamorph.precondition import precondition, _precondition_length
from anamorph.smoothing import hc_smooth, laplacian_smooth
from anamorph.roots import adjust_soma_roots, normalize_soma_sections
from anamorph.analysis import _numeric_curve_checks, approximate_analysis
from anamorph.collapse import (greedy_edge_collapse, _SourcePriorityQueue,
                              _source_face_neighborhood, _source_neighborhood_average)
from anamorph.remesh import isotropic_remesh
from anamorph.mesh import Mesh
from anamorph.model import prepare_morphology


class GeometryTests(unittest.TestCase):
    def test_source_precondition_norm_preserves_extreme_scales(self):
        for scale in (1e-200, 1e200):
            for mode in ("source", "source-x87"):
                with self.subTest(scale=scale, mode=mode):
                    self.assertEqual(_precondition_length((0., 0., 0.), (scale, scale, 0.), mode),
                                     math.dist((0., 0., 0.), (scale, scale, 0.)))

    def test_individual_surface_command_uses_precondition_norm(self):
        from anamorph.analytic_mesher import render_individual_surfaces
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/"threshold.swc"
            source.write_text("1 1 1 2 2 .1 -1\n2 3 1 2 3 .18266480860323492 1\n"
                              "3 3 -.2582743098573599 1.6248239331868253 1.7301055515322439 "
                              ".18266480860323492 2\n", encoding="utf-8")
            counts = []
            for mode in ("robust", "source-x87"):
                output = root/(mode+".obj")
                run = subprocess.run([sys.executable, "-m", "anamorph", "generate-individual", str(source),
                                      str(output), "--source-surface-roots", "--soma-refinements", "0",
                                      "--precondition-norm-mode", mode], text=True, capture_output=True, timeout=30)
                self.assertEqual(run.returncode, 0, run.stderr)
                actual = read_obj(output)
                expected = render_individual_surfaces(read_swc(source), soma_refinements=0,
                                                      preserve_surface_roots=False, precondition_norm_mode=mode)
                self.assertEqual((len(actual.vertices), len(actual.faces)), (len(expected.vertices), len(expected.faces)))
                counts.append(len(actual.vertices))
            self.assertNotEqual(*counts)

    def setUp(self):
        self.tree = Morphology([
            Node(1, 1, 0, 0, 0, 1, -1),
            Node(2, 3, 1, 0, 0, .25, 1),
            Node(3, 3, 2, 0, 0, .25, 2),
            Node(4, 3, 3, 0, 0, .25, 3),
            Node(5, 3, 2, 1, 0, .25, 3),
            Node(6, 3, 4, 0, 0, .25, 4),
        ])

    def test_partition_covers_edges_exactly_once(self):
        paths = partition_paths(self.tree)
        self.assertIn((2, 3, 4, 6), paths)
        self.assertIn((3, 5), paths)
        edges = [(a, b) for p in paths for a, b in zip(p, p[1:])]
        self.assertEqual(set(edges), {(a.id, b.id) for a, b in self.tree.neurite_segments()})
        self.assertEqual(len(edges), len(set(edges)))
        self.assertIn((2, 3), partition_paths(self.tree, "simple-neurite-paths"))

    def test_nearly_parallel_branch_angles_follow_source_snap_and_ties(self):
        tree = Morphology([Node(1, 1, -2., 0., 0., .25, -1), Node(2, 3, -1., 0., 0., .05, 1),
                           Node(3, 3, 0., 0., 0., .05, 2), Node(4, 3, 1., 2e-7, 0., .05, 3),
                           Node(5, 3, 1., -1e-7, 0., .05, 3)])
        for strategy in ("max-chordal-depth", "min-angle"):
            for angle in (1e-5, 90.):
                with self.subTest(strategy=strategy, angle=angle):
                    self.assertEqual(partition_paths(tree, strategy, filter_angle=angle),
                                     [(2, 3, 4), (3, 5)])
            self.assertEqual(partition_paths(tree, strategy, filter_angle=0.), [(2, 3), (3, 4), (3, 5)])

    def test_partition_filter_degree_conversion_matches_source_at_boundary(self):
        tree = Morphology([Node(1, 1, -2., 0., 0., .25, -1), Node(2, 3, -1., 0., 0., .05, 1),
                           Node(3, 3, 0., 0., 0., .05, 2),
                           Node(4, 3, .5652931960631212, -.5610766494745429, 0., .05, 3),
                           Node(5, 3, 0., 1., 0., .05, 3)])
        angle = 44.785515214127635
        for strategy in ("max-chordal-depth", "min-angle"):
            with self.subTest(strategy=strategy):
                self.assertEqual(partition_paths(tree, strategy, filter_angle=angle), [(2, 3, 4), (3, 5)])

    def test_spline_and_path_partition_are_stable_across_coordinate_scales(self):
        points = ((0., 0., 0.), (1., 0., 0.), (2., .5, 0.), (3., .2, 0.))
        baseline = Spline.interpolate(points)
        baseline_paths = partition_paths(self.tree)
        for scale in (1e-200, 1., 1e200):
            with self.subTest(scale=scale):
                spline = Spline.interpolate(tuple(tuple(scale*x for x in point) for point in points))
                tree = Morphology([Node(node.id, node.kind, node.x*scale, node.y*scale, node.z*scale,
                                        node.radius*scale, node.parent) for node in self.tree.nodes.values()])
                self.assertEqual(partition_paths(tree), baseline_paths)
                for segment in range(len(points)-1):
                    for t in (0., .25, .5, .75, 1.):
                        actual = spline.evaluate_segment(segment, t)
                        expected = baseline.evaluate_segment(segment, t)
                        self.assertLess(math.dist(tuple(value/scale for value in actual), expected), 1e-12)
        for scale in (1e-320, 1.3e308):
            self.assertAlmostEqual(math.hypot(*_unit((scale, scale, 0.))), 1.)

    def test_unrepresentable_spline_derivatives_are_reported(self):
        points = ((0., 0., 0.), (8e307, 0., 0.), (0., 0., 0.), (8e307, 0., 0.))
        with self.assertRaisesRegex(ValueError, "spline endpoint derivative exceeds float range"):
            Spline.interpolate(points)

    def test_precondition_subdivision_penalty_is_scale_invariant(self):
        for radius0, radius1, expected_segments in ((.1, .1, 20), (.1, .2, 13), (.2, .1, 13)):
            outputs = []
            for scale in (1e-200, 1., 1e200):
                with self.subTest(radius0=radius0, radius1=radius1, scale=scale):
                    tree = Morphology([Node(1, 1, 0., 0., 0., scale, -1),
                                       Node(2, 3, scale, 0., 0., radius0*scale, 1),
                                       Node(3, 3, 21*scale, 0., 0., radius1*scale, 2)])
                    result = precondition(tree)
                    segments = list(result.morphology.neurite_segments())
                    self.assertEqual((len(segments), result.split_segments, result.collapsed_segments),
                                     (expected_segments, 1, 0))
                    outputs.append(sorted((node.x/scale, node.radius/scale)
                                          for node in result.morphology.nodes.values() if node.kind == 3))
            for actual in outputs[1:]:
                for point, expected in zip(actual, outputs[0]):
                    self.assertLess(math.dist(point, expected), 1e-12)

    def test_precondition_reports_unrepresentable_split_requirements(self):
        tree = Morphology([Node(1, 1, 0., 0., 0., 1., -1), Node(2, 3, 0., 0., 0., 1e-320, 1),
                           Node(3, 3, 1., 0., 0., 1e-320, 2)])
        with self.assertRaisesRegex(ValueError, "subdivisions, exceeding the limit"):
            precondition(tree, max_splits_per_segment=100)
        tree = Morphology([Node(1, 1, 0., 0., 0., 1., -1), Node(2, 3, -1e308, 0., 0., 1., 1),
                           Node(3, 3, 1e308, 0., 0., 1., 2)])
        with self.assertRaisesRegex(ValueError, "length exceeds float range"):
            precondition(tree)

    def test_precondition_avoids_overflow_in_neighbor_threshold_and_collapse_midpoint(self):
        # beta*(ru+rv) is finite mathematically even though ru+rv overflows.
        tree = Morphology([Node(1, 1, -3., 0., 0., 1., -1)]+
                          [Node(i, 3, 3.*(i-2), 0., 0., 1e308, i-1) for i in range(2, 6)])
        result = precondition(tree, alpha=1e-308, beta=1e-308, gamma=1.)
        self.assertEqual((result.split_segments, result.collapsed_segments), (0, 0))
        # A required middle-edge collapse has representable averages although
        # the direct coordinate and radius sums overflow.
        tree = Morphology([Node(1, 1, 0., 0., 0., 1., -1), Node(2, 3, 5e307, 0., 0., 1e308, 1),
                           Node(3, 3, 9e307, 0., 0., 1e308, 2), Node(4, 3, 1e308, 0., 0., 1e308, 3),
                           Node(5, 3, 1.4e308, 0., 0., 1e308, 4)])
        result = precondition(tree, alpha=.2, beta=.01, gamma=1.)
        self.assertEqual(result.collapsed_segments, 1)
        self.assertEqual(result.morphology.nodes[3].x, 9.5e307)
        self.assertEqual(result.morphology.nodes[3].radius, 1e308)

    def test_partition_groups_paths_by_neurite_root(self):
        tree = Morphology([Node(1, 1, 0, 0, 0, 1, -1), Node(2, 3, 1, 0, 0, .1, 1),
                           Node(3, 3, 2, 0, 0, .1, 2), Node(4, 3, 3, 0, 0, .1, 3),
                           Node(5, 3, 2, 1, 0, .1, 3), Node(6, 3, -1, 0, 0, .1, 1),
                           Node(7, 3, -2, 0, 0, .1, 6), Node(8, 3, -3, 0, 0, .1, 7),
                           Node(9, 3, -2, -1, 0, .1, 7)])
        self.assertEqual(partition_paths(tree), [(2, 3, 4), (3, 5), (6, 7, 8), (7, 9)])

    def test_laplacian_smoothing_uses_simultaneous_updates_and_keeps_isolated_vertices(self):
        mesh = Mesh(((0., 0., 0.), (2., 0., 0.), (0., 2., 0.), (9., 9., 9.)), ((0, 1, 2),))
        smoothed = laplacian_smooth(mesh, smoothing_factor=.5, iterations=1)
        self.assertEqual(smoothed.vertices, ((.5, .5, 0.), (1., .5, 0.), (.5, 1., 0.), (9., 9., 9.)))
        self.assertEqual(smoothed.faces, mesh.faces)

    def test_preconditioning_preserves_edge_order_for_branch_ties(self):
        tree = read_swc(Path(__file__).parent / "fixtures" / "branch_tie.swc")
        prepared = prepare_morphology(tree).morphology
        self.assertEqual(prepared.children[4], (10, 6))
        self.assertIn((12, 3, 4, 10), partition_paths(prepared))
        # Radius scaling and root readjustment must not erase that ordering.
        scaled = prepare_morphology(tree, scale_radius=.9).morphology
        self.assertEqual(scaled.children[4], (10, 6))

    def test_radius_scaling_repositions_roots_after_preconditioning(self):
        tree = Morphology([Node(1, 1, 0, 0, 0, 1, -1),
                           Node(2, 3, 1.5, 0, 0, .2, 1),
                           Node(3, 3, 2.5, 0, 0, .2, 2)])
        result = prepare_morphology(tree, preconditioning=False, scale_radius=.5).morphology
        self.assertAlmostEqual(result.nodes[1].radius, .5)
        self.assertAlmostEqual(result.nodes[2].radius, .1)
        soma_children = result.children[1]
        self.assertEqual(len(soma_children), 1)
        new_root = result.nodes[soma_children[0]]
        self.assertAlmostEqual(math.dist(new_root.position, result.nodes[1].position), .5)
        self.assertAlmostEqual(new_root.radius, .1)
        self.assertEqual(tree.nodes[1].radius, 1.)
        with self.assertRaisesRegex(ValueError, "radius scale"):
            prepare_morphology(tree, scale_radius=.09)

    def test_path_tree_breadth_first_order_differs_from_vertex_depth_order(self):
        nodes = [Node(1, 1, 0, 0, 0, 1, -1)]
        nodes.extend(Node(i, 3, i, 0, 0, .1, i-1) for i in range(2, 11))
        nodes.extend([Node(11, 3, 3, 1, 0, .1, 3), Node(12, 3, 3, 2, 0, .1, 11),
                      Node(13, 3, 3, 1, 1, .1, 11), Node(14, 3, 9, 1, 0, .1, 9)])
        tree = Morphology(nodes)
        main = (2, 3, 4, 5, 6, 7, 8, 9, 10)
        shallow_branch, grandchild, distant_branch = (3, 11, 12), (11, 13), (9, 14)
        self.assertEqual(order_paths_breadth_first(tree, [main, shallow_branch, grandchild, distant_branch]),
                         [main, shallow_branch, distant_branch, grandchild])

    def test_spline_interpolates_and_is_linear_for_collinear_points(self):
        points = [(0., 0., 0.), (1., 0., 0.), (3., 0., 0.)]
        for strategy in ("chord-length", "centripetal", "uniform"):
            curve = Spline.interpolate(points, parametrization=strategy)
            for i in range(2):
                self.assertEqual(curve.evaluate_segment(i, 0), points[i])
                self.assertEqual(curve.evaluate_segment(i, 1), points[i+1])
                self.assertAlmostEqual(curve.evaluate_segment(i, .5)[1], 0)
        curve = Spline.interpolate(points)
        self.assertAlmostEqual(curve.evaluate_segment(1, .5)[0], 2.)
        velocity, acceleration = curve.derivatives_segment(1, .5)
        self.assertAlmostEqual(velocity[0], 2.)
        self.assertAlmostEqual(acceleration[0], 0.)
        coeff = curve.power_coefficients(1)
        self.assertAlmostEqual(sum(coeff[0][i]*(.5**i) for i in range(4)), 2.)
        # Match the checked-in uniform-parametrization implementation, which
        # divides interior indices by the number of vertices, not segments.
        self.assertEqual(Spline.interpolate(points, parametrization="uniform").parameters, (0., 1/3, 1.))

    def test_spline_has_c2_continuity_across_nonuniform_segments(self):
        points = [(0.,0.,0.), (1.,.2,.1), (2.,1.5,.4), (4.,2.,-.3)]
        for parameterization in ("chord-length", "centripetal", "uniform"):
            curve = Spline.interpolate(points, parametrization=parameterization)
            for i in range(len(points)-2):
                first_l, second_l = curve.derivatives_segment(i, 1.)
                first_r, second_r = curve.derivatives_segment(i+1, 0.)
                h_l = curve.parameters[i+1]-curve.parameters[i]
                h_r = curve.parameters[i+2]-curve.parameters[i+1]
                for axis in range(3):
                    self.assertAlmostEqual(first_l[axis]/h_l, first_r[axis]/h_r, places=9)
                    self.assertAlmostEqual(second_l[axis]/h_l**2, second_r[axis]/h_r**2, places=9)

    def test_sphere_voxel_mesh_is_closed_and_approximately_spherical(self):
        sphere = Morphology([Node(1, 1, 0, 0, 0, 1, -1)])
        mesh = generate_voxel_mesh(sphere, .3)
        stats = mesh.statistics()
        self.assertEqual(stats.boundary_edges, 0)
        self.assertEqual(stats.nonmanifold_edges, 0)
        self.assertEqual(stats.euler_characteristic, 2)
        self.assertAlmostEqual(stats.volume, 4*math.pi/3, delta=.6)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"sphere.obj"
            mesh.write_obj(path)
            self.assertEqual(read_obj(path).statistics().faces, stats.faces)

    def test_branched_mesh_and_grid_limit(self):
        mesh = generate_voxel_mesh(self.tree, .2)
        stats = mesh.statistics()
        self.assertEqual(stats.boundary_edges, 0)
        self.assertEqual(stats.nonmanifold_edges, 0)
        self.assertEqual(stats.euler_characteristic, 2)
        with self.assertRaisesRegex(ValueError, "grid requires"):
            generate_voxel_mesh(self.tree, .01, max_grid_points=1000)

    def test_voxel_grid_near_surface_grid_points_has_no_zero_area_faces(self):
        tree = Morphology([Node(1, 1, 0, 0, 0, 1, -1),
                           Node(2, 3, 1.25, 0, 0, .25, 1),
                           Node(3, 3, 2.25, 0, 0, .23, 2),
                           Node(4, 3, 3.25, 0, 0, .20, 3),
                           Node(5, 3, 4.25, .4, 0, .16, 4),
                           Node(6, 3, 3.6, -.9, 0, .15, 4),
                           Node(7, 3, 4., -1.7, 0, .12, 6)])
        stats = generate_voxel_mesh(tree, .04).statistics()
        self.assertEqual((stats.boundary_edges, stats.nonmanifold_edges, stats.euler_characteristic), (0, 0, 2))

    def test_voxel_grid_rejects_insufficient_coordinate_precision(self):
        sphere = Morphology([Node(1, 1, 1e12, 0, 0, 1, -1)])
        with self.assertRaisesRegex(ValueError, 'coordinate precision'):
            generate_voxel_mesh(sphere, .04)

    def test_hc_smoothing_preserves_topology_and_does_not_mutate_input(self):
        sphere = Morphology([Node(1, 1, 0, 0, 0, 1, -1)])
        mesh = generate_voxel_mesh(sphere, .4)
        smoothed = hc_smooth(mesh, iterations=2)
        self.assertEqual(mesh.faces, smoothed.faces)
        self.assertNotEqual(mesh.vertices, smoothed.vertices)
        self.assertEqual(smoothed.statistics().boundary_edges, 0)
        self.assertEqual(hc_smooth(mesh, iterations=0), mesh)

    def test_preconditioning_preserves_original_and_collapses_short_edges(self):
        tree = Morphology([Node(1, 1, 0, 0, 0, 1, -1), Node(2, 3, 1, 0, 0, 1, 1),
                           Node(3, 3, 1.5, 0, 0, 1, 2), Node(4, 3, 5, 0, 0, 1, 3)])
        result = precondition(tree, gamma=100)
        self.assertEqual(len(tree.nodes), 4)
        self.assertEqual(result.collapsed_segments, 1)
        self.assertEqual(result.morphology.nodes[4].parent, 2)
        self.assertEqual(result.morphology.nodes[2].position, (1, 0, 0))

    def test_preconditioning_splits_long_segment(self):
        tree = Morphology([Node(1, 1, 0, 0, 0, 1, -1), Node(2, 3, 1, 0, 0, 1, 1),
                           Node(3, 3, 21, 0, 0, 1, 2)])
        result = precondition(tree, alpha=.1, beta=.1, gamma=5)
        self.assertEqual(result.split_segments, 1)
        self.assertEqual(len(result.morphology.nodes), 6)
        self.assertEqual(result.collapsed_segments, 0)

    def test_generate_command_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            swc = Path(directory)/"sphere.swc"
            output = Path(directory)/"sphere.obj"
            swc.write_text("1 1 0 0 0 1 -1\n", encoding="utf-8")
            result = subprocess.run([sys.executable, "-m", "anamorph", "generate-voxel", str(swc), str(output),
                                     "--spacing", ".3", "--edge-collapse", "--smooth-iterations", "1"],
                                    text=True, capture_output=True, check=True)
            self.assertIn("experimental voxel meshing", result.stderr)
            self.assertGreater(json.loads(result.stdout)["faces"], 0)
            self.assertEqual(json.loads(result.stdout)["method"], "voxel")
            self.assertEqual(read_obj(output).statistics().euler_characteristic, 2)
            rejected = subprocess.run([sys.executable, "-m", "anamorph", "generate-voxel", str(swc), str(swc),
                                       "--spacing", ".3"], text=True, capture_output=True)
            self.assertEqual(rejected.returncode, 1)
            self.assertIn("output path must differ", rejected.stderr)
            self.assertTrue(swc.read_text(encoding="utf-8").startswith("1 1"))

    def test_soma_roots_outside_and_inside(self):
        outside = Morphology([Node(1, 1, 0, 0, 0, 2, -1), Node(2, 3, 4, 0, 0, .5, 1)])
        adjusted = adjust_soma_roots(outside)
        root = adjusted.nodes[adjusted.nodes[2].parent]
        self.assertEqual(root.position, (2., 0., 0.))
        self.assertEqual(root.parent, 1)
        inside = Morphology([Node(1, 1, 0, 0, 0, 2, -1), Node(2, 3, 1, 0, 0, .5, 1),
                             Node(3, 3, 3, 0, 0, 1.5, 2)])
        adjusted = adjust_soma_roots(inside)
        self.assertEqual(adjusted.nodes[2].position, (2., 0., 0.))
        self.assertEqual(adjusted.nodes[2].radius, 1.)
        with self.assertRaisesRegex(ValueError, "branches or ends inside"):
            adjust_soma_roots(Morphology([Node(1, 1, 0, 0, 0, 2, -1), Node(2, 3, 1, 0, 0, .5, 1)]))
        boundary = Morphology([Node(1,1,0,0,0,2,-1), Node(2,3,2,0,0,.5,1)])
        adjusted_boundary = adjust_soma_roots(boundary)
        self.assertEqual(adjusted_boundary.nodes[2].parent, 1)
        adjusted_boundary = adjust_soma_roots(boundary, preserve_surface_roots=False)
        self.assertNotEqual(adjusted_boundary.nodes[2].parent, 1)
        self.assertEqual(adjusted_boundary.nodes[adjusted_boundary.nodes[2].parent].position, (2.,0.,0.))

    def test_inside_root_surface_placement_is_scale_invariant(self):
        results = []
        for scale in (1e-200, 1e-150, 1., 1e150, 1e200):
            tree = Morphology([Node(1, 1, 0., 0., 0., scale, -1),
                               Node(2, 3, .2*scale, .3*scale, 0., .1*scale, 1),
                               Node(3, 3, 1.4*scale, .5*scale, .3*scale, .2*scale, 2)])
            root = adjust_soma_roots(tree).nodes[2]
            normalized = tuple(value/scale for value in root.position)
            self.assertAlmostEqual(math.dist(normalized, (0., 0., 0.)), 1., places=14)
            parameter = (normalized[0]-.2)/1.2
            self.assertGreater(parameter, 0.)
            self.assertLess(parameter, 1.)
            self.assertAlmostEqual(normalized[1], .3+.2*parameter, places=14)
            self.assertAlmostEqual(normalized[2], .3*parameter, places=14)
            self.assertAlmostEqual(root.radius/scale, .1+.1*parameter, places=14)
            results.append(normalized)
        for point in results[1:]:
            for actual, expected in zip(point, results[0]):
                self.assertAlmostEqual(actual, expected, places=14)

    def test_inside_root_absorbs_surface_sample_and_preserves_branch_order(self):
        for scale in (1e-200, 1., 1e200):
            with self.subTest(scale=scale):
                nodes = [Node(1, 1, 0., 0., 0., scale, -1),
                         Node(2, 3, .2*scale, 0., 0., .2*scale, 1),
                         Node(3, 3, .5*scale, 0., 0., .15*scale, 2),
                         Node(4, 3, scale, 0., 0., .1*scale, 3),
                         Node(5, 3, 2*scale, .2*scale, 0., .08*scale, 4),
                         Node(6, 3, 2*scale, -.2*scale, 0., .09*scale, 4)]
                tree = Morphology(nodes)
                order = dict(tree.children)
                order[4] = (6, 5)
                tree = Morphology(nodes, child_order=order)
                adjusted = adjust_soma_roots(tree)
                self.assertEqual(set(adjusted.nodes), {1, 2, 5, 6})
                self.assertEqual(adjusted.nodes[2].position, (scale, 0., 0.))
                self.assertEqual(adjusted.nodes[2].radius, .1*scale)
                self.assertEqual(adjusted.children[2], (6, 5))
                self.assertEqual((adjusted.nodes[5].parent, adjusted.nodes[6].parent), (2, 2))
                source_style = adjust_soma_roots(tree, preserve_surface_roots=False)
                self.assertEqual(source_style.nodes[2].position, source_style.nodes[4].position)
                self.assertEqual(source_style.children[4], (6, 5))

    def test_outside_root_with_unrepresentable_distance_uses_finite_direction(self):
        tree = Morphology([Node(1, 1, -1e308, 0., 0., 1e308, -1),
                           Node(2, 3, 1e308, 0., 0., 1., 1)])
        prepared = adjust_soma_roots(tree)
        inserted = prepared.nodes[prepared.nodes[2].parent]
        self.assertEqual(inserted.position, (0., 0., 0.))
        self.assertEqual(inserted.parent, 1)
        tree = Morphology([Node(1, 1, 0., 0., 0., 1e-200, -1),
                           Node(2, 3, 1e200, 0., 0., 1., 1)])
        prepared = adjust_soma_roots(tree)
        inserted = prepared.nodes[prepared.nodes[2].parent]
        self.assertEqual(inserted.position, (1e-200, 0., 0.))

    def test_soma_section_centroid_radius_and_neurite_reattachment(self):
        tree = Morphology([Node(1, 1, 0, 0, 0, 1, -1), Node(2, 1, 2, 0, 0, 2, 1),
                           Node(3, 1, 1, 3, 0, 1, 2), Node(4, 3, 5, 0, 0, .2, 2),
                           Node(5, 2, 0, 5, 0, .2, 3)])
        normalized = normalize_soma_sections(tree)
        self.assertEqual(normalized.nodes[1].position, (1., 1., 0.))
        self.assertEqual(normalized.nodes[1].radius, 2)
        self.assertEqual((normalized.nodes[4].parent, normalized.nodes[5].parent), (1, 1))
        self.assertEqual(len(tree.nodes), 5)
        with self.assertRaisesRegex(ValueError, "below a neurite"):
            normalize_soma_sections(Morphology([Node(1, 1, 0, 0, 0, 1, -1), Node(2, 3, 2, 0, 0, .2, 1),
                                               Node(3, 1, 3, 0, 0, .2, 2)]))

    def test_soma_centroid_survives_intermediate_sum_overflow(self):
        from fractions import Fraction
        tree = Morphology([Node(1, 1, 1e308, 0, 0, 1, -1), Node(2, 1, 1e308, 0, 0, 1, 1),
                           Node(3, 1, -1e308, 0, 0, 1, 1)])
        center = normalize_soma_sections(tree).nodes[1].x
        self.assertEqual(center, float(Fraction(1e308)/3))
        with self.assertRaisesRegex(ValueError, "centroid sum exceeds"):
            normalize_soma_sections(tree, source_soma_centroid=True)

    def test_source_soma_centroid_reproduces_sequential_rounding(self):
        from anamorph.analytic_network import analyze_analytic_partial
        from anamorph.source_analysis import trace_source_analysis
        from anamorph.source_jobs import plan_source_analysis_jobs
        tree = Morphology([Node(1, 1, 1e16, 0, 0, 1, -1), Node(2, 1, 1, 1, 0, 1, 1),
                           Node(3, 1, -1e16, 2, 0, 1, 2)])
        self.assertEqual(normalize_soma_sections(tree).nodes[1].position, (1/3, 1., 0.))
        self.assertEqual(normalize_soma_sections(tree, source_soma_centroid=True).nodes[1].position, (0., 1., 0.))
        self.assertEqual(prepare_morphology(tree, preconditioning=False,
                                            source_soma_centroid=True).morphology.nodes[1].position, (0., 1., 0.))
        with self.assertRaisesRegex(ValueError, "must be boolean"):
            normalize_soma_sections(tree, source_soma_centroid=1)
        for analyze in (analyze_analytic_partial, trace_source_analysis, plan_source_analysis_jobs):
            with self.subTest(analyzer=analyze.__name__), self.assertRaisesRegex(ValueError, "requires morphology preparation"):
                analyze(tree, prepare=False, source_soma_centroid=True)
            with self.subTest(analyzer=analyze.__name__), self.assertRaisesRegex(ValueError, "requires morphology preparation"):
                analyze(tree, prepare=False, precondition_norm_mode="source")

    def test_source_soma_centroid_reaches_generation_and_analysis_commands(self):
        from unittest.mock import patch
        from anamorph.analytic_mesher import assemble_redblue_checked
        from anamorph.analytic_network import analyze_analytic_partial
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/"soma.swc"
            source.write_text("1 1 1e16 0 0 1 -1\n2 1 1 1 0 1 1\n3 1 -1e16 2 0 1 2\n", encoding="utf-8")
            tree = read_swc(source)
            with patch("anamorph.analytic_mesher.analyze_analytic_partial", wraps=analyze_analytic_partial) as analysis:
                mesh, report = assemble_redblue_checked(tree, analysis_job_mode="source", analysis_gate="source",
                                                         preconditioning=False, soma_refinements=0,
                                                         source_soma_centroid=True, precondition_norm_mode="source-x87")
            self.assertTrue(analysis.call_args.kwargs["source_soma_centroid"])
            self.assertEqual(analysis.call_args.kwargs["precondition_norm_mode"], "source-x87")
            self.assertTrue(report.source_jobs_clean)
            self.assertEqual((min(v[0] for v in mesh.vertices)+max(v[0] for v in mesh.vertices))/2, 0.)
            for command_name in ("generate-redblue", "generate-individual", "export-amv"):
                for source_mode, expected in ((False, 1/3), (True, 0.)):
                    with self.subTest(command=command_name, source_mode=source_mode):
                        output = root/(command_name+("-source" if source_mode else "-stable"))
                        args = [sys.executable, "-m", "anamorph", command_name, str(source), str(output),
                                "--no-precondition"]
                        if command_name != "export-amv":
                            args += ["--soma-refinements", "0"]
                        if command_name == "generate-redblue":
                            args.append("--force")
                        if source_mode:
                            args += ["--source-soma-centroid", "--precondition-norm-mode", "source-x87"]
                        run = subprocess.run(args, text=True, capture_output=True, check=True, timeout=30)
                        if command_name == "export-amv":
                            actual = json.loads(run.stdout)["soma_position"][0]
                        else:
                            result = read_obj(output)
                            actual = (min(v[0] for v in result.vertices)+max(v[0] for v in result.vertices))/2
                        self.assertAlmostEqual(actual, expected, places=14)

            overflow = root/"overflow.swc"
            overflow.write_text("1 1 1e308 0 0 1 -1\n2 1 1e308 1 0 1 1\n3 1 -1e308 2 0 1 2\n", encoding="utf-8")
            for command_name in ("analyze-analytic", "plan-source-jobs", "trace-source-jobs",
                                 "export-amv", "generate-individual", "generate-redblue"):
                with self.subTest(command=command_name, overflow=True):
                    args = [sys.executable, "-m", "anamorph", command_name, str(overflow)]
                    if command_name in ("export-amv", "generate-individual", "generate-redblue"):
                        args.append(str(root/(command_name+"-overflow")))
                    if command_name == "generate-redblue":
                        args.append("--force")
                    args += ["--no-precondition", "--source-soma-centroid"]
                    run = subprocess.run(args, text=True, capture_output=True, timeout=30)
                    self.assertNotEqual(run.returncode, 0)
                    self.assertIn("source soma centroid sum exceeds float range", run.stderr)

    def test_soma_normalization_keeps_root_and_descendant_edge_order(self):
        nodes = [Node(1, 1, 0, 0, 0, 1, -1), Node(2, 1, 0, .1, 0, 1, 1),
                 Node(3, 1, 0, -.1, 0, 1, 1), Node(4, 1, 0, .2, 0, 1, 2),
                 Node(5, 1, 0, .3, 0, 1, 2), Node(8, 3, 2, 0, 0, .1, 1),
                 Node(9, 3, 0, 2, 0, .1, 2), Node(10, 3, -2, 0, 0, .1, 3),
                 Node(11, 3, 0, -2, 0, .1, 4), Node(12, 3, 2, 2, 0, .1, 5),
                 Node(13, 3, 0, 3, 0, .1, 9), Node(14, 3, 1, 3, 0, .1, 9)]
        tree = Morphology(nodes)
        custom = dict(tree.children)
        custom[9] = (14, 13)
        tree = Morphology(nodes, child_order=custom)
        result = normalize_soma_sections(tree)
        # Direct root children precede redirected children; the source's
        # section traversal visits the last nested sibling first.
        self.assertEqual(result.children[1], (8, 9, 12, 11, 10))
        self.assertEqual(result.children[9], (14, 13))

    def test_inserted_soma_root_keeps_neurite_path_order(self):
        tree = Morphology([Node(1, 1, 0, 0, 0, 1, -1), Node(2, 3, 1.5, 0, 0, .1, 1),
                           Node(3, 3, 2.5, 0, 0, .1, 2), Node(4, 3, -.2, 0, 0, .1, 1),
                           Node(5, 3, -1.4, 0, 0, .1, 4), Node(6, 3, -2.5, 0, 0, .1, 5)])
        prepared = prepare_morphology(tree).morphology
        paths = partition_paths(prepared)
        self.assertEqual(tuple(path[0] for path in paths), prepared.children[1])
        self.assertGreater(prepared.nodes[paths[0][1]].x, 0)
        self.assertLess(prepared.nodes[paths[1][1]].x, 0)

    def test_model_preparation_merges_soma_before_adjusting_root(self):
        tree = Morphology([Node(1, 1, 0, 0, 0, 1, -1), Node(2, 1, 2, 0, 0, 2, 1),
                           Node(3, 3, 5, 0, 0, .2, 2)])
        prepared = prepare_morphology(tree, preconditioning=False).morphology
        self.assertEqual(prepared.nodes[1].position, (1., 0., 0.))
        root = prepared.nodes[prepared.nodes[3].parent]
        self.assertEqual(root.position, (3., 0., 0.))
        self.assertEqual(root.parent, 1)
        self.assertEqual(len(tree.nodes), 3)

    def test_model_preparation_rejects_incompatible_neurite_types_and_radii(self):
        tree = Morphology([Node(1,1,0,0,0,1,-1), Node(2,3,2,0,0,.2,1), Node(3,2,3,0,0,.2,2)])
        with self.assertRaisesRegex(ValueError, "changes neurite type"):
            prepare_morphology(tree)
        tiny = Morphology([Node(1,1,0,0,0,1,-1), Node(2,3,2,0,0,1e-9,1)])
        with self.assertRaisesRegex(ValueError, "1e-8"):
            prepare_morphology(tiny)

    def test_secondary_distance_uses_source_code_neighborhood(self):
        tree = Morphology([Node(1,1,0,0,0,1,-1), Node(2,3,2,0,0,.3,1), Node(3,3,3,0,0,.2,2),
                           Node(4,3,3.5,0,0,.1,3), Node(5,3,4.5,0,0,5,4), Node(6,3,3,1,0,.4,3)])
        self.assertEqual(tree.secondary_distance_radii(tree.nodes[3], tree.nodes[4]), (.4, .4))
        violation = next(item for item in tree.minimum_distance_violations() if item.edge == (3, 4))
        self.assertFalse(violation.primary)
        self.assertTrue(violation.secondary)

    def test_distance_violations_handle_overflowing_lengths_and_thresholds(self):
        def tree(radius):
            return Morphology([Node(1, 1, 0., 1., 0., 1., -1),
                               Node(2, 3, -7e307, 0., 0., radius, 1),
                               Node(3, 3, -1e308, 0., 0., radius, 2),
                               Node(4, 3, 1e308, 0., 0., radius, 3),
                               Node(5, 3, 9e307, 0., 0., radius, 4)])

        separated = tree(9e307)
        self.assertNotIn((3, 4), separated.primary_distance_violations(alpha=2.))
        self.assertNotIn((3, 4), {item.edge for item in separated.minimum_distance_violations()})

        touching = tree(1e308)
        self.assertIn((3, 4), touching.primary_distance_violations(alpha=2.))
        hit = next(item for item in touching.minimum_distance_violations() if item.edge == (3, 4))
        self.assertEqual((hit.primary, hit.severe_primary, hit.secondary), (True, False, True))
        self.assertTrue(math.isinf(hit.length))
        self.assertEqual(Fraction(*hit.exact_length_squared), (2*Fraction(1e308))**2)

    def test_distance_violations_compare_exact_values_at_threshold(self):
        for scale in (2.**-600, 1., 2.**600):
            threshold = scale/2
            for radius in (math.nextafter(threshold, 0.), threshold, math.nextafter(threshold, math.inf)):
                with self.subTest(scale=scale, radius=radius):
                    tree = Morphology([Node(1, 1, 0., scale, 0., scale, -1),
                                       Node(2, 3, 0., 0., 0., radius, 1),
                                       Node(3, 3, scale, 0., 0., radius, 2)])
                    expected = Fraction(scale) <= 2*Fraction(radius)
                    self.assertEqual((2, 3) in tree.primary_distance_violations(alpha=2.), expected)
                    self.assertEqual((2, 3) in {item.edge for item in tree.minimum_distance_violations()}, expected)

    def test_sampled_analysis_detects_crossing_neurites(self):
        tree = Morphology([Node(1, 1, -3, 0, 0, .5, -1), Node(2, 3, -2, 0, 0, .1, 1),
                           Node(3, 3, 2, 0, 0, .1, 2), Node(4, 1, 0, -3, 0, .5, -1),
                           Node(5, 3, 0, -2, 0, .1, 4), Node(6, 3, 0, 2, 0, .1, 5)])
        warnings = approximate_analysis(tree)
        self.assertTrue(any(w.kind == "neurite_proximity" for w in warnings))
        with self.assertRaisesRegex(ValueError, "segment pairs"):
            approximate_analysis(tree, max_pairs=0)

    def test_numeric_curvature_warning_on_parabolic_segment(self):
        # x=u, y=u² has maximum curvature 2 at u=0.
        curve = Spline((0., 1.), ((0., 0., 0.), (1., 1., 0.)), ((0., 2., 0.), (0., 2., 0.)))
        self.assertEqual(_numeric_curve_checks(curve, 0, .1), (False, False))
        self.assertEqual(_numeric_curve_checks(curve, 0, 1.), (False, True))

    def test_edge_collapse_preserves_closed_mesh_topology(self):
        sphere = Morphology([Node(1, 1, 0, 0, 0, 1, -1)])
        mesh = generate_voxel_mesh(sphere, .3)
        result = greedy_edge_collapse(mesh, depth=2)
        stats = result.statistics()
        self.assertLess(len(result.faces), len(mesh.faces))
        self.assertEqual((stats.euler_characteristic, stats.boundary_edges, stats.nonmanifold_edges), (2, 0, 0))
        self.assertLess(stats.aspect_ratio_max, mesh.statistics().aspect_ratio_max)

    def test_edge_collapse_stops_when_all_candidates_are_unsafe(self):
        # Collapsing any tetrahedron edge would produce duplicate faces.  The
        # postponed candidates must reach a fixed point rather than loop.
        mesh = Mesh(((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.)),
                    ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)))
        result = greedy_edge_collapse(mesh, alpha=1., area_floor=2.,
                                      area_ceiling=2., depth=0)
        self.assertEqual(result, mesh)
        source = greedy_edge_collapse(mesh, alpha=1., area_floor=2., area_ceiling=2., depth=0, safety="source")
        self.assertEqual((len(source.vertices), len(source.faces)), (3, 2))
        # The original's topological rule can leave coincident opposing faces.
        self.assertEqual(set(source.faces[0]), set(source.faces[1]))

    def test_collapse_neighborhood_uses_source_edge_order_and_sequential_sum(self):
        faces = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))
        edge_faces = {}
        for index, face in enumerate(faces):
            for a, b in zip(face, face[1:]+face[:1]):
                edge_faces.setdefault(tuple(sorted((a, b))), set()).add(index)
        # The source expands faces at the requested depth, so depth zero
        # includes the first neighbor ring in triangle-edge order.
        self.assertEqual(_source_face_neighborhood(0, faces, edge_faces, 0), [0, 2, 3, 1])
        metrics = ((1., 1e16), (1., 1.), (1., 1.), (2., 3.))
        self.assertEqual(_source_neighborhood_average(0, [0, 1, 2], metrics, 1.5), 1e16/3)
        self.assertEqual(_source_neighborhood_average(3, [3], metrics, 1.5), 3.)

    def test_collapse_queue_reuses_priority_slots_without_reviving_stale_entries(self):
        queue = _SourcePriorityQueue()
        for face in (1, 2, 3):
            queue.insert(face, 0.)
        self.assertEqual(queue.pop()[1], 1)
        queue.insert(4, 0.)  # Reuses slot 0 and wins the next tie.
        self.assertEqual(queue.pop()[1], 4)
        queue.discard(2)
        queue.insert(5, 0.)
        self.assertEqual(queue.pop()[1], 5)
        queue.insert(3, 1.)  # Replacing an active face leaves a stale heap entry.
        self.assertEqual(queue.pop(), (1., 3))
        self.assertFalse(queue)

    def test_queue_key_change_preserves_tie_order_and_stale_entries(self):
        queue = _SourcePriorityQueue()
        for value in (1, 2, 3):
            queue.insert(value, 0.)
        self.assertEqual(queue.pop()[1], 1)
        self.assertTrue(queue.change_key(3, 0.))
        self.assertFalse(queue.change_key(99, 0.))
        queue.insert(4, 0.)
        self.assertEqual([queue.pop()[1] for _ in range(3)], [4, 2, 3])
        self.assertFalse(queue)

    def test_long_segment_subdivision_uses_rounded_source_penalty(self):
        tie = 10*math.sqrt(6)
        for length, expected in ((math.nextafter(tie, -math.inf), 2), (tie, 2),
                                 (math.nextafter(tie, math.inf), 3)):
            with self.subTest(length=length):
                tree = Morphology([Node(1, 3, 0, 0, 0, 1, -1), Node(2, 3, length, 0, 0, 1, 1)])
                result = precondition(tree).morphology
                self.assertEqual(len(result.nodes)-1, expected)

    def test_remesh_reduces_edge_length_spread_and_keeps_topology(self):
        from math import dist
        sphere = Morphology([Node(1, 1, 0, 0, 0, 1, -1)])
        before = generate_voxel_mesh(sphere, .3)
        after = isotropic_remesh(before, .22, iterations=4)
        def spread(mesh):
            edges = {tuple(sorted((u, v))) for a, b, c in mesh.faces for u, v in ((a, b), (b, c), (c, a))}
            lengths = sorted(dist(mesh.vertices[u], mesh.vertices[v]) for u, v in edges)
            return lengths[int(.95*(len(lengths)-1))] / lengths[int(.05*(len(lengths)-1))]
        self.assertLess(spread(after), spread(before))
        stats = after.statistics()
        self.assertEqual((stats.euler_characteristic, stats.boundary_edges, stats.nonmanifold_edges), (2, 0, 0))
        self.assertLess(abs(stats.volume-before.statistics().volume)/before.statistics().volume, .1)

    def test_remesh_adaptive_collapse_checks_local_intersections(self):
        from anamorph.canal_mesh import ico_sphere
        from anamorph.mesh import dot
        from anamorph.mesh_intersections import find_self_intersections
        from anamorph.remesh import _Topology
        base = ico_sphere((0., 0., 0.), 1., 1)
        cases = (
            ((10, 38), True, {
                1: (.131084491933802, -.2591802486566075, 1.0667416232751794),
                10: (.25247799283127736, -1.0918022037329205, -.10627723762628308),
                18: (.5900003828962214, -.38567434197780437, .879672412922545),
                23: (-.14828520915881221, -.7360244645221424, .5763857392226786),
                37: (.6753361017712092, -.21575326233313263, -.03962842963562552),
                38: (.16686277578954653, -.5958017447590119, .7167538634584161),
                39: (-.008893218041802875, -1.1938236847090573, .22813415998697462),
                40: (.5814580795279098, -.9222479244144067, -.7466492545680257),
                41: (.6576064464222084, -.2725445882532127, -.21481863909074334)}),
            ((1, 18), False, {
                1: (-.23888921393792278, -.5817437505962106, .6845764349858344),
                7: (.5589968107924355, .10795814137052295, .764487921372852),
                18: (.6171131124593874, -.18091840349581845, .9834554739769126),
                19: (.4966253651744973, .44979418476242966, .8664567052745282),
                20: (.25118735093941036, -.03031048444915435, 1.3200827720943922),
                21: (-.2235708273787208, -.574488183641132, .5449348838871867),
                23: (-.43571570363715195, -1.1565137886579486, .42769813575720944),
                37: (.580316881561423, -.25427054044995523, .3533692709682417),
                38: (.5907807960663356, -.8768942799913184, .4263340403181996)}),
        )
        for edge, should_collapse, changes in cases:
            with self.subTest(edge=edge):
                vertices = list(base.vertices)
                for index, point in changes.items():
                    vertices[index] = point
                before = Mesh(tuple(vertices), base.faces)
                self.assertFalse(find_self_intersections(before, max_hits=1, include_adjacent=True)[0])
                topology = _Topology(before)
                u, v = edge
                midpoint = tuple((a+b)/2 for a, b in zip(vertices[u], vertices[v]))
                affected = topology.vertex_faces[u] | topology.vertex_faces[v]
                incident = topology.edges[edge]
                def quality(face, position=None):
                    normal = topology.normal(face, ({u, v}, position)) if position is not None else topology.normal(face)
                    points = [position if position is not None and i in edge else topology.vertices[i] for i in face]
                    denominator = sum(math.dist(points[i], points[(i+1)%3])**2 for i in range(3))
                    return 2*math.sqrt(3)*math.sqrt(dot(normal, normal))/denominator
                old_quality = min(quality(topology.faces[fid]) for fid in affected)
                midpoint_quality = min(quality(topology.faces[fid], midpoint) for fid in affected-incident)
                if should_collapse:
                    self.assertLess(midpoint_quality, .35)
                else:
                    self.assertTrue(any(dot(topology.normal(topology.faces[fid]),
                                            topology.normal(topology.faces[fid], ({u, v}, midpoint))) <= 0
                                        for fid in affected-incident))
                self.assertEqual(topology.collapse(edge, 2.), should_collapse)
                if should_collapse:
                    self.assertNotEqual(topology.vertices[u], midpoint)
                    self.assertGreaterEqual(min(quality(topology.faces[fid]) for fid in topology.vertex_faces[u]),
                                            old_quality*(1-1e-12))
                after = topology.finish()
                stats = after.statistics()
                self.assertEqual((stats.boundary_edges, stats.nonmanifold_edges), (0, 0))
                self.assertFalse(find_self_intersections(after, max_hits=1, include_adjacent=True)[0])

    def test_remesh_rejects_open_mesh(self):
        open_mesh = Mesh(((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)), ((0, 1, 2),))
        with self.assertRaisesRegex(ValueError, "closed"):
            isotropic_remesh(open_mesh, .5)


if __name__ == "__main__":
    unittest.main()
