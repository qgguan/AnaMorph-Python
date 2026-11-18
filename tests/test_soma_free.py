from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from anamorph.analytic_mesher import (assemble_redblue_neuron, assemble_redblue_to_obj,
                                      render_individual_surfaces)
from anamorph.analytic_network import analyze_analytic_partial
from anamorph.mesh import read_obj
from anamorph.mesh_intersections import find_self_intersections
from anamorph.model import prepare_morphology
from anamorph.redblue import RedBlueDisjoint, RedBlueNumericalEdgeCase, _edge_faces
from anamorph.swc import Morphology, Node, read_swc
from anamorph.workflow import run_workflow


ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT/"examples"/"no_soma.swc"
OPTIONS = {"angular_segments": 8, "soma_refinements": 2, "max_outer_attempts": 24}


def straight(kind=3):
    return Morphology([Node(1, kind, 0, 0, 0, .2, -1), Node(2, kind, 1, 0, 0, .2, 1),
                       Node(3, kind, 2, 0, 0, .15, 2)])


def root_branch():
    return Morphology([Node(1, 3, 0, 0, 0, .2, -1), Node(2, 3, 1, .6, 0, .15, 1),
                       Node(3, 3, 1, -.6, 0, .15, 1), Node(4, 3, -1, 0, .2, .15, 1)])


def mixed():
    return Morphology([Node(1, 1, 0, 0, 0, .5, -1), Node(2, 3, .7, 0, 0, .15, 1),
                       Node(3, 3, 2, 0, 0, .15, 2), Node(4, 3, 0, 4, 0, .2, -1),
                       Node(5, 3, 2, 4, 0, .2, 4)])


def oriented_faces(mesh):
    faces = []
    for face in mesh.faces:
        vertices = tuple(mesh.vertices[i] for i in face)
        faces.append(min(vertices, vertices[1:]+vertices[:1], vertices[2:]+vertices[:2]))
    return Counter(faces)


class SomaFreeTests(unittest.TestCase):
    def assert_closed_without_intersection_candidates(self, mesh, components=1):
        statistics = mesh.statistics()
        self.assertEqual((statistics.boundary_edges, statistics.nonmanifold_edges), (0, 0))
        self.assertEqual(statistics.euler_characteristic, 2*components)
        _edge_faces(mesh)
        hits, truncated = find_self_intersections(mesh, max_hits=1, include_adjacent=True)
        self.assertEqual(hits, [])
        self.assertFalse(truncated)

    def test_preparation_preserves_soma_free_root_and_scaling(self):
        tree = Morphology([Node(1, 3, 0, 0, 0, .2, -1), Node(2, 3, .1, 0, 0, .2, 1),
                           Node(3, 3, 1, 0, 0, .15, 2)])
        for preconditioning in (False, True):
            prepared = prepare_morphology(tree, preconditioning=preconditioning, scale_radius=.5)
            root = prepared.morphology.nodes[1]
            self.assertEqual((root.position, root.parent, root.kind, root.radius), ((0, 0, 0), -1, 3, .1))
            self.assertIsNotNone(prepared.source_edge_order)

    def test_straight_neurite_has_a_closed_flat_start(self):
        for kind in (2, 3, 4):
            for preconditioning in (False, True):
                with self.subTest(kind=kind, preconditioning=preconditioning):
                    mesh = assemble_redblue_neuron(straight(kind), preconditioning=preconditioning, **OPTIONS)
                    self.assert_closed_without_intersection_candidates(mesh)
                    self.assertIn((0., 0., 0.), mesh.vertices)
                    self.assertGreaterEqual(min(point[0] for point in mesh.vertices), -1e-12)
                    self.assertGreater(max(point[0] for point in mesh.vertices), 2.)
                    individual = render_individual_surfaces(straight(kind), preconditioning=preconditioning,
                                                            angular_segments=8, soma_refinements=1)
                    self.assert_closed_without_intersection_candidates(individual)

    def test_branches_and_root_junction_are_closed(self):
        for name, tree in (("branch", read_swc(EXAMPLE)), ("root junction", root_branch())):
            with self.subTest(case=name):
                mesh = assemble_redblue_neuron(tree, **OPTIONS)
                self.assert_closed_without_intersection_candidates(mesh)
                self.assertGreater(max(point[1] for point in mesh.vertices), .6)
                self.assertLess(min(point[1] for point in mesh.vertices), -.6)
                if name == "root junction":
                    self.assertLess(min(point[0] for point in mesh.vertices), -1.)
                individual = render_individual_surfaces(tree, angular_segments=8, soma_refinements=1)
                self.assertGreater(len(individual.faces), 0)
                if name == "root junction":
                    self.assertEqual(individual.statistics().euler_characteristic, 8)

    def test_multiple_soma_free_and_mixed_components(self):
        multiple = Morphology([Node(1, 3, 0, 0, 0, .2, -1), Node(2, 3, 2, 0, 0, .2, 1),
                               Node(3, 3, 0, 4, 0, .2, -1), Node(4, 3, 2, 4, 0, .2, 3)])
        for name, tree in (("multiple neurites", multiple), ("mixed", mixed())):
            with self.subTest(case=name):
                mesh = assemble_redblue_neuron(tree, **OPTIONS)
                self.assert_closed_without_intersection_candidates(mesh, components=2)
                self.assertGreater(max(point[1] for point in mesh.vertices), 4.)
                self.assertGreater(len(render_individual_surfaces(tree, angular_segments=8,
                                                                 soma_refinements=1).faces), 0)

    def test_short_root_branches_are_joined_or_contained(self):
        for length in (.05, .1, .2, .3):
            tree = Morphology([Node(1, 3, 0, 0, 0, .2, -1), Node(2, 3, length, 0, 0, .05, 1),
                               Node(3, 3, 0, length, 0, .05, 1)])
            with self.subTest(length=length):
                mesh = assemble_redblue_neuron(tree, **OPTIONS)
                self.assert_closed_without_intersection_candidates(mesh)
                if length <= .1:
                    # The complete capped tips are inside the root junction.
                    self.assertEqual(len(mesh.faces), 320)
                else:
                    self.assertGreater(len(mesh.faces), 320)

    def test_short_initial_segments_can_continue_beyond_root_junction(self):
        tree = Morphology([Node(1, 3, 0, 0, 0, .2, -1), Node(2, 3, .1, 0, 0, .12, 1),
                           Node(3, 3, 0, .1, 0, .12, 1), Node(4, 3, 1, 0, 0, .1, 2),
                           Node(5, 3, 0, 1, 0, .1, 3)])
        mesh = assemble_redblue_neuron(tree, preconditioning=False, **OPTIONS)
        self.assert_closed_without_intersection_candidates(mesh)
        self.assertGreater(max(point[0] for point in mesh.vertices), 1.)
        self.assertGreater(max(point[1] for point in mesh.vertices), 1.)

    def test_ambiguous_containment_does_not_silently_discard_a_branch(self):
        tree = Morphology([Node(1, 3, 0, 0, 0, .2, -1), Node(2, 3, .05, 0, 0, .05, 1),
                           Node(3, 3, 0, .05, 0, .05, 1)])
        with patch("anamorph.analytic_mesher._point_inside_surface",
                   side_effect=RedBlueNumericalEdgeCase("ambiguous containment")):
            with self.assertRaises(RedBlueDisjoint):
                assemble_redblue_neuron(tree, **OPTIONS)

    def test_streamed_output_matches_in_memory_faces(self):
        for name, tree in (("branch", read_swc(EXAMPLE)), ("root junction", root_branch()), ("mixed", mixed())):
            with self.subTest(case=name), tempfile.TemporaryDirectory() as folder:
                expected = assemble_redblue_neuron(tree, **OPTIONS)
                path = Path(folder)/"streamed.obj"
                summary = assemble_redblue_to_obj(tree, path, flush_face_limit=1, **OPTIONS)
                actual = read_obj(path)
                self.assertGreater(summary["flushes"], 0)
                self.assertEqual(oriented_faces(actual), oriented_faces(expected))

    def test_isolated_neurite_is_explicitly_rejected(self):
        tree = Morphology([Node(1, 3, 0, 0, 0, .2, -1)])
        for generate in (assemble_redblue_neuron, render_individual_surfaces):
            with self.subTest(generate=generate.__name__), self.assertRaisesRegex(ValueError, "no segment to render"):
                generate(tree)
        report = analyze_analytic_partial(tree, prepare=True)
        self.assertTrue(any("no segment to analyze" in item for item in report.unresolved))

    def test_workflow_generates_and_remeshes_soma_free_neuron(self):
        with tempfile.TemporaryDirectory() as folder:
            result = run_workflow(EXAMPLE, Path(folder)/"output", angular_segments=8, analyze=True)
            report = json.loads(Path(result["detailed_report"]).read_text(encoding="utf-8"))
            self.assertTrue(report["analysis"]["performed"])
            self.assertEqual(report["analysis"]["report"]["segment_count"], 3)
            self.assertEqual(report["raw"]["self_intersections"]["status"], "none_found")
            self.assertEqual(report["remeshed"]["self_intersections"]["status"], "none_found")
            for name in ("raw_mesh", "remeshed_mesh"):
                self.assert_closed_without_intersection_candidates(read_obj(result[name]))

    def test_direct_analytic_commands_accept_soma_free_swc(self):
        with tempfile.TemporaryDirectory() as folder:
            for command in ("generate-redblue", "generate-individual"):
                with self.subTest(command=command):
                    output = Path(folder)/(command+".obj")
                    args = [sys.executable, "-m", "anamorph", command, str(EXAMPLE), str(output),
                            "--angular-segments", "8", "--soma-refinements", "2"]
                    if command == "generate-redblue":
                        args.append("--check-output")
                    run = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, timeout=60)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    self.assertGreater(len(read_obj(output).faces), 0)


if __name__ == "__main__":
    unittest.main()
