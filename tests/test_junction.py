import json
import math
from pathlib import Path
import tempfile
import unittest

from anamorph.analytic_mesher import assemble_redblue_neuron
from anamorph.collapse import greedy_edge_collapse
from anamorph.junction import blend_soma_junctions, fill_and_smooth_soma_junctions
from anamorph.mesh_intersections import find_self_intersections
from anamorph.model import prepare_morphology
from anamorph.smoothing import taubin_smooth
from anamorph.swc import read_swc
from anamorph.workflow import run_workflow


ROOT = Path(__file__).resolve().parent.parent


class SomaJunctionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = read_swc(ROOT/"examples"/"branched.swc")
        cls.prepared = prepare_morphology(cls.tree).morphology
        raw = assemble_redblue_neuron(cls.tree, angular_segments=8, soma_refinements=2)
        cls.collapsed = greedy_edge_collapse(raw, safety="source")

    def test_blend_adds_material_locally_and_keeps_connectivity(self):
        blended, report = blend_soma_junctions(self.collapsed, self.prepared)
        self.assertEqual(report["attachments"], 1)
        self.assertEqual(report["attachments_blended"], 1)
        self.assertGreater(report["vertices_moved"], 0)
        self.assertGreater(report["max_displacement"], 0)
        self.assertEqual(blended.faces, self.collapsed.faces)
        self.assertGreater(blended.statistics().volume, self.collapsed.statistics().volume)
        far_vertices = [i for i, (x, _, _) in enumerate(self.collapsed.vertices) if x > 2.]
        self.assertTrue(far_vertices)
        self.assertTrue(all(blended.vertices[i] == self.collapsed.vertices[i] for i in far_vertices))
        self.assertEqual(find_self_intersections(blended, max_hits=1, include_adjacent=True)[0], [])

    def test_fillet_remains_wider_after_taubin_smoothing(self):
        blended, _ = blend_soma_junctions(self.collapsed, self.prepared)
        original = taubin_smooth(self.collapsed)
        smoothed = taubin_smooth(blended)
        indexes = [i for i, (x, y, z) in enumerate(original.vertices) if .95 < x < 1.30 and math.hypot(y, z) > .1]
        self.assertTrue(indexes)
        increase = [math.hypot(*smoothed.vertices[i][1:])-math.hypot(*original.vertices[i][1:]) for i in indexes]
        self.assertGreater(sum(increase)/len(increase), .02)
        self.assertEqual(find_self_intersections(smoothed, max_hits=1, include_adjacent=True)[0], [])

    def test_soma_free_and_disabled_blends_do_not_move_vertices(self):
        no_soma = prepare_morphology(read_swc(ROOT/"examples"/"no_soma.swc")).morphology
        unchanged, report = blend_soma_junctions(self.collapsed, no_soma)
        self.assertEqual(unchanged, self.collapsed)
        self.assertEqual(report["attachments"], 0)
        unchanged, report = blend_soma_junctions(self.collapsed, self.prepared, width_factor=0.)
        self.assertEqual(unchanged, self.collapsed)
        self.assertEqual(report["vertices_moved"], 0)

    def test_two_attachments_are_both_blended(self):
        tree = read_swc(ROOT/"examples"/"mixed_soma_roots.swc")
        mesh = assemble_redblue_neuron(tree, angular_segments=8, soma_refinements=2)
        mesh = greedy_edge_collapse(mesh, safety="source")
        blended, report = blend_soma_junctions(mesh, prepare_morphology(tree).morphology)
        self.assertEqual((report["attachments"], report["attachments_blended"]), (2, 2))
        self.assertGreater(report["vertices_moved"], 0)
        self.assertEqual(find_self_intersections(blended, max_hits=1, include_adjacent=True)[0], [])

    def test_invalid_blend_width_is_rejected(self):
        for width in (-1., 4.1, float("nan"), float("inf")):
            with self.subTest(width=width), self.assertRaises(ValueError):
                blend_soma_junctions(self.collapsed, self.prepared, width_factor=width)

    def test_mortar_widens_the_root_and_keeps_distant_geometry(self):
        baseline = taubin_smooth(self.collapsed)
        mortar, report = fill_and_smooth_soma_junctions(baseline, self.prepared)
        self.assertEqual(report["method"], "mortar")
        self.assertEqual((report["attachments"], report["attachments_blended"]), (1, 1))
        self.assertEqual(mortar.faces, baseline.faces)
        indexes = [i for i, (x, y, z) in enumerate(baseline.vertices)
                   if .98 < x < 1.30 and math.hypot(y, z) > .1]
        self.assertTrue(indexes)
        widening = [math.hypot(*mortar.vertices[i][1:])-math.hypot(*baseline.vertices[i][1:]) for i in indexes]
        self.assertGreater(sum(widening)/len(widening), .02)
        far = [i for i, (x, _, _) in enumerate(baseline.vertices) if x > 2.]
        self.assertTrue(all(mortar.vertices[i] == baseline.vertices[i] for i in far))
        self.assertEqual(find_self_intersections(mortar, max_hits=1, include_adjacent=True)[0], [])

    def test_mortar_leaves_soma_free_neuron_unchanged(self):
        no_soma = prepare_morphology(read_swc(ROOT/"examples"/"no_soma.swc")).morphology
        result, report = fill_and_smooth_soma_junctions(self.collapsed, no_soma)
        self.assertEqual(result, self.collapsed)
        self.assertEqual(report["attachments"], 0)

    def test_workflow_mortar_is_applied_after_smoothing(self):
        with tempfile.TemporaryDirectory() as folder:
            result = run_workflow(ROOT/"examples"/"branched.swc", Path(folder)/"result",
                                  angular_segments=8, soma_refinements=2, smoothing="taubin",
                                  soma_blend_method="mortar", soma_blend_width=3.)
            report = json.loads(Path(result["detailed_report"]).read_text(encoding="utf-8"))
            blend = report["settings"]["soma_junction_blend"]
            self.assertEqual(blend["method"], "mortar")
            self.assertEqual(blend["relaxation_iterations"], 12)
            self.assertEqual(report["remeshed"]["self_intersections"]["status"], "none_found")
            self.assertIn("after smoothing", Path(result["report"]).read_text(encoding="utf-8"))

    def test_workflow_records_the_soma_blend(self):
        with tempfile.TemporaryDirectory() as folder:
            result = run_workflow(ROOT/"examples"/"branched.swc", Path(folder)/"result",
                                  angular_segments=8, soma_refinements=2, smoothing="taubin")
            report = json.loads(Path(result["detailed_report"]).read_text(encoding="utf-8"))
            blend = report["settings"]["soma_junction_blend"]
            self.assertEqual((blend["attachments"], blend["attachments_blended"]), (1, 1))
            self.assertGreater(blend["vertices_moved"], 0)
            self.assertEqual(report["remeshed"]["self_intersections"]["status"], "none_found")
            self.assertIn("Soma junction blend", Path(result["report"]).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
