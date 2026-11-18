import tempfile
import unittest
import json
import random
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from anamorph.mesh import Mesh, read_obj
from anamorph.canal_mesh import ico_sphere
from anamorph.analytic_mesher import assemble_redblue_neuron, assemble_redblue_to_obj, _flushable_faces
from anamorph.analytic_network import _bounding_box
from anamorph.partial_obj import PartialObjWriter
from anamorph.redblue import (RedBlueBoundaryInteraction, _edge_faces, intersection_face_circles,
                              redblue_union_with_retries)
from anamorph.swc import read_swc


def oriented_faces(mesh):
    result = []
    for face in mesh.faces:
        points = tuple(mesh.vertices[i] for i in face)
        result.append(min(points, points[1:]+points[:1], points[2:]+points[:2]))
    return sorted(result)


class PartialObjTests(unittest.TestCase):
    def setUp(self):
        self.mesh = Mesh(((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.)),
                         ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)))

    def test_multiple_flushes_preserve_shared_vertices_and_orientation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"result.obj"
            with PartialObjWriter(self.mesh) as writer:
                after = writer.flush([0])
                self.assertEqual(len(after.faces), 3)
                # A later operation may renumber vertices while leaving the
                # written boundary and its active one-ring unchanged.
                mapping = tuple(reversed(range(len(after.vertices))))
                reordered = Mesh(tuple(reversed(after.vertices)),
                                 tuple(tuple(mapping[i] for i in face) for face in after.faces))
                writer.replace_mesh(reordered, mapping)
                writer.flush([1])
                writer.finalize(path)
            result = read_obj(path)
            self.assertEqual(len(result.vertices), 4)
            self.assertEqual(oriented_faces(result), oriented_faces(self.mesh))
            self.assertEqual(result.statistics().boundary_edges, 0)

    def test_flush_selection_keeps_disconnected_vertex_fans_together(self):
        mesh = ico_sphere((0., 0., 0.), 1., 2)
        def boxes_for(face_ids):
            centers = [tuple(sum(mesh.vertices[v][axis] for v in mesh.faces[fid])/3 for axis in range(3))
                       for fid in face_ids]
            return [tuple((x-1e-7, x+1e-7) for x in point) for point in centers]
        boxes = boxes_for((156, 295, 62, 95))
        def intersects(face, box):
            return all(min(mesh.vertices[v][axis] for v in face) <= box[axis][1] and
                       max(mesh.vertices[v][axis] for v in face) >= box[axis][0] for axis in range(3))
        protected = {fid for fid, face in enumerate(mesh.faces) if any(intersects(face, box) for box in boxes)}
        halo = {vertex for fid in protected for vertex in mesh.faces[fid]}
        naive = {fid for fid, face in enumerate(mesh.faces) if not halo.intersection(face)}
        remaining = Mesh(mesh.vertices, tuple(face for i, face in enumerate(mesh.faces) if i not in naive))
        with self.assertRaisesRegex(ValueError, "vertex-manifold"):
            _edge_faces(remaining, allow_boundary=True)
        selected = set(_flushable_faces(mesh, boxes))
        self.assertEqual(selected, set(_flushable_faces(mesh, iter(boxes))))
        self.assertTrue(selected)
        self.assertLess(selected, naive)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"flushed.obj"
            with PartialObjWriter(mesh) as writer:
                active = writer.flush(selected)
                _edge_faces(active, allow_boundary=True)
                writer.finalize(path)
            self.assertEqual(oriented_faces(read_obj(path)), oriented_faces(mesh))
        rng = random.Random(91878)
        for _ in range(50):
            selected = set(_flushable_faces(mesh, boxes_for(rng.sample(range(len(mesh.faces)), rng.randrange(1, 5)))))
            if selected:
                remaining = Mesh(mesh.vertices, tuple(face for i, face in enumerate(mesh.faces) if i not in selected))
                _edge_faces(remaining, allow_boundary=True)

    def test_rejects_changed_or_removed_boundary_and_leaves_target_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"existing.obj"
            path.write_text("original", encoding="utf-8")
            with PartialObjWriter(self.mesh) as writer:
                after = writer.flush([0])
                identity = tuple(range(len(after.vertices)))
                altered = Mesh(after.vertices, (tuple(reversed(after.faces[0])),)+after.faces[1:])
                with self.assertRaisesRegex(ValueError, "face next to"):
                    writer.replace_mesh(altered, identity)
                mapping = (None,)+identity[1:]
                with self.assertRaisesRegex(ValueError, "boundary vertex"):
                    writer.replace_mesh(after, mapping)
                with self.assertRaisesRegex(ValueError, "invalid face index"):
                    writer.flush([-1])
                self.assertEqual(path.read_text(encoding="utf-8"), "original")
                writer.finalize(path)
            self.assertEqual(oriented_faces(read_obj(path)), oriented_faces(self.mesh))

    def test_finalizing_without_prior_flush(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"result.obj"
            with PartialObjWriter(self.mesh) as writer:
                writer.finalize(path)
            self.assertEqual(oriented_faces(read_obj(path)), oriented_faces(self.mesh))

    def test_flushed_union_matches_union_of_complete_spheres(self):
        red = ico_sphere((0., 0., 0.), 1., 2)
        blue = ico_sphere((.6, .2, .1), .8, 2)
        face = min(range(len(red.faces)), key=lambda j: sum(
            .6*red.vertices[i][0]+.2*red.vertices[i][1]+.1*red.vertices[i][2] for i in red.faces[j]))
        complete = redblue_union_with_retries(red, blue, max_attempts=16, max_growth_factor=4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"union.obj"
            with PartialObjWriter(red) as writer:
                active = writer.flush([face])
                # Public union still requires a closed mesh by default.
                with self.assertRaisesRegex(ValueError, "closed edge-manifold"):
                    redblue_union_with_retries(active, blue)
                union = redblue_union_with_retries(active, blue, max_attempts=16, max_growth_factor=4,
                                                    return_vertex_maps=True, _allow_red_boundary=True)
                writer.replace_mesh(union.mesh, union.red_vertex_map)
                writer.finalize(path)
            result = read_obj(path)
            self.assertEqual(oriented_faces(result), oriented_faces(complete))
            self.assertEqual(result.statistics().boundary_edges, 0)

    def test_union_rejects_intersection_at_flushed_boundary(self):
        red = ico_sphere((0., 0., 0.), 1., 2)
        blue = ico_sphere((.6, .2, .1), .8, 2)
        circle, _ = intersection_face_circles(red, blue)
        with PartialObjWriter(red) as writer:
            active = writer.flush([circle[0].incident_faces[0]])
            with self.assertRaises(RedBlueBoundaryInteraction):
                redblue_union_with_retries(active, blue, _allow_red_boundary=True)

    def test_branched_assembly_with_repeated_flushes_matches_full_mesh(self):
        root = Path(__file__).resolve().parent.parent
        settings = dict(angular_segments=8, soma_refinements=2, max_outer_attempts=20, max_inner_attempts=20)
        for filename in ("examples/branched.swc", "tests/fixtures/branch_tie.swc"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                tree = read_swc(root/filename)
                expected = assemble_redblue_neuron(tree, **settings)
                path = Path(directory)/"stream.obj"
                with patch("anamorph.analytic_mesher._bounding_box", wraps=_bounding_box) as box_calls:
                    report = assemble_redblue_to_obj(tree, path, flush_face_limit=1, **settings)
                result = read_obj(path)
                bounded_curves = [id(call.args[0]) for call in box_calls.call_args_list]
                self.assertTrue(bounded_curves)
                self.assertEqual(len(bounded_curves), len(set(bounded_curves)))
                self.assertGreaterEqual(report["flushes"], 2)
                self.assertEqual((report["vertices"], report["faces"]), (len(result.vertices), len(result.faces)))
                self.assertLess(report["peak_active_faces"], len(result.faces))
                self.assertEqual(oriented_faces(result), oriented_faces(expected))
                self.assertEqual((result.statistics().boundary_edges, result.statistics().nonmanifold_edges), (0, 0))

    def test_failed_assembly_does_not_replace_existing_output(self):
        root = Path(__file__).resolve().parent.parent
        tree = read_swc(root/"examples/branched.swc")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"existing.obj"
            path.write_text("untouched", encoding="utf-8")
            # Deliberately violate the face-selection rule. The operation must
            # fail without publishing a partial or invalid OBJ.
            with patch("anamorph.analytic_mesher._flushable_faces",
                       side_effect=lambda mesh, boxes: tuple(range(len(mesh.faces)-1))):
                with self.assertRaises((ValueError, RuntimeError)):
                    assemble_redblue_to_obj(tree, path, flush_face_limit=1, angular_segments=8,
                                            soma_refinements=2, max_outer_attempts=12, max_inner_attempts=20)
            self.assertEqual(path.read_text(encoding="utf-8"), "untouched")

    def test_cli_flush_with_analysis_and_whole_mesh_checks(self):
        root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as directory:
            output, processed = Path(directory)/"raw.obj", Path(directory)/"processed.obj"
            command = [sys.executable, "-m", "anamorph", "generate-redblue",
                       str(root/"examples"/"branched.swc"), str(output), "--force", "--angular-segments", "8",
                       "--soma-refinements", "2", "--outer-attempts", "20", "--inner-attempts", "20",
                       "--flush-face-limit", "1"]
            plain = subprocess.run(command+["--require-no-findings"], text=True, capture_output=True, check=True)
            info = json.loads(plain.stdout)
            self.assertGreaterEqual(info["partial_flush"]["flushes"], 2)
            self.assertFalse(info["partial_flush"]["reloaded_for_whole_mesh_work"])
            self.assertFalse(info["analysis"]["complete"])
            original = read_obj(output)
            checked = subprocess.run(command+["--check-output", "--edge-collapse", "--postprocess-output", str(processed)],
                                     text=True, capture_output=True, check=True)
            info = json.loads(checked.stdout)
            self.assertTrue(info["partial_flush"]["reloaded_for_whole_mesh_work"])
            self.assertEqual(info["output_check"]["checked_surfaces"], ["raw", "processed"])
            self.assertEqual(oriented_faces(read_obj(output)), oriented_faces(original))
            self.assertEqual(len(read_obj(processed).faces), info["postprocess_output"]["faces"])
            output.write_text("untouched", encoding="utf-8")
            invalid = subprocess.run(command[:-1]+["0"], text=True, capture_output=True)
            self.assertNotEqual(invalid.returncode, 0)
            self.assertEqual(output.read_text(encoding="utf-8"), "untouched")


if __name__ == "__main__":
    unittest.main()
