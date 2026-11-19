# Based on AnaMorph (https://github.com/NeuroBox3D/AnaMorph).
# Recommended citations retained from the original source:
# Mörschel K, Breit M, Queisser G. Generating neuron geometries for detailed
# three-dimensional simulations using AnaMorph. Neuroinformatics (2017).
# Grein S, Stepniewski M, Reiter S, Knodel MM, Queisser G. 1D-3D hybrid
# modelling - from multi-compartment models to full resolution models in space
# and time. Frontiers in Neuroinformatics 8, 68 (2014).
# Breit M, Stepniewski M, Grein S, Gottmann P, Reinhardt L, Queisser G.
# Anatomically detailed and large-scale simulations studying synapse loss
# and synchrony using NeuroBox. Frontiers in Neuroanatomy 10 (2016).
# See LICENSE and NOTICE.md for license and attribution terms.
"""Local outward blending of soma and primary neurite attachments."""

from math import hypot, isfinite

from .mesh import Mesh, cross, dot, sub
from .smoothing import _neighbors


def _unit_face_normal(vertices, face):
    a, b, c = (vertices[i] for i in face)
    u, v = sub(b, a), sub(c, a)
    scale = max(*(abs(x) for x in u), *(abs(x) for x in v))
    if not scale or not isfinite(scale):
        raise ValueError("invalid triangle at soma junction")
    normal = cross(tuple(x/scale for x in u), tuple(x/scale for x in v))
    length = hypot(*normal)
    if not length:
        raise ValueError("degenerate triangle at soma junction")
    return tuple(x/length for x in normal)


def blend_soma_junctions(mesh: Mesh, tree, *, width_factor: float = 2.):
    """Add a local fillet to soma attachments in a *prepared* morphology.

    Uses a smooth-min blend between a soma sphere and its primary neurite's
    root cylinder to displace nearby surface vertices outwards. No vertices or
    faces are added. A local orientation check backs off the displacement if
    necessary; this does not prove the absence of global intersections.
    """
    if not isfinite(width_factor) or width_factor < 0 or width_factor > 4:
        raise ValueError("soma blend width factor must be finite and between 0 and 4")
    roots = [(tree.nodes[node.parent], node) for node in tree.nodes.values()
             if node.kind in (2, 3, 4) and node.parent in tree.nodes and tree.nodes[node.parent].kind == 1]
    report = {"width_factor": width_factor, "attachments": len(roots), "attachments_blended": 0,
              "vertices_moved": 0, "displacement_scale": 1.0, "max_displacement": 0.0}
    if not width_factor or not roots:
        return mesh, report
    displacements = [(0., 0., 0.) for _ in mesh.vertices]
    magnitudes = [0. for _ in mesh.vertices]
    for soma, root in roots:
        axis = sub(root.position, soma.position)
        axis_length = hypot(*axis)
        if not axis_length:
            raise ValueError(f"soma attachment {root.id} has no direction")
        axis = tuple(v/axis_length for v in axis)
        width = min(width_factor * root.radius, .5 * soma.radius)
        if width == 0:
            continue
        affected = False
        for index, position in enumerate(mesh.vertices):
            relative = sub(position, soma.position)
            length = hypot(*relative)
            if not length or abs(length-soma.radius) > width + root.radius:
                continue
            axial = dot(relative, axis)
            if axial <= 0:
                continue
            radial = tuple(relative[j]-axial*axis[j] for j in range(3))
            rho = hypot(*radial)
            if not rho:
                continue
            sphere_distance = length-soma.radius
            cylinder_distance = rho-root.radius
            difference = cylinder_distance-sphere_distance
            if abs(difference) >= width:
                continue
            # The smooth-min correction is largest at the original intersection
            # and falls to zero with zero slope at the ends of the blend band.
            strength = width*.25*(1-abs(difference)/width)**2
            affected = affected or strength > 0
            if strength <= magnitudes[index]:
                continue
            sphere_weight = .5+.5*difference/width
            direction = tuple(sphere_weight*relative[j]/length + (1-sphere_weight)*radial[j]/rho
                              for j in range(3))
            direction_length = hypot(*direction)
            if not direction_length:
                continue
            displacements[index] = tuple(strength*v/direction_length for v in direction)
            magnitudes[index] = strength
        if affected:
            report["attachments_blended"] += 1
    moved = {i for i, amount in enumerate(magnitudes) if amount > 0}
    report["vertices_moved"] = len(moved)
    if not moved:
        return mesh, report
    affected_faces = [face for face in mesh.faces if any(i in moved for i in face)]
    original_normals = [_unit_face_normal(mesh.vertices, face) for face in affected_faces]
    scale = 1.
    for _ in range(9):
        vertices = tuple(tuple(v[j]+scale*displacements[i][j] for j in range(3))
                         for i, v in enumerate(mesh.vertices))
        try:
            valid = all(dot(old, _unit_face_normal(vertices, face)) > 0
                        for face, old in zip(affected_faces, original_normals))
        except ValueError:
            valid = False
        if valid:
            report["displacement_scale"] = scale
            report["max_displacement"] = scale*max(magnitudes)
            return Mesh(vertices, mesh.faces), report
        scale *= .5
    raise ValueError("soma junction blending would invert mesh triangles; try a smaller blend width")


def fill_and_smooth_soma_junctions(mesh: Mesh, tree, *, width_factor: float = 3.):
    """Build and locally smooth an outward soma-to-neurite fillet.

    The smooth-min blend supplies material at the attachment; weighted local
    relaxation softens its tessellation, then a smoothly tapered radial floor
    helps prevent narrowing at the neurite root. This is an approximate shape
    adjustment; an intersection check of the result remains necessary.
    """
    blended, report = blend_soma_junctions(mesh, tree, width_factor=width_factor)
    report["method"] = "mortar"
    report["initial_blend_scale"] = report["displacement_scale"]
    if not width_factor or not report["vertices_moved"]:
        return blended, report
    roots = [(tree.nodes[node.parent], node) for node in tree.nodes.values()
             if node.kind in (2, 3, 4) and node.parent in tree.nodes and tree.nodes[node.parent].kind == 1]
    frames = []
    weights = [0.] * len(mesh.vertices)
    anchors = [None] * len(mesh.vertices)
    for soma, root in roots:
        axis = sub(root.position, soma.position)
        length = hypot(*axis)
        axis = tuple(v/length for v in axis)
        frames.append((soma, root, axis))
        extent = min(width_factor*root.radius, .5*soma.radius)
        for i, position in enumerate(blended.vertices):
            relative = sub(position, soma.position)
            axial = dot(relative, axis)
            if axial <= 0:
                continue
            radial = hypot(*(relative[j]-axial*axis[j] for j in range(3)))
            distance = hypot(axial-soma.radius, radial-root.radius)
            t = max(0., 1.-distance/extent)
            weight = t*t*(3-2*t)
            if weight > weights[i]:
                reference = sub(mesh.vertices[i], soma.position)
                reference_axial = dot(reference, axis)
                reference_radius = hypot(*(reference[j]-reference_axial*axis[j] for j in range(3)))
                weights[i] = weight
                anchors[i] = (soma.position, axis, reference_radius)
    neighbors = _neighbors(blended)
    current = blended.vertices
    # Relax only a short band around the attachment; all other vertices remain
    # exactly where the chosen remeshing/smoothing stage left them.
    for _ in range(12):
        relaxed = []
        for i, (p, ns) in enumerate(zip(current, neighbors)):
            if not ns or not weights[i]:
                relaxed.append(p)
                continue
            candidate = tuple(p[k]+.35*weights[i]*sum(current[j][k]-p[k] for j in ns)/len(ns) for k in range(3))
            centre, axis, minimum_radius = anchors[i]
            relative = sub(candidate, centre)
            axial = dot(relative, axis)
            radial = tuple(relative[k]-axial*axis[k] for k in range(3))
            radius = hypot(*radial)
            if radius and radius < minimum_radius:
                candidate = tuple(candidate[k]+(minimum_radius-radius)*radial[k]/radius for k in range(3))
            relaxed.append(candidate)
        current = tuple(relaxed)
    def smoothstep(t):
        t = max(0., min(1., t))
        return t*t*(3-2*t)
    corrections = [(0., 0., 0.)] * len(current)
    magnitudes = [0.] * len(current)
    for soma, root, axis in frames:
        radius = root.radius
        for i, position in enumerate(current):
            relative = sub(position, soma.position)
            axial = dot(relative, axis)
            distance = axial-soma.radius
            if distance < -.5*radius or distance > 2*radius:
                continue
            radial = tuple(relative[j]-axial*axis[j] for j in range(3))
            rho = hypot(*radial)
            if not rho or rho > 2*radius:
                continue
            influence = smoothstep((distance+.5*radius)/(.5*radius)) * (1-smoothstep(max(0., distance)/(2*radius)))
            desired = radius*(1 + .35*min(width_factor/3., 1.)*influence)
            displacement = influence*max(0., desired-rho)
            if displacement > magnitudes[i]:
                magnitudes[i] = displacement
                corrections[i] = tuple(displacement*value/rho for value in radial)
    candidate = tuple(tuple(point[k]+corrections[i][k] for k in range(3)) for i, point in enumerate(current))
    changed = {i for i, (a, b) in enumerate(zip(mesh.vertices, candidate)) if a != b}
    faces = [face for face in mesh.faces if any(i in changed for i in face)]
    normals = [_unit_face_normal(mesh.vertices, face) for face in faces]
    scale = 1.
    for _ in range(9):
        vertices = tuple(tuple(a[k]+scale*(b[k]-a[k]) for k in range(3))
                         for a, b in zip(mesh.vertices, candidate))
        try:
            valid = all(dot(old, _unit_face_normal(vertices, face)) > 0 for face, old in zip(faces, normals))
        except ValueError:
            valid = False
        if valid:
            report["vertices_moved"] = len(changed)
            report["relaxation_iterations"] = 12
            report["displacement_scale"] = scale
            report["max_displacement"] = max((hypot(*sub(vertices[i], mesh.vertices[i])) for i in changed), default=0.)
            return Mesh(vertices, mesh.faces), report
        scale *= .5
    raise ValueError("soma junction filling would invert mesh triangles; try a smaller blend width")
