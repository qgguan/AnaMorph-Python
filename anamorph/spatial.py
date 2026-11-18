# Based on AnaMorph (https://github.com/NeuroBox3D/AnaMorph).
# Original code copyright 2013-2017 G-CSC, Goethe University Frankfurt - Queisser group.
# Recommended citations retained from the original source:
# Mörschel K, Breit M, Queisser G. Generating neuron geometries for detailed
# three-dimensional simulations using AnaMorph. Neuroinformatics (2017).
# Grein S, Stepniewski M, Reiter S, Knodel MM, Queisser G. 1D-3D hybrid
# modelling - from multi-compartment models to full resolution models in space
# and time. Frontiers in Neuroinformatics 8, 68 (2014).
# Breit M, Stepniewski M, Grein S, Gottmann P, Reinhardt L, Queisser G.
# Anatomically detailed and large-scale simulations studying synapse loss
# and synchrony using NeuroBox. Frontiers in Neuroanatomy 10 (2016).
# See LICENSE and NOTICE.md for the license and additional attribution terms.

"""Axis-aligned box overlap queries shared by network analysis stages."""
from __future__ import annotations


def _overlap(a, b):
    return all(x0 <= y1 and y0 <= x1 for (x0, x1), (y0, y1) in zip(a, b))


class _BoxIndex:
    """Balanced box hierarchy whose query results retain input order.

    Bounds are used directly, including infinite conservative bounds, without
    midpoint arithmetic that can overflow for distant coordinates.
    """

    def __init__(self, boxes, *, leaf_size=8):
        if leaf_size < 1:
            raise ValueError("box-index leaf size must be positive")
        self.boxes = tuple(boxes)

        def build(indices):
            if not indices:
                return None
            bounds = tuple((min(self.boxes[i][axis][0] for i in indices),
                            max(self.boxes[i][axis][1] for i in indices)) for axis in range(3))
            if len(indices) <= leaf_size:
                return bounds, tuple(indices), None, None
            axis = max(range(3), key=lambda j: 0. if bounds[j][0] == bounds[j][1]
                       else bounds[j][1]-bounds[j][0])
            indices.sort(key=lambda i: (self.boxes[i][axis][0], self.boxes[i][axis][1], i))
            middle = len(indices)//2
            return bounds, (), build(indices[:middle]), build(indices[middle:])

        self.root = build(list(range(len(self.boxes))))

    def query(self, box):
        matches = []
        stack = [self.root] if self.root is not None else []
        while stack:
            bounds, indices, left, right = stack.pop()
            if not _overlap(bounds, box):
                continue
            if indices:
                matches.extend(i for i in indices if _overlap(self.boxes[i], box))
            else:
                stack.extend((left, right))
        matches.sort()
        return matches
