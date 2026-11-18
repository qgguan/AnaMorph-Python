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

"""Small numerical polynomial tools for one-dimensional geometry checks.

Coefficients are in ascending power order. The derivative-subdivision root
routine here is separate from the translated Bernstein-clipping routines.
"""
from __future__ import annotations


def evaluate(p, x):
    result = 0.0
    for c in reversed(p):
        result = result*x+c
    return result


def derivative(p):
    return [i*p[i] for i in range(1, len(p))] or [0.0]


def add(a, b):
    return [(a[i] if i < len(a) else 0.0)+(b[i] if i < len(b) else 0.0)
            for i in range(max(len(a), len(b)))]


def multiply(a, b):
    c = [0.0]*(len(a)+len(b)-1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            c[i+j] += x*y
    return c


def roots_unit_interval(coefficients, *, tolerance=1e-10):
    """Find roots on [0,1] using derivative subdivision and bisection.

    Recursive derivative roots split the domain into monotone intervals, also
    exposing even-multiplicity roots. Floating-point tolerance means this is a
    numerical diagnostic, not a certified or exhaustive root solver.
    """
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    p = list(coefficients)
    while len(p) > 1 and p[-1] == 0:
        p.pop()
    if len(p) <= 1:
        return []
    scale = sum(abs(x) for x in p)
    if not scale:
        return []
    p = [x/scale for x in p]
    critical = roots_unit_interval(derivative(p), tolerance=tolerance)
    points = [0.0] + [x for x in critical if tolerance < x < 1-tolerance] + [1.0]
    roots = []
    for x in points:
        if abs(evaluate(p, x)) <= tolerance:
            roots.append(x)
    for lo, hi in zip(points, points[1:]):
        left, right = evaluate(p, lo), evaluate(p, hi)
        if left*right >= 0:
            continue
        while hi-lo > tolerance:
            mid = (lo+hi)/2
            value = evaluate(p, mid)
            if left*value <= 0:
                hi = mid
            else:
                lo, left = mid, value
        roots.append((lo+hi)/2)
    roots.sort()
    unique = []
    for root in roots:
        if not unique or root-unique[-1] > 2*tolerance:
            unique.append(root)
    return unique
