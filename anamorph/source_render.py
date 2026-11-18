# Based on AnaMorph (https://github.com/NeuroBox3D/AnaMorph).
# Original code copyright 2013-2017 G-CSC, Goethe University Frankfurt.
# Recommended citations retained from the original source:
# Mörschel K, Breit M, Queisser G. Generating neuron geometries for detailed
# three-dimensional simulations using AnaMorph. Neuroinformatics (2017).
# Grein S, Stepniewski M, Reiter S, Knodel MM, Queisser G. 1D-3D hybrid
# modelling - from multi-compartment models to full resolution models in space
# and time. Frontiers in Neuroinformatics 8, 68 (2014).
# Breit M, Stepniewski M, Grein S, Gottmann P, Reinhardt L, Queisser G.
# Anatomically detailed and large-scale simulations studying synapse loss
# and synchrony using NeuroBox. Frontiers in Neuroanatomy 10 (2016).
# See LICENSE and NOTICE.md for attribution and license terms.
"""Source-order Bézier polynomials for canal render-vector selection."""
from math import isfinite

from .bernstein import BernsteinPolynomial, ClippingConvergenceError
from .source_canal_polynomials import _add, _components, _derivative, _multiply, _subtract
from .source_univariate import source_bezier_clip_roots


def source_render_polynomials(segment, axis):
    """Build the source's unscaled squared cross, speed, and stationary polynomials."""
    if len(axis) != 3 or not all(isfinite(x) for x in axis):
        raise ValueError("render vector must contain three finite coordinates")
    try:
        velocity = tuple(_derivative(component) for component in _components(segment))
        cross = tuple(tuple(velocity[j][k]*axis[i] - velocity[i][k]*axis[j] for k in range(3))
                      for i, j in ((1, 2), (2, 0), (0, 1)))
        p = _add(*(_multiply(component, component) for component in cross))
        q = _add(*(_multiply(component, component) for component in velocity))
        numerator = _subtract(_multiply(_derivative(p), q), _multiply(p, _derivative(q)))
        return tuple(BernsteinPolynomial(coefficients) for coefficients in (p, q, numerator))
    except (ValueError, OverflowError) as error:
        raise ClippingConvergenceError("source render-vector polynomial exceeds float range") from error


def source_render_vector_score(segment, axis, *, tolerance=1e-6, max_intervals=100000):
    """Trace the source's numerical minimum of squared sine along one segment.

    As in the C++ routine, candidates are endpoints and midpoints of intervals
    returned by source-order Bézier clipping. A zero or non-finite speed, an
    out-of-domain candidate, or an exhausted work limit is unresolved.
    """
    p, q, numerator = source_render_polynomials(segment, axis)
    roots = source_bezier_clip_roots(numerator, tolerance=tolerance, max_intervals=max_intervals)
    values = []
    for parameter in (0., 1., *(root.midpoint for root in roots)):
        if not 0 <= parameter <= 1:
            raise ClippingConvergenceError("source render-vector candidate is outside the parameter domain")
        numerator_value, speed_squared = p.evaluate(parameter), q.evaluate(parameter)
        if not isfinite(numerator_value) or not isfinite(speed_squared) or speed_squared <= 0:
            raise ClippingConvergenceError("source render-vector score has zero or non-finite squared speed")
        value = numerator_value/speed_squared
        if not isfinite(value):
            raise ClippingConvergenceError("source render-vector score exceeds float range")
        values.append(value)
    return min(values)
