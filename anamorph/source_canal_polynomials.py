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
# See LICENSE and NOTICE.md for the license and attribution terms.
"""Source-order univariate canal check polynomials for comparison tracing."""
from dataclasses import dataclass
from math import comb, isfinite

from .bernstein import BernsteinPolynomial, ClippingConvergenceError, RootInterval
from .source_univariate import source_bezier_clip_roots


@dataclass(frozen=True, slots=True)
class SourceScalarCandidate:
    """A source-style numerical candidate; its threshold is not a proof."""
    parameter: float
    value: float
    interval: RootInterval


def _multiply(p, q):
    m, n = len(p)-1, len(q)-1
    result = []
    for k in range(m+n+1):
        value = 0.
        for i in range(max(0, k-n), min(m, k)+1):
            term = p[i]*q[k-i]
            term *= comb(m, i)
            term *= comb(n, k-i)
            term /= comb(m+n, k)
            value += term
        result.append(value)
    return tuple(result)


def _elevate(p, degree):
    n = len(p)-1
    result = []
    for k in range(degree+1):
        value = 0.
        for i in range(max(0, k-degree+n), min(n, k)+1):
            term = p[i]*comb(n, i)
            term *= comb(degree-n, k-i)
            term /= comb(degree, k)
            value += term
        result.append(value)
    return tuple(result)


def _add(*polys):
    result = list(polys[0])
    for poly in polys[1:]:
        result = [x+y for x, y in zip(result, poly)]
    return tuple(result)


def _subtract(p, q):
    return tuple(a-b for a, b in zip(p, q))


def _derivative(p):
    return tuple((len(p)-1)*(p[i+1]-p[i]) for i in range(len(p)-1))


def _components(segment):
    controls = getattr(segment, "source_controls", None)
    if controls is not None:
        return controls
    return tuple(BernsteinPolynomial.from_power(axis).coefficients for axis in segment.power)


def _finish(coefficients):
    if not all(isfinite(value) for value in coefficients):
        raise ClippingConvergenceError("source canal polynomial exceeds float range")
    return BernsteinPolynomial(coefficients)


def source_regularity_polynomial(segment):
    """Build the unscaled source polynomial for squared centerline speed."""
    try:
        velocity = tuple(_derivative(axis) for axis in _components(segment))
        return _finish(_add(*(_multiply(axis, axis) for axis in velocity)))
    except (ValueError, OverflowError) as error:
        raise ClippingConvergenceError("source regularity polynomial exceeds float range") from error


def source_soma_stationary_polynomial(segment, center):
    """Build the source's unscaled soma-distance stationary polynomial."""
    if len(center) != 3 or not all(isfinite(value) for value in center):
        raise ValueError("soma center must have three finite coordinates")
    try:
        components = _components(segment)
        velocity = tuple(_derivative(axis) for axis in components)
        displacement = tuple(tuple(c-center[i] for c in axis) for i, axis in enumerate(components))
        return _finish(_add(*(_multiply(velocity[i], displacement[i]) for i in range(3))))
    except (ValueError, OverflowError) as error:
        raise ClippingConvergenceError("source soma stationary polynomial exceeds float range") from error


def source_local_self_intersection_polynomial(segment):
    """Build the degree-12 source local self-intersection polynomial."""
    try:
        velocity = tuple(_derivative(axis) for axis in _components(segment))
        acceleration = tuple(_derivative(axis) for axis in velocity)
        cross = tuple(_subtract(_multiply(velocity[i], acceleration[j]),
                                _multiply(velocity[j], acceleration[i]))
                      for i, j in ((1, 2), (2, 0), (0, 1)))
        cross_squared = _add(*(_multiply(axis, axis) for axis in cross))
        radius_squared = segment.max_radius*segment.max_radius
        cross_squared = tuple(value*radius_squared for value in cross_squared)
        speed_squared = _add(*(_multiply(axis, axis) for axis in velocity))
        speed_sixth = _multiply(_multiply(speed_squared, speed_squared), speed_squared)
        return _finish(_subtract(_elevate(cross_squared, 12), speed_sixth))
    except (ValueError, OverflowError) as error:
        raise ClippingConvergenceError("source local self-intersection polynomial exceeds float range") from error


def _value(poly, parameter):
    if not 0 <= parameter <= 1:
        raise ClippingConvergenceError("source candidate midpoint lies outside the unit interval")
    value = poly.evaluate(parameter)
    if not isfinite(value):
        raise ClippingConvergenceError("source candidate value exceeds float range")
    return value


def source_regularity_candidates(segment, *, tolerance=1e-6, max_intervals=100000):
    """Trace the source REG polynomial, clipping and threshold decision."""
    poly = source_regularity_polynomial(segment)
    threshold = max(map(abs, poly.coefficients))*1e-10
    roots = source_bezier_clip_roots(poly, tolerance=tolerance, max_intervals=max_intervals)
    result = []
    for root in roots:
        t = root.midpoint
        value = _value(poly, t)
        if value <= threshold:
            result.append(SourceScalarCandidate(t, value, root))
    return result


def source_local_self_intersection_candidates(segment, *, tolerance=1e-6, max_intervals=100000):
    """Trace the source LSI endpoint, clipping and threshold decision."""
    poly = source_local_self_intersection_polynomial(segment)
    threshold = max(map(abs, poly.coefficients))*1e-10
    roots = source_bezier_clip_roots(poly, tolerance=tolerance, max_intervals=max_intervals)
    result = []
    for root in (RootInterval(0., 0.), RootInterval(1., 1.), *roots):
        t = root.midpoint
        value = _value(poly, t)
        if value > -threshold:
            result.append(SourceScalarCandidate(t, value, root))
    return result


def source_soma_neurite_candidates(segment, center, soma_radius, *, is_root_segment=False,
                                    tolerance=1e-6, max_intervals=100000):
    """Trace SONS, including the source's polynomial-dependent distance margin."""
    if not isfinite(soma_radius) or soma_radius <= 0:
        raise ValueError("soma radius must be finite and positive")
    poly = source_soma_stationary_polynomial(segment, center)
    threshold = segment.max_radius+soma_radius
    threshold *= threshold
    threshold += max(map(abs, poly.coefficients))*1e-10
    if not isfinite(threshold):
        raise ClippingConvergenceError("source soma threshold exceeds float range")
    roots = source_bezier_clip_roots(poly, tolerance=tolerance, max_intervals=max_intervals)
    components = tuple(BernsteinPolynomial(axis) for axis in _components(segment))
    result = []
    for root in (RootInterval(0., 0.), RootInterval(1., 1.), *roots):
        t = root.midpoint
        if is_root_segment and t <= 2*tolerance:
            continue
        difference = tuple(center[i]-_value(components[i], t) for i in range(3))
        distance = difference[0]*difference[0]+difference[1]*difference[1]+difference[2]*difference[2]
        if not isfinite(distance):
            raise ClippingConvergenceError("source soma distance exceeds float range")
        if distance <= threshold:
            result.append(SourceScalarCandidate(t, distance, root))
    return result
