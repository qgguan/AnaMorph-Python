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
"""Bounded comparison trace of the source's univariate Bézier clipping.

This intentionally preserves source-specific floating-point and midpoint
behavior. It is not the geometric solver or a certified root enclosure.
"""
from collections import deque
from math import inf, isfinite

from .bernstein import BernsteinPolynomial, ClippingConvergenceError, RootInterval


class SourceUnivariateFailure(ClippingConvergenceError):
    """A failure corresponding to a source univariate solver exception."""


def _source_hull(poly):
    """Construct the source's slope-scanned hull, including its 1e-10 tie rule."""
    n, c = poly.degree, poly.coefficients
    # The degree-zero specialization returns an empty hull, and its caller
    # dereferences it. Do not reproduce the resulting undefined behavior.
    if n == 0:
        raise ClippingConvergenceError("source degree-zero hull is undefined")
    if n == 1:
        return [(0., c[0])]
    hull, i = [(0., c[0])], 0
    while i < n:
        slope, choice = -inf, i
        for j in range(i+1, n+1):
            candidate = n*(c[j]-c[i])/(j-i)
            if not isfinite(candidate):
                raise ClippingConvergenceError("source hull slope exceeds float range")
            if abs(candidate-slope) < 1e-10 or candidate > slope:
                slope, choice = candidate, j
        i = choice
        hull.append((i/n, c[i]))
    while i > 0:
        slope, choice = -inf, i
        for j in range(i-1, -1, -1):
            candidate = n*(c[i]-c[j])/(i-j)
            if not isfinite(candidate):
                raise ClippingConvergenceError("source hull slope exceeds float range")
            if abs(candidate-slope) < 1e-10 or candidate > slope:
                slope, choice = candidate, j
        i = choice
        hull.append((i/n, c[i]))
    hull.pop()
    return hull


def _source_interval(hull, left, right, epsilon):
    low, high, crossings, in_strip = inf, -inf, 0, True
    for point, following in zip(hull, hull[1:]+hull[:1]):
        x, y = point
        nx, ny = following
        if abs(y) < epsilon:
            low, high = min(low, x), max(high, x)
        elif (y < -epsilon and ny > -epsilon) or (y > epsilon and ny < -epsilon):
            in_strip = False
            x0, y0, x1, y1 = (x, y, nx, ny) if x < nx else (nx, ny, x, y)
            zero = x0-y0*(x1-x0)/(y1-y0)
            if not isfinite(zero):
                raise ClippingConvergenceError("source hull intersection exceeds float range")
            if zero < low:
                low = max(0., zero)
            if zero > high:
                high = min(1., zero)
            crossings += 1
    if crossings > 2:
        raise SourceUnivariateFailure("source hull has more than two axis crossings")
    if low == inf:
        return None
    return left+low*(right-left), left+high*(right-left), in_strip


def source_bezier_clip_roots(poly, *, left=0., right=1., tolerance=1e-6,
                             epsilon=1e-11, max_intervals=100000):
    """Trace the original queue and clipping rules with a work limit.

    Source midpoint roots are reported at ``middle +/- tolerance``, while
    subdivision excludes only ``middle +/- tolerance/4``. Returned intervals
    may overlap or extend beyond the requested domain. Degree zero is
    unresolved because the original implementation dereferences an empty hull.
    """
    if (not isinstance(poly, BernsteinPolynomial) or
            not all(isfinite(x) for x in (left, right, tolerance, epsilon)) or
            not 0 <= left <= right <= 1 or tolerance <= 0 or epsilon < 0 or
            type(max_intervals) is not int or max_intervals < 1):
        raise ValueError("invalid source Bézier clipping settings")
    if poly.degree == 0:
        raise ClippingConvergenceError("source degree-zero hull is undefined")
    roots, queue, visited = [], deque([(left, right, poly.restrict(left, right))]), 0
    while queue:
        lo, hi, current = queue.popleft()
        if not 0 <= lo <= hi <= 1:
            raise ClippingConvergenceError("source midpoint subdivision produced an invalid interval",
                                           partial_roots=roots, processed=visited, pending=len(queue)+1)
        while True:
            if visited >= max_intervals:
                raise ClippingConvergenceError("source Bézier clipping exceeded the interval limit",
                                               partial_roots=roots, processed=visited, pending=len(queue)+1)
            visited += 1
            bounds = _source_interval(_source_hull(current), lo, hi, epsilon)
            if bounds is None:
                break
            new_lo, new_hi, in_strip = bounds
            old_size, new_size = hi-lo, new_hi-new_lo
            if in_strip or new_size < tolerance:
                roots.append(RootInterval(new_lo, new_hi))
                break
            try:
                current = poly.restrict(new_lo, new_hi)
            except (ValueError, OverflowError) as error:
                raise ClippingConvergenceError(f"source polynomial restriction unresolved: {error}",
                                               partial_roots=roots, processed=visited, pending=len(queue)+1) from error
            lo, hi = new_lo, new_hi
            if new_size <= .5*old_size:
                continue
            middle = (lo+hi)/2
            if abs(current.evaluate(.5)) < epsilon:
                roots.append(RootInterval(middle-tolerance, middle+tolerance))
                relative = tolerance/(4*(hi-lo))
                lpoly = current.split(.5-relative)[0]
                rpoly = current.split(.5+relative)[1]
                queue.extend(((lo, middle-tolerance, lpoly), (middle+tolerance, hi, rpoly)))
            else:
                lpoly, rpoly = current.split(.5)
                queue.extend(((middle, hi, rpoly), (lo, middle, lpoly)))
            break
    return roots
