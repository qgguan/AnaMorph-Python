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
"""Source-order cubic spline construction for Windows reference comparisons."""
from math import isfinite
from dataclasses import dataclass
from fractions import Fraction

from .bernstein import BernsteinPolynomial
from .source_float import x87_sqrt


@dataclass(frozen=True, slots=True)
class SourceCanalSegment:
    """Source controls paired with the port's local power representation."""
    power: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]
    radius0: float
    radius1: float
    source_controls: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]

    def __post_init__(self):
        if (len(self.power) != 3 or len(self.source_controls) != 3 or
                any(len(component) != 4 for component in (*self.power, *self.source_controls)) or
                not all(isfinite(value) for component in (*self.power, *self.source_controls) for value in component) or
                not all(isfinite(radius) and radius > 0 for radius in (self.radius0, self.radius1))):
            raise ValueError("source canal segment requires finite cubic coordinates and positive radii")

    @property
    def max_radius(self):
        return max(self.radius0, self.radius1)

    def point(self, t):
        """Evaluate the source's Bézier spine by de Casteljau subdivision."""
        return tuple(BernsteinPolynomial(component).evaluate(t) for component in self.source_controls)

    def tangent(self, t):
        """Evaluate the source's Bézier derivative at the local parameter."""
        return tuple(BernsteinPolynomial(component).derivative().evaluate(t) for component in self.source_controls)


def source_root_cylinder(start, end, radius):
    """Build the source's cubic Bézier cylinder from a soma to a root."""
    if len(start) != 3 or len(end) != 3:
        raise ValueError("source root cylinder requires two 3-D points")
    direction = tuple(b-a for a, b in zip(start, end))
    power = tuple((start[i], direction[i], 0., 0.) for i in range(3))
    controls = tuple((start[i], start[i]+direction[i]/3., start[i]+direction[i]*(2./3.), end[i])
                     for i in range(3))
    return SourceCanalSegment(power, radius, radius, controls)


def exact_source_distance_squared(first, second, t, u):
    """Squared distance on the represented source Bézier curves, exactly.

    The source controls and the separately constructed local power curves can
    differ by rounding. This verifies a source candidate against the controls
    used by the source-style solver, not against those other coefficients.
    """
    def point(segment, parameter):
        parameter = Fraction(parameter)
        coordinates = []
        for component in segment.source_controls:
            values = [Fraction(value) for value in component]
            while len(values) > 1:
                values = [(1-parameter)*a + parameter*b for a, b in zip(values, values[1:])]
            coordinates.append(values[0])
        return coordinates
    a, b = point(first, t), point(second, u)
    return sum(((x-y)**2 for x, y in zip(a, b)), Fraction(0))


def source_spline_controls(points, *, parametrization="chord-length", start_direction=None):
    """Return knots and cubic Bézier controls using the source's operation order.

    This follows the checked-in normalized tridiagonal system, global power
    coefficients, and Bernstein interval clipping, using the tested Windows
    x87 square root. It is for source comparisons and raises when the source's
    unscaled arithmetic cannot represent the requested geometry.
    """
    points = tuple(tuple(point) for point in points)
    if (len(points) < 2 or any(len(point) != 3 or not all(isfinite(x) for x in point) for point in points)):
        raise ValueError("source spline requires at least two finite 3-D points")
    if parametrization not in ("chord-length", "centripetal", "uniform"):
        raise ValueError("invalid source spline parametrization")

    def difference(a, b):
        return tuple(a[i]-b[i] for i in range(3))

    def length(vector):
        square = vector[0]*vector[0]+vector[1]*vector[1]+vector[2]*vector[2]
        if not isfinite(square) or square <= 0:
            raise ValueError("source spline length is not representable")
        return x87_sqrt(square)

    chords = [length(difference(b, a)) for a, b in zip(points, points[1:])]
    if parametrization == "uniform":
        parameters = [0.]+[i/len(points) for i in range(1, len(points)-1)]+[1.]
    else:
        parameters = [0.]
        for chord in chords:
            parameters.append(parameters[-1]+(chord if parametrization == "chord-length" else x87_sqrt(chord)))
        total = parameters[-1]
        if not isfinite(total) or total <= 0:
            raise ValueError("source spline parameter length is not representable")
        parameters = [0.]+[value/total for value in parameters[1:-1]]+[1.]
    h = [parameters[i+1]-parameters[i] for i in range(len(points)-1)]
    if any(not isfinite(value) or value <= 0 for value in h):
        raise ValueError("source spline parameter spacing is not representable")
    start = tuple(start_direction) if start_direction is not None else difference(points[1], points[0])
    if len(start) != 3 or not all(isfinite(value) for value in start):
        raise ValueError("invalid source spline start direction")
    end = difference(points[-1], points[-2])
    start_norm, end_norm = length(start), length(end)
    if parametrization == "chord-length":
        # Reuse the source's left-to-right accumulated chord length. Python
        # 3.12 and newer may use compensated summation for sum(float_values),
        # which can change spline controls and subsequent solver decisions.
        start_scale = end_scale = total
    else:
        start_scale, end_scale = chords[0]/h[0], chords[-1]/h[-1]
    d0 = tuple(value/start_norm*start_scale for value in start)
    dn = tuple(value/end_norm*end_scale for value in end)
    count = len(points)
    controls = [[None]*3 for _ in range(count-1)]
    for axis in range(3):
        values = [point[axis] for point in points]
        lower, diagonal, upper, rhs = [0.]*count, [2.]*count, [0.]*count, [0.]*count
        upper[0], lower[-1] = 1., 1.
        rhs[0] = 6./h[0]*((values[1]-values[0])/h[0]-d0[axis])
        rhs[-1] = 6./h[-1]*(dn[axis]-(values[-1]-values[-2])/h[-1])
        for i in range(1, count-1):
            lower[i] = h[i-1]/(h[i-1]+h[i])
            upper[i] = h[i]/(h[i-1]+h[i])
            rhs[i] = 6./(h[i-1]+h[i])*((values[i+1]-values[i])/h[i]-(values[i]-values[i-1])/h[i-1])
        for i in range(count-1):
            factor = -lower[i+1]/diagonal[i]
            diagonal[i+1] += factor*upper[i]
            rhs[i+1] += factor*rhs[i]
        moments = [0.]*count
        moments[-1] = rhs[-1]/diagonal[-1]
        for i in range(count-2, -1, -1):
            moments[i] = (rhs[i]-upper[i]*moments[i+1])/diagonal[i]
        for i in range(count-1):
            a = (values[i+1]-values[i])/h[i]-h[i]/6.*(moments[i+1]-moments[i])
            alpha = values[i]
            gamma = moments[i]/2.
            beta = (-moments[i]*h[i])/2.+a
            delta = (moments[i+1]-moments[i])/(6.*h[i])
            t = parameters[i]
            power = (alpha+t*(-beta+t*(gamma-delta*t)), beta+t*(-2.*gamma+t*(3.*delta)),
                     gamma+-3.*delta*t, delta)
            if not all(isfinite(value) for value in power):
                raise ValueError("source spline coefficients exceed float range")
            poly = BernsteinPolynomial.from_power(power)
            right = poly.split(parameters[i])[1]
            controls[i][axis] = right.split((parameters[i+1]-parameters[i])/(1.-parameters[i]))[0].coefficients
    return tuple(parameters), tuple(tuple(axis for axis in segment) for segment in controls)
