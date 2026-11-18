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

"""Analytic canal-segment intersection primitives using Bernstein clipping.

Based on AnaMorph (https://github.com/NeuroBox3D/AnaMorph), especially
NLM_CellNetwork::checkCanalSegmentRegularity, checkSomaNeuriteIntersection,
checkNeuriteLocalSelfIntersection and checkNeuriteNeuriteIntersection.

These functions are building blocks for the full analysis, not yet the C++
network-level intersection classifier. Numerical clipping is not certified.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import fsum, inf, isfinite, isqrt, nextafter, ulp

from .bernstein import (BernsteinPolynomial, BiBernsteinPolynomial, ClippingConvergenceError, RootInterval,
                        RootRectangle, _univariate_gcd, bezier_clip_roots, bilinear_clip_roots)
from .geometry import Spline
from .polynomial import add, derivative, evaluate, multiply


@dataclass(frozen=True, slots=True)
class CanalSegment:
    """A cubic centerline segment and its two endpoint radii."""
    power: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]
    radius0: float
    radius1: float

    def __post_init__(self):
        if (len(self.power) != 3 or any(len(p) != 4 for p in self.power) or
                not all(isfinite(c) for p in self.power for c in p) or
                not all(isfinite(r) and r > 0 for r in (self.radius0, self.radius1))):
            raise ValueError("canal segment requires finite cubic coordinates and positive radii")

    @classmethod
    def from_spline(cls, spline: Spline, index: int, radius0: float, radius1: float):
        return cls(spline.power_coefficients(index), radius0, radius1)

    @property
    def max_radius(self):
        return max(self.radius0, self.radius1)

    def point(self, t: float):
        if not 0 <= t <= 1:
            raise ValueError("parameter must lie in [0, 1]")
        return tuple(evaluate(component, t) for component in self.power)


@dataclass(frozen=True, slots=True)
class CurveCandidate:
    """Intersection evidence; exact_parameters overrides rounded t and u if present.

    Each exact parameter is a (numerator, denominator) pair. This is used
    when an exact affine minimum cannot be represented by binary floats.
    If the squared distance overflows or underflows a float, distance_squared
    is None and exact_distance_squared contains its numerator and denominator.
    For ``parallel_line_tangency``, t and u are approximations: an exact
    witnessing parameter lies in each reported parameter bound, and the
    reported distance is the proven minimum rather than the midpoint distance.
    """
    t: float
    u: float
    distance_squared: float | None
    source: str
    parameter_bounds: tuple[tuple[float, float], tuple[float, float]] | None = None
    exact_parameters: tuple[tuple[int, int], tuple[int, int]] | None = None
    exact_distance_squared: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class ScalarCandidate:
    parameter: float
    value: float | None
    source: str
    parameter_bounds: tuple[float, float]
    exact_value: tuple[int, int] | None = None


class IntersectionSolveError(ClippingConvergenceError):
    """An incomplete solve that retains independently verified candidates."""

    def __init__(self, failures, candidates):
        self.failures = tuple(failures)
        self.candidates = tuple(candidates)
        super().__init__("; ".join(self.failures))


def _sum_squares(vectors):
    total = [0.0]
    for value in vectors:
        total = add(total, multiply(value, value))
    return total


def _unit_scale_polynomial(polynomial):
    """Scale coefficients without changing roots, avoiding an absolute-eps blind spot.

    The original solvers use an absolute clipping epsilon. Very small input
    coordinates can otherwise make a nonconstant polynomial look identically
    zero, returning a wide candidate whose midpoint misses an actual root.
    Keeping zero polynomials unchanged preserves their special-case behavior.
    """
    if isinstance(polynomial, BernsteinPolynomial):
        scale = max(abs(c) for c in polynomial.coefficients)
        if not scale:
            return polynomial
        values = tuple(c/scale for c in polynomial.coefficients)
        if any(original and not value for original, value in zip(polynomial.coefficients, values)):
            raise ClippingConvergenceError("polynomial coefficient scale ratio exceeds float range")
        return BernsteinPolynomial(values)
    scale = max(abs(c) for row in polynomial.coefficients for c in row)
    if not scale:
        return polynomial
    values = tuple(tuple(c/scale for c in row) for row in polynomial.coefficients)
    if any(original and not value for original_row, row in zip(polynomial.coefficients, values)
           for original, value in zip(original_row, row)):
        raise ClippingConvergenceError("polynomial coefficient scale ratio exceeds float range")
    return BiBernsteinPolynomial(values)


def _curve_roots(polynomial, *, tolerance, max_intervals=100000):
    return bezier_clip_roots(_unit_scale_polynomial(polynomial), tolerance=tolerance, max_intervals=max_intervals)


def _validate_interval_limit(max_intervals):
    if type(max_intervals) is not int or max_intervals < 1:
        raise ValueError("interval limit must be a positive integer")


def _normalized_velocity(segment):
    """Scale before differentiation and products to preserve geometric units."""
    scale = max(abs(c) for component in segment.power for c in component[1:])
    if not scale:
        return [[0., 0., 0.] for _ in range(3)], 0.
    velocity = [[i*(component[i]/scale) for i in range(1, 4)] for component in segment.power]
    if any(component[i] and (not velocity[axis][i-1] or
                             not velocity[axis][i-1]*velocity[axis][i-1])
           for axis, component in enumerate(segment.power) for i in range(1, 4)):
        raise ClippingConvergenceError("centerline coefficient scale ratio exceeds float range")
    return velocity, scale


def _exact_local_polynomial(segment, parameter):
    """LSI polynomial at a float parameter, using the exact input floats."""
    t = Fraction(parameter)
    velocity = [sum((i*Fraction(component[i])*t**(i-1) for i in range(1, 4)), Fraction(0))
                for component in segment.power]
    acceleration = [sum((i*(i-1)*Fraction(component[i])*t**(i-2) for i in range(2, 4)), Fraction(0))
                    for component in segment.power]
    cross_product = [velocity[i]*acceleration[j]-velocity[j]*acceleration[i]
                     for i, j in ((1, 2), (2, 0), (0, 1))]
    speed_squared = sum((v*v for v in velocity), Fraction(0))
    return Fraction(segment.max_radius)**2*sum((v*v for v in cross_product), Fraction(0))-speed_squared**3


def _exact_speed_squared(segment, parameter):
    t = Fraction(parameter)
    velocity = [sum((i*Fraction(component[i])*t**(i-1) for i in range(1, 4)), Fraction(0))
                for component in segment.power]
    return sum((v*v for v in velocity), Fraction(0))


def _report_fraction(value, description):
    """Return a finite, non-underflowed float for an exact reported value."""
    try:
        result = float(value)
    except OverflowError as error:
        raise ClippingConvergenceError(f"{description} exceeds float range") from error
    if value and not result:
        raise ClippingConvergenceError(f"{description} underflows float range")
    return result


def _report_exact_value(value, description):
    """Represent an exact value as a finite float or a rational pair."""
    try:
        return _report_fraction(value, description), None
    except ClippingConvergenceError:
        return None, (value.numerator, value.denominator)


def _report_distance_squared(value):
    """Represent an exact squared distance as a float or rational pair."""
    return _report_exact_value(value, "candidate squared distance")


def _exact_regularity_intervals(segment, tolerance, max_intervals):
    """Enclose common real roots of the exact represented derivatives.

    Cubic centerline derivatives have degree at most two. Their rational gcd
    has precisely the common complex roots, so a monotonic partition at the
    quadratic's vertex isolates every distinct common root on [0, 1].
    """
    common = []
    for component in segment.power:
        velocity = [i*Fraction(component[i]) for i in range(1, 4)]
        while velocity and not velocity[-1]:
            velocity.pop()
        if not velocity:
            continue
        if len(velocity) == 1:
            return []
        common = _univariate_gcd(common, velocity)
        if len(common) == 1:
            return []
    if not common:
        return [RootInterval(0., 1., certified=True)]

    def at(t):
        result = Fraction(0)
        for coefficient in reversed(common):
            result = result*t+coefficient
        return result

    points = [Fraction(0), Fraction(1)]
    if len(common) == 2:
        root = -common[0]/common[1]
        if 0 < root < 1:
            points.insert(1, root)
    elif len(common) == 3:
        vertex = -common[1]/(2*common[2])
        if 0 < vertex < 1:
            points.insert(1, vertex)
    values = [at(point) for point in points]
    brackets = [(point, point) for point, value in zip(points, values) if not value]
    remaining = min(max_intervals, 2048)
    for lo, hi, vlo, vhi in zip(points, points[1:], values, values[1:]):
        if vlo*vhi >= 0:
            continue
        while hi-lo > Fraction(tolerance) and remaining:
            middle = (lo+hi)/2
            value = at(middle)
            remaining -= 1
            if not value:
                lo = hi = middle
                break
            if (value > 0) == (vlo > 0):
                lo, vlo = middle, value
            else:
                hi = middle
        brackets.append((lo, hi))

    def outward(value, *, lower):
        rounded = float(value)
        represented = Fraction(rounded)
        if (lower and represented > value) or (not lower and represented < value):
            rounded = nextafter(rounded, -inf if lower else inf)
        return rounded
    return [RootInterval(outward(lo, lower=True), outward(hi, lower=False), certified=True)
            for lo, hi in sorted(brackets)]


def regularity_candidates(segment: CanalSegment, *, tolerance: float = 1e-6, max_intervals: int = 100000):
    """Candidate zero-speed intervals for the centerline, as in REG analysis."""
    _validate_interval_limit(max_intervals)
    if not isfinite(tolerance) or tolerance <= 0:
        raise ValueError("solver tolerance must be positive and finite")
    try:
        velocity, _ = _normalized_velocity(segment)
    except ClippingConvergenceError as error:
        evidence = _exact_regularity_intervals(segment, tolerance, max_intervals)
        raise IntersectionSolveError((f"regularity normalization: {error}",), evidence) from error
    speed_squared = _sum_squares(velocity)
    polynomial = BernsteinPolynomial.from_power(speed_squared, degree=4)
    scale = max(abs(c) for c in polynomial.coefficients)
    failures = []
    try:
        roots = _curve_roots(polynomial, tolerance=tolerance, max_intervals=max_intervals)
    except ClippingConvergenceError as error:
        failures.append(f"regularity roots: {error}")
        roots = error.partial_roots
    result = [root for root in roots if polynomial.evaluate(root.midpoint) <= 1e-10*scale]
    # Numerical clipping can miss a repeated root after rounded polynomial
    # products. Keep independently proved common derivative roots as well.
    for exact in _exact_regularity_intervals(segment, tolerance, max_intervals):
        covering = next((i for i, root in enumerate(result)
                         if root.left <= exact.left and exact.right <= root.right), None)
        if covering is None:
            result.append(exact)
        else:
            root = result[covering]
            result[covering] = RootInterval(root.left, root.right, certified=True)
    result.sort(key=lambda root: (root.left, root.right))
    if failures:
        raise IntersectionSolveError(failures, result)
    return result


def local_self_intersection_candidates(segment: CanalSegment, *, tolerance: float = 1e-6,
                                       with_bounds: bool = False, max_intervals: int = 100000):
    """Candidate focal points based on the paper's maximum-radius LSI polynomial.

    Return tuples (parameter, polynomial value), including endpoints and root
    candidate midpoints whose value meets the C++ threshold. A value outside
    the float range is a Fraction, or is stored in exact_value with bounds.
    """
    _validate_interval_limit(max_intervals)
    velocity, velocity_scale = _normalized_velocity(segment)
    if not velocity_scale:
        return []
    acceleration = [derivative(component) for component in velocity]
    cross = [add(multiply(velocity[i], acceleration[j]),
                 [-v for v in multiply(velocity[j], acceleration[i])])
             for i, j in ((1, 2), (2, 0), (0, 1))]
    speed_squared = _sum_squares(velocity)
    norm = max(velocity_scale, segment.max_radius)
    radius_factor, velocity_factor = segment.max_radius/norm, velocity_scale/norm
    radius_weight, velocity_weight = radius_factor*radius_factor, velocity_factor*velocity_factor
    cross_squared = _sum_squares(cross)
    speed_sixth = multiply(multiply(speed_squared, speed_squared), speed_squared)
    if ((not radius_weight and any(cross_squared)) or
            (not velocity_weight and any(speed_sixth))):
        raise ClippingConvergenceError("local self-intersection scale ratio exceeds float range")
    radius_term = [radius_weight*c for c in cross_squared]
    speed_term = [velocity_weight*v for v in speed_sixth]
    if (any(c and not value for c, value in zip(cross_squared, radius_term)) or
            any(c and not value for c, value in zip(speed_sixth, speed_term))):
        raise ClippingConvergenceError("local self-intersection polynomial underflows float range")
    check = BernsteinPolynomial.from_power(add(radius_term, [-v for v in speed_term]), degree=12)
    scale = max(abs(c) for c in check.coefficients)
    failures = []
    try:
        roots = _curve_roots(check, tolerance=tolerance, max_intervals=max_intervals)
    except ClippingConvergenceError as error:
        failures.append(f"local self-intersection roots: {error}")
        roots = error.partial_roots
    parameters = [(0., "endpoint", (0., 0.)), (1., "endpoint", (1., 1.))]
    parameters.extend((r.midpoint, "root_interval", (r.left, r.right)) for r in roots)
    hits = []
    for t, source, bounds in parameters:
        if check.evaluate(t) > -1e-10*scale:
            value, exact = _report_exact_value(_exact_local_polynomial(segment, t),
                                               "local self-intersection value")
            hits.append(ScalarCandidate(t, value, source, bounds, exact))
    if failures:
        raise IntersectionSolveError(failures, hits)
    return hits if with_bounds else [(hit.parameter, hit.value if hit.value is not None else Fraction(*hit.exact_value))
                                     for hit in hits]


def soma_neurite_candidates(segment: CanalSegment, center: tuple[float, float, float], soma_radius: float,
                            *, is_root_segment: bool = False, tolerance: float = 1e-6,
                            with_bounds: bool = False, max_intervals: int = 100000):
    """Find verified candidate overlaps of a soma and the bounding neurite pipe.

    The stationary-root intervals are numerical, not certified. If a root
    interval cannot be separated or refined to an overlap witness, raise an
    IntersectionSolveError and retain any independently verified candidates.
    With bounds, a distance outside the float range is carried in exact_value;
    without bounds, that distance is returned as a Fraction.
    """
    _validate_interval_limit(max_intervals)
    if len(center) != 3 or not all(isfinite(x) for x in center) or not isfinite(soma_radius) or soma_radius <= 0:
        raise ValueError("soma center and radius must be finite, with positive radius")
    displacement = [list(segment.power[i]) for i in range(3)]
    for i in range(3):
        displacement[i][0] -= center[i]
    if not all(isfinite(c) for component in displacement for c in component):
        raise ClippingConvergenceError("soma-relative curve displacement exceeds float range")
    scale = max(abs(c) for component in displacement for c in component)
    normalized = [[c/scale for c in component] for component in displacement] if scale else displacement
    distance_squared = _sum_squares(normalized)
    # The source uses (gamma-center)·gamma', i.e. half the derivative
    # of squared distance. Its fixed absolute clipping epsilon makes this
    # scaling relevant even though both polynomials have the same roots.
    stationary_power = [c/2 for c in derivative(distance_squared)]
    stationary = BernsteinPolynomial.from_power(stationary_power, degree=5)
    parameters = [(0., "endpoint", (0., 0.)), (1., "endpoint", (1., 1.))]
    failures = []
    try:
        roots = _curve_roots(stationary, tolerance=tolerance, max_intervals=max_intervals)
    except ClippingConvergenceError as error:
        failures.append(f"stationary roots: {error}")
        roots = error.partial_roots
    parameters.extend((r.midpoint, "root_interval", (r.left, r.right)) for r in roots)
    point = CanalSegment(tuple((coordinate, 0., 0., 0.) for coordinate in center), soma_radius, soma_radius)
    threshold = (Fraction(soma_radius)+Fraction(segment.max_radius))**2
    hits = []
    for t, source, bounds in parameters:
        if is_root_segment and t <= 2*tolerance:
            continue
        try:
            witness, ambiguous = _refine_distance_candidate(segment, point, t, 0., source,
                                                              (bounds, (0., 0.)), threshold,
                                                              excluded=(lambda x, _: x <= 2*tolerance)
                                                              if is_root_segment else None,
                                                              stationary_system=([[c] for c in stationary_power], [[0.]]))
        except ClippingConvergenceError as error:
            failures.append(f"candidate {source} at {bounds}: {error}")
            continue
        if witness is not None:
            hits.append(ScalarCandidate(witness.t, witness.distance_squared, source, bounds,
                                        witness.exact_distance_squared))
        elif ambiguous:
            failures.append(f"candidate {source} at {bounds}: distance remains ambiguous")
    if failures:
        raise IntersectionSolveError(failures, hits)
    return hits if with_bounds else [(hit.parameter, hit.value if hit.value is not None else Fraction(*hit.exact_value))
                                     for hit in hits]


def _distance_system(first: CanalSegment, second: CanalSegment):
    """Build a scale-normalized curve-distance stationary system."""
    # d(t,u) = gamma(t)-delta(u); p=d·gamma', q=d·delta'.
    # Subtract origins exactly before scaling: dividing two large, nearby
    # origins first can erase their small but representable separation.
    components = []
    for a, b in zip(first.power, second.power):
        components.append((Fraction(a[0])-Fraction(b[0]),
                           *(Fraction(c) for c in a[1:]), *(Fraction(c) for c in b[1:])))
    scale = max(abs(c) for component in components for c in component)
    normalized = []
    for component in components:
        values = tuple(float(c/scale) for c in component) if scale else (0.,)*7
        if any(original and (not value or not value*value) for original, value in zip(component, values)):
            raise ClippingConvergenceError("curve-pair coefficient scale ratio exceeds float range")
        normalized.append(values)
    p = [[0.0]*4 for _ in range(6)]
    q = [[0.0]*6 for _ in range(4)]
    for component in normalized:
        a = component[:4]
        b = (0., *component[4:])
        delta = [[0.0]*4 for _ in range(4)]
        delta[0][0] = a[0]-b[0]
        for i in range(1, 4):
            delta[i][0] = a[i]
            delta[0][i] = -b[i]
        va, vb = derivative(a), derivative(b)
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    p[i+k][j] += delta[i][j]*va[k]
                    q[i][j+k] += delta[i][j]*vb[k]
    return p, q


def _edge_power(matrix, *, fixed_axis: str, fixed_value: float):
    if fixed_axis == "x":
        return [sum(matrix[i][j]*fixed_value**i for i in range(len(matrix))) for j in range(len(matrix[0]))]
    return [sum(matrix[i][j]*fixed_value**j for j in range(len(matrix[0]))) for i in range(len(matrix))]


def _exact_distance_at(first, second, t, u):
    """Squared distance for the binary-float coefficients and parameters."""
    t, u = Fraction(t), Fraction(u)
    def point(component, parameter):
        value = Fraction(0)
        for coefficient in reversed(component):
            value = value*parameter+Fraction(coefficient)
        return value
    delta = tuple(point(a, t)-point(b, u) for a, b in zip(first.power, second.power))
    return sum((x*x for x in delta), Fraction(0))


def _relative_displacement(first, second, t, u):
    """Evaluate curve differences without adding small offsets to large origins."""
    try:
        displacement = tuple(fsum((a[0], -b[0], t*evaluate(a[1:], t), -u*evaluate(b[1:], u)))
                             for a, b in zip(first.power, second.power))
    except (OverflowError, ValueError) as error:
        raise ClippingConvergenceError("relative curve displacement exceeds float range") from error
    if not all(isfinite(value) for value in displacement):
        raise ClippingConvergenceError("relative curve displacement exceeds float range")
    return displacement


def _interval_distance_lower_bound(first, second, bounds):
    """Exact interval-arithmetic lower bound for two parameter intervals."""
    def component_bounds(component, interval):
        lo, hi = map(Fraction, interval)
        lower = upper = Fraction(0)
        for coefficient in reversed(component):
            products = (lower*lo, lower*hi, upper*lo, upper*hi)
            lower, upper = min(products)+Fraction(coefficient), max(products)+Fraction(coefficient)
        return lower, upper
    total = Fraction(0)
    for a, b in zip(first.power, second.power):
        a0, a1 = component_bounds(a, bounds[0])
        b0, b1 = component_bounds(b, bounds[1])
        gap = max(Fraction(0), a0-b1, b0-a1)
        total += gap*gap
    return total


def _refine_distance_candidate(first, second, t, u, source, bounds, threshold, *, excluded=None, inclusive=True,
                               stationary_system=None):
    """Refine a stationary candidate or flag an undecidable root rectangle.

    Newton steps are confined to the returned rectangle. Any accepted point
    is checked against the exact rational values of the input floats. If no
    point is found and interval arithmetic cannot separate the two curve
    pieces, the candidate remains unresolved. The clipping rectangle itself
    is numerical and is not a certified root enclosure.
    """
    from math import isfinite
    from .mesh import dot
    lower_bound = _interval_distance_lower_bound(first, second, bounds)
    if (lower_bound > threshold if inclusive else lower_bound >= threshold):
        return None, False
    va, vb = tuple(map(derivative, first.power)), tuple(map(derivative, second.power))
    aa, ab = tuple(map(derivative, va)), tuple(map(derivative, vb))
    if stationary_system is not None:
        def evaluate_system(matrix, x, y):
            return evaluate([evaluate(row, y) for row in matrix], x)
        def partial_t(matrix):
            return [[i*value for value in matrix[i]] for i in range(1, len(matrix))] or [[0.]]
        def partial_u(matrix):
            return [[j*row[j] for j in range(1, len(row))] or [0.] for row in matrix]
        p_matrix, q_matrix = stationary_system
        jacobian = (partial_t(p_matrix), partial_u(p_matrix), partial_t(q_matrix), partial_u(q_matrix))
    t_bounds, u_bounds = bounds
    exact_witness = None
    for _ in range(16):
        if excluded is None or not excluded(t, u):
            distance2 = _exact_distance_at(first, second, t, u)
            if (distance2 <= threshold if inclusive else distance2 < threshold):
                value, exact_distance = _report_distance_squared(distance2)
                candidate = CurveCandidate(t, u, value, source, bounds,
                                           exact_distance_squared=exact_distance)
                if value is not None:
                    return candidate, False
                # Keep the proof but continue refinement: an exact crossing
                # or a closer point may have a representable float distance.
                exact_witness = candidate
        if stationary_system is None:
            d = _relative_displacement(first, second, t, u)
            p, q = tuple(evaluate(c, t) for c in va), tuple(evaluate(c, u) for c in vb)
            p2, q2 = tuple(evaluate(c, t) for c in aa), tuple(evaluate(c, u) for c in ab)
            f, g = dot(d, p), dot(d, q)
            j00, j01 = dot(p, p)+dot(d, p2), -dot(p, q)
            j10, j11 = dot(p, q), -dot(q, q)+dot(d, q2)
        else:
            f, g = evaluate_system(p_matrix, t, u), evaluate_system(q_matrix, t, u)
            j00, j01, j10, j11 = (evaluate_system(matrix, t, u) for matrix in jacobian)
        if t_bounds[0] == t_bounds[1]:
            if not j11 or not isfinite(j11):
                break
            next_t, next_u = t, u-g/j11
        elif u_bounds[0] == u_bounds[1]:
            if not j00 or not isfinite(j00):
                break
            next_t, next_u = t-f/j00, u
        else:
            determinant = j00*j11-j01*j10
            if not determinant or not isfinite(determinant):
                break
            next_t = t-(f*j11-j01*g)/determinant
            next_u = u-(j00*g-f*j10)/determinant
        if not all(isfinite(x) for x in (next_t, next_u)):
            break
        # A root can lie on a clipping boundary; roundoff in a Newton step
        # may land just outside it. Projection still evaluates only points
        # within the candidate rectangle, and witness acceptance is exact.
        next_t = max(t_bounds[0], min(t_bounds[1], next_t))
        next_u = max(u_bounds[0], min(u_bounds[1], next_u))
        if (next_t, next_u) == (t, u):
            break
        t, u = next_t, next_u
    return (exact_witness, False) if exact_witness is not None else (None, True)


def _global_self_intersection_system(segment: CanalSegment):
    """Factor the trivial diagonal from the self-distance stationary system.

    For each cubic component, (gamma(t)-gamma(u))/(t-u) is
    c1+c2*(t+u)+c3*(t²+tu+u²). The C++ implementation builds this through
    its precomputed Bernstein G polynomials; this is the same quotient in
    the power basis, before degree elevation.
    """
    p = [[0.0]*3 for _ in range(5)]
    q = [[0.0]*5 for _ in range(3)]
    _, scale = _normalized_velocity(segment)
    for component in segment.power:
        c1, c2, c3 = (c/scale for c in component[1:]) if scale else (0., 0., 0.)
        divided = [[0.0]*3 for _ in range(3)]
        divided[0][0] = c1
        divided[1][0] = divided[0][1] = c2
        divided[2][0] = divided[1][1] = divided[0][2] = c3
        velocity = (c1, 2*c2, 3*c3)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    p[i+k][j] += divided[i][j]*velocity[k]
                    q[i][j+k] += divided[i][j]*velocity[k]
    return p, q


def global_self_intersection_candidates(segment: CanalSegment, *,
                                        univariate_tolerance: float = 1e-6,
                                        bivariate_tolerance: float = 1e-4,
                                        max_rectangles: int = 100000, max_intervals: int = 100000):
    """Candidate global self-intersections of one canal bounding pipe.

    The diagonal of identical parameters is factored out as in AnaMorph.
    Detected unresolved systems raise ClippingConvergenceError; numerical
    clipping does not identify every non-isolated system.
    """
    _validate_interval_limit(max_intervals)
    failures = []
    p = q = None
    try:
        p, q = _global_self_intersection_system(segment)
        bp = BiBernsteinPolynomial.from_power(p).elevate(7, 7)
        bq = BiBernsteinPolynomial.from_power(q).elevate(7, 7)
        roots = bilinear_clip_roots(_unit_scale_polynomial(bp), _unit_scale_polynomial(bq),
                                    tolerance=bivariate_tolerance, max_rectangles=max_rectangles)
    except ClippingConvergenceError as error:
        failures.append(f"interior: {error}")
        roots = error.partial_roots
    points = [(0., 1., "corner", ((0., 0.), (1., 1.)))]
    points.extend((*r.midpoint, "interior", ((r.x0, r.x1), (r.y0, r.y1))) for r in roots)
    try:
        _, original_q = _distance_system(segment, segment)
    except ClippingConvergenceError as error:
        failures.append(f"edge system: {error}")
        original_q = None
    for endpoint in ((0., 1.) if original_q is not None else ()):
        edge = BernsteinPolynomial.from_power(_edge_power(original_q, fixed_axis="x", fixed_value=endpoint))
        try:
            edge_roots = _curve_roots(edge, tolerance=univariate_tolerance, max_intervals=max_intervals)
        except ClippingConvergenceError as error:
            failures.append(f"edge t={endpoint:g}: {error}")
            edge_roots = error.partial_roots
        points.extend((endpoint, r.midpoint, "edge", ((endpoint, endpoint), (r.left, r.right)))
                      for r in edge_roots)
    threshold = (2*Fraction(segment.max_radius))**2
    offset = 2*max(univariate_tolerance, bivariate_tolerance)
    result = []
    def excluded(t, u):
        return (t <= offset and u <= offset) or (t >= 1-offset and u >= 1-offset)
    for t, u, source, bounds in points:
        t_range, u_range = bounds
        if ((t_range[1] <= offset and u_range[1] <= offset) or
                (t_range[0] >= 1-offset and u_range[0] >= 1-offset)):
            continue
        try:
            candidate, ambiguous = _refine_distance_candidate(segment, segment, t, u, source, bounds, threshold,
                                                                excluded=excluded, inclusive=False,
                                                                stationary_system=(p, q) if p is not None else None)
        except ClippingConvergenceError as error:
            failures.append(f"candidate: {error}")
            continue
        if candidate is not None:
            result.append(candidate)
        elif ambiguous:
            failures.append(f"candidate {source} at {bounds}: distance remains ambiguous")
    if failures:
        raise IntersectionSolveError(failures, result)
    return result


def _distance_candidates(first, second, univariate_tolerance, bivariate_tolerance, max_rectangles,
                         max_intervals, blacklist=()):
    points = [(t, u, "corner", ((t, t), (u, u))) for t in (0., 1.) for u in (0., 1.)]
    try:
        p, q = _distance_system(first, second)
    except ClippingConvergenceError as error:
        return points, [f"system: {error}"], None
    # The C++ solver elevates the (5,3) and (3,5) systems to (5,5).
    bp = BiBernsteinPolynomial.from_power(p).elevate(5, 5)
    bq = BiBernsteinPolynomial.from_power(q).elevate(5, 5)
    failures = []
    try:
        roots = bilinear_clip_roots(_unit_scale_polynomial(bp), _unit_scale_polynomial(bq),
                                    tolerance=bivariate_tolerance, max_rectangles=max_rectangles,
                                    blacklist=blacklist)
    except ClippingConvergenceError as error:
        failures.append(f"interior: {error}")
        roots = error.partial_roots
    points.extend((*r.midpoint, "interior", ((r.x0, r.x1), (r.y0, r.y1))) for r in roots)
    for endpoint in (0., 1.):
        try:
            roots_x = _curve_roots(BernsteinPolynomial.from_power(_edge_power(q, fixed_axis="x", fixed_value=endpoint)),
                                   tolerance=univariate_tolerance, max_intervals=max_intervals)
        except ClippingConvergenceError as error:
            failures.append(f"edge t={endpoint:g}: {error}")
            roots_x = error.partial_roots
        for r in roots_x:
            points.append((endpoint, r.midpoint, "edge", ((endpoint, endpoint), (r.left, r.right))))
        try:
            roots_y = _curve_roots(BernsteinPolynomial.from_power(_edge_power(p, fixed_axis="y", fixed_value=endpoint)),
                                   tolerance=univariate_tolerance, max_intervals=max_intervals)
        except ClippingConvergenceError as error:
            failures.append(f"edge u={endpoint:g}: {error}")
            roots_y = error.partial_roots
        for r in roots_y:
            points.append((r.midpoint, endpoint, "edge", ((r.left, r.right), (endpoint, endpoint))))
    return points, failures, (p, q)


def _affine_distance_data(first, second):
    """Exact closest points for two affine curves over the unit square.

    Fractions represent the supplied binary floating-point coefficients
    exactly. All four edge minima and the interior stationary point, when
    unique, cover the minimum of the convex squared-distance quadratic.
    """
    if any(component[2] != 0 or component[3] != 0
           for segment in (first, second) for component in segment.power):
        return None
    start_a = tuple(Fraction(component[0]) for component in first.power)
    start_b = tuple(Fraction(component[0]) for component in second.power)
    u = tuple(Fraction(component[1]) for component in first.power)
    v = tuple(Fraction(component[1]) for component in second.power)
    w = tuple(a-b for a, b in zip(start_a, start_b))
    def dot(a, b):
        return sum((x*y for x, y in zip(a, b)), Fraction(0))
    def clamp(value):
        return max(Fraction(0), min(Fraction(1), value))
    aa, bb, cc, dd, ee = dot(u, u), dot(u, v), dot(v, v), dot(u, w), dot(v, w)
    determinant = aa*cc-bb*bb
    candidates = []
    for t in (Fraction(0), Fraction(1)):
        candidates.append((t, clamp((ee+bb*t)/cc) if cc else Fraction(0)))
    for s in (Fraction(0), Fraction(1)):
        candidates.append((clamp((bb*s-dd)/aa) if aa else Fraction(0), s))
    if determinant > 0:
        t, s = (bb*ee-cc*dd)/determinant, (aa*ee-bb*dd)/determinant
        if 0 <= t <= 1 and 0 <= s <= 1:
            candidates.append((t, s))
    def distance_squared(pair):
        t, s = pair
        delta = tuple(w[i]+t*u[i]-s*v[i] for i in range(3))
        return dot(delta, delta)
    closest = min(candidates, key=distance_squared)
    return determinant == 0, closest, distance_squared(closest)


def _source_initial_linear_fit(first, second):
    """Exact L2 fit slopes of the source's unscaled stationary system.

    Its Legendre projection has x/y slopes 12 times the integral of each
    polynomial against (x-1/2) or (y-1/2). Separability lets us integrate
    the two cubic centerlines directly, using their represented coefficients.
    """
    a = tuple(tuple(map(Fraction, axis)) for axis in first.power)
    b = tuple(tuple(map(Fraction, axis)) for axis in second.power)
    def mean(axis):
        return sum((value/Fraction(i+1) for i, value in enumerate(axis)), Fraction(0))
    def moment(degree):
        return Fraction(degree, 2*(degree+1)*(degree+2))
    px = py = qx = qy = Fraction(0)
    for aa, bb in zip(a, b):
        amean, bmean = mean(aa), mean(bb)
        amoment = sum((value*moment(i) for i, value in enumerate(aa)), Fraction(0))
        bmoment = sum((value*moment(i) for i, value in enumerate(bb)), Fraction(0))
        px += sum(((value-(bmean if i == 0 else 0))*j*aa[j]*moment(i+j-1)
                   for i, value in enumerate(aa) for j in range(1, 4)), Fraction(0))
        qy += sum((((amean if i == 0 else 0)-value)*j*bb[j]*moment(i+j-1)
                   for i, value in enumerate(bb) for j in range(1, 4)), Fraction(0))
        py -= bmoment*sum(aa[1:])
        qx += amoment*sum(bb[1:])
    return tuple(12*value for value in (px, py, qx, qy))


def _source_pair_singular_risk(first, second):
    """Flag pairs whose source bivariate solver may encounter singularity.

    The source uses full-pivot elimination and an absolute 1e-11 threshold.
    This exact-input estimate includes a rounding allowance. Its Bernstein
    construction and fit are floating point, so this is a warning rather than
    proof of what that executable will return.
    """
    px, py, qx, qy = _source_initial_linear_fit(first, second)
    peak = max(abs(px), abs(py), abs(qx), abs(qy))
    if not peak:
        return True
    determinant = abs(px*qy-py*qx)
    # det / largest matrix element is the remaining full-pivot value.
    # Account heuristically for rounding in coefficients and fit sums. The
    # coefficient bound includes absolute positions because source basis
    # conversion can lose detail before it subtracts the two curves.
    coefficient_bound = Fraction(0)
    for axis_a, axis_b in zip(first.power, second.power):
        aa, bb = tuple(map(Fraction, axis_a)), tuple(map(Fraction, axis_b))
        position = sum((abs(x) for x in aa+bb), Fraction(0))
        velocity = max(sum((i*abs(aa[i]) for i in range(1, 4)), Fraction(0)),
                       sum((i*abs(bb[i]) for i in range(1, 4)), Fraction(0)))
        coefficient_bound += position*velocity
    allowance = Fraction(4e-11)+Fraction(256*ulp(1.))*max(peak, coefficient_bound)
    if determinant <= allowance*peak:
        return True
    # Parallel straight centerlines have a shared displacement factor in
    # their stationary system. The source can encounter a singular fit after
    # subdivision even when its initial fit was nonsingular. This is a risk
    # indication, not a claim that the source necessarily throws.
    vectors = [tuple(Fraction(axis[degree]) for axis in curve.power)
               for curve in (first, second) for degree in (1, 2, 3)]
    if not any(any(vectors[i]) for i in range(3)) or not any(any(vectors[i]) for i in range(3, 6)):
        return False
    direction = next(vector for vector in vectors if any(vector))
    axis = next(i for i, value in enumerate(direction) if value)
    return all(vector[i]*direction[axis] == vector[axis]*direction[i]
               for vector in vectors for i in range(3))


def _affine_distance_hits(first, second, data):
    _, (t, u), distance_squared = data
    threshold = Fraction(first.max_radius)+Fraction(second.max_radius)
    if distance_squared > threshold*threshold:
        return []
    value, exact_distance = _report_distance_squared(distance_squared)
    rounded_t, rounded_u = float(t), float(u)
    exact_parameters = (((t.numerator, t.denominator), (u.numerator, u.denominator))
                        if Fraction(rounded_t) != t or Fraction(rounded_u) != u else None)
    return [CurveCandidate(rounded_t, rounded_u, value, "linear_minimum", exact_parameters=exact_parameters,
                           exact_distance_squared=exact_distance)]


def _translated_curve_distance(first, second):
    """Exact minimum for a perpendicular translation of the same curve.

    The second curve may be parameterized in the opposite direction. If the
    nonconstant coefficients agree after reversal and the translation is
    orthogonal to each, the squared distance is at least the translation's
    squared length, with equality at matching parameters. Rational arithmetic
    makes these identities properties of the represented input floats.
    """
    if all(tuple(a[1:]) == tuple(b[1:]) for a, b in zip(first.power, second.power)):
        second_start = tuple(Fraction(b[0]) for b in second.power)
    elif all(a[3] == -b[3] for a, b in zip(first.power, second.power)):
        # Coefficients of b(1-t), computed exactly to avoid cancellation,
        # overflow and accidental equality after a rounded reversal.
        reversed_power = []
        for component in second.power:
            b0, b1, b2, b3 = map(Fraction, component)
            reversed_power.append((b0+b1+b2+b3, -b1-2*b2-3*b3, b2+3*b3, -b3))
        if any(any(Fraction(a[degree]) != b[degree] for degree in (1, 2, 3))
               for a, b in zip(first.power, reversed_power)):
            return None
        second_start = tuple(b[0] for b in reversed_power)
    else:
        return None
    offset = tuple(Fraction(a[0])-b0 for a, b0 in zip(first.power, second_start))
    if any(sum((offset[axis]*Fraction(first.power[axis][degree]) for axis in range(3)), Fraction(0))
           for degree in (1, 2, 3)):
        return None
    return sum((value*value for value in offset), Fraction(0))


_CUBE_RESIDUES = tuple((modulus, frozenset(pow(i, 3, modulus) for i in range(modulus)))
                       for modulus in (7, 9, 13, 19, 37))


def _integer_cube_root_if_exact(value):
    """Return the exact nonnegative integer cube root, or None."""
    if any(value % modulus not in residues for modulus, residues in _CUBE_RESIDUES):
        return None
    if value < 2:
        return value
    root = 1 << ((value.bit_length()+2)//3)
    while True:
        next_root = (2*root+value//(root*root))//3
        if next_root >= root:
            return root if root*root*root == value else None
        root = next_root


def _rational_root_if_exact(value, degree):
    """Return the exact rational square or cube root, or None."""
    if degree == 2:
        if value < 0:
            return None
        numerator, denominator = isqrt(value.numerator), isqrt(value.denominator)
        if numerator*numerator != value.numerator or denominator*denominator != value.denominator:
            return None
        return Fraction(numerator, denominator)
    sign = -1 if value < 0 else 1
    numerator = _integer_cube_root_if_exact(abs(value.numerator))
    denominator = _integer_cube_root_if_exact(value.denominator)
    if numerator is None or denominator is None:
        return None
    return Fraction(sign*numerator, denominator)


def _shifted_translated_curve_minimum(first, second):
    """Exact minimum when the curves share a parameter subinterval.

    Detect second(u) = first(alpha + speed*u) + offset, with nonzero rational
    speed and offset perpendicular to the nonconstant coefficients. If the
    parameter domains overlap, the minimum is attained on that overlap. The
    proof and witness use exact rational values of the input floats.
    """
    cubic_axis = next((i for i in range(3) if first.power[i][3]), None)
    if cubic_axis is not None:
        ratio = Fraction(second.power[cubic_axis][3])/Fraction(first.power[cubic_axis][3])
        speed = _rational_root_if_exact(ratio, 3)
        if speed is None or not speed:
            return None
        speeds = (speed,)
    else:
        quadratic_axis = next((i for i in range(3) if first.power[i][2]), None)
        if quadratic_axis is None or any(second.power[i][3] for i in range(3)):
            return None
        ratio = Fraction(second.power[quadratic_axis][2])/Fraction(first.power[quadratic_axis][2])
        speed = _rational_root_if_exact(ratio, 2)
        if speed is None or not speed:
            return None
        speeds = (speed, -speed)
    a = tuple(tuple(map(Fraction, component)) for component in first.power)
    b = tuple(tuple(map(Fraction, component)) for component in second.power)
    for speed in speeds:
        if cubic_axis is not None:
            alpha = (b[cubic_axis][2]/(speed*speed)-a[cubic_axis][2])/(3*a[cubic_axis][3])
        else:
            alpha = (b[quadratic_axis][1]/speed-a[quadratic_axis][1])/(2*a[quadratic_axis][2])
        if any(b[i][3] != speed**3*a[i][3] or
               b[i][2] != speed**2*(a[i][2]+3*a[i][3]*alpha) or
               b[i][1] != speed*(a[i][1]+2*a[i][2]*alpha+3*a[i][3]*alpha*alpha)
               for i in range(3)):
            continue
        offset = tuple(a[i][0]+alpha*(a[i][1]+alpha*(a[i][2]+alpha*a[i][3]))-b[i][0]
                       for i in range(3))
        if any(sum((offset[i]*a[i][degree] for i in range(3)), Fraction(0)) for degree in (1, 2, 3)):
            continue
        lo = max(Fraction(0), min(alpha, alpha+speed))
        hi = min(Fraction(1), max(alpha, alpha+speed))
        if lo > hi:
            continue
        t = (lo+hi)/2
        u = (t-alpha)/speed
        return sum((value*value for value in offset), Fraction(0)), t, u
    return None


def _parallel_monotone_minimum(first, second):
    """Prove the minimum for parallel straight curves, one affine.

    The other curve may traverse its line at a nonuniform cubic speed. Exact
    rational derivative signs establish that its image is its endpoint
    interval. An affine inverse then supplies an exact witness inside any
    overlapping projection intervals. Return None if the proof does not apply.
    """
    def affine_nonconstant(segment):
        return (not any(axis[2] or axis[3] for axis in segment.power) and
                any(axis[1] for axis in segment.power))
    if affine_nonconstant(first):
        straight, other, swapped = first, second, False
    elif affine_nonconstant(second):
        straight, other, swapped = second, first, True
    else:
        return None
    a = tuple(tuple(map(Fraction, axis)) for axis in straight.power)
    b = tuple(tuple(map(Fraction, axis)) for axis in other.power)
    direction = tuple(axis[1] for axis in a)
    axis = next(i for i, value in enumerate(direction) if value)
    if any(b[i][degree]*direction[axis] != b[axis][degree]*direction[i]
           for degree in (1, 2, 3) for i in range(3)):
        return None
    # q is the scalar coordinate in the direction of the affine curve,
    # relative to its start. The perpendicular displacement is constant.
    norm_squared = sum((value*value for value in direction), Fraction(0))
    q = [sum(((b[i][degree]-(a[i][0] if degree == 0 else 0))*direction[i]
              for i in range(3)), Fraction(0))/norm_squared for degree in range(4)]
    def evaluate_q(t):
        return q[0]+t*(q[1]+t*(q[2]+t*q[3]))
    derivatives = [q[1], q[1]+2*q[2]+3*q[3]]
    if q[3]:
        vertex = -q[2]/(3*q[3])
        if 0 < vertex < 1:
            derivatives.append(q[1]+2*q[2]*vertex+3*q[3]*vertex*vertex)
    if min(derivatives) < 0 < max(derivatives):
        return None
    q0, q1 = evaluate_q(Fraction(0)), evaluate_q(Fraction(1))
    low, high = max(Fraction(0), min(q0, q1)), min(Fraction(1), max(q0, q1))
    if low > high:
        # The closest points of disjoint intervals are endpoint pairs.
        t, u = min(((t, u) for t in (Fraction(0), Fraction(1))
                    for u in (Fraction(0), Fraction(1))),
                   key=lambda pair: abs(pair[0]-evaluate_q(pair[1])))
    else:
        # Prefer an endpoint. If only an interior portion overlaps, binary
        # search for a dyadic parameter with a projection in the overlap.
        u = next((x for x, value in ((Fraction(0), q0), (Fraction(1), q1))
                  if low <= value <= high), None)
        if u is None:
            left, right = Fraction(0), Fraction(1)
            target = (low+high)/2
            for _ in range(2048):
                u = (left+right)/2
                value = evaluate_q(u)
                if low <= value <= high:
                    break
                if (value < target) == (q1 > q0):
                    left = u
                else:
                    right = u
            else:
                return None
        t = evaluate_q(u)
    if swapped:
        t, u = u, t
    return _exact_distance_at(first, second, t, u), t, u


def _parallel_reversing_affine_candidates(first, second, tolerance):
    """Prove some parallel-line decisions when one curve reverses direction.

    Exact rational Bernstein bounds exclude subintervals, and exact sampled
    pairs establish positives. A sign-changing projection bracket also proves
    a tangent witness even if its parameter is irrational. Return None if the
    bounded search cannot establish either result.
    """
    def affine(segment):
        return (not any(axis[2] or axis[3] for axis in segment.power) and
                any(axis[1] for axis in segment.power))
    if affine(first):
        straight, other, swapped = first, second, False
    elif affine(second):
        straight, other, swapped = second, first, True
    else:
        return None
    a = tuple(tuple(map(Fraction, axis)) for axis in straight.power)
    b = tuple(tuple(map(Fraction, axis)) for axis in other.power)
    direction = tuple(axis[1] for axis in a)
    axis = next(i for i, value in enumerate(direction) if value)
    if any(b[i][degree]*direction[axis] != b[axis][degree]*direction[i]
           for degree in (1, 2, 3) for i in range(3)):
        return None
    norm_squared = sum((value*value for value in direction), Fraction(0))
    q = tuple(sum(((b[i][degree]-(a[i][0] if degree == 0 else 0))*direction[i]
                   for i in range(3)), Fraction(0))/norm_squared for degree in range(4))
    derivative = [q[1], q[1]+2*q[2]+3*q[3]]
    if q[3]:
        vertex = -q[2]/(3*q[3])
        if 0 < vertex < 1:
            derivative.append(q[1]+2*q[2]*vertex+3*q[3]*vertex*vertex)
    if not min(derivative) < 0 < max(derivative):
        return None
    offset = tuple(b[i][0]-a[i][0] for i in range(3))
    projection = sum((offset[i]*direction[i] for i in range(3)), Fraction(0))
    perpendicular = sum((value*value for value in offset), Fraction(0))-projection*projection/norm_squared
    threshold = (Fraction(first.max_radius)+Fraction(second.max_radius))**2
    if perpendicular > threshold:
        return []

    def eval_q(u):
        return q[0]+u*(q[1]+u*(q[2]+u*q[3]))

    def witness(u):
        t = max(Fraction(0), min(Fraction(1), eval_q(u)))
        t, u = (u, t) if swapped else (t, u)
        distance = _exact_distance_at(first, second, t, u)
        if distance > threshold:
            return None
        value, exact_distance = _report_distance_squared(distance)
        rt, ru = float(t), float(u)
        exact_parameters = (((t.numerator, t.denominator), (u.numerator, u.denominator))
                            if Fraction(rt) != t or Fraction(ru) != u else None)
        bounds = None if exact_parameters is not None else ((rt, rt), (ru, ru))
        return CurveCandidate(rt, ru, value, "parallel_line_witness", bounds,
                              exact_parameters, exact_distance)

    # Non-dyadic rational extrema matter at exact tangency, where no nearby
    # sample is itself a witness. Solve the linear/quadratic derivative exactly.
    extrema = []
    if not q[3] and q[2]:
        extrema.append(-q[1]/(2*q[2]))
    elif q[3]:
        discriminant = (2*q[2])**2-12*q[3]*q[1]
        if discriminant >= 0:
            numerator, denominator = isqrt(discriminant.numerator), isqrt(discriminant.denominator)
            if numerator*numerator == discriminant.numerator and denominator*denominator == discriminant.denominator:
                square_root = Fraction(numerator, denominator)
                extrema.extend(((-2*q[2]-square_root)/(6*q[3]), (-2*q[2]+square_root)/(6*q[3])))
    for root in extrema:
        if 0 <= root <= 1:
            hit = witness(root)
            if hit is not None:
                return [hit]

    controls = (q[0], q[0]+q[1]/3, q[0]+2*q[1]/3+q[2]/3, sum(q))
    stack = [(Fraction(0), Fraction(1), controls)]
    for _ in range(256):
        if not stack:
            return []
        lo, hi, c = stack.pop()
        low, high = min(c), max(c)
        separation = max(Fraction(0), -high, low-1)
        if perpendicular+norm_squared*separation*separation > threshold:
            continue
        middle = (lo+hi)/2
        for u in (lo, middle, hi):
            hit = witness(u)
            if hit is not None:
                return [hit]
        if perpendicular == threshold:
            # A bracket straddling either end of the affine interval proves
            # existence by continuity, even without a rational root.
            for left, right in ((lo, middle), (middle, hi)):
                qleft, qright = eval_q(left), eval_q(right)
                target = next((Fraction(bound) for bound in (0, 1)
                               if min(qleft, qright) < bound < max(qleft, qright)), None)
                if target is None:
                    continue
                while right-left > Fraction(tolerance):
                    mid = (left+right)/2
                    qmid = eval_q(mid)
                    hit = witness(mid)
                    if hit is not None:
                        return [hit]
                    if (qleft-target)*(qmid-target) < 0:
                        right = mid
                    else:
                        left, qleft = mid, qmid
                low_float, high_float = float(left), float(right)
                if Fraction(low_float) > left:
                    low_float = nextafter(low_float, -inf)
                if Fraction(high_float) < right:
                    high_float = nextafter(high_float, inf)
                fixed = float(target)
                free = (low_float+high_float)/2
                bounds = (((low_float, high_float), (fixed, fixed)) if swapped else
                          ((fixed, fixed), (low_float, high_float)))
                t, u = (free, fixed) if swapped else (fixed, free)
                value, exact_distance = _report_distance_squared(perpendicular)
                return [CurveCandidate(t, u, value, "parallel_line_tangency", bounds,
                                       exact_distance_squared=exact_distance)]
        c01, c12, c23 = (c[0]+c[1])/2, (c[1]+c[2])/2, (c[2]+c[3])/2
        c012, c123 = (c01+c12)/2, (c12+c23)/2
        cmid = (c012+c123)/2
        stack.append((middle, hi, (cmid, c123, c23, c[3])))
        stack.append((lo, middle, (c[0], c01, c012, cmid)))
    return None


def _parallel_nonlinear_line_candidates(first, second, tolerance):
    """Resolve two monotone straight cubic curves using exact projections.

    Returns None when the proof does not apply, otherwise a list of candidates.
    A touching minimum can require an irrational parameter: its interval is a
    rigorous existence witness, not an assertion about the rounded midpoint.
    """
    a = tuple(tuple(map(Fraction, component)) for component in first.power)
    b = tuple(tuple(map(Fraction, component)) for component in second.power)
    direction = next((tuple(axes[i][degree] for i in range(3))
                      for axes in (a, b) for degree in (1, 2, 3)
                      if any(axes[i][degree] for i in range(3))), None)
    if direction is None:
        return None
    axis = next(i for i, value in enumerate(direction) if value)
    if any(axes[i][degree]*direction[axis] != axes[axis][degree]*direction[i]
           for axes in (a, b) for degree in (1, 2, 3) for i in range(3)):
        return None
    norm_squared = sum((value*value for value in direction), Fraction(0))
    def project(axes):
        return tuple(sum(((axes[i][degree]-(a[i][0] if degree == 0 else 0))*direction[i]
                          for i in range(3)), Fraction(0))/norm_squared for degree in range(4))
    qa, qb = project(a), project(b)
    def evaluate_q(q, t):
        return q[0]+t*(q[1]+t*(q[2]+t*q[3]))
    def monotone(q):
        derivative = [q[1], q[1]+2*q[2]+3*q[3]]
        if q[3]:
            vertex = -q[2]/(3*q[3])
            if 0 < vertex < 1:
                derivative.append(q[1]+2*q[2]*vertex+3*q[3]*vertex*vertex)
        return not (min(derivative) < 0 < max(derivative))
    if not monotone(qa) or not monotone(qb):
        return None
    a0, a1 = evaluate_q(qa, Fraction(0)), evaluate_q(qa, Fraction(1))
    b0, b1 = evaluate_q(qb, Fraction(0)), evaluate_q(qb, Fraction(1))
    low = max(min(a0, a1), min(b0, b1))
    high = min(max(a0, a1), max(b0, b1))
    threshold = (Fraction(first.max_radius)+Fraction(second.max_radius))**2
    def exact_hit(t, u, source="parallel_line_witness"):
        distance = _exact_distance_at(first, second, t, u)
        if distance > threshold:
            return None
        value, exact_distance = _report_distance_squared(distance)
        rt, ru = float(t), float(u)
        exact_parameters = (((t.numerator, t.denominator), (u.numerator, u.denominator))
                            if Fraction(rt) != t or Fraction(ru) != u else None)
        bounds = None if exact_parameters is not None else ((rt, rt), (ru, ru))
        return CurveCandidate(rt, ru, value, source, bounds, exact_parameters, exact_distance)
    if low > high:
        t, u = min(((t, u) for t in (Fraction(0), Fraction(1)) for u in (Fraction(0), Fraction(1))),
                   key=lambda pair: _exact_distance_at(first, second, *pair))
        hit = exact_hit(t, u, "parallel_line_minimum")
        return [] if hit is None else [hit]
    displacement = tuple(b[i][0]-a[i][0] for i in range(3))
    projection = sum((displacement[i]*direction[i] for i in range(3)), Fraction(0))
    perpendicular_squared = (sum((value*value for value in displacement), Fraction(0)) -
                             projection*projection/norm_squared)
    if perpendicular_squared > threshold:
        return []
    # One of the four endpoints lies inside the other closed projection
    # interval whenever the two intervals overlap.
    if min(b0, b1) <= a0 <= max(b0, b1):
        fixed, target, free_q, fixed_first = Fraction(0), a0, qb, True
    elif min(b0, b1) <= a1 <= max(b0, b1):
        fixed, target, free_q, fixed_first = Fraction(1), a1, qb, True
    elif min(a0, a1) <= b0 <= max(a0, a1):
        fixed, target, free_q, fixed_first = Fraction(0), b0, qa, False
    else:
        fixed, target, free_q, fixed_first = Fraction(1), b1, qa, False
    def parameters(free):
        return (fixed, free) if fixed_first else (free, fixed)
    left, right = Fraction(0), Fraction(1)
    qleft, qright = evaluate_q(free_q, left), evaluate_q(free_q, right)
    for free, qvalue in ((left, qleft), (right, qright)):
        if qvalue == target:
            return [exact_hit(*parameters(free), source="parallel_line_minimum")]
    # q is monotone and brackets target; an exact equality always exists.
    for _ in range(2048):
        midpoint = (left+right)/2
        qmid = evaluate_q(free_q, midpoint)
        hit = exact_hit(*parameters(midpoint))
        if hit is not None:
            return [hit]
        if (qmid < target) == (qright > qleft):
            left, qleft = midpoint, qmid
        else:
            right, qright = midpoint, qmid
        if perpendicular_squared == threshold and right-left <= Fraction(tolerance):
            # Preserve enclosure when converting exact dyadic bounds to floats.
            lo, hi = float(left), float(right)
            if Fraction(lo) > left:
                lo = nextafter(lo, -inf)
            if Fraction(hi) < right:
                hi = nextafter(hi, inf)
            value, exact_distance = _report_distance_squared(perpendicular_squared)
            bounds = (((float(fixed), float(fixed)), (lo, hi)) if fixed_first else
                      ((lo, hi), (float(fixed), float(fixed))))
            t, u = ((float(fixed), (lo+hi)/2) if fixed_first else ((lo+hi)/2, float(fixed)))
            return [CurveCandidate(t, u, value, "parallel_line_tangency", bounds,
                                   exact_distance_squared=exact_distance)]
    return None


def neurite_neurite_candidates(first: CanalSegment, second: CanalSegment, *,
                               univariate_tolerance: float = 1e-6, bivariate_tolerance: float = 1e-4,
                               max_rectangles: int = 100000, max_intervals: int = 100000):
    """Return stationary/corner candidates whose bounding pipes intersect.

    This translates the nonadjacent segment criterion: interior stationary
    points, stationary points on all four edges, and the four corners.
    Exactly parallel affine curves, many parallel straight curves with
    nonuniform or reversing cubic parameterizations, and overlapping
    portions of the same cubic curve with a perpendicular translation use
    direct geometric tests. Other affine curves use a direct minimum if interior
    clipping fails or returns no witness.
    An unresolved edge search remains reported even when a witness is known.
    Other unresolved systems raise ClippingConvergenceError rather than report clean.
    A positive affine result can carry its squared distance as an exact
    rational when that value cannot be represented as a finite nonzero float.
    """
    _validate_interval_limit(max_intervals)
    if (not isfinite(univariate_tolerance) or univariate_tolerance <= 0 or
            not isfinite(bivariate_tolerance) or bivariate_tolerance <= 0 or max_rectangles < 1):
        raise ValueError("solver tolerances and rectangle limit must be positive")
    affine = _affine_distance_data(first, second)
    if affine is not None and affine[0]:
        return _affine_distance_hits(first, second, affine)
    translated_distance = _translated_curve_distance(first, second)
    if translated_distance is not None:
        minimum = translated_distance, Fraction(1, 2), Fraction(1, 2)
    elif affine is None:
        minimum = _shifted_translated_curve_minimum(first, second)
    else:
        minimum = None
    source = "translated_minimum"
    if minimum is None and affine is None:
        minimum = _parallel_monotone_minimum(first, second)
        source = "parallel_line_minimum"
    if minimum is None and affine is None:
        reversing = _parallel_reversing_affine_candidates(first, second, bivariate_tolerance)
        if reversing is not None:
            return reversing
    if minimum is None and affine is None:
        parallel = _parallel_nonlinear_line_candidates(first, second, bivariate_tolerance)
        if parallel is not None:
            return parallel
    if minimum is not None:
        translated_distance, t, u = minimum
        threshold = Fraction(first.max_radius)+Fraction(second.max_radius)
        if translated_distance > threshold*threshold:
            return []
        value, exact_distance = _report_distance_squared(translated_distance)
        rounded_t, rounded_u = float(t), float(u)
        exact_parameters = (((t.numerator, t.denominator), (u.numerator, u.denominator))
                            if Fraction(rounded_t) != t or Fraction(rounded_u) != u else None)
        bounds = None if exact_parameters is not None else ((rounded_t, rounded_t), (rounded_u, rounded_u))
        return [CurveCandidate(rounded_t, rounded_u, value, source, bounds,
                               exact_parameters=exact_parameters,
                               exact_distance_squared=exact_distance)]
    points, failures, system = _distance_candidates(first, second, univariate_tolerance, bivariate_tolerance,
                                                     max_rectangles, max_intervals)
    threshold = (Fraction(first.max_radius)+Fraction(second.max_radius))**2
    result = []
    for t, u, source, bounds in points:
        try:
            candidate, ambiguous = _refine_distance_candidate(first, second, t, u, source, bounds, threshold,
                                                               stationary_system=system)
        except ClippingConvergenceError as error:
            failures.append(f"candidate: {error}")
            continue
        if candidate is not None:
            result.append(candidate)
        elif ambiguous:
            failures.append(f"candidate {source} at {bounds}: distance remains ambiguous")
    # The exact affine minimum also protects against a silently omitted root.
    # Keep source-style candidates and unresolved edge searches when present;
    # a proven minimum witness can accompany the latter without implying that
    # all stationary candidates were found.
    if affine is not None and not result:
        exact_hits = _affine_distance_hits(first, second, affine)
        if not failures:
            return exact_hits
        result.extend(exact_hits)
    if failures:
        if (affine is not None and
                not any(failure.startswith("edge ") for failure in failures) and
                any(failure.startswith(("system:", "interior:", "candidate ", "candidate:"))
                    for failure in failures)):
            return _affine_distance_hits(first, second, affine)
        raise IntersectionSolveError(failures, result)
    return result


def adjacent_neurite_candidates(first: CanalSegment, second: CanalSegment, *,
                                first_end_second_start: bool,
                                univariate_tolerance: float = 1e-6, bivariate_tolerance: float = 1e-4,
                                max_rectangles: int = 100000, check_end_start: bool = False,
                                max_intervals: int = 100000):
    """Translate the adjacent-segment intersection check in this C++ version.

    In the checked-in C++ source, consecutive end-to-start segments return false
    immediately, unless the optional supplementary check is enabled. For two
    segments sharing their starts, the source blacklists
    the [0,.01]² corner to allow their intended junction. This is a numerical
    heuristic and not a proof that intersections in that region are harmless.
    """
    _validate_interval_limit(max_intervals)
    t = 1. if first_end_second_start else 0.
    # The shared endpoint is evaluated exactly from the input coefficients.
    # Allow local geometric error plus a small number of coefficient ulps,
    # rather than a unit-size or coordinate-magnitude tolerance floor.
    used_coefficients = ([c for component in first.power for c in component] if t else
                         [component[0] for component in first.power])
    used_coefficients.extend(component[0] for component in second.power)
    rounding = max(Fraction(ulp(c)) for c in used_coefficients)
    tolerance = Fraction(1e-9)*max(Fraction(first.max_radius), Fraction(second.max_radius))+16*rounding
    if _exact_distance_at(first, second, t, 0.) > tolerance*tolerance:
        raise ValueError("adjacent canal segments must share the specified endpoint")
    if first_end_second_start and not check_end_start:
        return []
    offset = .01
    blacklist = (RootRectangle(1.-offset, 1., 0., offset) if first_end_second_start
                 else RootRectangle(0., offset, 0., offset))
    points, failures, system = _distance_candidates(first, second, univariate_tolerance, bivariate_tolerance,
                                                     max_rectangles, max_intervals, blacklist=(blacklist,))
    threshold = (Fraction(first.max_radius)+Fraction(second.max_radius))**2
    result = []
    def excluded(t, u):
        return (t >= 1.-offset if first_end_second_start else t <= offset) and u <= offset
    for t, u, source, bounds in points:
        t_range, u_range = bounds
        if ((t_range[0] >= 1.-offset if first_end_second_start else t_range[1] <= offset) and
                u_range[1] <= offset):
            continue
        try:
            candidate, ambiguous = _refine_distance_candidate(first, second, t, u, source, bounds, threshold,
                                                               excluded=excluded, stationary_system=system)
        except ClippingConvergenceError as error:
            failures.append(f"candidate: {error}")
            continue
        if candidate is not None:
            result.append(candidate)
        elif ambiguous:
            failures.append(f"candidate {source} at {bounds}: distance remains ambiguous")
    if failures:
        raise IntersectionSolveError(failures, result)
    return result
