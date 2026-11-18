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

"""Bernstein polynomials and univariate Bézier clipping.

Based on AnaMorph (https://github.com/NeuroBox3D/AnaMorph), specifically
PolyAlg::convertBasis and PolyAlg::BezClip_roots. Results are floating-point
root *candidates*, not mathematically certified interval enclosures.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from math import comb, frexp, isfinite, ldexp, ulp
from typing import Sequence


class ClippingConvergenceError(RuntimeError):
    """An incomplete search, optionally retaining numerical root candidates."""

    def __init__(self, message, *, partial_roots=(), processed=None, pending=None):
        super().__init__(message)
        self.partial_roots = tuple(partial_roots)
        self.processed = processed
        self.pending = pending


@dataclass(frozen=True, slots=True)
class RootInterval:
    """A root candidate; ``certified`` marks a separately proved enclosure."""
    left: float
    right: float
    certified: bool = False

    @property
    def midpoint(self) -> float:
        return (self.left+self.right)/2


@dataclass(frozen=True, slots=True)
class BernsteinPolynomial:
    coefficients: tuple[float, ...]

    def __post_init__(self):
        if not self.coefficients or not all(isfinite(x) for x in self.coefficients):
            raise ValueError("coefficients must be nonempty and finite")

    @classmethod
    def from_power(cls, coefficients: Sequence[float], degree: int | None = None):
        """Convert ascending power coefficients to Bernstein coefficients."""
        power = tuple(coefficients)
        if not power:
            raise ValueError("coefficients must be nonempty")
        degree = len(power)-1 if degree is None else degree
        if degree < len(power)-1:
            raise ValueError("Bernstein degree must be at least the power degree")
        power += (0.0,)*(degree+1-len(power))
        result = []
        for i in range(degree+1):
            value = power[0]
            factor = 1.0
            for j in range(1, i+1):
                factor *= (i-j+1)/(degree-j+1)
                value += factor*power[j]
            result.append(value)
        return cls(tuple(result))

    @property
    def degree(self) -> int:
        return len(self.coefficients)-1

    def evaluate(self, t: float) -> float:
        if not 0 <= t <= 1:
            raise ValueError("Bernstein parameter must lie in [0, 1]")
        values = list(self.coefficients)
        for size in range(len(values)-1, 0, -1):
            for i in range(size):
                values[i] = (1-t)*values[i]+t*values[i+1]
        return values[0]

    def split(self, t: float) -> tuple[BernsteinPolynomial, BernsteinPolynomial]:
        if not 0 <= t <= 1:
            raise ValueError("split parameter must lie in [0, 1]")
        work = list(self.coefficients)
        left, right = [work[0]], [work[-1]]
        while len(work) > 1:
            work = [(1-t)*a+t*b for a, b in zip(work, work[1:])]
            left.append(work[0])
            right.append(work[-1])
        return BernsteinPolynomial(tuple(left)), BernsteinPolynomial(tuple(reversed(right)))

    def restrict(self, left: float, right: float) -> BernsteinPolynomial:
        if not 0 <= left <= right <= 1:
            raise ValueError("restriction must lie within [0, 1]")
        if left == right:
            return BernsteinPolynomial((self.evaluate(left),)*(self.degree+1))
        poly = self if left == 0 else self.split(left)[1]
        return poly if right == 1 else poly.split((right-left)/(1-left))[0]

    def derivative(self) -> BernsteinPolynomial:
        n = self.degree
        return BernsteinPolynomial(tuple(n*(b-a) for a, b in zip(self.coefficients, self.coefficients[1:]))) if n else BernsteinPolynomial((0.0,))

    def elevate(self, target_degree: int) -> BernsteinPolynomial:
        if target_degree < self.degree:
            raise ValueError("cannot lower a Bernstein polynomial's degree")
        coeffs = self.coefficients
        while len(coeffs)-1 < target_degree:
            n = len(coeffs)
            coeffs = (coeffs[0],) + tuple((i/n)*coeffs[i-1]+(1-i/n)*coeffs[i] for i in range(1, n)) + (coeffs[-1],)
        return BernsteinPolynomial(coeffs)

    def multiply(self, other: BernsteinPolynomial) -> BernsteinPolynomial:
        m, n = self.degree, other.degree
        return BernsteinPolynomial(tuple(
            sum(comb(m, i)*comb(n, k-i)*self.coefficients[i]*other.coefficients[k-i]
                for i in range(max(0, k-n), min(m, k)+1))/comb(m+n, k)
            for k in range(m+n+1)))


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    def turn(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    points = sorted(set(points))
    if len(points) <= 1:
        return points
    lower = []
    for p in points:
        while len(lower) >= 2 and turn(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and turn(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _hull_strip_bounds(poly: BernsteinPolynomial, eps: float) -> tuple[float, float] | None:
    """Project the control polygon's convex hull inside |y| <= eps onto x."""
    if poly.degree == 0:
        return (0.0, 1.0) if abs(poly.coefficients[0]) <= eps else None
    hull = _convex_hull([(i/poly.degree, y) for i, y in enumerate(poly.coefficients)])
    xs = [x for x, y in hull if abs(y) <= eps]
    if len(hull) > 1:
        for (x0, y0), (x1, y1) in zip(hull, hull[1:]+hull[:1]):
            if y0 == y1:
                continue
            for boundary in (-eps, eps):
                t = (boundary-y0)/(y1-y0)
                if 0 <= t <= 1:
                    xs.append(x0+t*(x1-x0))
    return (max(0.0, min(xs)), min(1.0, max(xs))) if xs else None


def _merge_intervals(candidates, tolerance):
    merged = []
    for root in sorted(candidates, key=lambda r: (r.left, r.right)):
        if merged and root.left <= merged[-1].right + tolerance*0.01:
            previous = merged.pop()
            merged.append(RootInterval(previous.left, max(previous.right, root.right)))
        else:
            merged.append(root)
    return merged


def bezier_clip_roots(poly: BernsteinPolynomial, *, left: float = 0.0, right: float = 1.0,
                      tolerance: float = 1e-6, eps: float = 1e-11,
                      max_intervals: int = 100000) -> list[RootInterval]:
    """Isolate root candidates using control-hull clipping and subdivision.

    A control polygon entirely inside the epsilon strip, including a numerical
    roundoff allowance, is returned as a candidate interval, which can be wider
    than `tolerance`. The allowance is heuristic, not a certified enclosure.
    Unlike the C++ midpoint special case, both halves are retained so nearby
    roots are not silently discarded. Intersecting/adjacent candidates are
    merged. An exhausted work limit raises, retaining completed numerical
    candidates in ``error.partial_roots``; the remaining search is incomplete.
    """
    if not all(isfinite(x) for x in (left, right, tolerance, eps)) or not 0 <= left <= right <= 1:
        raise ValueError("invalid clipping domain")
    if tolerance <= 0 or eps < 0 or max_intervals < 1:
        raise ValueError("tolerance and work limit must be positive; eps must be nonnegative")
    # This initial hull check compares the supplied coefficients directly and
    # needs no allowance for arithmetic. Subsequent subdivision and hull
    # clipping do round, so even eps=0 must leave a small numerical margin.
    if min(poly.coefficients) > eps or max(poly.coefficients) < -eps:
        return []
    magnitude = max(abs(c) for c in poly.coefficients)
    if magnitude <= eps:
        return [RootInterval(left, right)]
    _, exponent = frexp(magnitude)
    scaled = tuple(ldexp(value, -exponent) for value in poly.coefficients)
    if any(ldexp(value, exponent) != original for value, original in zip(scaled, poly.coefficients)):
        raise ClippingConvergenceError("univariate coefficient scale ratio exceeds float range")
    poly = BernsteinPolynomial(scaled)
    eps = ldexp(eps, -exponent)
    margin = max(eps, 16*(poly.degree+1)*ulp(max(abs(c) for c in poly.coefficients)))
    if left == right:
        return [RootInterval(left, right)] if abs(poly.evaluate(left)) <= margin else []
    queue = deque([(left, right)])
    candidates = []
    visited = 0
    while queue:
        lo, hi = queue.popleft()
        visited += 1
        if visited > max_intervals:
            raise ClippingConvergenceError("Bézier clipping exceeded the interval limit",
                                           partial_roots=_merge_intervals(candidates, tolerance),
                                           processed=visited-1, pending=len(queue)+1)
        current = poly.restrict(lo, hi)
        bounds = _hull_strip_bounds(current, margin)
        if bounds is None:
            continue
        if max(abs(x) for x in current.coefficients) <= margin:
            candidates.append(RootInterval(lo, hi))
            continue
        new_lo = lo + (hi-lo)*bounds[0]
        new_hi = lo + (hi-lo)*bounds[1]
        if new_hi-new_lo <= tolerance:
            candidates.append(RootInterval(new_lo, new_hi))
        elif new_hi-new_lo <= (hi-lo)*0.5:
            queue.append((new_lo, new_hi))
        else:
            mid = (new_lo+new_hi)/2
            if mid == new_lo or mid == new_hi:
                raise ClippingConvergenceError("Bézier clipping reached floating-point resolution",
                                               partial_roots=_merge_intervals(candidates, tolerance),
                                               processed=visited, pending=len(queue)+1)
            queue.append((new_lo, mid))
            queue.append((mid, new_hi))
    return _merge_intervals(candidates, tolerance)


@dataclass(frozen=True, slots=True)
class RootRectangle:
    x0: float
    x1: float
    y0: float
    y1: float

    @property
    def midpoint(self) -> tuple[float, float]:
        return ((self.x0+self.x1)/2, (self.y0+self.y1)/2)


@dataclass(frozen=True, slots=True)
class BiBernsteinPolynomial:
    """Tensor-product Bernstein coefficients indexed by x, then y."""
    coefficients: tuple[tuple[float, ...], ...]

    def __post_init__(self):
        if (not self.coefficients or not self.coefficients[0] or
                any(len(row) != len(self.coefficients[0]) for row in self.coefficients) or
                not all(isfinite(c) for row in self.coefficients for c in row)):
            raise ValueError("bivariate coefficients must form a nonempty finite rectangle")

    @property
    def degree_x(self):
        return len(self.coefficients)-1

    @property
    def degree_y(self):
        return len(self.coefficients[0])-1

    @classmethod
    def from_power(cls, coefficients: Sequence[Sequence[float]]):
        """Convert rectangular ascending-power coefficients in both variables."""
        coeffs = tuple(tuple(row) for row in coefficients)
        if not coeffs or not coeffs[0] or any(len(row) != len(coeffs[0]) for row in coeffs):
            raise ValueError("bivariate power coefficients must be rectangular")
        xdegree, ydegree = len(coeffs)-1, len(coeffs[0])-1
        by_x = [BernsteinPolynomial.from_power([coeffs[i][j] for i in range(xdegree+1)]).coefficients
                for j in range(ydegree+1)]
        return cls(tuple(BernsteinPolynomial.from_power([by_x[j][i] for j in range(ydegree+1)]).coefficients
                         for i in range(xdegree+1)))

    def evaluate(self, x: float, y: float) -> float:
        return BernsteinPolynomial(tuple(BernsteinPolynomial(row).evaluate(y) for row in self.coefficients)).evaluate(x)

    def restrict(self, x0: float, x1: float, y0: float, y1: float) -> BiBernsteinPolynomial:
        if not 0 <= x0 <= x1 <= 1 or not 0 <= y0 <= y1 <= 1:
            raise ValueError("restriction must lie within the unit square")
        after_x = [BernsteinPolynomial(tuple(self.coefficients[i][j] for i in range(self.degree_x+1)))
                   .restrict(x0, x1).coefficients for j in range(self.degree_y+1)]
        return BiBernsteinPolynomial(tuple(
            BernsteinPolynomial(tuple(after_x[j][i] for j in range(self.degree_y+1)))
            .restrict(y0, y1).coefficients for i in range(self.degree_x+1)))

    def elevate(self, degree_x: int, degree_y: int) -> BiBernsteinPolynomial:
        if degree_x < self.degree_x or degree_y < self.degree_y:
            raise ValueError("cannot lower a Bernstein polynomial's degree")
        after_x = [BernsteinPolynomial(tuple(self.coefficients[i][j] for i in range(self.degree_x+1)))
                   .elevate(degree_x).coefficients for j in range(self.degree_y+1)]
        return BiBernsteinPolynomial(tuple(
            BernsteinPolynomial(tuple(after_x[j][i] for j in range(self.degree_y+1)))
            .elevate(degree_y).coefficients for i in range(degree_x+1)))

    def linear_fit(self) -> tuple[float, float, float, float]:
        """Return constant, x, y coefficients and the Bernstein residual bound.

        This is the source algorithm's L2 projection on the shifted Legendre
        basis {1, 2x-1, 2y-1}, converted to the ordinary linear basis.
        """
        m, n = self.degree_x, self.degree_y
        average = sum(sum(row) for row in self.coefficients)/((m+1)*(n+1))
        cx = 3*sum(self.coefficients[i][j]*(2*(i+1)/(m+2)-1)
                   for i in range(m+1) for j in range(n+1))/((m+1)*(n+1))
        cy = 3*sum(self.coefficients[i][j]*(2*(j+1)/(n+2)-1)
                   for i in range(m+1) for j in range(n+1))/((m+1)*(n+1))
        a, b, c = average-cx-cy, 2*cx, 2*cy
        residual = max(abs(self.coefficients[i][j] -
                           (a+b*(i/m if m else 0.0)+c*(j/n if n else 0.0)))
                       for i in range(m+1) for j in range(n+1))
        return a, b, c, residual


def _clip_half_plane(polygon: list[tuple[float, float]], a: float, b: float, c: float):
    """Intersect a convex polygon with a+b*x+c*y <= 0."""
    if not polygon:
        return []
    result = []
    previous = polygon[-1]
    fprev = a+b*previous[0]+c*previous[1]
    for current in polygon:
        fcur = a+b*current[0]+c*current[1]
        if (fprev <= 0) != (fcur <= 0):
            t = fprev/(fprev-fcur)
            result.append((previous[0]+t*(current[0]-previous[0]),
                           previous[1]+t*(current[1]-previous[1])))
        if fcur <= 0:
            result.append(current)
        previous, fprev = current, fcur
    return result


def _proportional_vectors(first_values, second_values) -> bool:
    pairs = tuple(zip(first_values, second_values))
    reference = next(((a, b) for a, b in pairs if a or b), None)
    if reference is None:
        return True
    first, second = reference
    if not first:
        return all(not a for a, _ in pairs)
    if not second:
        return all(not b for _, b in pairs)
    # Equal exact products necessarily round to equal float products. This
    # fast rejection avoids rational arithmetic for normal independent jobs.
    if any(a*second != b*first for a, b in pairs):
        return False
    first, second = Fraction(first), Fraction(second)
    return all(Fraction(a)*second == Fraction(b)*first for a, b in pairs)


def _exactly_proportional(p: BiBernsteinPolynomial, q: BiBernsteinPolynomial) -> bool:
    """Detect exactly dependent coefficient vectors in the same basis."""
    if (p.degree_x, p.degree_y) != (q.degree_x, q.degree_y):
        return False
    return _proportional_vectors((c for row in p.coefficients for c in row),
                                 (c for row in q.coefficients for c in row))


def _shared_axis_root(p: BiBernsteinPolynomial, q: BiBernsteinPolynomial, *, axis: str) -> bool:
    """Detect a shared axis factor whose endpoint signs guarantee a root."""
    if axis == "x":
        if p.degree_x != q.degree_x or p.degree_x == 0:
            return False
        vectors = [tuple(poly.coefficients[i][j] for i in range(poly.degree_x+1))
                   for poly in (p, q) for j in range(poly.degree_y+1)]
    else:
        if p.degree_y != q.degree_y or p.degree_y == 0:
            return False
        vectors = [row for poly in (p, q) for row in poly.coefficients]
    reference = next((vector for vector in vectors if any(vector)), None)
    if reference is None:
        return False
    if not (reference[0] <= 0 <= reference[-1] or reference[-1] <= 0 <= reference[0]):
        return False
    return all(_proportional_vectors(reference, vector) for vector in vectors)


def _exact_affine(poly: BiBernsteinPolynomial):
    """Return exact (constant, x, y) coefficients for an affine polynomial."""
    m, n = poly.degree_x, poly.degree_y
    base = poly.coefficients[0][0]
    dx = poly.coefficients[-1][0]-base if m else 0.
    dy = poly.coefficients[0][-1]-base if n else 0.
    magnitude = max(abs(value) for row in poly.coefficients for value in row)
    margin = 32*ulp(magnitude)
    if isfinite(dx) and isfinite(dy):
        for i, row in enumerate(poly.coefficients):
            for j, value in enumerate(row):
                estimate = base+dx*(i/m if m else 0.)+dy*(j/n if n else 0.)
                if isfinite(estimate) and abs(value-estimate) > margin:
                    return None
    a = Fraction(base)
    b = Fraction(poly.coefficients[-1][0])-a if m else Fraction(0)
    c = Fraction(poly.coefficients[0][-1])-a if n else Fraction(0)
    if not b and not c:
        return None
    if any(Fraction(value) != a+(b*Fraction(i, m) if m else 0)+(c*Fraction(j, n) if n else 0)
           for i, row in enumerate(poly.coefficients) for j, value in enumerate(row)):
        return None
    return a, b, c


def _exact_power_coefficients(poly: BiBernsteinPolynomial):
    """Convert tensor-product Bernstein coefficients to exact power basis."""
    def convert(values):
        n = len(values)-1
        return [comb(n, k)*sum(((-1)**(k-i)*comb(k, i)*values[i] for i in range(k+1)), Fraction(0))
                for k in range(n+1)]
    columns = [convert([Fraction(poly.coefficients[i][j]) for i in range(poly.degree_x+1)])
               for j in range(poly.degree_y+1)]
    return [convert([columns[j][i] for j in range(poly.degree_y+1)])
            for i in range(poly.degree_x+1)]


def _shared_affine_line(first: BiBernsteinPolynomial, second: BiBernsteinPolynomial) -> bool:
    """Prove an affine factor is shared and crosses a line segment in the square."""
    affine = _exact_affine(first)
    if affine is None:
        return False
    a, b, c = affine
    corners = (a, a+b, a+c, a+b+c)
    if not (min(corners) < 0 < max(corners) or sum(value == 0 for value in corners) >= 2):
        return False
    power = _exact_power_coefficients(second)
    if not b:
        y = -a/c
        return all(sum((coefficient*y**j for j, coefficient in enumerate(row)), Fraction(0)) == 0
                   for row in power)
    # Evaluate the power polynomial at x=-(a+c*y)/b with polynomial Horner
    # arithmetic. A zero remainder proves divisibility by the affine factor.
    constant, slope = -a/b, -c/b
    remainder = [Fraction(0)]
    for row in reversed(power):
        result = [Fraction(0)]*(len(remainder)+1)
        for j, value in enumerate(remainder):
            result[j] += constant*value
            result[j+1] += slope*value
        for j, value in enumerate(row):
            if j == len(result):
                result.append(value)
            else:
                result[j] += value
        remainder = result
    return all(value == 0 for value in remainder)


def _exact_polynomial_divides(divisor, dividend) -> bool:
    """Test exact divisibility of two rational power-basis polynomials."""
    def terms(matrix):
        return {(i, j): value for i, row in enumerate(matrix) for j, value in enumerate(row) if value}
    denominator, remainder = terms(divisor), terms(dividend)
    if not denominator or not remainder:
        return False
    leading = max(denominator)
    if leading == (0, 0):
        return False
    while remainder:
        power = max(remainder)
        if power[0] < leading[0] or power[1] < leading[1]:
            return False
        shift = (power[0]-leading[0], power[1]-leading[1])
        factor = remainder[power]/denominator[leading]
        for (i, j), coefficient in denominator.items():
            key = (i+shift[0], j+shift[1])
            value = remainder.get(key, Fraction(0))-factor*coefficient
            if value:
                remainder[key] = value
            else:
                remainder.pop(key, None)
    return True


def _shared_quadratic_with_affine_cofactors(p: BiBernsteinPolynomial,
                                             q: BiBernsteinPolynomial) -> bool:
    """Prove a shared sign-changing quadratic, even between sampling lines.

    For cubic polynomials p=L1*H and q=L2*H, find affine L1,L2 by solving
    q*L1-p*L2=0 over the exact rationals. Verify both divisions, then find
    the exact extrema of H on the unit square. Opposite signs prove that its
    zero set contains a non-isolated curve in the domain.
    """
    if (max(p.degree_x, p.degree_y, q.degree_x, q.degree_y) > 7 or
            max(p.degree_x+p.degree_y, q.degree_x+q.degree_y) < 3):
        return False
    def may_have_cubic_total_degree(poly):
        if poly.degree_x <= 3 and poly.degree_y <= 3:
            return True
        magnitude = max(abs(value) for row in poly.coefficients for value in row)
        if not magnitude:
            return True
        _, exponent = frexp(magnitude)
        differences = [[ldexp(value, -exponent) for value in row] for row in poly.coefficients]
        if any(original and not value for original_row, row in zip(poly.coefficients, differences)
               for original, value in zip(original_row, row)):
            return True  # Let the exact check decide when scaling loses data.
        for i in range(poly.degree_x+1):
            row = differences[0][:]
            for j in range(poly.degree_y+1):
                if i+j > 3 and abs(row[0]) > 64*(2**(i+j))*ulp(1.):
                    return False
                row = [b-a for a, b in zip(row, row[1:])]
            differences = [[b-a for a, b in zip(first, second)]
                           for first, second in zip(differences, differences[1:])]
        return True
    if not may_have_cubic_total_degree(p) or not may_have_cubic_total_degree(q):
        return False
    def terms(poly):
        matrix = _exact_power_coefficients(poly)
        return {(i, j): value for i, row in enumerate(matrix) for j, value in enumerate(row) if value}
    first, second = terms(p), terms(q)
    if not first or not second:
        return False
    degrees = (max(i+j for i, j in first), max(i+j for i, j in second))
    # Lower-degree common quadratics are proportional and are handled earlier.
    if max(degrees) != 3 or min(degrees) < 2:
        return False
    shifts = ((0, 0), (1, 0), (0, 1))
    rows = {}
    for offset, polynomial, multiplier in ((0, second, 1), (3, first, -1)):
        for col, (di, dj) in enumerate(shifts, offset):
            for (i, j), value in polynomial.items():
                row = rows.setdefault((i+di, j+dj), [Fraction(0)]*6)
                row[col] += multiplier*value
    matrix = [row for row in rows.values() if any(row)]
    pivots = []
    for column in range(6):
        index = next((i for i in range(len(pivots), len(matrix)) if matrix[i][column]), None)
        if index is None:
            continue
        pivot_row = len(pivots)
        matrix[pivot_row], matrix[index] = matrix[index], matrix[pivot_row]
        factor = matrix[pivot_row][column]
        matrix[pivot_row] = [value/factor for value in matrix[pivot_row]]
        for i, row in enumerate(matrix):
            if i != pivot_row and row[column]:
                factor = row[column]
                matrix[i] = [a-factor*b for a, b in zip(row, matrix[pivot_row])]
        pivots.append(column)
    free_columns = [column for column in range(6) if column not in pivots]

    def divide(dividend, coefficients):
        divisor = {power: value for power, value in zip(shifts, coefficients) if value}
        if not divisor:
            return None
        remainder, quotient = dict(dividend), {}
        leading = max(divisor)
        while remainder:
            power = max(remainder)
            if power[0] < leading[0] or power[1] < leading[1]:
                return None
            shift = (power[0]-leading[0], power[1]-leading[1])
            factor = remainder[power]/divisor[leading]
            quotient[shift] = quotient.get(shift, Fraction(0))+factor
            for (i, j), value in divisor.items():
                key = (i+shift[0], j+shift[1])
                remaining = remainder.get(key, Fraction(0))-factor*value
                if remaining:
                    remainder[key] = remaining
                else:
                    remainder.pop(key, None)
        return {power: value for power, value in quotient.items() if value}

    def changes_sign(quadratic):
        if not quadratic or max(i+j for i, j in quadratic) != 2:
            return False
        a, b, c = (quadratic.get(power, Fraction(0)) for power in ((2, 0), (1, 1), (0, 2)))
        d, e, f = (quadratic.get(power, Fraction(0)) for power in ((1, 0), (0, 1), (0, 0)))
        zero, one = Fraction(0), Fraction(1)
        points = [(x, y) for x in (zero, one) for y in (zero, one)]
        for side in (zero, one):
            if a:
                x = -(b*side+d)/(2*a)
                if zero < x < one:
                    points.append((x, side))
            if c:
                y = -(b*side+e)/(2*c)
                if zero < y < one:
                    points.append((side, y))
        determinant = 4*a*c-b*b
        if determinant:
            x, y = (b*e-2*c*d)/determinant, (b*d-2*a*e)/determinant
            if zero < x < one and zero < y < one:
                points.append((x, y))
        values = [a*x*x+b*x*y+c*y*y+d*x+e*y+f for x, y in points]
        return min(values) < 0 < max(values)

    for free in free_columns:
        vector = [Fraction(0)]*6
        vector[free] = Fraction(1)
        for row, pivot in reversed(list(zip(matrix, pivots))):
            vector[pivot] = -sum((row[j]*vector[j] for j in free_columns), Fraction(0))
        first_quotient, second_quotient = divide(first, vector[:3]), divide(second, vector[3:])
        if first_quotient is not None and first_quotient == second_quotient and changes_sign(first_quotient):
            return True
    return False


def _shared_sign_changing_factor(p: BiBernsteinPolynomial, q: BiBernsteinPolynomial) -> bool:
    """Prove one polynomial divides the other and changes sign in the square."""
    grid = (0., .5, 1.)
    def likely_change(poly):
        values = [poly.evaluate(x, y) for x in grid for y in grid]
        return min(values) < 0 < max(values)
    candidates = (likely_change(p), likely_change(q))
    if not any(candidates):
        return False
    powers = (_exact_power_coefficients(p), _exact_power_coefficients(q))
    def exact_change(power):
        values = []
        for x in map(Fraction, grid):
            for y in map(Fraction, grid):
                values.append(sum((coefficient*x**i*y**j for i, row in enumerate(power)
                                   for j, coefficient in enumerate(row)), Fraction(0)))
        return min(values) < 0 < max(values)
    return any(candidates[i] and _exact_polynomial_divides(powers[i], powers[1-i]) and
               exact_change(powers[i]) for i in (0, 1))


def _trim_polynomial(values):
    values = list(values)
    while values and not values[-1]:
        values.pop()
    return values


def _univariate_remainder(first, second):
    result = _trim_polynomial(first)
    while len(result) >= len(second) and result:
        factor = result[-1]/second[-1]
        shift = len(result)-len(second)
        for i, coefficient in enumerate(second):
            result[i+shift] -= factor*coefficient
        result = _trim_polynomial(result)
    return result


def _univariate_gcd(first, second):
    first, second = _trim_polynomial(first), _trim_polynomial(second)
    while second:
        first, second = second, _univariate_remainder(first, second)
    return [value/first[-1] for value in first] if first else []


def _has_unit_interval_root(polynomial):
    """Decide whether an exact rational univariate polynomial has a root in [0,1]."""
    def at(poly, x):
        value = Fraction(0)
        for coefficient in reversed(poly):
            value = value*x+coefficient
        return value
    polynomial = _trim_polynomial(polynomial)
    if len(polynomial) < 2:
        return False
    if not at(polynomial, 0) or not at(polynomial, 1):
        return True
    derivative = [i*polynomial[i] for i in range(1, len(polynomial))]
    sequence = [polynomial, _trim_polynomial(derivative)]
    while sequence[-1]:
        remainder = _univariate_remainder(sequence[-2], sequence[-1])
        if not remainder:
            break
        sequence.append([-value for value in remainder])
    def variations(x):
        signs = [1 if value > 0 else -1 for poly in sequence if (value := at(poly, x))]
        return sum(first != second for first, second in zip(signs, signs[1:]))
    return variations(0) > variations(1)


def _modular_shared_x_candidate(p: BiBernsteinPolynomial, q: BiBernsteinPolynomial) -> bool:
    """Cheap one-sided filter; false negatives only skip the optional exact test."""
    prime = 1000000007
    def specialize(poly, numerator, denominator):
        y = numerator*pow(denominator, -1, prime) % prime
        bernstein_x = []
        for row in poly.coefficients:
            values = []
            for value in row:
                a, b = value.as_integer_ratio()
                values.append((a % prime)*pow(b % prime, -1, prime) % prime)
            while len(values) > 1:
                values = [((1-y)*a+y*b) % prime for a, b in zip(values, values[1:])]
            bernstein_x.append(values[0])
        n = poly.degree_x
        result = [(comb(n, k)*sum((-1)**(k-i)*comb(k, i)*bernstein_x[i] for i in range(k+1))) % prime
                  for k in range(n+1)]
        return _trim_polynomial(result)
    def degree_gcd(first, second):
        while second:
            remainder = first[:]
            inverse = pow(second[-1], -1, prime)
            while len(remainder) >= len(second) and remainder:
                factor = remainder[-1]*inverse % prime
                offset = len(remainder)-len(second)
                for i, value in enumerate(second):
                    remainder[i+offset] = (remainder[i+offset]-factor*value) % prime
                remainder = _trim_polynomial(remainder)
            first, second = second, remainder
        return len(first)-1
    for numerator, denominator in ((1, 2), (1, 3)):
        first, second = specialize(p, numerator, denominator), specialize(q, numerator, denominator)
        if not first or not second or degree_gcd(first, second) > 0:
            return True
    return False


def _shared_x_factor_with_real_candidate(p: BiBernsteinPolynomial, q: BiBernsteinPolynomial,
                                         *, max_samples: int = 129) -> bool:
    """Prove a shared x-dependent factor and locate a possible real common root.

    The resultant's y degree is bounded by m_p*n_q + m_q*n_p. If exact
    specialized univariate gcds are nonconstant at more distinct y values
    than that bound, the resultant vanishes identically. The grid-based
    search may miss factors; it never treats an unproved factor as shared.
    """
    if not _modular_shared_x_candidate(p, q):
        return False
    first, second = _exact_power_coefficients(p), _exact_power_coefficients(q)
    def degrees(power):
        terms = [(i, j) for i, row in enumerate(power) for j, value in enumerate(row) if value]
        return (max(i for i, _ in terms), max(j for _, j in terms)) if terms else (0, 0)
    mx, my = degrees(first)
    nx, ny = degrees(second)
    if not mx or not nx:
        return False
    bound = mx*ny+nx*my
    # Detection is supplementary to the numerical search. Respect its work
    # limit and avoid an unbounded exact-algebra prepass on high-degree input.
    if bound+1 > min(max_samples, 129):
        return False
    samples = min(2*bound+3, int(max_samples), 129)
    zeros, real_candidates = 0, 0
    for index in range(samples):
        y = Fraction(index, samples-1)
        def specialize(power):
            values = []
            for row in power:
                value = Fraction(0)
                for coefficient in reversed(row):
                    value = value*y+coefficient
                values.append(value)
            return _trim_polynomial(values)
        left, right = specialize(first), specialize(second)
        if not left or not right:
            zeros += 1
        else:
            gcd = _univariate_gcd(left, right)
            if len(gcd) > 1:
                zeros += 1
                if _has_unit_interval_root(gcd):
                    real_candidates += 1
            elif len(left) == mx+1 and len(right) == nx+1:
                return False
        if zeros > bound and real_candidates >= 2:
            return True
    return False


def bilinear_clip_roots(p: BiBernsteinPolynomial, q: BiBernsteinPolynomial, *,
                        tolerance: float = 1e-4, eps: float = 1e-11,
                        max_rectangles: int = 100000,
                        blacklist: Sequence[RootRectangle] = ()) -> list[RootRectangle]:
    """Find bivariate root candidates by linear fits, fat-line clipping and subdivision.

    Uses the same Legendre projection and Bernstein residual bounds as the
    original bivariate linear clipping method. Clips the two fat lines directly
    against the unit square, including singular/parallel linear fits. Floating
    point arithmetic is not interval-certified. Identically zero and exactly
    proportional same-basis systems, shared axis roots, detected shared affine
    lines and sign-changing polynomial divisors are unresolved; other
    non-isolated systems may evade detection. An exhausted work limit also raises
    ClippingConvergenceError. Completed numerical candidates are retained in
    the exception's ``partial_roots`` attribute.
    """
    if (not isfinite(tolerance) or not isfinite(eps) or tolerance <= 0 or eps < 0 or
            type(max_rectangles) is not int or max_rectangles < 1):
        raise ValueError("tolerance and work limit must be positive; eps must be nonnegative")
    if any(not 0 <= r.x0 <= r.x1 <= 1 or not 0 <= r.y0 <= r.y1 <= 1 for r in blacklist):
        raise ValueError("blacklist rectangles must lie within the unit square")
    if any(r.x0 == 0 and r.x1 == 1 and r.y0 == 0 and r.y1 == 1 for r in blacklist):
        return []
    if any(min(c for row in poly.coefficients for c in row) > eps or
           max(c for row in poly.coefficients for c in row) < -eps for poly in (p, q)):
        return []
    # Restriction, Legendre fitting, and clipping perform rounded arithmetic
    # even when the caller requests eps=0. Leave a scale-dependent numerical
    # margin so those operations do not silently discard an exact root merely
    # because a computed hull or fat line moved by a few ulps. The initial
    # coefficient-hull check above uses the requested eps directly because it
    # does not perform arithmetic on the input coefficients. This allowance is
    # heuristic and does not make the returned boxes certified enclosures.
    if all(c == 0 for row in p.coefficients for c in row) or all(c == 0 for row in q.coefficients for c in row):
        raise ClippingConvergenceError("system contains an identically zero polynomial and may have non-isolated roots")
    if _exactly_proportional(p, q):
        raise ClippingConvergenceError("system contains dependent polynomials and may have non-isolated roots")
    if _shared_axis_root(p, q, axis="x") or _shared_axis_root(p, q, axis="y"):
        raise ClippingConvergenceError("system has a shared axis root and non-isolated solutions")
    if _shared_affine_line(p, q) or _shared_affine_line(q, p):
        raise ClippingConvergenceError("system has a shared affine root line and non-isolated solutions")
    if _shared_sign_changing_factor(p, q):
        raise ClippingConvergenceError("system has a shared sign-changing factor and non-isolated solutions")
    if _shared_quadratic_with_affine_cofactors(p, q):
        raise ClippingConvergenceError("system has a shared factor: quadratic root curve with non-isolated solutions")
    def transpose(poly):
        return BiBernsteinPolynomial(tuple(tuple(poly.coefficients[i][j] for i in range(poly.degree_x+1))
                                           for j in range(poly.degree_y+1)))
    if (_shared_x_factor_with_real_candidate(p, q, max_samples=max_rectangles) or
            _shared_x_factor_with_real_candidate(transpose(p), transpose(q), max_samples=max_rectangles)):
        raise ClippingConvergenceError("system has a shared factor and may have non-isolated solutions")
    def normalized(poly):
        magnitude = max(abs(c) for row in poly.coefficients for c in row)
        _, exponent = frexp(magnitude)
        coefficients = tuple(tuple(ldexp(value, -exponent) for value in row) for row in poly.coefficients)
        # A common power-of-two scaling is exact unless it loses low bits to
        # underflow. Refuse to silently solve a changed polynomial in that case.
        if any(ldexp(value, exponent) != original
               for row, original_row in zip(coefficients, poly.coefficients)
               for value, original in zip(row, original_row)):
            raise ClippingConvergenceError("bivariate coefficient scale ratio exceeds float range")
        try:
            scaled_eps = ldexp(eps, -exponent)
        except OverflowError:
            scaled_eps = float("inf")
        scaled = BiBernsteinPolynomial(coefficients)
        margin = max(scaled_eps, 16*(scaled.degree_x+1)*(scaled.degree_y+1)*
                     ulp(max(abs(c) for row in coefficients for c in row)))
        return scaled, margin
    # Work at a bounded magnitude so finite input coefficients cannot overflow
    # the Legendre projection and make an intersecting system appear disjoint.
    (p, p_margin), (q, q_margin) = normalized(p), normalized(q)
    allowances = p_margin, q_margin
    queue = deque([RootRectangle(0., 1., 0., 1.)])
    roots = []
    visited = 0
    while queue:
        rect = queue.popleft()
        visited += 1
        if visited > max_rectangles:
            raise ClippingConvergenceError("bivariate clipping exceeded the rectangle limit; system may have non-isolated roots",
                                           partial_roots=roots, processed=visited-1, pending=len(queue)+1)
        if any(r.x0 <= rect.x0 <= rect.x1 <= r.x1 and r.y0 <= rect.y0 <= rect.y1 <= r.y1 for r in blacklist):
            continue
        rp = p.restrict(rect.x0, rect.x1, rect.y0, rect.y1)
        rq = q.restrict(rect.x0, rect.x1, rect.y0, rect.y1)
        # The convex hull property eliminates rectangles where either function
        # is bounded strictly on one side of zero.
        if any(min(c for row in r.coefficients for c in row) > margin or
               max(c for row in r.coefficients for c in row) < -margin
               for r, margin in zip((rp, rq), allowances)):
            continue
        polygon = [(0., 0.), (1., 0.), (1., 1.), (0., 1.)]
        for polynomial, margin in zip((rp, rq), allowances):
            a, b, c, delta = polynomial.linear_fit()
            delta += margin
            polygon = _clip_half_plane(polygon, a-delta, b, c)
            polygon = _clip_half_plane(polygon, -a-delta, -b, -c)
            if not polygon:
                break
        if not polygon:
            continue
        xlo, xhi = max(0., min(x for x, _ in polygon)), min(1., max(x for x, _ in polygon))
        ylo, yhi = max(0., min(y for _, y in polygon)), min(1., max(y for _, y in polygon))
        dx, dy = rect.x1-rect.x0, rect.y1-rect.y0
        clipped = RootRectangle(rect.x0+dx*xlo, rect.x0+dx*xhi,
                                rect.y0+dy*ylo, rect.y0+dy*yhi)
        wx, wy = clipped.x1-clipped.x0, clipped.y1-clipped.y0
        if any(r.x0 <= clipped.x0 <= clipped.x1 <= r.x1 and r.y0 <= clipped.y0 <= clipped.y1 <= r.y1
               for r in blacklist):
            continue
        if wx <= tolerance and wy <= tolerance:
            # The fat-line intersections can have a tiny bounding box even
            # though the original polynomial has one sign on that box. Check
            # the clipped box itself before accepting a root candidate.
            restricted = (p.restrict(clipped.x0, clipped.x1, clipped.y0, clipped.y1),
                          q.restrict(clipped.x0, clipped.x1, clipped.y0, clipped.y1))
            if all(min(value for row in item.coefficients for value in row) <= margin and
                   max(value for row in item.coefficients for value in row) >= -margin
                   for item, margin in zip(restricted, allowances)):
                roots.append(clipped)
            continue
        if wx*wy < .9*dx*dy and (wx < .9*dx or wy < .9*dy):
            queue.append(clipped)
        elif wx >= wy:
            mid = (clipped.x0+clipped.x1)/2
            if mid in (clipped.x0, clipped.x1):
                raise ClippingConvergenceError("bivariate clipping reached floating-point resolution",
                                               partial_roots=roots, processed=visited, pending=len(queue)+1)
            queue.extend((RootRectangle(clipped.x0, mid, clipped.y0, clipped.y1),
                          RootRectangle(mid, clipped.x1, clipped.y0, clipped.y1)))
        else:
            mid = (clipped.y0+clipped.y1)/2
            if mid in (clipped.y0, clipped.y1):
                raise ClippingConvergenceError("bivariate clipping reached floating-point resolution",
                                               partial_roots=roots, processed=visited, pending=len(queue)+1)
            queue.extend((RootRectangle(clipped.x0, clipped.x1, clipped.y0, mid),
                          RootRectangle(clipped.x0, clipped.x1, mid, clipped.y1)))
    # Keep distinct rectangles; adjacent boxes may both enclose a boundary root.
    return roots
