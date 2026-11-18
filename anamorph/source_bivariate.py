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
"""Source-order Bernstein construction for the curve-pair solver.

The floating-point operations follow BLRCanalSurface::computeIntersectionSystem
and its degree elevation. This is separate from the scale-normalized systems
used by the more robust numerical analysis path.
"""
from functools import lru_cache
from dataclasses import dataclass
from math import comb, isfinite

from .bernstein import BernsteinPolynomial, BiBernsteinPolynomial, ClippingConvergenceError, RootRectangle
from .source_canal_polynomials import _components as _source_components


class SourceBivariateFailure(ClippingConvergenceError):
    """A failure corresponding to a source bivariate solver exception."""


@dataclass(frozen=True, slots=True)
class SourceGlobalCandidate:
    """A source-style global self-intersection candidate, not a certificate."""
    t: float
    u: float
    distance_squared: float
    rectangle: RootRectangle
    source: str


@dataclass(frozen=True, slots=True)
class SourcePairCandidate:
    """A source-style neurite-pair candidate, not a certificate."""
    t: float
    u: float
    distance_squared: float
    rectangle: RootRectangle
    source: str


def _bi_product(p, q):
    """Multiply Bernstein coefficient matrices in source loop order."""
    m, n, r, s = len(p)-1, len(p[0])-1, len(q)-1, len(q[0])-1
    result = [[0.]*(n+s+1) for _ in range(m+r+1)]
    for k in range(m+r+1):
        for l in range(n+s+1):
            for i in range(max(0, k-r), min(k, m)+1):
                for j in range(max(0, l-s), min(l, n)+1):
                    term = p[i][j]*q[k-i][l-j]
                    term *= comb(m, i)
                    term *= comb(r, k-i)
                    term *= comb(n, j)
                    term *= comb(s, l-j)
                    term /= comb(m+r, k)*comb(n+s, l)
                    result[k][l] += term
    return result


def _bi_source_power(coeff):
    """Convert a square power matrix using the C++ nested summation order."""
    n = len(coeff)-1
    result = [[0.]*(n+1) for _ in range(n+1)]
    for k in range(n+1):
        for l in range(n+1):
            ifac = jfac = 1.
            value = coeff[0][0]
            for j in range(1, l+1):
                jfac *= (l-j+1)/(n-j+1)
                value += jfac*coeff[0][j]
            for i in range(1, k+1):
                ifac *= (k-i+1)/(n-i+1)
                jfac = 1.
                value += ifac*coeff[i][0]
                for j in range(1, l+1):
                    jfac *= (l-j+1)/(n-j+1)
                    value += ifac*jfac*coeff[i][j]
            result[k][l] = value
    return result


@lru_cache(maxsize=1)
def _source_global_g():
    """Precompute the source's degree-three G_ij matrices."""
    bases = ((1., -3., 3., -1.), (0., 3., -6., 3.),
             (0., 0., 3., -3.), (0., 0., 0., 1.))
    divided = []
    for basis in bases:
        power = [[0.]*3 for _ in range(3)]
        for k in range(1, 4):
            for m in range(k):
                power[m][k-1-m] += basis[k]
        divided.append(_bi_source_power(power))
    g = []
    for i in range(4):
        row = []
        for j in range(4):
            by_j = [[float(y == j) for y in range(4)] for _ in range(4)]
            by_i = [[float(y == i) for y in range(4)] for _ in range(4)]
            first, second = _bi_product(by_j, divided[i]), _bi_product(by_i, divided[j])
            row.append(tuple(tuple(first[x][y]-second[x][y] for y in range(6)) for x in range(6)))
        g.append(tuple(row))
    return tuple(g)


def source_global_stationary_bernstein(segment):
    """Return the source's elevated (7,7) global self-intersection system.

    Builds the original G_ij divided-distance matrices and follows the
    source's unscaled multiplication and degree-elevation order.
    """
    try:
        components = _source_components(segment)
        velocity = [tuple(3.*(axis[i+1]-axis[i]) for i in range(3)) for axis in components]
        divided = [[[0.]*6 for _ in range(6)] for _ in range(3)]
        g = _source_global_g()
        for i in range(4):
            for j in range(4):
                for axis in range(3):
                    for x in range(6):
                        for y in range(6):
                            divided[axis][x][y] += g[i][j][x][y]*components[axis][i]
        pparts, qparts = [], []
        for axis in range(3):
            pparts.append(_bi_product(divided[axis], [[v] for v in velocity[axis]]))
            qparts.append(_bi_product(divided[axis], [list(velocity[axis])]))
        p = [[pparts[0][i][j]+pparts[1][i][j]+pparts[2][i][j] for j in range(6)] for i in range(8)]
        q = [[qparts[0][i][j]+qparts[1][i][j]+qparts[2][i][j] for j in range(8)] for i in range(6)]
        pe = _bi_product(p, [[1., 1., 1.]])
        qe = _bi_product(q, [[1.], [1.], [1.]])
        if not all(isfinite(value) for matrix in (pe, qe) for row in matrix for value in row):
            raise ClippingConvergenceError("source global stationary coefficients exceed float range")
        return BiBernsteinPolynomial(tuple(map(tuple, pe))), BiBernsteinPolynomial(tuple(map(tuple, qe)))
    except (OverflowError, ValueError) as error:
        raise ClippingConvergenceError("source global stationary coefficients exceed float range") from error


def source_stationary_bernstein(first, second):
    """Return the source's elevated (5,5) stationary polynomials.

    Input centerlines are cubic power coefficients. As in the checked-in C++,
    they are converted to degree-three Bernstein form before the derivative,
    products and simultaneous two-degree elevations are computed. Extreme
    coordinates can overflow this source-style arithmetic; this is reported
    as unresolved rather than passed to the numerical solver.
    """
    try:
        a = _source_components(first)
        b = _source_components(second)
        da = [tuple(3.*(axis[i+1]-axis[i]) for i in range(3)) for axis in a]
        db = [tuple(3.*(axis[i+1]-axis[i]) for i in range(3)) for axis in b]
        per_axis_p, per_axis_q = [], []
        for axis in range(3):
            diff = [[a[axis][i]-b[axis][j] for j in range(4)] for i in range(4)]
            p = [[0.]*4 for _ in range(6)]
            q = [[0.]*6 for _ in range(4)]
            for k in range(6):
                for l in range(4):
                    value = 0.
                    for i in range(max(0, k-2), min(k, 3)+1):
                        term = diff[i][l]*da[axis][k-i]
                        term *= comb(3, i)
                        term *= comb(2, k-i)
                        term *= comb(3, l)
                        term /= comb(5, k)*comb(3, l)
                        value += term
                    p[k][l] = value
            for k in range(4):
                for l in range(6):
                    value = 0.
                    for j in range(max(0, l-2), min(l, 3)+1):
                        term = diff[k][j]*db[axis][l-j]
                        term *= comb(3, k)
                        term *= comb(3, j)
                        term *= comb(2, l-j)
                        term /= comb(3, k)*comb(5, l)
                        value += term
                    q[k][l] = value
            per_axis_p.append(p)
            per_axis_q.append(q)
        p = [[per_axis_p[0][i][j]+per_axis_p[1][i][j]+per_axis_p[2][i][j]
              for j in range(4)] for i in range(6)]
        q = [[per_axis_q[0][i][j]+per_axis_q[1][i][j]+per_axis_q[2][i][j]
              for j in range(6)] for i in range(4)]
        pe, qe = [[0.]*6 for _ in range(6)], [[0.]*6 for _ in range(6)]
        for k in range(6):
            for l in range(6):
                for j in range(max(0, l-2), min(l, 3)+1):
                    term = p[k][j]*comb(5, k)
                    term *= comb(3, j)
                    term *= comb(2, l-j)
                    term /= comb(5, k)*comb(5, l)
                    pe[k][l] += term
                for i in range(max(0, k-2), min(k, 3)+1):
                    term = q[i][l]*comb(3, i)
                    term *= comb(2, k-i)
                    term *= comb(5, l)
                    term /= comb(5, k)*comb(5, l)
                    qe[k][l] += term
        if not all(isfinite(value) for matrix in (pe, qe) for row in matrix for value in row):
            raise ClippingConvergenceError("source-style stationary coefficients exceed float range")
        return (BiBernsteinPolynomial(tuple(map(tuple, pe))),
                BiBernsteinPolynomial(tuple(map(tuple, qe))))
    except (OverflowError, ValueError) as error:
        raise ClippingConvergenceError("source-style stationary coefficients exceed float range") from error


@lru_cache(maxsize=2)
def _source_fit_weights(degree):
    """Construct the source's shifted-Legendre fit matrices."""
    n = degree
    inner = [[comb(n, i)*comb(n, j)/((2.*n+1.)*comb(2*n, i+j))
              for j in range(n+1)] for i in range(n+1)]
    legendre = [-1.]+[(comb(n-1, k-1)-comb(n-1, k))/comb(n, k) for k in range(1, n)]+[1.]
    weights = [[0.]*(n+1) for _ in range(n+1)]
    xweights = [[0.]*(n+1) for _ in range(n+1)]
    yweights = [[0.]*(n+1) for _ in range(n+1)]
    for i in range(n+1):
        for j in range(n+1):
            for k in range(n+1):
                for l in range(n+1):
                    weights[i][j] += inner[i][k]*inner[j][l]
                    xweights[i][j] += legendre[k]*inner[i][k]*inner[j][l]
                    yweights[i][j] += legendre[l]*inner[i][k]*inner[j][l]
    return weights, xweights, yweights, legendre


def _source_fit(poly):
    """Source-order Legendre coefficients and Bernstein residual bound."""
    n = poly.degree_x
    if n not in (5, 7) or poly.degree_y != n:
        raise ValueError("source rectangle clipping requires degree (5,5) or (7,7) polynomials")
    weights, xweights, yweights, legendre = _source_fit_weights(n)
    c0 = cx = cy = 0.
    for i in range(n+1):
        for j in range(n+1):
            value = poly.coefficients[i][j]
            c0 += value*weights[i][j]
            cx += value*xweights[i][j]
            cy += value*yweights[i][j]
    cx *= 3.
    cy *= 3.
    residual = 0.
    for i in range(n+1):
        for j in range(n+1):
            approximation = (c0+legendre[i]*cx)+legendre[j]*cy
            residual = max(residual, abs(poly.coefficients[i][j]-approximation))
    if not all(isfinite(value) for value in (c0, cx, cy, residual)):
        raise ClippingConvergenceError("source-style linear fit exceeds float range")
    return c0, cx, cy, residual


def source_initial_fit(first, second):
    """Return source-order floating-point slopes for the first clipping fit."""
    p, q = source_stationary_bernstein(first, second)
    _, px, py, _ = _source_fit(p)
    _, qx, qy, _ = _source_fit(q)
    result = tuple(2.*value for value in (px, py, qx, qy))
    if not all(isfinite(value) for value in result):
        raise ClippingConvergenceError("source-style initial fit exceeds float range")
    return result


def source_initial_singular_fallback(first, second, *, epsilon=1e-11, tolerance=1e-4):
    """Whether the source's first full-pivot fit takes its throwing branch.

    This follows the unscaled degree-(5,5) initial fit and the source's
    absolute pivot threshold. A flat row instead takes a different path and
    returns False here. Later iterations and platform-specific rounding can
    still change the full source result.
    """
    if not all(isfinite(value) and value > 0 for value in (epsilon, tolerance)):
        raise ValueError("source pivot epsilon and tolerance must be positive and finite")
    if tolerance > 1.:
        # C++ constructs the system first, then rejects a domain whose two
        # dimensions are already below the requested tolerance.
        source_stationary_bernstein(first, second)
        return True
    px, py, qx, qy = source_initial_fit(first, second)
    original = ((px, py), (qx, qy))
    matrix = [[px, py], [qx, qy]]
    largest, row, column = abs(matrix[0][0]), 0, 0
    for i, j in ((0, 1), (1, 0), (1, 1)):
        if abs(matrix[i][j]) > largest:
            largest, row, column = abs(matrix[i][j]), i, j
    if largest < epsilon:
        return False
    if row:
        matrix[0], matrix[1] = matrix[1], matrix[0]
    if column:
        matrix[0][0], matrix[0][1] = matrix[0][1], matrix[0][0]
        matrix[1][0], matrix[1][1] = matrix[1][1], matrix[1][0]
    factor = -matrix[1][0]/matrix[0][0]
    remaining = matrix[1][1]+factor*matrix[0][1]
    if not isfinite(remaining):
        raise ClippingConvergenceError("source-style initial elimination exceeds float range")
    return (abs(remaining) < epsilon and
            max(abs(original[0][0]), abs(original[0][1])) >= epsilon and
            max(abs(original[1][0]), abs(original[1][1])) >= epsilon)


def _source_linear_solve(matrix, right_sides, epsilon):
    """Source full-pivot elimination for four right-hand sides."""
    a = [list(row) for row in matrix]
    b = [list(row) for row in right_sides]
    largest, pivot_row, pivot_col = abs(a[0][0]), 0, 0
    for i, j in ((0, 1), (1, 0), (1, 1)):
        if abs(a[i][j]) > largest:
            largest, pivot_row, pivot_col = abs(a[i][j]), i, j
    if pivot_row:
        a[0], a[1] = a[1], a[0]
        for row in b:
            row[0], row[1] = row[1], row[0]
    if pivot_col:
        for row in a:
            row[0], row[1] = row[1], row[0]
    if abs(a[0][0]) < epsilon:
        return None
    factor = -a[1][0]/a[0][0]
    a[1][0] += factor*a[0][0]
    a[1][1] += factor*a[0][1]
    for row in b:
        row[1] += factor*row[0]
    if abs(a[1][1]) < epsilon:
        return None
    solutions = []
    for row in b:
        y = row[1]/a[1][1]
        x = (row[0]-a[0][1]*y)/a[0][0]
        solutions.append((y, x) if pivot_col else (x, y))
    if not all(isfinite(value) for pair in solutions for value in pair):
        raise ClippingConvergenceError("source-style linear solve exceeds float range")
    return solutions


def source_clip_rectangle(p, q, rectangle, *, x_frozen=False, y_frozen=False,
                          epsilon=1e-11, pivot_epsilon=1e-11):
    """Port one source bivariate clipping step; return a rectangle or None.

    Inputs are degree-(5,5) or degree-(7,7) polynomials already restricted to the
    supplied rectangle and transformed to the unit square. Like the source,
    parallel non-flat fits and some frozen-axis cases raise instead of clipping.
    This operation is numerical and does not certify root containment.
    """
    if (not all(isfinite(value) for value in (rectangle.x0, rectangle.x1, rectangle.y0, rectangle.y1,
                                               epsilon, pivot_epsilon)) or
            not 0 <= rectangle.x0 <= rectangle.x1 <= 1 or not 0 <= rectangle.y0 <= rectangle.y1 <= 1 or
            epsilon < 0 or pivot_epsilon <= 0):
        raise ValueError("invalid source clipping rectangle or tolerance")
    if x_frozen and y_frozen:
        raise SourceBivariateFailure("source clipping cannot freeze both axes")
    if (p.degree_x, p.degree_y) != (q.degree_x, q.degree_y):
        raise ValueError("source clipping polynomials must have the same degree")
    p0, px, py, dp = _source_fit(p)
    q0, qx, qy, dq = _source_fit(q)
    matrix = ((2*px, 2*py), (2*qx, 2*qy))
    if not x_frozen and not y_frozen:
        rhs = ((px+py-p0+dp, qx+qy-q0+dq), (px+py-p0+dp, qx+qy-q0-dq),
               (px+py-p0-dp, qx+qy-q0+dq), (px+py-p0-dp, qx+qy-q0-dq))
        solutions = _source_linear_solve(matrix, rhs, pivot_epsilon)
        if solutions is None:
            for row, offset, radius in ((matrix[0], p0-px-py, dp), (matrix[1], q0-qx-qy, dq)):
                if max(abs(row[0]), abs(row[1])) < pivot_epsilon:
                    return None if offset+radius < -epsilon or offset-radius > epsilon else rectangle
            raise SourceBivariateFailure("source clipping found parallel non-flat fits")
        xmin, xmax = min(point[0] for point in solutions), max(point[0] for point in solutions)
        ymin, ymax = min(point[1] for point in solutions), max(point[1] for point in solutions)
        if xmax < 0. or xmin > 1. or ymax < 0. or ymin > 1.:
            return None
        return RootRectangle(rectangle.x0+xmin*(rectangle.x1-rectangle.x0) if xmin > 0. else rectangle.x0,
                             rectangle.x0+xmax*(rectangle.x1-rectangle.x0) if xmax < 1. else rectangle.x1,
                             rectangle.y0+ymin*(rectangle.y1-rectangle.y0) if ymin > 0. else rectangle.y0,
                             rectangle.y0+ymax*(rectangle.y1-rectangle.y0) if ymax < 1. else rectangle.y1)

    free = 1 if x_frozen else 0
    candidates = []
    for c0, cx, cy, radius in ((p0, px, py, dp), (q0, qx, qy, dq)):
        boundary = (1., 0.) if x_frozen else (0., 1.)
        matrix = ((2*cx, 2*cy), boundary)
        rhs = ((cx+cy-c0-radius, 0.), (cx+cy-c0+radius, 0.),
               (cx+cy-c0-radius, 1.), (cx+cy-c0+radius, 1.))
        # The source's frozen-axis calls use the linear solver's default
        # epsilon, not the supplied pivot_epsilon.
        solutions = _source_linear_solve(matrix, rhs, 1e-11)
        if solutions is None:
            candidates.append(None)
            continue
        low = min(point[free] for point in solutions)
        high = max(point[free] for point in solutions)
        low -= .01*low
        high += .01*high
        if low > 1. or high < 0.:
            return None
        candidates.append((max(0., low), min(1., high)))
    if all(candidate is None for candidate in candidates):
        raise SourceBivariateFailure("source clipping has two singular frozen-axis fits")
    p_interval, q_interval = candidates
    if p_interval is None:
        low, high = q_interval
    elif q_interval is None or p_interval[1]-p_interval[0] < q_interval[1]-q_interval[0]:
        low, high = p_interval
    else:
        low, high = q_interval
    if x_frozen:
        return RootRectangle(rectangle.x0, rectangle.x1,
                             rectangle.y0+low*(rectangle.y1-rectangle.y0),
                             rectangle.y0+high*(rectangle.y1-rectangle.y0))
    return RootRectangle(rectangle.x0+low*(rectangle.x1-rectangle.x0),
                         rectangle.x0+high*(rectangle.x1-rectangle.x0),
                         rectangle.y0, rectangle.y1)


def source_split_xy(poly):
    """Reproduce the checked-in source's four-way split, including its defect.

    The source overwrites the computed right-x half with two further y-splits
    of the already split left-down half. This function exists for source
    comparison; it must not be used as a mathematically correct subdivision.
    The returned order is left-down, right-down, right-up, left-up.
    """
    if poly.degree_x not in (5, 7) or poly.degree_y != poly.degree_x:
        raise ValueError("source subdivision requires degree (5,5) or (7,7) polynomials")
    left = poly.restrict(0., .5, 0., 1.)
    left_down = left.restrict(0., 1., 0., .5)
    left_up = left.restrict(0., 1., .5, 1.)
    right_down = left_down.restrict(0., 1., 0., .5)
    right_up = left_down.restrict(0., 1., .5, 1.)
    return left_down, right_down, right_up, left_up


def source_bilinear_clip_roots(p, q, *, tolerance=1e-4, epsilon=1e-11,
                               pivot_epsilon=1e-11, max_rectangles=10000, blacklist=()):
    """Trace the source's bivariate clipping loop with an explicit work limit.

    This comparison implementation deliberately reproduces the checked-in
    source's four-way split and frozen-axis root-output defects. Its results
    are numerical source-style candidates, not certified root enclosures.
    """
    if (not all(isfinite(value) and value > 0 for value in (tolerance, pivot_epsilon)) or
            not isfinite(epsilon) or epsilon < 0 or type(max_rectangles) is not int or max_rectangles < 1):
        raise ValueError("invalid source bivariate solver settings")
    if (p.degree_x not in (5, 7) or p.degree_y != p.degree_x or
            q.degree_x != p.degree_x or q.degree_y != p.degree_y):
        raise ValueError("source bivariate solver requires matching degree (5,5) or (7,7) polynomials")
    if tolerance > 1.:
        raise SourceBivariateFailure("source bivariate input rectangle is smaller than tolerance")
    forbidden = tuple(blacklist)
    if any(not isinstance(box, RootRectangle) or
           not 0 <= box.x0 <= box.x1 <= 1 or not 0 <= box.y0 <= box.y1 <= 1 for box in forbidden):
        raise ValueError("invalid source bivariate blacklist")
    roots, stack = [], [(RootRectangle(0., 1., 0., 1.), p, q, False, False)]
    visited = 0
    # These outputs are mutable across calls in the C++ loop. In the
    # y-frozen convergence branch, the source accidentally emits these
    # stale y values together with the old x values.
    last_y0, last_y1 = 0., 1.

    def restrict_original(poly, box):
        if box.x0 == 1. or box.y0 == 1.:
            raise ClippingConvergenceError("source restriction reaches an unrepresentable boundary")
        return poly.restrict(box.x0, box.x1, box.y0, box.y1)

    while stack:
        box, current_p, current_q, x_converged, y_converged = stack.pop()
        while True:
            if any(box.x0 >= item.x0 and box.x1 <= item.x1 and
                   box.y0 >= item.y0 and box.y1 <= item.y1 for item in forbidden):
                break
            width, height = box.x1-box.x0, box.y1-box.y0
            x_frozen, y_frozen = x_converged, y_converged
            if not x_converged and not y_converged:
                if not width or not height:
                    raise ClippingConvergenceError("source rectangle reached floating-point resolution",
                                                   partial_roots=roots, processed=visited, pending=len(stack)+1)
                if max(width/height, height/width) > 1e5:
                    x_frozen, y_frozen = (False, True) if width > height else (True, False)
            visited += 1
            if visited > max_rectangles:
                raise ClippingConvergenceError("source bivariate clipping exceeded the rectangle limit",
                                               partial_roots=roots, processed=visited-1, pending=len(stack)+1)
            try:
                clipped = source_clip_rectangle(current_p, current_q, box, x_frozen=x_frozen,
                                                y_frozen=y_frozen, epsilon=epsilon, pivot_epsilon=pivot_epsilon)
            except SourceBivariateFailure as error:
                raise SourceBivariateFailure(str(error), partial_roots=roots,
                                             processed=visited, pending=len(stack)+1) from error
            if clipped is None:
                break
            if not y_frozen:
                last_y0, last_y1 = clipped.y0, clipped.y1
            new_width, new_height = clipped.x1-clipped.x0, clipped.y1-clipped.y0
            subdivide = False
            if not x_frozen and not y_frozen:
                if new_width < tolerance:
                    x_converged = x_frozen = True
                if new_height < tolerance:
                    y_converged = y_frozen = True
                if x_converged and y_converged:
                    roots.append(clipped)
                    break
                try:
                    current_p, current_q = restrict_original(p, clipped), restrict_original(q, clipped)
                except (ValueError, ClippingConvergenceError) as error:
                    raise ClippingConvergenceError(f"source restriction unresolved: {error}",
                                                   partial_roots=roots, processed=visited, pending=len(stack)+1) from error
                subdivide = (new_width/width)*(new_height/height) > .9
            elif x_frozen and not y_frozen:
                if new_height < tolerance:
                    y_converged = y_frozen = True
                    if x_converged:
                        roots.append(RootRectangle(box.x0, box.x1, clipped.y0, clipped.y1))
                        break
                    x_frozen = False
                try:
                    current_p, current_q = restrict_original(p, clipped), restrict_original(q, clipped)
                except (ValueError, ClippingConvergenceError) as error:
                    raise ClippingConvergenceError(f"source restriction unresolved: {error}",
                                                   partial_roots=roots, processed=visited, pending=len(stack)+1) from error
                if not y_converged:
                    subdivide = new_height/height > .5
            elif y_frozen and not x_frozen:
                if new_width < tolerance:
                    x_converged = x_frozen = True
                    if y_converged:
                        roots.append(RootRectangle(box.x0, box.x1, last_y0, last_y1))
                        break
                    y_frozen = False
                try:
                    current_p, current_q = restrict_original(p, clipped), restrict_original(q, clipped)
                except (ValueError, ClippingConvergenceError) as error:
                    raise ClippingConvergenceError(f"source restriction unresolved: {error}",
                                                   partial_roots=roots, processed=visited, pending=len(stack)+1) from error
                if not x_converged:
                    subdivide = new_width/width > .5
            else:
                raise SourceBivariateFailure("source clipping cannot freeze both axes", partial_roots=roots,
                                             processed=visited, pending=len(stack)+1)
            box = clipped
            if not subdivide:
                continue
            if not x_frozen and not y_frozen:
                xm, ym = (box.x0+box.x1)/2, (box.y0+box.y1)/2
                if xm in (box.x0, box.x1) or ym in (box.y0, box.y1):
                    raise ClippingConvergenceError("source subdivision reached floating-point resolution",
                                                   partial_roots=roots, processed=visited, pending=len(stack)+1)
                boxes = (RootRectangle(box.x0, xm, box.y0, ym), RootRectangle(xm, box.x1, box.y0, ym),
                         RootRectangle(xm, box.x1, ym, box.y1), RootRectangle(box.x0, xm, ym, box.y1))
                ps, qs = source_split_xy(current_p), source_split_xy(current_q)
            elif x_frozen and not y_frozen:
                ym = (box.y0+box.y1)/2
                if ym in (box.y0, box.y1):
                    raise ClippingConvergenceError("source subdivision reached floating-point resolution",
                                                   partial_roots=roots, processed=visited, pending=len(stack)+1)
                boxes = (RootRectangle(box.x0, box.x1, box.y0, ym), RootRectangle(box.x0, box.x1, ym, box.y1))
                ps = (current_p.restrict(0., 1., 0., .5), current_p.restrict(0., 1., .5, 1.))
                qs = (current_q.restrict(0., 1., 0., .5), current_q.restrict(0., 1., .5, 1.))
            elif y_frozen and not x_frozen:
                xm = (box.x0+box.x1)/2
                if xm in (box.x0, box.x1):
                    raise ClippingConvergenceError("source subdivision reached floating-point resolution",
                                                   partial_roots=roots, processed=visited, pending=len(stack)+1)
                boxes = (RootRectangle(box.x0, xm, box.y0, box.y1), RootRectangle(xm, box.x1, box.y0, box.y1))
                ps = (current_p.restrict(0., .5, 0., 1.), current_p.restrict(.5, 1., 0., 1.))
                qs = (current_q.restrict(0., .5, 0., 1.), current_q.restrict(.5, 1., 0., 1.))
            else:
                raise SourceBivariateFailure("source subdivision cannot freeze both axes", partial_roots=roots,
                                             processed=visited, pending=len(stack)+1)
            stack.extend((child, cp, cq, x_converged, y_converged) for child, cp, cq in zip(boxes, ps, qs))
            break
    return roots


def source_global_self_intersection_candidates(segment, *, univariate_tolerance=1e-6,
                                               bivariate_tolerance=1e-4,
                                               max_rectangles=10000, max_intervals=100000,
                                               _suppressed_failures=None):
    """Trace source GSI candidates, including its bivariate exception behavior.

    The original catches a bivariate solver exception but later resets its
    result to false. Completed rectangles and edge candidates are still
    considered; an exception alone is not a positive GSI result.
    """
    from .source_canal_polynomials import source_soma_stationary_polynomial
    from .source_univariate import source_bezier_clip_roots

    if not all(isfinite(value) and value > 0 for value in (univariate_tolerance, bivariate_tolerance)):
        raise ValueError("source tolerances must be positive and finite")
    p, q = source_global_stationary_bernstein(segment)
    try:
        roots = source_bilinear_clip_roots(p, q, tolerance=bivariate_tolerance, max_rectangles=max_rectangles)
    except SourceBivariateFailure as error:
        if _suppressed_failures is not None:
            _suppressed_failures.append(error)
        roots = error.partial_roots
    points = [(RootRectangle(0., 0., 1., 1.), "corner")]
    points.extend((root, "interior") for root in roots)
    components = tuple(BernsteinPolynomial(axis) for axis in _source_components(segment))
    for endpoint in (0., 1.):
        center = tuple(poly.evaluate(endpoint) for poly in components)
        edge = source_soma_stationary_polynomial(segment, center)
        edge_roots = source_bezier_clip_roots(edge, tolerance=univariate_tolerance, max_intervals=max_intervals)
        points.extend((RootRectangle(endpoint, endpoint, root.left, root.right), "edge") for root in edge_roots)
    threshold = 2.*segment.max_radius
    threshold *= threshold
    offset = 2.*max(univariate_tolerance, bivariate_tolerance)
    if not isfinite(threshold):
        raise ClippingConvergenceError("source global distance threshold exceeds float range")
    result = []
    for rectangle, source in points:
        t, u = rectangle.midpoint
        if (t <= offset and u <= offset) or (t >= 1.-offset and u >= 1.-offset):
            continue
        if not 0 <= t <= 1 or not 0 <= u <= 1:
            raise ClippingConvergenceError("source global candidate midpoint lies outside the unit square")
        difference = tuple(component.evaluate(t)-component.evaluate(u) for component in components)
        distance = difference[0]*difference[0]+difference[1]*difference[1]+difference[2]*difference[2]
        if not isfinite(distance):
            raise ClippingConvergenceError("source global candidate distance exceeds float range")
        if distance < threshold:
            result.append(SourceGlobalCandidate(t, u, distance, rectangle, source))
    return result


def source_neurite_pair_edge_polynomials(first, second):
    """Build source-order pair edge polynomials x0, x1, y0, y1."""
    from .source_canal_polynomials import source_soma_stationary_polynomial

    try:
        a = tuple(BernsteinPolynomial(axis) for axis in _source_components(first))
        b = tuple(BernsteinPolynomial(axis) for axis in _source_components(second))
        x = tuple(source_soma_stationary_polynomial(second, tuple(poly.evaluate(t) for poly in a))
                  for t in (0., 1.))
        y = tuple(source_soma_stationary_polynomial(first, tuple(poly.evaluate(t) for poly in b))
                  for t in (0., 1.))
        return x+y
    except (OverflowError, ValueError) as error:
        raise ClippingConvergenceError("source pair edge polynomial exceeds float range") from error


def source_neurite_pair_candidates(first, second, *, adjacent=False, first_end_second_start=False,
                                    univariate_tolerance=1e-6, bivariate_tolerance=1e-4,
                                    max_rectangles=10000, max_intervals=100000):
    """Trace source nonadjacent or adjacent pair candidates and thresholds.

    Source bivariate exceptions propagate as ``SourceBivariateFailure``:
    both checked-in callers return a positive result without a witness for
    that failure, except that end-to-start pairs return clear before solving.
    """
    from .source_univariate import source_bezier_clip_roots

    if not all(isfinite(value) and value > 0 for value in (univariate_tolerance, bivariate_tolerance)):
        raise ValueError("source tolerances must be positive and finite")
    if first_end_second_start and not adjacent:
        raise ValueError("end-to-start is only meaningful for adjacent pairs")
    p, q = source_stationary_bernstein(first, second)
    if adjacent and first_end_second_start:
        return []
    blacklist = (RootRectangle(0., .01, 0., .01),) if adjacent else ()
    roots = source_bilinear_clip_roots(p, q, tolerance=bivariate_tolerance,
                                       max_rectangles=max_rectangles, blacklist=blacklist)
    points = [(RootRectangle(t, t, u, u), "corner") for t in (0., 1.) for u in (0., 1.)]
    points.extend((root, "interior") for root in roots)
    first_components = tuple(BernsteinPolynomial(axis) for axis in _source_components(first))
    second_components = tuple(BernsteinPolynomial(axis) for axis in _source_components(second))
    edges = source_neurite_pair_edge_polynomials(first, second)
    for endpoint, edge in zip((0., 1.), edges[:2]):
        edge_roots = source_bezier_clip_roots(edge, tolerance=univariate_tolerance, max_intervals=max_intervals)
        points.extend((RootRectangle(endpoint, endpoint, root.midpoint, root.midpoint), "edge")
                      for root in edge_roots)
    for endpoint, edge in zip((0., 1.), edges[2:]):
        edge_roots = source_bezier_clip_roots(edge, tolerance=univariate_tolerance, max_intervals=max_intervals)
        points.extend((RootRectangle(root.midpoint, root.midpoint, endpoint, endpoint), "edge")
                      for root in edge_roots)
    threshold = first.max_radius+second.max_radius
    threshold *= threshold
    if not isfinite(threshold):
        raise ClippingConvergenceError("source pair distance threshold exceeds float range")
    result = []
    for rectangle, source in points:
        t, u = rectangle.midpoint
        if adjacent and t <= .01 and u <= .01:
            continue
        if not 0 <= t <= 1 or not 0 <= u <= 1:
            raise ClippingConvergenceError("source pair candidate midpoint lies outside the unit square")
        difference = tuple(a.evaluate(t)-b.evaluate(u) for a, b in zip(first_components, second_components))
        distance = difference[0]*difference[0]+difference[1]*difference[1]+difference[2]*difference[2]
        if not isfinite(distance):
            raise ClippingConvergenceError("source pair candidate distance exceeds float range")
        if distance <= threshold:
            result.append(SourcePairCandidate(t, u, distance, rectangle, source))
    return result
