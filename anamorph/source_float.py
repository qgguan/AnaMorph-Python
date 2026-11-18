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
# See LICENSE and NOTICE.md for attribution and license terms.
"""Optional arithmetic for comparing the tested Windows x87 reference."""
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from functools import lru_cache
from math import cos, frexp, isfinite, isqrt, ldexp, sin


_PI = Decimal("3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647")


def x87_sqrt(value):
    """Emulate x87 sqrt rounded to 64-bit precision, then binary64.

    The tested MinGW reference uses an x87 instruction with the usual
    round-to-nearest-even control word. A double rounding can differ by one
    ulp from the correctly rounded binary64 square root. Other builds and
    floating-point control settings may use different arithmetic.
    """
    if not isfinite(value) or value < 0:
        raise ValueError("x87 square root requires a nonnegative finite value")
    if value == 0:
        return value
    numerator, denominator = value.as_integer_ratio()
    exponent = (frexp(value)[1]-1)//2
    shift = 2*(63-exponent)
    if shift >= 0:
        numerator <<= shift
    else:
        denominator <<= -shift
    mantissa = isqrt(numerator//denominator)
    comparison = 4*numerator-denominator*(2*mantissa+1)**2
    if comparison > 0 or (comparison == 0 and mantissa % 2):
        mantissa += 1
    return ldexp(float(mantissa), exponent-63)


def _round_extended_to_double(value):
    """Round a Decimal to a 64-bit binary significand, then binary64."""
    if not value:
        return float(value)
    sign = -1 if value < 0 else 1
    numerator, denominator = abs(value).as_integer_ratio()
    exponent = numerator.bit_length()-denominator.bit_length()
    if (numerator < denominator << exponent if exponent >= 0 else numerator << -exponent < denominator):
        exponent -= 1
    shift = 63-exponent
    if shift >= 0:
        numerator <<= shift
    else:
        denominator <<= -shift
    mantissa, remainder = divmod(numerator, denominator)
    if 2*remainder > denominator or (2*remainder == denominator and mantissa % 2):
        mantissa += 1
    return sign*ldexp(float(mantissa), exponent-63)


def x87_sincos(value):
    """Approximate the local reference's long-double sin/cos for render angles.

    A high-precision series is rounded through an x87-style significand.
    This is not an exact port of the reference libm: its approximation may
    differ by an ulp. Outside [-64, 64], use the host's trigonometric library.
    """
    if not isfinite(value):
        raise ValueError("render angle must be finite")
    if value == 0:
        return value, 1.
    if abs(value) > 64:
        return sin(value), cos(value)
    return _x87_sincos_cached(value)


@lru_cache(maxsize=4096)
def _x87_sincos_cached(value):
    with localcontext() as context:
        context.prec = 110
        angle = Decimal(value)
        angle -= (angle/(2*_PI)).to_integral_value(rounding=ROUND_HALF_EVEN)*(2*_PI)
        sine = sin_term = angle
        cosine = cos_term = Decimal(1)
        square = angle*angle
        for i in range(1, 120):
            sin_term = -sin_term*square/Decimal((2*i)*(2*i+1))
            cos_term = -cos_term*square/Decimal((2*i-1)*(2*i))
            sine += sin_term
            cosine += cos_term
            if abs(sin_term) < Decimal("1e-105") and abs(cos_term) < Decimal("1e-105"):
                break
        return _round_extended_to_double(sine), _round_extended_to_double(cosine)
