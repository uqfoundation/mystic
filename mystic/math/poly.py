#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2026 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
tools for polynomial functions
"""
#__all__ = ['polyeval', 'poly1d']

def polyeval(coeffs, x):
    """evaluate the polynomial defined by coeffs at evaluation points, x

thus, ``[a3, a2, a1, a0]`` yields ``a3 x^3 + a2 x^2 + a1 x^1 + a0``

Args:
    coeffs (list[float]): polynomial coefficients
    x (array[float]): array of points to evaluate the polynomial

Returns:
    array of evaluations of the polynomial

Examples:
    >>> x = numpy.array([1, 2, 3, 4, 5])
    >>> polyeval([1, 0, 0], x)
    array([ 1,  4,  9, 16, 25])
    >>> polyeval([0, 1, -1], x)
    array([0, 1, 2, 3, 4])
    """
    # The effect is this:
    #    return reduce(lambda x1, x2: x1 * x + x2, coeffs, 0)
    # However, the for loop used below is faster by about 50%.
#   from numpy import asarray
#   x = asarray(x) #FIXME: converting to numpy.array slows by 10x
    val = 0*x
    for c in coeffs:
       val = c + val*x #FIXME: requires x to be a numpy.array
    return val

def poly1d(coeff):
    """generate a 1-D polynomial instance from a list of coefficients"""
    from numpy import poly1d as npoly1d
    return npoly1d(coeff)


# End of file
