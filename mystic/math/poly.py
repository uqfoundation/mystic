#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
tools for polynomial functions
"""
from numpy import asarray
from numpy import poly1d as npoly1d

def polyeval(coeffs, x):
    """takes list of coefficients & evaluation points, returns f(x)
thus, [a3, a2, a1, a0] yields  a3 x^3 + a2 x^2 + a1 x^1 + a0"""
    # The effect is this:
    #    return reduce(lambda x1, x2: x1 * x + x2, coeffs, 0)
    # However, the for loop used below is faster by about 50%.
#   x = asarray(x) #FIXME: converting to numpy.array slows by 10x
    val = 0*x
    for c in coeffs:
       val = c + val*x #FIXME: requires x to be a numpy.array
    return val

def poly1d(coeff):
    """generates a 1-D polynomial instance from a list of coefficients
using numpy.poly1d(coeffs)"""
    return npoly1d(coeff)


# End of file
