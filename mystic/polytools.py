#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                       Patrick Hung & Mike McKerns, Caltech
#                        (C) 1998-2008  All Rights Reserved
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

"""
Functions involving 1D polynomials

Main function exported is polyeval
"""

def polyeval(coeffs, x):
    """
Takes an iterable of coefficients -> to be interpreted as
[a3, a2, a1, a0] -> a3 x^3 + a2 x^2 + a1 x^1 + a0

x is anything that supports the * operator.

# Note:
# The effect is this:
#    return reduce(lambda x1, x2: x1 * x + x2, coeffs, 0)
#
# But the for loop used below is faster by about 50%.
    """
    val = 0*x
    for c in coeffs:
       val = c + val*x
    return val



def coefficients_to_polynomial(coeff):
    """
>>> coefficients_to_polynomial([1,2,3])
poly1d([1, 2, 3])
    """  
    import numpy
    return numpy.poly1d(coeff)


if __name__=='__main__':
    import doctest
    doctest.testmod(verbose=True)


# End of file
