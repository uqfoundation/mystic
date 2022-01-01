#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Author: Alta Fang (altafang @caltech and alta @princeton)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
__doc__ = _doc = """
This is drawn from examples in the NAG Library, with the 'peaks' function
definition found in [1].

References:
    1. Numerical Algorithms Group, "NAG Library", Oxford UK, Mark 24,
       2013. http://www.nag.co.uk/numeric/CL/nagdoc_cl24/pdf/E05/e05jbc.pdf
"""
from .abstract_model import AbstractFunction

from math import exp

class Peaks(AbstractFunction):
    __doc__ = \
    """a peaks function generator

A peaks function [1] is essentially flat, with three wells and three
peaks near the origin. The global minimum is separated from the local
minima by peaks.

The generated function f(x) is identical to the 'peaks' function in
section 10 of [1], and requires len(x) == 2.
    """ + _doc
    def __init__(self, ndim=2): # is 2-dimensional
        AbstractFunction.__init__(self, ndim=ndim)
        return

    def function(self,coeffs):
        r"""evaluates an 2-dimensional peaks function for a list of coeffs

f(x) = f_0(x) - f_1(x) - f_2(x)

Where:
f_0(x) = 3 * (1 - x_0)^2 * \exp(-x_0^2 - (x_1 + 1)^2)
and
f_1(x) = 10 * (.2 * x_0 - x_0^3 - x_1^5) * \exp(-x_0^2 - x_1^2)
and
f_2(x) = \exp(-(x_0 + 1)^2 - x_1^2) / 3

Inspect with mystic_model_plotter using::
    mystic.models.peaks -b "-5:5:.1, -5:5:.1" -d

The minimum is f(x)=-6.551133332835841 at x=(0.22827892, -1.62553496)"""
       #x = asarray(x) #XXX: converting to numpy.array slows by 10x
        x,y = coeffs
        result = 3.*(1. - x)**2*exp(-x**2 - (y + 1.)**2) - \
                10.*(x*(1./5.) - x**3 - y**5)*exp(-x**2 - y**2) - \
                1./3.*exp(-(x + 1.)**2 - y**2)
        return result

    minimizers = [(0.22827892, -1.62553496),(-1.34739625,  0.20451886),(0.29644556,  0.3201962)]
    pass

# cleanup
del _doc

# prepared instances
peaks = Peaks().function

# End of file
