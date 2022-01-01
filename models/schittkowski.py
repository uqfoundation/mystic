#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Author: Alta Fang (altafang @caltech and alta @princeton)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
__doc__ = _doc = """
This is part of Hock and Schittkowski's test suite in [1], with function
definitions drawn from [1] and [2].

References:
    1. Hock, W. and Schittkowski, K. "Test Examples for Nonlinear Programming
       Codes", Lecture Notes in Economics and Mathematical Systems, Vol. 187,
       Springer, 1981. http://www.ai7.uni-bayreuth.de/test_problem_coll.pdf
    2. Paviani, D.A. "A new method for the solution of the general
       nonlinear programming problem", Ph.D. dissertation, The University
       of Texas, Austin, TX, 1969.
"""
from .abstract_model import AbstractFunction

from math import sin, cos, sqrt, pi, exp, log
from numpy import inf

class Paviani(AbstractFunction):
    __doc__ = \
    """a Paviani's function generator

Paviani's function [1,2] is a relatively flat basin that
quickly jumps to infinity for x_i >= 10 or x_i <= 2. The
global minimum is located near the corner of one of the
basin corners. There are local minima in the corners
ajacent to the global minima.

The generated function f(x) is identical to function (110)
of [1], where len(x) >= 0.
    """ + _doc
    def __init__(self, ndim=10): # is n-dimensional
        AbstractFunction.__init__(self, ndim=ndim)
        return

    def function(self,coeffs):
        r"""evaluates Paviani's function for a list of coeffs

f(x) = f_0(x) - f_1(x)

Where:
f_0(x) = \sum_(i=0)^(N-1) (\ln(x_i - 2)^2 + \ln(10 - x_i)^2)
and
f_1(x) = \prod_(i=0)^(N-1) x_(i)^(.2)

Inspect with mystic_model_plotter using::
    mystic.models.paviani -b "2:10:.1, 2:10:.1" -d

For N=1, the minimum is f(x)=2.133838 at x_i=8.501586,
for N=3, the minimum is f(x)=7.386004 at x_i=8.589578,
for N=5, the minimum is f(x)=9.730525 at x_i=8.740743,
for N=8, the minimum is f(x)=-3.411859 at x_i=9.086900,
for N=10, the minimum is f(x)=-45.778470 at x_i=9.350241."""
        s = 0.
        for x in coeffs:
            if x >= 10. or x <= 2.:
                s = inf
                break
            s += log(x-2.)**2 + log(10.-x)**2
        p = reduce(lambda x,y:x*y, coeffs)**(.2)
        return s - p

    minimizers = None #XXX: there are also some local minima
    pass


# cleanup
del _doc

# prepared instances
paviani = Paviani().function


# End of file
