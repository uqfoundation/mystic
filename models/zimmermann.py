#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2014 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE
__doc__ = _doc = """
This is part of Storn's "Differential Evolution" test suite, as defined
in [2], with 'Zimmermann' function definitions drawn from [3].

References::
    [1] Storn, R. and Price, K. "Differential Evolution - A Simple and
    Efficient Heuristic for Global Optimization over Continuous Spaces"
    Journal of Global Optimization 11: 341-359, 1997.

    [2] Storn, R. and Price, K. "Differential Evolution - A Simple and
    Efficient Heuristic for Global Optimization over Continuous Spaces"
    TR-95-012, ICSI, 1995. http://www.icsi.berkeley.edu/~storn/TR-95-012.pdf

    [3] Zimmermann, W. "Operations Research" Oldenbourg Munchen, Wien, 1990.
"""
from abstract_model import AbstractFunction

class Zimmermann(AbstractFunction):
    __doc__ = \
    """a Zimmermann function generator

A Zimmermann function [1,2,3] poses difficulty for minimizers
as the minimum is located at the corner of the constrained region.
A penalty is applied to all values outside the constrained region,
creating a local minimum.

The generated function f(x) is a modified version of equation (24-26)
of [2], and requires len(x) == 2.
    """ + _doc
    def __init__(self, ndim=2):
        AbstractFunction.__init__(self, ndim=ndim)
        return

    def function(self,coeffs):
        """evaluates a Zimmermann function for a list of coeffs

f(x) = max(f_0(x), p_i(x)), with i = 0,1,2,3

Where:
f_0(x) = 9 - x_0 - x_1
with for x_0 < 0:
p_0(x) = -100 * x_0
and for x_1 < 0:
p_1(x) = -100 * x_1
and for c_2(x) > 16 and c_3(x) > 14:
p_i(x) = 100 * c_i(x), with i = 2,3
c_2(x) = (x_0 - 3)^2 + (x_1 - 2)^2
c_3(x) = x_0 * x_1
Otherwise, p_i(x)=0 for i=0,1,2,3 and c_i(x)=0 for i=2,3.

Inspect with mystic_model_plotter using::
    mystic.models.zimmermann -b "-5:10:.1, -5:10:.1" -d -x 1

The minimum is f(x)=0.0 at x=(7.0,2.0)"""
        x0, x1 = coeffs #must provide 2 values (x0,y0)
        f8 = 9 - x0 - x1
        #XXX: apply penalty p(k) = 100 + 100*k; k = |f(x) - c(x)|
        c0,c1,c2,c3 = 0,0,0,0
        if x0 < 0: c0 = -100 * x0
        if x1 < 0: c1 = -100 * x1
        xx =  (x0-3.)*(x0-3) + (x1-2.)*(x1-2)
        if xx > 16: c2 = 100 * (xx-16)
        if x0 * x1 > 14: c3 = 100 * (x0*x1-14.)
        return max(f8,c0,c1,c2,c3)

    minimizers = [(7., 2.), (2.35477650,  5.94832200)]
   #minima = [0.0, 0.69690150]
    pass

# cleanup
del _doc

# prepared instances
zimmermann = Zimmermann().function

# End of file
