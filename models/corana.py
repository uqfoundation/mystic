#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2014 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE
__doc__ = _doc = """
This is part of Storn's "Differential Evolution" test suite, as defined
in [2], with 'Corana' function definitions drawn from [3,4].

References::
    [1] Storn, R. and Price, K. "Differential Evolution - A Simple and
    Efficient Heuristic for Global Optimization over Continuous Spaces"
    Journal of Global Optimization 11: 341-359, 1997.

    [2] Storn, R. and Price, K. "Differential Evolution - A Simple and
    Efficient Heuristic for Global Optimization over Continuous Spaces"
    TR-95-012, ICSI, 1995. http://www.icsi.berkeley.edu/~storn/TR-95-012.pdf

    [3] Ingber, L. "Simulated Annealing: Practice Versus Theory" J. of
    Mathematical and Computer Modeling 18(11), 29-57, 1993.

    [4] Corana, A. and Marchesi, M. and Martini, C. and Ridella, S.
    "Minimizing Multimodal Functions of Continuous Variables with the
    'Simulated Annealing Algorithm'" ACM Transactions on Mathematical
    Software, March, 272-280, 1987.
"""
from abstract_model import AbstractFunction

from numpy import asarray
from math import pow
from numpy import sign, floor

class Corana(AbstractFunction):
    __doc__ = \
    """a Corana's parabola function generator

Corana's parabola function [1,2,3,4] defines a paraboloid whose
axes are parallel to the coordinate axes. This funciton has a
large number of wells that increase in depth with proximity to
the origin. The global minimum is a plateau around the origin.

The generated function f(x) is a modified version of equation (22)
of [2], where len(x) <= 4.
    """ + _doc
    def __init__(self, ndim=4): # is n-dimensional n=[1,4] (n=4 in ref)
        AbstractFunction.__init__(self, ndim=ndim)
        return

    def function(self,coeffs):
        """evaluates a 4-D Corana's parabola function for a list of coeffs

f(x) = \sum_(i=0)^(3) f_0(x)

Where for \abs(x_i - z_i) < 0.05:
f_0(x) = 0.15*(z_i - 0.05*\sign(z_i))^(2) * d_i
and otherwise:
f_0(x) = d_i * x_(i)^(2),
with z_i = \floor(\abs(x_i/0.2)+0.49999)*\sign(x_i)*0.2
and d_i = 1,1000,10,100.

For len(x) == 1, x = x_0,0,0,0;
for len(x) == 2, x = x_0,0,x_1,0;
for len(x) == 3, x = x_0,0,x_1,x_2;
for len(x) >= 4, x = x_0,x_1,x_2,x_3.

Inspect with mystic_model_plotter using::
    mystic.models.corana -b "-1:1:.01, -1:1:.01" -d -x 1

The minimum is f(x)=0 for \abs(x_i) < 0.05 for all i."""
        d = [1., 1000., 10., 100.]
        _d = [0, 3, 1, 2]   # ordering for lower dimensions
       #x = asarray(coeffs) #XXX: converting to numpy.array slows by 10x
        x = [0.]*4 # ensure that there are 4 coefficients
        if len(coeffs) < 4:
            _x = x[:]
            _x[:len(coeffs)]=coeffs
            for i in range(4):
                x[_d.index(i)] = _x[i]
        else:
            x = coeffs
        r = 0
        for j in range(4):
            zj =  floor( abs(x[j]/0.2) + 0.49999 ) * sign(x[j]) * 0.2
            if abs(x[j]-zj) < 0.05:
                r += 0.15 * pow(zj - 0.05*sign(zj), 2) * d[j]
            else:
                r += d[j] * x[j] * x[j]
        return r

    minimizers = None #FIXME: degenerate minimum... (-0.05, 0.05)
                      # minimum is f(x)=0 for \abs(x_i) < 0.05 for all i."""
    pass

# cleanup
del _doc

# prepared instances
corana = Corana().function

# End of file
