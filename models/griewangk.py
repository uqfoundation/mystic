#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2014 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE
__doc__ = _doc = """
This is part of Storn's "Differential Evolution" test suite, as defined
in [2], with 'Griewangk' function definitions drawn from [3].

References::
    [1] Storn, R. and Price, K. "Differential Evolution - A Simple and
    Efficient Heuristic for Global Optimization over Continuous Spaces"
    Journal of Global Optimization 11: 341-359, 1997.

    [2] Storn, R. and Price, K. "Differential Evolution - A Simple and
    Efficient Heuristic for Global Optimization over Continuous Spaces"
    TR-95-012, ICSI, 1995. http://www.icsi.berkeley.edu/~storn/TR-95-012.pdf

    [3] Griewangk, A.O. "Generalized Descent for Global Optimization"
    Journal of Optimization Theory and Applications 34: 11-39, 1981.
"""
from abstract_model import AbstractFunction

from numpy import asarray
from math import cos, sqrt

class Griewangk(AbstractFunction):
    __doc__ = \
    """a Griewangk's function generator

Griewangk's function [1,2,3] is a multi-dimensional cosine
function that provides several periodic local minima, with
the global minimum at the origin. The local minima are 
fractionally more shallow than the global minimum, such that
when viewed at a very coarse scale the function appears as
a multi-dimensional parabola similar to De Jong's sphere.

The generated function f(x) is a modified version of equation (23)
of [2], where len(x) >= 0.
    """ + _doc
    def __init__(self, ndim=10): # is n-dimensional (n=10 in ref)
        AbstractFunction.__init__(self, ndim=ndim)
        return

    def function(self,coeffs):
        """evaluates an N-dimensional Griewangk's function for a list of coeffs

f(x) = f_0(x) - f_1(x) + 1

Where:
f_0(x) = \sum_(i=0)^(N-1) x_(i)^(2) / 4000.
and:
f_1(x) = \prod_(i=0)^(N-1) \cos( x_i / (i+1)^(1/2) )

Inspect with mystic_model_plotter using::
    mystic.models.griewangk -b "-10:10:.1, -10:10:.1" -d -x 5

The minimum is f(x)=0.0 for x_i=0.0"""
       #x = asarray(x) #XXX: converting to numpy.array slows by 10x
        term1 = sum([c*c for c in coeffs])/4000.
        term2 = 1
        for i in range(len(coeffs)):
            term2 = term2 * cos( coeffs[i] / sqrt(i+1.0) )
        return term1 - term2 + 1


    minimizers = [0.] #XXX: there are many periodic local minima
    pass

# cleanup
del _doc

# prepared instances
griewangk = Griewangk().function

# End of file
