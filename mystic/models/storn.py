#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
__doc__ = _doc = """
This is part of Storn's "Differential Evolution" test suite, as defined
in [2], with 'Corana' function definitions drawn from [3,4], 'Griewangk'
function definitions drawn from [5], and 'Zimmermann' function definitions
drawn from [6].

References:
    1. Storn, R. and Price, K. "Differential Evolution - A Simple and
       Efficient Heuristic for Global Optimization over Continuous Spaces"
       Journal of Global Optimization 11: 341-359, 1997.
    2. Storn, R. and Price, K. "Differential Evolution - A Simple and
       Efficient Heuristic for Global Optimization over Continuous Spaces"
       TR-95-012, ICSI, 1995. http://www.icsi.berkeley.edu/~storn/TR-95-012.pdf
    3. Ingber, L. "Simulated Annealing: Practice Versus Theory" J. of
       Mathematical and Computer Modeling 18(11), 29-57, 1993.
    4. Corana, A. and Marchesi, M. and Martini, C. and Ridella, S.
       "Minimizing Multimodal Functions of Continuous Variables with the
       'Simulated Annealing Algorithm'" ACM Transactions on Mathematical
       Software, March, 272-280, 1987.
    5. Griewangk, A.O. "Generalized Descent for Global Optimization"
       Journal of Optimization Theory and Applications 34: 11-39, 1981.
    6. Zimmermann, W. "Operations Research" Oldenbourg Munchen, Wien, 1990.
"""
from .abstract_model import AbstractFunction

from numpy import asarray
from math import pow, cos, sqrt
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
        r"""evaluates a 4-D Corana's parabola function for a list of coeffs

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

class Griewangk(AbstractFunction):
    __doc__ = \
    """a Griewangk's function generator

Griewangk's function [1,2,5] is a multi-dimensional cosine
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
        r"""evaluates an N-dimensional Griewangk's function for a list of coeffs

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

class Zimmermann(AbstractFunction):
    __doc__ = \
    """a Zimmermann function generator

A Zimmermann function [1,2,6] poses difficulty for minimizers
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
        r"""evaluates a Zimmermann function for a list of coeffs

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
corana = Corana().function
griewangk = Griewangk().function
zimmermann = Zimmermann().function

# End of file
