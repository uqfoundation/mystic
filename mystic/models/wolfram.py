#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
__doc__ = _doc = """
This is drawn from Mathematica's example suites, with the 'fOsc3D'
function definition found in [1], and the 'XXX' function found in [2].

References:
    1. Trott, M. "The Mathematica GuideBook for Numerics", Springer-Verlag,
       New York, 2006.
    2. Champion, B. and Strzebonski, A. "Wolfram Mathematica Tutorial
       Collection on Constrained Optimization", Wolfram Research, USA, 2008.
       http://reference.wolfram.com/language/guide/Optimization.html
"""
from .abstract_model import AbstractFunction

from math import sin, exp

class fOsc3D(AbstractFunction):
    __doc__ = \
    """a fOsc3D function generator

A fOsc3D function [1] for positive x_1 values yields
small sinusoidal oscillations on a flat plane, where
a sinkhole containing the global minimum and a few local
minima is found in a small region near the origin.
For negative x_1 values, a parabolic penalty is applied
that decreases as the x_1 appoaches zero.

The generated function f(x) is identical to equation (75)
of section 1.10 of [1], and requires len(x) == 2.
    """ + _doc
    def __init__(self, ndim=2): # is 2-dimensional
        AbstractFunction.__init__(self, ndim=ndim)
        return

    def function(self,coeffs):
        r"""evaluates the fOsc3D function for a list of coeffs

f(x) = f_0(x) + p(x)

Where:
f_0(x) = -4 * \exp(-x_(0)^2 - x_(1)^2) + \sin(6*x_(0)) * \sin(5*x_(1))
with for x_1 < 0:
p(x) = 100.*x_(1)^2
and otherwise:
p(x) = 0.

Inspect with mystic_model_plotter using::
    mystic.models.fosc3d -b "-5:5:.1, 0:5:.1" -d

The minimum is f(x)=-4.501069742528923 at x=(-0.215018, 0.240356)"""
        x,y = coeffs
        func =  -4. * exp( -x*x - y*y ) + sin(6. * x) * sin(5. *y)
        penalty = 0
        if y < 0: penalty = 100.*y*y
        return func + penalty

    minimizers = [(-0.215018, 0.240356)] #XXX: there are many local minima
    pass


class NMinimize51(AbstractFunction):
    __doc__ = \
    """a NMinimize51 function generator

A NMinimize51 function [2] has many local minima. The
minima are periodic over parameter space, and modulate
the surface of a parabola at the coarse scale.  The
global minimum is located at the deepest of the many
periodic wells.

The generated function f(x) is identical to equation (51)
of the 'NMinimize' section in [2], and requires len(x) == 2.
    """ + _doc
    def __init__(self, ndim=2): # is 2-dimensional
        AbstractFunction.__init__(self, ndim=ndim)
        return

    def function(self,coeffs):
        r"""evaluates the NMinimize51 function for a list of coeffs

f(x) = f_0(x) + f_1(x)

Where:
f_0(x) = \exp(\sin(50*x_0)) + \sin(60*\exp(x_1)) + \sin(70*\sin(x_0))
and
f_1(x) = \sin(\sin(80*x_1)) - \sin(10*(x_0 + x_1)) + (x_(0)^2 + x_(1)^2)/4

Inspect with mystic_model_plotter using::
    mystic.models.nmin51 -b "-5:5:.1, 0:5:.1" -d

The minimum is f(x)=-3.306869 at x=(-0.02440313,0.21061247)"""
        x,y = coeffs
        return exp(sin(50.*x)) + sin(60.*exp(y)) + sin(70.*sin(x)) + \
               sin(sin(80.*y)) - sin(10.*(x + y)) + 1./4.*(x**2 + y**2)

    minimizers = [(-0.02440313,0.21061247)] #XXX: there are many local minima
    pass


# cleanup
del _doc

# prepared instances
fosc3d = fOsc3D().function
nmin51 = NMinimize51().function

# End of file
