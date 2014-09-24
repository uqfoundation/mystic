#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2014 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE
__doc__ = _doc = """
This is drawn from the Mathematica GuideBook's example suite, with the
'fOsc3D' function definition found in [1].

References::
    [1] Trott, M. "The Mathematica GuideBook for Numerics", Springer-Verlag,
    New York, 2006.
"""
from abstract_model import AbstractFunction

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
        """evaluates the fOsc3D function for a list of coeffs

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

# cleanup
del _doc

# prepared instances
fosc3d = fOsc3D().function

# End of file
