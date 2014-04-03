#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2014 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE
"""
Corana's function

References::
    [1] Storn, R. and Price, K. Differential Evolution - A Simple and Efficient
    Heuristic for Global Optimization over Continuous Spaces. Journal of Global
    Optimization 11: 341-359, 1997.

    [2] Storn, R. and Price, K.
    (Same title as above, but as a technical report.)
    http://www.icsi.berkeley.edu/~storn/deshort1.ps
"""
from abstract_model import AbstractFunction

from numpy import asarray
from math import pow
from numpy import sign, floor

class Corana(AbstractFunction):
    """Corana's function:
a multi-minima function, Equation (22) of [2]"""

    def __init__(self):
        AbstractFunction.__init__(self)
        return

    def function(self,coeffs):
        """evaluates the Corana function for a list of coeffs

minimum is f(x)=0.0 at xi=0.0"""
        d = [1., 1000., 10., 100.]
       #x = asarray(coeffs) #XXX: converting to numpy.array slows by 10x
        x = coeffs
        r = 0
        for j in range(4):
            zj =  floor( abs(x[j]/0.2) + 0.49999 ) * sign(x[j]) * 0.2
            if abs(x[j]-zj) < 0.05:
                r += 0.15 * pow(zj - 0.05*sign(zj), 2) * d[j]
            else:
                r += d[j] * x[j] * x[j]
        return r

#   def forward(self,pts):
#       """n-dimensional Corana; returns f(xi) for each xi in pts"""
#       return AbstractFunction.forward(self,pts)

    pass


# prepared instances
corana = Corana()

def corana1d(x):
    """Corana in 1D; coeffs = (x,0,0,0)"""
    return corana([x[0], 0, 0, 0])

def corana2d(x):
    """Corana in 2D; coeffs = (x,0,y,0)"""
    return corana([x[0], 0, x[1], 0])

def corana3d(x):
    """Corana in 3D; coeffs = (x,0,y,z)"""
    return corana([x[0], 0, x[1], x[2]])

# End of file
