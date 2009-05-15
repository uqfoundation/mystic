#!/usr/bin/env python

"""
the fOsc3D Mathematica function

References::
    [4] Mathematica guidebook
"""
from abstract_model import AbstractFunction

from math import sin, exp

class fOsc3D(AbstractFunction):
    """fOsc3D Mathematica function:
fOsc3D[x_,y_] := -4 Exp[(-x^2 - y^2)] + Sin[6 x] Sin[5 y]"""

    def __init__(self):
        AbstractFunction.__init__(self)
        return

    def function(self,coeffs):
        """evaluates the fOsc3D function for a list of coeffs

minimum is f(x)=? at x=(?,?)"""
        x,y = coeffs
        func =  -4. * exp( -x*x - y*y ) + sin(6. * x) * sin(5. *y)
        penalty = 0
        if y < 0: penalty = 100.*y*y
        return func + penalty

#   def forward(self,pts):
#       """2-D fOsc3D; returns f(xi,yi) for pts=(x,y)"""
#       return AbstractFunction.forward(self,pts)

    pass


# prepared instances
fosc3d = fOsc3D()

# End of file
