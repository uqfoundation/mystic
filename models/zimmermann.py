#!/usr/bin/env python

"""
Zimmermann's function

References::
    [1] Storn, R. and Price, K. Differential Evolution - A Simple and Efficient
    Heuristic for Global Optimization over Continuous Spaces. Journal of Global
    Optimization 11: 341-359, 1997.

    [2] Storn, R. and Price, K.
    (Same title as above, but as a technical report.)
    http://www.icsi.berkeley.edu/~storn/deshort1.ps
"""
from abstract_model import AbstractFunction

class Zimmermann(AbstractFunction):
    """Zimmermann function:
a non-continuous function, Equation (24-26) of [2]"""

    def __init__(self):
        AbstractFunction.__init__(self)
        return

    def function(self,coeffs):
        """evaluates the Zimmermann function for a list of coeffs

minimum is f(x)=0.0 at x=(7.0,2.0)"""
        x0, x1 = coeffs #must provide 2 values (x0,y0)
        f8 = 9 - x0 - x1
        c0,c1,c2,c3 = 0,0,0,0
        if x0 < 0: c0 = -100 * x0
        if x1 < 0: c1 = -100 * x1
        xx =  (x0-3.)*(x0-3) + (x1-2.)*(x1-2)
        if xx > 16: c2 = 100 * (xx-16)
        if x0 * x1 > 14: c3 = 100 * (x0*x1-14.)
        return max(f8,c0,c1,c2,c3)

#   def forward(self,pts):
#       """2-D Zimmermann; returns f(xi,yi) for pts=(x,y)"""
#       return AbstractFunction.forward(self,pts)

    pass


# prepared instances
zimmermann = Zimmermann()

# End of file
