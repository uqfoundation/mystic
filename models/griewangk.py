#!/usr/bin/env python

"""
References:

[1] Storn, R. and Price, K. Differential Evolution - A Simple and Efficient
Heuristic for Global Optimization over Continuous Spaces. Journal of Global
Optimization 11: 341-359, 1997.

[2] Storn, R. and Price, K. (Same title as above, but as a technical report.)
http://www.icsi.berkeley.edu/~storn/deshort1.ps
"""
from dejong import AbstractFunction

from numpy import sum as numpysum
from numpy import asarray
from numpy import cos, sqrt

class Griewangk(AbstractFunction):
    """Griewangk function:
a multi-minima function, Eq. (23) of [2]"""

    def __init__(self):
        AbstractFunction.__init__(self)
        return

    def function(self,coeffs):
        """evaluates the Griewangk function for a list of coeffs
minimum is f(x)=0.0 at xi=0.0"""
        # ensure that there are 10 coefficients
        x = [0]*10
        x[:len(coeffs)]=coeffs
        x = asarray(x)

        term1 = numpysum([c*c for c in x])/4000
        term2 = 1
        for i in range(10):
            term2 = term2 * cos( x[i] / sqrt(i+1.0) )
        return term1 - term2 + 1


    def forward(self,pts):
        """10-D Griewangk; returns f(xi,yi,...) for pts=(x,y,...)"""
        return AbstractFunction.forward(self,pts)

    pass


# prepared instances
griewangk = Griewangk()

# End of file
