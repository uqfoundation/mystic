#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2014 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE
"""
Griewangk's function

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
from math import cos, sqrt

class Griewangk(AbstractFunction):
    """Griewangk function:
a multi-minima function, Equation (23) of [2]"""

    def __init__(self):
        AbstractFunction.__init__(self)
        return

    def function(self,coeffs):
        """evaluates the Griewangk function for a list of coeffs

minimum is f(x)=0.0 at xi=0.0"""
        # ensure that there are 10 coefficients
        x = [0]*10
        x[:len(coeffs)]=coeffs
       #x = asarray(x) #XXX: converting to numpy.array slows by 10x

        term1 = sum([c*c for c in x])/4000
        term2 = 1
        for i in range(10):
            term2 = term2 * cos( x[i] / sqrt(i+1.0) )
        return term1 - term2 + 1


#   def forward(self,pts):
#       """10-D Griewangk; returns f(xi,yi,...) for pts=(x,y,...)"""
#       return AbstractFunction.forward(self,pts)

    pass


# prepared instances
griewangk = Griewangk()

# End of file
