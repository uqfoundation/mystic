#!/usr/bin/env python

"""
References:

None.
"""
from dejong import AbstractFunction

from numpy import absolute as abs
from numpy import asarray
from numpy import sin, pi

class Wavy1(AbstractFunction):
    """Wavy function #1:
a simple multi-minima function"""

    def __init__(self):
        AbstractFunction.__init__(self)
        return

    def function(self,coeffs):
        """evaluates the Wavy1 function for a list of coeffs
minimum is f(x)=0.0 at xi=0.0"""
        x = asarray(coeffs)
        return abs(x+3.*sin(x+pi)+pi)

    def forward(self,pts):
        """n-D Wavy; returns f(xi,yi,...) for pts=(x,y,...)"""
        return AbstractFunction.forward(self,pts)

    pass


class Wavy2(Wavy1):
    """Wavy function #2:
a simple multi-minima function"""

    def __init__(self):
        Wavy1.__init__(self)
        return

    def function(self,coeffs):
        """evaluates the Wavy2 function for a list of coeffs
minimum is f(x)=0.0 at xi=0.0"""
        x = asarray(coeffs)
        return 4 *sin(x)+sin(4*x) + sin(8*x)+sin(16*x)+sin(32*x)+sin(64*x)

    pass


# prepared instances
wavy1 = Wavy1()
wavy2 = Wavy2()


# End of file
