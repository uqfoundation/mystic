#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2014 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE
"""
simple sine-based multi-minima functions

References::
    None.
"""
from abstract_model import AbstractFunction

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

minimum is f(x)=0.0 at xi=-3.141592653589793"""
        x = asarray(coeffs) #XXX: must be numpy.array
        return abs(x+3.*sin(x+pi)+pi)

#   def forward(self,pts):
#       """n-D Wavy; returns f(xi,yi,...) for pts=(x,y,...)"""
#       return AbstractFunction.forward(self,pts)

    pass


class Wavy2(Wavy1):
    """Wavy function #2:
a simple multi-minima function"""

    def __init__(self):
        Wavy1.__init__(self)
        return

    def function(self,coeffs):
        """evaluates the Wavy2 function for a list of coeffs

(degenerate) minimum is f(x)=-6.9 at xi=-26.92"""
        x = asarray(coeffs) #XXX: must be a numpy.array
        return 4 *sin(x)+sin(4*x) + sin(8*x)+sin(16*x)+sin(32*x)+sin(64*x)

    pass


# prepared instances
wavy1 = Wavy1()
wavy2 = Wavy2()


# End of file
