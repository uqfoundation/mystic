#!/usr/bin/env python

"""
References:

[1] Storn, R. and Price, K. Differential Evolution - A Simple and Efficient
Heuristic for Global Optimization over Continuous Spaces. Journal of Global
Optimization 11: 341-359, 1997.

[2] Storn, R. and Price, K. (Same title as above, but as a technical report.)
http://www.icsi.berkeley.edu/~storn/deshort1.ps
"""
from abstract_model import AbstractFunction

from numpy import sum as numpysum
from numpy import asarray, transpose
from math import floor
import random
from math import pow

class Rosenbrock(AbstractFunction):
    """Rosenbrock function:
A modified second De Jong function, Eq. (18) of [2]"""

    def __init__(self):
        AbstractFunction.__init__(self)
        return

    def function(self,coeffs):
        """evaluates n-dimensional Rosenbrock function for a list of coeffs
minimum is f(x)=0.0 at xi=1.0"""
        #ensure that there are 2 coefficients
        x = [1]*2
        x[:len(coeffs)]=coeffs
        x = asarray(x) #XXX: must be a numpy.array
        return numpysum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

#   def forward(self,pts):
#       """n-dimensional Rosenbrock; returns f(xi,yi,...) for pts=(x,y,...)"""
#       return AbstractFunction.forward(self,pts)

    pass
 

class Step(AbstractFunction):
    """De Jong's step function:
The third De Jong function, Eq. (19) of [2]"""

    def __init__(self):
        AbstractFunction.__init__(self)
        return

    def function(self,coeffs):
        """evaluates n-dimensional De Jong step function for a list of coeffs
minimum is f(x)=0.0 at xi=-5-n where n=[0.0,0.12]"""
        f = 30.
        for c in coeffs:
            if abs(c) <= 5.12:
                f += floor(c)
            elif c > 5.12:
                f += 30 * (c - 5.12)
            else:
                f += 30 * (5.12 - c)
        return f

#   def forward(self,pts):
#       """n-dimensional De Jong step; returns f(xi,yi,...) for pts=(x,y,...)"""
#       return AbstractFunction.forward(self,pts)

    pass


class Quartic(AbstractFunction):
    """De Jong's quartic function:
The modified fourth De Jong function, Eq. (20) of [2]"""

    def __init__(self):
        AbstractFunction.__init__(self)
        return

    def function(self,coeffs):
        """evaluates n-dimensional De Jong quartic function for a list of coeffs
minimum is f(x)=random, but statistically at xi=0"""
        f = 0.
        for j, c in enumerate(coeffs):
            f += pow(c,4) * (j+1.0) + random.random()
        return f

#   def forward(self,pts):
#       """n-dimensional De Jong quartic; returns f(xi,yi,...) for pts=(x,y,...)"""
#       return AbstractFunction.forward(self,pts)

    pass


class Shekel(AbstractFunction):
    """Shekel's function:
The modified fifth De Jong function, Eq. (21) of [2]"""

    def __init__(self):
        AbstractFunction.__init__(self)
        return

    def function(self,coeffs):
        """evaluates 2-D Shekel's function at (x,y)
minimum is f(x)=0.0 at x(-32,-32)"""
        A = [-32., -16., 0., 16., 32.]
        a1 = A * 5
        a2 = reduce(lambda x1,x2: x1+x2, [[c] * 5 for c in A])

        x,y=coeffs
        r = 0.0
        for i in range(25):
            r += 1.0/ (1.0*i + pow(x-a1[i],6) + pow(y-a2[i],6) + 1e-15)
        return 1.0/(0.002 + r)

#   def forward(self,pts):
#       """2-D Shekel; returns f(xi,yi) for pts=(x,y)"""
#       return AbstractFunction.forward(self,pts)

    pass

# prepared instances
rosen = Rosenbrock()
step = Step()
quartic = Quartic()
shekel = Shekel()

# End of file
