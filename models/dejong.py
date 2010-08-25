#!/usr/bin/env python

"""
Rosenbrock's function, De Jong's step function, De Jong's quartic function,
and Shekel's function

References::
    [1] Storn, R. and Price, K. Differential Evolution - A Simple and Efficient
    Heuristic for Global Optimization over Continuous Spaces. Journal of Global
    Optimization 11: 341-359, 1997.

    [2] Storn, R. and Price, K.
    (Same title as above, but as a technical report.)
    http://www.icsi.berkeley.edu/~storn/deshort1.ps
"""
from abstract_model import AbstractFunction

from numpy import sum as numpysum
from numpy import asarray, transpose
from numpy import zeros_like, diag, zeros, atleast_1d
from math import floor
import random
from math import pow

class Rosenbrock(AbstractFunction):
    """Rosenbrock function:
A modified second De Jong function, Equation (18) of [2]"""

    def __init__(self):
        AbstractFunction.__init__(self)
        return

    def function(self,coeffs):
        """evaluates n-dimensional Rosenbrock function for a list of coeffs

minimum is f(x)=0.0 at xi=1.0"""
        x = [1]*2 # ensure that there are 2 coefficients
        x[:len(coeffs)]=coeffs
        x = asarray(x) #XXX: must be a numpy.array
        return numpysum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)#,axis=0)

    #def forward(self,pts):
    #    """n-dimensional Rosenbrock; returns f(xi,yi,...) for pts=(x,y,...)"""
    #    return AbstractFunction.forward(self,pts)

    def derivative(self,coeffs):
        """evaluates n-dimensional Rosenbrock derivative for a list of coeffs

minimum is f'(x)=[0.0]*n at x=[1.0]*n;  x must have len >= 2"""
        l = len(coeffs)
        x = [0]*l #XXX: ensure that there are 2 coefficients ?
        x[:l]=coeffs
        x = asarray(x) #XXX: must be a numpy.array
        xm = x[1:-1]
        xm_m1 = x[:-2]
        xm_p1 = x[2:]
        der = zeros_like(x)
        der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
        der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
        der[-1] = 200*(x[-1]-x[-2]**2)
        return list(der)

    def hessian(self, coeffs):
        """evaluates n-dimensional Rosenbrock hessian for the given coeffs

coeffs must have len >= 2"""
        x = atleast_1d(coeffs)
        H = diag(-400*x[:-1],1) - diag(400*x[:-1],-1)
        diagonal = zeros(len(x), dtype=x.dtype)
        diagonal[0] = 1200*x[0]-400*x[1]+2
        diagonal[-1] = 200
        diagonal[1:-1] = 202 + 1200*x[1:-1]**2 - 400*x[2:]
        H = H + diag(diagonal)
        return H

    def hessian_product(self, coeffs, p):
        """evaluates n-dimensional Rosenbrock hessian product for the given coeffs

both p and coeffs must have len >= 2"""
        #XXX: not well-tested
        p = atleast_1d(p)
        x = atleast_1d(coeffs)
        Hp = zeros(len(x), dtype=x.dtype)
        Hp[0] = (1200*x[0]**2 - 400*x[1] + 2)*p[0] - 400*x[0]*p[1]
        Hp[1:-1] = -400*x[:-2]*p[:-2]+(202+1200*x[1:-1]**2-400*x[2:])*p[1:-1] \
                   -400*x[1:-1]*p[2:]
        Hp[-1] = -400*x[-2]*p[-2] + 200*p[-1]
        return Hp

    pass
 

class Step(AbstractFunction):
    """De Jong's step function:
The third De Jong function, Equation (19) of [2]"""

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
The modified fourth De Jong function, Equation (20) of [2]"""

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
The modified fifth De Jong function, Equation (21) of [2]"""

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
