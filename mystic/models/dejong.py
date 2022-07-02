#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
__doc__ = _doc = """
This is part of Storn's "Differential Evolution" test suite, as defined
in [2], with 'De Jong' function definitions drawn from [3].

References:
    1. Storn, R. and Price, K. "Differential Evolution - A Simple and
       Efficient Heuristic for Global Optimization over Continuous Spaces"
       Journal of Global Optimization 11: 341-359, 1997.
    2. Storn, R. and Price, K. "Differential Evolution - A Simple and
       Efficient Heuristic for Global Optimization over Continuous Spaces"
       TR-95-012, ICSI, 1995. http://www.icsi.berkeley.edu/~storn/TR-95-012.pdf
    3. Ingber, L. and Rosen, B. "Genetic Algorithms and Very Fast
       Simulated Reannealing: A Comparison" J. of Mathematical and Computer
       Modeling 16(11), 87-100, 1992.
"""
from .abstract_model import AbstractFunction

from numpy import sum as numpysum
from numpy import asarray, transpose, inf, ones_like
from numpy import zeros_like, diag, zeros, atleast_1d
from math import floor
import random
from math import pow
from mystic.tools import permutations
from functools import reduce

class Sphere(AbstractFunction):
    __doc__ = \
    """a De Jong spherical function generator

De Jong's spherical function [1,2,3] is considered to be a simple
task for every serious minimization method. The minimum is located
at the center of the N-dimensional spehere.  There are no local
minima.

The generated function f(x) is identical to equation (17) of [2],
where len(x) >= 0.
    """ + _doc
    def __init__(self, ndim=3): # is n-dimensional (n=3 in ref)
        AbstractFunction.__init__(self, ndim=ndim)
        return

    def function(self,coeffs):
        r"""evaluates an N-dimensional spherical function for a list of coeffs

f(x) = \sum_(i=0)^(N-1) x_(i)^2

Inspect with mystic_model_plotter using::
    mystic.models.sphere -b "-5:5:.1, -5:5:.1" -d

The minimum is f(x)=0.0 at x_i=0.0 for all i"""
        f = 0.
        for c in coeffs:
            f += c*c
        return f

    minimizers = [0.]
    pass


class Rosenbrock(AbstractFunction):
    __doc__ = \
    """a Rosenbrock's Saddle function generator

Rosenbrock's Saddle function [1,2,3] has the reputation of being
a difficult minimization problem. In two dimensions, the function
is a saddle with an inverted basin, where the global minimum
occurs along the rim of the inverted basin.

The generated function f(x) is a modified version of equation (18)
of [2], where len(x) >= 0.
    """ + _doc
    def __init__(self, ndim=2, axis=None): # is n-dimensional (n=2 in ref)
        self.axis = axis
        AbstractFunction.__init__(self, ndim=ndim)
        return

    def function(self,coeffs):
        r"""evaluates an N-dimensional Rosenbrock saddle for a list of coeffs

f(x) = \sum_(i=0)^(N-2) 100*(x_(i+1) - x_(i)^(2))^(2) + (1 - x_(i))^(2)

Inspect with mystic_model_plotter using::
    mystic.models.rosen -b "-3:3:.1, -1:5:.1, 1" -d -x 1

The minimum is f(x)=0.0 at x_i=1.0 for all i"""
        coeffs = asarray(coeffs) #XXX: must be a numpy.array
        x = ones_like(coeffs) #XXX: ensure > 1 coeffs ?
        x[:len(coeffs)]=coeffs
        return numpysum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0, axis=self.axis).tolist()

    def derivative(self,coeffs): #XXX: axis?
        """evaluates an N-dimensional Rosenbrock derivative for a list of coeffs

The minimum is f'(x)=[0.0]*n at x=[1.0]*n, where len(x) >= 2."""
        coeffs = asarray(coeffs) #XXX: must be a numpy.array
        x = zeros_like(coeffs) #XXX: ensure > 1 coeffs ?
        x[:len(coeffs)]=coeffs
        xm = x[1:-1]
        xm_m1 = x[:-2]
        xm_p1 = x[2:]
        der = zeros_like(x)
        der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
        der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
        der[-1] = 200*(x[-1]-x[-2]**2)
        return der.tolist()

    def hessian(self, coeffs): #XXX: axis?
        """evaluates an N-dimensional Rosenbrock hessian for the given coeffs

The function f''(x) requires len(x) >= 2."""
        x = atleast_1d(coeffs)
        H = diag(-400*x[:-1],1) - diag(400*x[:-1],-1)
        diagonal = zeros(len(x), dtype=x.dtype)
        diagonal[0] = 1200*x[0]-400*x[1]+2
        diagonal[-1] = 200
        diagonal[1:-1] = 202 + 1200*x[1:-1]**2 - 400*x[2:]
        H = H + diag(diagonal)
        return H.tolist()

    def hessian_product(self, coeffs, p):
        """evaluates an N-dimensional Rosenbrock hessian product
for p and the given coeffs

The hessian product requires both p and coeffs to have len >= 2."""
        #XXX: not well-tested
        p = atleast_1d(p)
        x = atleast_1d(coeffs)
        Hp = zeros(len(x), dtype=x.dtype)
        Hp[0] = (1200*x[0]**2 - 400*x[1] + 2)*p[0] - 400*x[0]*p[1]
        Hp[1:-1] = -400*x[:-2]*p[:-2]+(202+1200*x[1:-1]**2-400*x[2:])*p[1:-1] \
                   -400*x[1:-1]*p[2:]
        Hp[-1] = -400*x[-2]*p[-2] + 200*p[-1]
        return Hp.tolist()

    minimizers = [1.] #NOTE: minima in lower dimensions occur along the ridge
    pass
 

class Step(AbstractFunction):
    __doc__ = \
    """a De Jong step function generator

De Jong's step function [1,2,3] has several plateaus, which pose
difficulty for many optimization algorithms. Degenerate global
minima occur for all x_i on the lowest plateau, with degenerate
local minima on all other plateaus.

The generated function f(x) is a modified version of equation (19)
of [2], where len(x) >= 0.
    """ + _doc
    def __init__(self, ndim=5): # is n-dimensional (n=5 in ref)
        AbstractFunction.__init__(self, ndim=ndim)
        return

    def function(self,coeffs):
        r"""evaluates an N-dimensional step function for a list of coeffs

f(x) = f_0(x) + p_i(x), with i=0,1

Where for abs(x_i) <= 5.12:
f_0(x) = 30 + \sum_(i=0)^(N-1) \floor x_i
and for x_i > 5.12:
p_0(x) = 30 * (1 + (x_i - 5.12))
and for x_i < 5.12:
p_1(x) = 30 * (1 + (5.12 - x_i))
Otherwise, f_0(x) = 0 and p_i(x)=0 for i=0,1.

Inspect with mystic_model_plotter using::
    mystic.models.step -b "-10:10:.2, -10:10:.2" -d -x 1

The minimum is f(x)=(30 - 6*N) for all x_i=[-5.12,-5)"""
        f = 30.
        for c in coeffs:
            if abs(c) <= 5.12:
                f += floor(c)
            elif c > 5.12:
                f += 30 * (c - 5.12)
            else:
                f += 30 * (5.12 - c)
        return f

    minimizers = None #FIXME: degenerate minimum... [-5.00000001]to[-5.12000000]
                 # minimum is f(x)=(30 - 6*N) for all x_i=[-5.12,-5.00000001]"""
    pass


class Quartic(AbstractFunction):
    __doc__ = \
    """a De Jong quartic function generator

De Jong's quartic function [1,2,3] is designed to test the
behavior of minimizers in the presence of noise. The function's
global minumum depends on the expectation value of a random
variable, and also includes several randomly distributed local
minima.

The generated function f(x) is a modified version of equation (20)
of [2], where len(x) >= 0.
    """ + _doc
    def __init__(self, ndim=30): # is n-dimensional (n=30 in ref)
        AbstractFunction.__init__(self, ndim=ndim)
        return

    def function(self,coeffs):
        r"""evaluates an N-dimensional quartic function for a list of coeffs

f(x) = \sum_(i=0)^(N-1) (x_(i)^4 * (i+1) + k_i)
 
Where k_i is a random variable with uniform distribution bounded by [0,1).

Inspect with mystic_model_plotter using::
    mystic.models.quartic -b "-3:3:.1, -3:3:.1" -d -x 1

The minimum is f(x)=N*E[k] for x_i=0.0, where E[k] is the expectation
of k, and thus E[k]=0.5 for a uniform distribution bounded by [0,1)."""
        f = 0.
        for j, c in enumerate(coeffs):
            f += pow(c,4) * (j+1.0) + random.random()
        return f

    minimizers = None #FIXME: statistical minimum... of f(x) <= N*0.5
    pass


class Shekel(AbstractFunction):
    __doc__ = \
    """a Shekel's Foxholes function generator

Shekel's Foxholes function [1,2,3] has a generally flat surface
with several narrow wells. The function's global minimum is at
(-32, -32), with local minima at (i,j) in (-32, -16, 0, 16, 32).

The generated function f(x) is a modified version of equation (21)
of [2], where len(x) == 2.
    """ + _doc
    def __init__(self, ndim=2):
        AbstractFunction.__init__(self, ndim=ndim)
        return

    def function(self,coeffs):
        r"""evaluates a 2-D Shekel's Foxholes function for a list of coeffs

f(x) = 1 / (0.002 + f_0(x))

Where:
f_0(x) = \sum_(i=0)^(24) 1 / (i + \sum_(j=0)^(1) (x_j - a_ij)^(6))
with a_ij=(-32,-16,0,16,32).
for j=0 and i=(0,1,2,3,4), a_i0=a_k0 with k=i \mod 5
also j=1 and i=(0,5,10,15,20), a_i1=a_k1 with k=i+k' and k'=(1,2,3,4).

Inspect with mystic_model_plotter using::
    mystic.models.shekel -b "-50:50:1, -50:50:1" -d -x 1

The minimum is f(x)=0 for x=(-32,-32)"""
        A = [-32., -16., 0., 16., 32.]
        a1 = A * 5
        a2 = reduce(lambda x1,x2: x1+x2, [[c] * 5 for c in A])

        x,y=coeffs
        r = 0.0
        for i in range(25):
#           r += 1.0/ (1.0*i + pow(x-a1[i],6) + pow(y-a2[i],6) + 1e-15)
            z = 1.0*i + pow(x-a1[i],6) + pow(y-a2[i],6)
            if z: r += 1.0/z
            else: r += inf
        return 1.0/(0.002 + r)

    minimizers = [(j,i) for (i,j) in sorted(list(permutations(range(-32,33,16),2))+[(i,i) for i in range(-32,33,16)])]
    pass

# cleanup
del _doc

# prepared instances
sphere = Sphere().function
rosen = Rosenbrock().function
step = Step().function
quartic = Quartic().function
shekel = Shekel().function
rosen0der = Rosenbrock(axis=0).function
rosen1der = Rosenbrock(axis=0).derivative

# End of file
