#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
1d model representation for polynomials

References:
    1. Storn, R. and Price, K. "Differential Evolution - A Simple and
       Efficient Heuristic for Global Optimization over Continuous Spaces"
       Journal of Global Optimization 11: 341-359, 1997.
    2. Storn, R. and Price, K. "Differential Evolution - A Simple and
       Efficient Heuristic for Global Optimization over Continuous Spaces"
       TR-95-012, ICSI, 1995. http://www.icsi.berkeley.edu/~storn/TR-95-012.pdf
    3. Storn, R. "Constrained Optimization" Dr. Dobb's Journal, May,
       119-123, 1995.
"""
from .abstract_model import AbstractModel

from numpy import sum as numpysum
from numpy import asarray
from mystic.math import polyeval, poly1d


class Polynomial(AbstractModel):
    """1-D Polynomial models and functions"""

    def __init__(self,name='poly',metric=lambda x: numpysum(x*x),sigma=1.0):
        AbstractModel.__init__(self,name,metric,sigma)
        return

    def evaluate(self,coeffs,x):
        """takes list of coefficients & evaluation points, returns f(x)
thus, [a3, a2, a1, a0] yields  a3 x^3 + a2 x^2 + a1 x^1 + a0"""
        return polyeval(coeffs,x)

    def ForwardFactory(self,coeffs):
        """generates a 1-D polynomial instance from a list of coefficients
using numpy.poly1d(coeffs)"""
        self.__forward__ = poly1d(coeffs)
        return self.__forward__

    pass


# coefficients for specific Chebyshev polynomials
from mystic.models._model_helper import chebyshev2coeffs, chebyshev4coeffs, \
                      chebyshev6coeffs, chebyshev8coeffs, chebyshev16coeffs

class Chebyshev(Polynomial):
    """Chebyshev polynomial models and functions,
including specific methods for Tn(z) n=2,4,6,8,16, Equation (27-33) of [2]

NOTE: default is T8(z)"""

    def __init__(self,order=8,name='poly',metric=lambda x: numpysum(x*x),sigma=1.0):
        Polynomial.__init__(self,name,metric,sigma)
        if order == 2:  self.coeffs = chebyshev2coeffs
        elif order == 4:  self.coeffs = chebyshev4coeffs
        elif order == 6:  self.coeffs = chebyshev6coeffs
        elif order == 8:  self.coeffs = chebyshev8coeffs
        elif order == 16:  self.coeffs = chebyshev16coeffs
        else: raise NotImplementedError("provide self.coeffs 'by hand'")
        return

    def __call__(self,*args,**kwds):
        return self.forward(*args,**kwds)

    def forward(self,x):
        """forward Chebyshev function""" #of order %s""" % (len(self.coeffs)-1)
        fwd = Polynomial.ForwardFactory(self,self.coeffs)
        return fwd(x)

    def ForwardFactory(self,coeffs):
        """generates a 1-D polynomial instance from a list of coefficients"""
        raise NotImplementedError("use Polynomial.ForwardFactory(coeffs)")

    def CostFactory(self,target,pts):
        """generates a cost function instance from list of coefficients & evaluation points"""
        raise NotImplementedError("use Polynomial.CostFactory(targets,pts)")

    def CostFactory2(self,pts,datapts,nparams):
        """generates a cost function instance from datapoints & evaluation points"""
        raise NotImplementedError("use Polynomial.CostFactory2(pts,datapts,nparams)")

    def cost(self,trial,M=61):
        """The costfunction for order-n Chebyshev fitting.
M evaluation points between [-1, 1], and two end points"""# % (len(self.coeffs)-1)
        #XXX: throw error when len(trial) != len(self.coeffs) ?
        myCost = chebyshevcostfactory(self.coeffs)
        return myCost(trial,M)

    pass

 
# faster implementation
def chebyshevcostfactory(target):
    def chebyshevcost(trial,M=61):
        """The costfunction for order-n Chebyshev fitting.
M evaluation points between [-1, 1], and two end points"""
        from mystic.models._model_helper import chebyshev
        return chebyshev(trial, target, M)
    return chebyshevcost

# prepared instances
poly = Polynomial()
chebyshev2 = Chebyshev(2)
chebyshev4 = Chebyshev(4)
chebyshev6 = Chebyshev(6)
chebyshev8 = Chebyshev(8)
chebyshev16 = Chebyshev(16)

# adjusted for parallel use
def chebyshev2cost(trial,M=61):
    """The costfunction for order-n Chebyshev fitting.
M evaluation points between [-1, 1], and two end points"""
    from mystic.models._model_helper import chebyshev, chebyshev2coeffs
    return chebyshev(trial, chebyshev2coeffs, M)
def chebyshev4cost(trial,M=61):
    """The costfunction for order-n Chebyshev fitting.
M evaluation points between [-1, 1], and two end points"""
    from mystic.models._model_helper import chebyshev, chebyshev4coeffs
    return chebyshev(trial, chebyshev4coeffs, M)
def chebyshev6cost(trial,M=61):
    """The costfunction for order-n Chebyshev fitting.
M evaluation points between [-1, 1], and two end points"""
    from mystic.models._model_helper import chebyshev, chebyshev6coeffs
    return chebyshev(trial, chebyshev6coeffs, M)
def chebyshev8cost(trial,M=61):
    """The costfunction for order-n Chebyshev fitting.
M evaluation points between [-1, 1], and two end points"""
    from mystic.models._model_helper import chebyshev, chebyshev8coeffs
    return chebyshev(trial, chebyshev8coeffs, M)
def chebyshev16cost(trial,M=61):
    """The costfunction for order-n Chebyshev fitting.
M evaluation points between [-1, 1], and two end points"""
    from mystic.models._model_helper import chebyshev, chebyshev16coeffs
    return chebyshev(trial, chebyshev16coeffs, M)


# End of file
