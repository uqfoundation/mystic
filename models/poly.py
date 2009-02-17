#!/usr/bin/env python

"""
References:

[1] Storn, R. and Price, K. Differential Evolution - A Simple and Efficient
Heuristic for Global Optimization over Continuous Spaces. Journal of Global
Optimization 11: 341-359, 1997.

[2] Storn, R. and Price, K. (Same title as above, but as a technical report.)
http://www.icsi.berkeley.edu/~storn/deshort1.ps
"""

from numpy import sum as numpysum
from numpy import asarray
from mystic.forward_model import CostFactory as CF
from numpy import poly1d as npoly1d

class AbstractModel(object):
    """abstract Model"""

    def __init__(self,name='dummy',metric=lambda x: numpysum(x*x),sigma=1.0):
        self.__name__ = name
        self.__metric__ = metric
        self.__sigma__ = sigma
        self.__forward__ = None
        self.__cost__ = None
        return

#   def forward(self,pts):
#       """takes points p=(x,y,...), returns f(xi,yi,...)"""
#       if self.__forward__ is None:
#           raise NotImplementedError, "construct w/ ForwardFactory(coeffs)"
#       return self.__forward__(pts)

#   def cost(self,coeffs):
#       """takes list of coefficients, return cost of metric(coeffs,target) at evalpts"""
#       if self.__cost__ is None:
#           raise NotImplementedError, "construct w/ CostFactory(target,pts)"
#       return self.__cost__(coeffs)

    def evaluate(self,coeffs,x):
        """takes lists of coefficients & evaluation points, returns f(x)"""
        raise NotImplementedError, "overwrite for each derived class"

    def ForwardFactory(self,coeffs):
        """generates a forward model instance from a list of coefficients"""
        raise NotImplementedError, "overwrite for each derived class"

    def CostFactory(self,target,pts):
        """generates a cost function instance from lists of coefficients & evaluation points"""
        datapts = self.evaluate(target,pts)
        F = CF()
        F.addModel(self.ForwardFactory,self.__name__,len(target))
        self.__cost__ = F.getCostFunction(evalpts=pts,observations=datapts,sigma=self.__sigma__,metric=self.__metric__)
        return self.__cost__

    def CostFactory2(self,pts,datapts,nparams):
        """generates a cost function instance from datapoints & evaluation points"""
        F = CF()
        F.addModel(self.ForwardFactory,self.__name__,nparams)
        self.__cost__ = F.getCostFunction(evalpts=pts,observations=datapts,sigma=self.__sigma__,metric=self.__metric__)
        return self.__cost__

    pass


class Polynomial(AbstractModel):
    """1-D Polynomial models and functions"""

    def __init__(self,name='poly',metric=lambda x: numpysum(x*x),sigma=1.0):
        AbstractModel.__init__(self,name,metric,sigma)
        return

    def evaluate(self,coeffs,x):
        """takes lists of coefficients & evaluation points, returns f(x)
thus, [a3, a2, a1, a0] yields  a3 x^3 + a2 x^2 + a1 x^1 + a0"""
        # The effect is this:
        #    return reduce(lambda x1, x2: x1 * x + x2, coeffs, 0)
        # However, the for loop used below is faster by about 50%.
        x = asarray(x)
        val = 0*x
        for c in coeffs:
           val = c + val*x
        return val

    def ForwardFactory(self,coeffs):
        """generates a 1-D polynomial instance from a list of coefficients
i.e. numpy.poly1d(coeffs)"""
        self.__forward__ = npoly1d(coeffs)
        return self.__forward__

    pass


# coefficients for specific Chebyshev polynomials
chebyshev8coeffs = [128., 0., -256., 0., 160., 0., -32., 0., 1.]
chebyshev16coeffs = [32768., 0., -131072., 0., 212992., 0., -180224., 0., 84480., 0., -21504., 0., 2688., 0., -128., 0., 1]

class Chebyshev(Polynomial):
    """Chebyshev polynomial models and functions:
with specific methods for T8(z) & T16(z), Eq. (27-33) of [2]
NOTE: default is T8(z)"""

    def __init__(self,order=8,name='poly',metric=lambda x: numpysum(x*x),sigma=1.0):
        Polynomial.__init__(self,name,metric,sigma)
        if order == 8:  self.coeffs = chebyshev8coeffs
        elif order == 16:  self.coeffs = chebyshev16coeffs
        else: raise NotImplementedError, "provide self.coeffs 'by hand'"
        return

    def __call__(self,*args,**kwds):
        return self.forward(*args,**kwds)

    def forward(self,x):
        """forward Chebyshev function""" #of order %s""" % (len(self.coeffs)-1)
        fwd = Polynomial.ForwardFactory(self,self.coeffs)
        return fwd(x)

    def ForwardFactory(self,coeffs):
        """generates a 1-D polynomial instance from a list of coefficients"""
        raise NotImplementedError, "use Polynomial.ForwardFactory(coeffs)"

    def CostFactory(self,target,pts):
        """generates a cost function instance from lists of coefficients & evaluation points"""
        raise NotImplementedError, "use Polynomial.CostFactory(targets,pts)"

    def CostFactory2(self,pts,datapts,nparams):
        """generates a cost function instance from datapoints & evaluation points"""
        raise NotImplementedError, "use Polynomial.CostFactory2(pts,datapts,nparams)"

    def __CostFactory(self,target,M):
        """target is a list of order-n Chebyshev polynomial coefficients
M is the desired number of evaluation points between [-1,1]"""
        def cost(trial):
            """Costfunction for Chebyshev order-%s polynomial.
Uses %s evaluation points between [-1, 1] w/ two end points""" % (len(target)-1,M)
            result=0.0
            x=-1.0
            dx = 2.0 / (M-1)
            for i in range(M):
                px = self.evaluate(trial, x)
                if px<-1 or px>1:
                    result += (1 - px) * (1 - px)
                x += dx

            px = self.evaluate(trial, 1.2) - self.evaluate(target, 1.2)
            if px<0: result += px*px

            px = self.evaluate(trial, -1.2) - self.evaluate(target, -1.2)
            if px<0: result += px*px

            return result
        return cost

    def cost(self,trial,M=61):
        """The costfunction for order-n Chebyshev fitting.
M evaluation points between [-1, 1], and two end points"""# % (len(self.coeffs)-1)
        #XXX: throw error when len(trial) != len(self.coeffs) ?
        myCost = self.__CostFactory(self.coeffs,M)
        return myCost(trial)

    pass
 

# prepared instances
poly = Polynomial()
chebyshev8 = Chebyshev(8)
chebyshev16 = Chebyshev(16)

polyeval = poly.evaluate
poly1d = poly.ForwardFactory
chebyshev8cost = chebyshev8.cost
chebyshev16cost = chebyshev16.cost


# End of file
