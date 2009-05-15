#!/usr/bin/env python

"""
Base classes for mystic's provided models
    AbstractFunction   -- evaluates f(x) for given evaluation points x
    AbstractModel      -- generates f(x,p) for given coefficients p

"""
from numpy import sum as numpysum
from mystic.forward_model import CostFactory as CF

class AbstractFunction(object):
    """Base class for mystic functions"""

    def __init__(self):
        """
Provides a base class for mystic functions.

The 'function' method must be overwritten, thus allowing
calls to the class instance to mimic calls to the function object.

For example, if function is overwritten with the Rosenbrock function:
    >>> rosen = Rosenbrock()
    >>> rosen(1,1,1)
    0.

Takes no inputs.
        """
        return

    def __call__(self,*args,**kwds):
        return self.function(*args,**kwds)

    def function(self,coeffs):
        """takes a list of coefficients x, returns f(x)"""
        raise NotImplementedError, "overwrite for each derived class"

#   def forward(self,pts):
#       """takes points p=(x,y,...), returns f(xi,yi,...)"""
#       pts = asarray(pts) #XXX: converting to numpy.array slows by 10x
#       return [self.function(i) for i in pts.transpose()] #FIXME: requires pts is a numpy.array
    pass


class AbstractModel(object):
    """Base class for mystic models"""

    def __init__(self,name='dummy',metric=lambda x: numpysum(x*x),sigma=1.0):
        """
Provides a base class for mystic models.

The 'evaluate' and 'ForwardFactory' methods must be overwritten, thus providing
a standard interface for generating a forward model factory and evaluating a
forward model.  Additionally, two common ways to generate a cost function are
built into the model.  For "standard models", the cost function generator will
work with no modifications.

See `mystic.models.poly` for a few basic examples.

Inputs:
    name    -- a name string for the model
    metric  -- the cost metric object
    sigma   -- a scaling factor applied to the raw cost
        """
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
        """takes list of coefficients & evaluation points, returns f(x)"""
        raise NotImplementedError, "overwrite for each derived class"

    def ForwardFactory(self,coeffs):
        """generates a forward model instance from a list of coefficients"""
        raise NotImplementedError, "overwrite for each derived class"

    def CostFactory(self,target,pts):
        """generates a cost function instance from list of coefficients 
and evaluation points"""
        datapts = self.evaluate(target,pts)
        F = CF()
        F.addModel(self.ForwardFactory,self.__name__,len(target))
        self.__cost__ = F.getCostFunction(evalpts=pts,observations=datapts,sigma=self.__sigma__,metric=self.__metric__)
        return self.__cost__

    def CostFactory2(self,pts,datapts,nparams):
        """generates a cost function instance from datapoints 
and evaluation points"""
        F = CF()
        F.addModel(self.ForwardFactory,self.__name__,nparams)
        self.__cost__ = F.getCostFunction(evalpts=pts,observations=datapts,sigma=self.__sigma__,metric=self.__metric__)
        return self.__cost__

    pass
