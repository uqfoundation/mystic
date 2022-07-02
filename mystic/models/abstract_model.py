#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Base classes for mystic's provided models::
    AbstractFunction   -- evaluates f(x) for given evaluation points x
    AbstractModel      -- generates f(x,p) for given coefficients p

"""
from numpy import sum as numpysum
from mystic.forward_model import CostFactory as CF

class AbstractFunction(object):
    """
Base class for mystic functions

The 'function' method must be overwritten, thus allowing
calls to the class instance to mimic calls to the function object.

For example, if function is overwritten with the Rosenbrock function:
    >>> rosen = Rosenbrock(ndim=3)
    >>> rosen([1,1,1])
    0.
   """

    def __init__(self, ndim=None):
        """
Provides a base class for mystic functions.

Takes optional input 'ndim' (number of dimensions).
        """
        if self.minimizers is None:
            self.minimizers = []
           #raise NotImplementedError('minimizers have not been provided')
        nmin = len(self.minimizers)

        # set the number of dimensions
        try: # fixed dimension
            self.ndim = len(self.minimizers[0] if nmin else None)
            fixed = True
            if ndim is None: ndim = self.ndim
        except TypeError: # not fixed
            self.ndim = ndim
            fixed = False
        if fixed and ndim != self.ndim:
            raise ValueError('number of dimensions is fixed (ndim=%d)' % self.ndim)
        elif not fixed and ndim is None:
            raise ValueError('number of dimensions must be set (ndim=None)')

        # set the minimizer (and adjust minimizers if not fixed)
        if not nmin:
            self.minimizers = list(self.minimizers)
            self.minimizer = None
        elif not fixed: #XXX: should be array instead of tuple?
            self.minimizers = [tuple([m]*ndim) for m in self.minimizers]
            self.minimizer = self.minimizers[0] # global *must* be first
        else:
            self.minimizers = list(self.minimizers)
            self.minimizer = self.minimizers[0] # global *must* be first

        # get the mimima
        self.minima = list(map(self.function, self.minimizers))
        self.minimum = min(self.minima) if self.minima else None
        if self.minima and self.minima.index(self.minimum):
            raise ValueError('global minimum must be at index = 0')
        return

    def __call__(self,*args,**kwds):
        coeffs = kwds['coeffs'] if 'coeffs' in kwds else (args[0] if len(args) else [])
        if len(coeffs) != self.ndim:
            raise ValueError('input length does not match ndim (ndim=%d)' % self.ndim)
        return self.function(*args,**kwds)

    def function(self,coeffs):
        """takes a list of coefficients x, returns f(x)"""
        raise NotImplementedError("overwrite function for each derived class")

#   def forward(self,pts):
#       """takes points p=(x,y,...), returns f(xi,yi,...)"""
#       pts = asarray(pts) #XXX: converting to numpy.array slows by 10x
#       return [self.function(i) for i in pts.transpose()]

    minimizers = None # not given; *must* set to 'list' or 'list of tuples'
    # - None          => not set / not known / no minimizers
    # - [1,5]         => global and local; input dimensions are not fixed
    # - [(1,2),(5,3)] => global and local; input dimensions are fixed
    pass


class AbstractModel(object):
    """
Base class for mystic models

The 'evaluate' and 'ForwardFactory' methods must be overwritten, thus providing
a standard interface for generating a forward model factory and evaluating a
forward model.  Additionally, two common ways to generate a cost function are
built into the model.  For "standard models", the cost function generator will
work with no modifications.

See `mystic.models.poly` for a few basic examples.
    """

    def __init__(self,name='dummy',metric=lambda x: numpysum(x*x),sigma=1.0):
        """
Provides a base class for mystic models.

Inputs::
    name    -- a name string for the model
    metric  -- the cost metric object  [default => lambda x: numpy.sum(x*x)]
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
        raise NotImplementedError("overwrite for each derived class")

    def ForwardFactory(self,coeffs):
        """generates a forward model instance from a list of coefficients"""
        raise NotImplementedError("overwrite for each derived class")

    def CostFactory(self,target,pts):
        """generates a cost function instance from list of coefficients 
and evaluation points"""
        datapts = self.evaluate(target,pts)
        F = CF()
        F.addModel(self.ForwardFactory,len(target),self.__name__)
        self.__cost__ = F.getCostFunction(evalpts=pts,observations=datapts,sigma=self.__sigma__,metric=self.__metric__)
        return self.__cost__

    def CostFactory2(self,pts,datapts,nparams):
        """generates a cost function instance from datapoints 
and evaluation points"""
        F = CF()
        F.addModel(self.ForwardFactory,nparams,self.__name__)
        self.__cost__ = F.getCostFunction(evalpts=pts,observations=datapts,sigma=self.__sigma__,metric=self.__metric__)
        return self.__cost__

    pass
