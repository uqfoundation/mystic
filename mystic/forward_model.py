#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                       Patrick Hung & Mike McKerns, Caltech
#                        (C) 1998-2008  All Rights Reserved
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

"""
This module contains classes that aid in constructing cost functions.
Cost function can easily be created by hand; however, mystic also
provides an automated method that allows the dynamic wrapping of 
forward models into cost function objects.

Usage
=====

The basic usage pattern for a cost factory is to generate a cost function
from a set of data points and a corresponding set of evaluation points.
The cost factory requires a "forward model factory", which is just a generator
of forward model instances from a list of coefficients. The following example
uses numpy.poly1d, which provides a factory for generating polynomials. An
expanded version of the following can be found in `mystic.examples.example12`.

    >>> # get a forward model factory, and generate some evaluation points
    >>> from numpy import array, sum, poly1d, random
    >>> ForwardFactory = poly1d
    >>> pts = 0.1*(numpy.array([range(101)])-50.)[0]
    >>> 
    >>> # we don't have real data, so generate some fake data from the model
    >>> target = [2.,-5.,3.]
    >>> datapts = [random.normal(0,1) + i for i in ForwardFactory(target)(pts)]
    >>> 
    >>> # get a cost factory
    >>> from mystic.forward_model import CostFactory
    >>> F = CostFactory()
    >>> 
    >>> # generate a cost function for the model factory
    >>> costmetric = lambda x: numpy.sum(x*x)
    >>> F.addModel(ForwardFactory, name='example', inputs=len(target))
    >>> costfunction = F.getCostFunction(evalpts=pts, observations=datapts,
    ...                                  sigma=1.0, metric=costmetric)
    >>>
    >>> # pass the cost function to the optimizer
    >>> initial_guess = [1.,-2.,1.]
    >>> solution = fmin_powell(costfunction, initial_guess)

In general, a user will be required to write their own model factory.
See the examples contained in `mystic.models` for more information.

The CostFactory can be used to couple models together into a single cost
function. For an example, see `mystic.examples.forward_model`.

"""

from mystic.filters import Identity, PickComponent
from mystic.filters import NullChecker

from inspect import getargspec
from numpy import pi, sqrt, array, mgrid, random, real, conjugate, arange, sum
#from numpy.random import rand


class ForwardModel1(object):
    """
A simple utility to 'prettify' object representations of forward models.
    """
    def __init__(self, func, inputs, outputs):
        """
Takes three initial inputs:
    func      -- the forward model
    inputs    -- the number of inputs
    outputs   -- the number of outputs

Example:
    >>> f = lambda x: sum(x*x)
    >>> fwd = ForwardModel1(f,1,1)
    >>> fwd
    func: ['x0'] -> ['y0']
    >>> fwd(1)
    1
    >>> fwd(2)
    4
        """
        self._func = func
        self._m = inputs
        self._n = outputs
        self._inNames = ['x%d'%i for i in range(self._m)]
        self._outNames = ['y%d'%i for i in range(self._n)]

    def __call__(self, *args, **kwds):
        return self._func(*args, **kwds)

    def __repr__(self):
        return 'func: %s -> %s' % (self._inNames, self._outNames)

class CostFactory(object):
    """
A cost function generator.
    """
    def __init__(self):
        """
CostFactory builds a list of forward model factories, and maintains a list
of associated model names and number of inputs. Can be used to combine several
models into a single cost function.

Takes no initial inputs.
        """
        self._names = []
        self._forwardFactories = []
        self._inputs = []
        self._inputFilters = {}
        self._outputFilters = []
        self._inputCheckers = []
        pass

    def addModel(self, model, name, inputs, outputFilter = Identity, inputChecker = NullChecker):
        """
Adds a forward model factory to the cost factory.

Inputs:
    model   -- a callable function factory object
    name    -- a string representing the model name
    inputs  -- number of input arguments to model
        """
        if name in self._names:
             print "Model [%s] already in database." % name
             raise AssertionError
        self._names.append(name)
        self._forwardFactories.append(model)
        self._inputs.append(inputs)
        self._outputFilters.append(outputFilter)
        self._inputCheckers.append(inputChecker)

    #XXX: addModelNew is a work in progress...
    '''
    def addModelNew(self, model, name, outputFilter = Identity, inputChecker = NullChecker):
        """
Adds a forward model factory to the cost factory.
The number of inputs is determined with inspect.getargspec.

Inputs:
    model   -- a callable function factory object
    name    -- a string representing the model name
        """
        #NOTE: better to replace "old-style" addModel above?
        if name in self._names:
             print "Model [%s] already in database." % name
             raise AssertionError
        self._names.append(name)
        self._forwardFactories.append(model)
        inputs = getargspec(model)[0]
        self._inputs.append(len(inputs))
        self._outputFilters.append(outputFilter)
        self._inputCheckers.append(inputChecker)
    '''

    def getForwardEvaluator(self, evalpts):
        """
Get a model factory that allows simultaneous evaluation of all forward models
for the same set of evaluation points.

Inputs:
    evalpts -- a list of evaluation points
        """
        #NOTE: does NOT go through inputChecker
        def _(params):
            out = []
            ind = 0
            for F, n, ofilt in zip(self._forwardFactories, self._inputs, self._outputFilters):
                m = F(params[ind:ind+n])
                out.append(ofilt(m(evalpts)))
                ind = ind+n
            return out
        return _

    def getVectorCostFunction(self, evalpts, observations):
        """
Get a vector cost function that allows simultaneous evaluation of all
forward models for the same set of evaluation points and observation points.

Inputs:
    evalpts -- a list of evaluation points
    observations -- a list of data points

The vector cost metric is hard-wired to be the sum of the difference of
getForwardEvaluator(evalpts) and the observations.

NOTE: Input parameters do NOT go through filters registered as inputCheckers.
        """
        def _(params):
            forward = self.getForwardEvaluator(evalpts)
            return sum(forward(params)) - observations
        return _

    def getCostFunction(self, evalpts, observations, sigma = None, metric = lambda x: sum(x*x)):
        """
Get a cost function that allows simultaneous evaluation of all forward models
for the same set of evaluation points and observation points.

Inputs:
    evalpts -- a list of evaluation points
    observations -- a list of data points
    sigma   -- a scaling factor applied to the raw cost
    metric  -- the cost metric object

The cost metric should be a function of one parameter (possibly an array)
that returns a scalar. The default is L2. When called, the "misfit" will
be passed in.

NOTE: Input parameters WILL go through filters registered as inputCheckers.
        """
        #XXX: better interface for sigma?
        def _(params):
            ind = 0
            for F, n, ofilt, icheck in zip(self._forwardFactories, self._inputs, \
                                       self._outputFilters, self._inputCheckers):
                # check input  #XXX: is this worthwile to do?
                my_params = params[ind:ind+n]
                checkQ = icheck(my_params, evalpts)
                if checkQ is not None:
                    # some parameters are out of range... returns "cost"
                    return checkQ

                Gm = F(params[ind:ind+n])
                if ind == 0:
                    x = ofilt(Gm(evalpts)) 
                else:
                    x = x + ofilt(Gm(evalpts)) 
                ind = ind+n
            if sigma == None:
                x = x - observations
            else:
                x = (x - observations) / sigma
            #return sum(real((conjugate(x)*x)))
            #return sum(x*x) 
            return metric(x)
        return _

    def getCostFunctionSlow(self, evalpts, observations):
        """
Get a cost function that allows simultaneous evaluation of all forward models
for the same set of evaluation points and observation points.

Inputs:
    evalpts -- a list of evaluation points
    observations -- a list of data points

The cost metric is hard-wired to be the sum of the real part of |x|^2,
where x is the VectorCostFunction for a given set of parameters.

NOTE: Input parameters do NOT go through filters registered as inputCheckers.
        """
        #XXX: update interface to allow metric?
        def _(params):
            v = self.getVectorCostFunction(evalpts, observations)
            x = v(params)
            return sum(real((conjugate(x)*x)))
        return _

    def getParameterList(self):
        """
Get a 'pretty' listing of the input parameters and corresponding models.
        """
        inputList = []
        for name, n in zip(self._names, self._inputs):
            inputList += ['%s.x%d' % (name, i) for i in range(n)]
        return inputList

    def getRandomParams(self):
        import random
        return array([random.random() for i in range(sum(self._inputs))])
    
    def __repr__(self):
        return "Input Parameter List: %s " % self.getParameterList()
            

if __name__=='__main__':
    pass


# End of file
