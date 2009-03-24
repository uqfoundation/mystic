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
A set of classes that aids in constructing cost functions.

TODO: <documentation here>
"""

from mystic.filters import Identity, PickComponent
from mystic.filters import NullChecker

from inspect import getargspec
from numpy import pi, sqrt, array, mgrid, random, real, conjugate, arange, sum
#from numpy.random import rand


class ForwardModel1(object):
    """
    TODO: docstring here
    """
    def __init__(self, func, inputs, outputs):
        self._func = func
        self._m = inputs
        self._n = outputs
        self._inNames = ['x%d'%i for i in range(self._m)]
        self._outNames = ['y%d'%i for i in range(self._n)]

    def __repr__(self):
        return 'func: %s -> %s' % (self._inNames, self._outNames)

class CostFactory(object):
    """
    TODO: docstring here
    """
    def __init__(self):
        self._names = []
        self._forwardFactories = []
        self._inputs = []
        self._inputFilters = {}
        self._outputFilters = []
        self._inputCheckers = []
        pass

    def addModel(self, model, name, inputs, outputFilter = Identity, inputChecker = NullChecker):
        """
    Adds a Model Factory (needs interface definition)
    inputs:
        model: a callable function factory
        name: a string
        inputs: number of arguments to model
        """
        if name in self._names:
             print "Model [%s] already in database." % name
             raise AssertionError
        self._names.append(name)
        self._forwardFactories.append(model)
        self._inputs.append(inputs)
        self._outputFilters.append(outputFilter)
        self._inputCheckers.append(inputChecker)

    def addModelNew(self, model, name, outputFilter = Identity, inputChecker = NullChecker):
        """
    Adds a Model Factory (new style). No need for inputs because it can 
    be obtained from inspect.getargspec
    inputs:
        model: a callable function factory
        name: a string
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

    def getForwardEvaluator(self, evalpts):
        """
    TODO: docstring here
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
    TODO: docstring here
        """
        def _(params):
            forward = self.getForwardEvaluator(evalpts)
            return sum(forward(params)) - observations
        return _

    def getCostFunction(self, evalpts, observations, sigma = None, metric = lambda x: sum(x*x)):

        """Input params WILL go through inputCheckers
the metric should be a function that takes one parameter (possibly a vector) 
and returns a scalar. The default is L2. When called, the "misfit" will be passed in."""
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
    TODO: docstring here
        """
        #XXX: update interface to allow metric?
        def _(params):
            v = self.getVectorCostFunction(evalpts, observations)
            x = v(params)
            return sum(real((conjugate(x)*x)))
        return _

    def getParameterList(self):
        """
    TODO: docstring here
        """
        inputList = []
        for name, n in zip(self._names, self._inputs):
            inputList += ['%s.x%d' % (name, i) for i in range(n)]
        return inputList

    def getRandomParams(self):
        """
    TODO: docstring here
        """
        import random
        return array([random.random() for i in range(sum(self._inputs))])
    
    def __repr__(self):
        """
    TODO: docstring here
        """
        return "Input Parameter List: %s " % self.getParameterList()
            

if __name__=='__main__':
    pass


# End of file
