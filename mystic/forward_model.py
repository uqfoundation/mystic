#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
This module contains classes that aid in constructing cost functions.
Cost function can easily be created by hand; however, mystic also
provides an automated method that allows the dynamic wrapping of 
forward models into cost function objects.

Usage
=====

The basic usage pattern for a cost factory is to generate a cost function
from a set of data points and a corresponding set of evaluation points.
The cost factory requires a "model factory", which is just a generator
of model function instances from a list of coefficients. The following example
uses numpy.poly1d, which provides a factory for generating polynomials. An
expanded version of the following can be found in `mystic.examples.example12`.

    >>> # get a model factory
    >>> import numpy as np
    >>> FunctionFactory = np.poly1d
    >>> 
    >>> # generate some evaluation points
    >>> xpts = 0.1 * np.arange(-50.,51.)
    >>> 
    >>> # we don't have real data, so generate fake data from target and model
    >>> target = [2.,-5.,3.]
    >>> ydata = FunctionFactory(target)(xpts)
    >>> noise = np.random.normal(0,1,size=len(ydata))
    >>> ydata = ydata + noise
    >>> 
    >>> # get a cost factory
    >>> from mystic.forward_model import CostFactory
    >>> C = CostFactory()
    >>> 
    >>> # generate a cost function for the model factory
    >>> metric = lambda x: np.sum(x*x)
    >>> C.addModel(FunctionFactory, inputs=len(target))
    >>> cost = C.getCostFunction(evalpts=xpts, observations=ydata,
    ...                                        sigma=1.0, metric=metric)
    >>>
    >>> # pass the cost function to the optimizer
    >>> initial_guess = [1.,-2.,1.]
    >>> solution = fmin_powell(cost, initial_guess)
    >>> print(solution)
    [ 2.00495233 -5.0126248   2.72873734]


In general, a user will be required to write their own model factory.
See the examples contained in `mystic.models` for more information.

The CostFactory can be used to couple models together into a single cost
function. For an example, see `mystic.examples.forward_model`.

"""

from mystic.filters import identity, component
from mystic.filters import null_check

#from inspect import getargspec
from numpy import pi, sqrt, array, mgrid, random, real, conjugate, arange, sum

__all__ = ['CostFactory']

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

    def addModel(self, model, inputs, name=None, outputFilter=identity, inputChecker=null_check):
        """
Adds a forward model factory to the cost factory.

Inputs:
    model   -- a callable function factory object
    inputs  -- number of input arguments to model
    name    -- a string representing the model name

Example:
    >>> import numpy as np
    >>> C = CostFactory()
    >>> C.addModel(np.poly, inputs=3)
        """
        if name is None:
            import dill
            name = dill.source.getname(model)
        if name is None:
            for i in range(len(self._names)+1):
                name = 'model'+str(i)
                if name not in self._names: break
        elif name in self._names:
            print("Model [%s] already in database." % name)
            raise AssertionError
        self._names.append(name)
        self._forwardFactories.append(model)
        self._inputs.append(inputs)
        self._outputFilters.append(outputFilter)
        self._inputCheckers.append(inputChecker)

    #XXX: addModelNew is a work in progress...
    '''
    def addModelNew(self, model, name, outputFilter=identity, inputChecker=null_check):
        """
Adds a forward model factory to the cost factory.
The number of inputs is determined with inspect.getargspec.

Inputs:
    model   -- a callable function factory object
    name    -- a string representing the model name
        """
        #NOTE: better to replace "old-style" addModel above?
        if name in self._names:
            print("Model [%s] already in database." % name)
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

Example:
    >>> import numpy as np
    >>> C = CostFactory()
    >>> C.addModel(np.poly, inputs=3)
    >>> F = C.getForwardEvaluator([1,2,3,4,5])
    >>> F([1,0,0])
    [array([ 1,  4,  9, 16, 25])]
    >>> F([0,1,0])
    [array([1, 2, 3, 4, 5])]
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

Example:
    >>> import numpy as np
    >>> C = CostFactory()
    >>> C.addModel(np.poly, inputs=3)
    >>> x = np.array([-2., -1., 0., 1., 2.])
    >>> y = np.array([-4., -2., 0., 2., 4.])
    >>> F = C.getVectorCostFunction(x, y)
    >>> F([1,0,0])
    0.0
    >>> F([2,0,0])
    10.0
        """
        def _(params):
            forward = self.getForwardEvaluator(evalpts)
            return sum(forward(params) - observations)
        return _

    def getCostFunction(self, evalpts, observations, sigma=None, metric=lambda x: sum(x*x)):
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

Example:
    >>> import numpy as np
    >>> C = CostFactory()
    >>> C.addModel(np.poly, inputs=3)
    >>> x = np.array([-2., -1., 0., 1., 2.])
    >>> y = np.array([-4., -2., 0., 2., 4.])
    >>> F = C.getCostFunction(x, y, metric=lambda x: np.sum(x))
    >>> F([1,0,0])
    0.0
    >>> F([2,0,0])
    10.0
    >>> F = C.getCostFunction(x, y)
    >>> F([2,0,0])
    34.0
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
            if sigma is None:
                x = x - observations
            else:
                x = (x - observations) / sigma
            #return sum(real((conjugate(x)*x)))
            #return sum(x*x) 
            return metric(x)
        return _

    def getCostFunctionSlow(self, evalpts, observations):
        """Get a cost function that allows simultaneous evaluation of all
forward models for the same set of evaluation points and observation points.

Args:
    evalpts (list(float)): a list of evaluation points (i.e. input).
    observations (list(float)): a list of data points (i.e. output).

Notes:
    The cost metric is hard-wired to be the sum of the real part of ``|x|^2``,
    where x is the VectorCostFunction for a given set of parameters.

    Input parameters do NOT go through filters registered as inputCheckers.

Examples:
    >>> import numpy as np
    >>> C = CostFactory()
    >>> C.addModel(np.poly, inputs=3)
    >>> x = np.array([-2., -1., 0., 1., 2.])
    >>> y = np.array([-4., -2., 0., 2., 4.])
    >>> F = C.getCostFunctionSlow(x, y)
    >>> F([1,0,0])
    0.0
    >>> F([2,0,0])
    100.0
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
