#!/usr/bin/env python

## Abstract Solver Class
# ...
# by mmckerns@caltech.edu

"""
module doc... 
"""
__all__ = ['AbstractSolver']


from mystic.tools import Null

import numpy
from numpy import shape, asarray, absolute, asfarray

abs = absolute



class AbstractSolver(object):
    """
    AbstractSolver base class for mystic optimizers.
    """
    
    def __init__(self, dim, **kwds):
        """
 constructor doc... <document class members here>
        """
        NP = 1
        if kwds.has_key('npop'): NP = kwds['npop']

        self.nDim             = dim
        self.nPop             = NP
        self.generations      = 0
        self.bestEnergy       = 0.0
        self.bestSolution     = [0.0] * dim
        self.trialSolution    = [0.0] * dim

        self._popEnergy_x0    = 1.0E20 #XXX: or numpy.inf?
        self.popEnergy	      = [0.0] * NP
        self.population	      = [[0.0 for i in range(dim)] for j in range(NP)]
        self.energy_history   = []
        self.signal_handler   = None

        self._handle_sigint   = False
        self._useStrictRange  = False
        self._strictMin       = []
        self._strictMax       = []
        self._maxiter         = None
        self._maxfun          = None


    def Solution(self):
        """return the best solution"""
        return self.bestSolution

    def SetStrictRanges(self, min, max):
        """ensure solution is within bounds
        - min, max: must be a sequence of length self.nDim
        - each min[i] should be <= the corresponding max[i]"""
        min = asarray(min); max = asarray(max)
        if numpy.any(( min > max ),0):
            raise ValueError, "each min[i] must be <= the corresponding max[i]"
        if len(min) != self.nDim:
            raise ValueError, "bounds array must be length %s" % self.nDim
        self._useStrictRange = True
        self._strictMin = min
        self._strictMax = max
        return

    def _clipGuessWithinRangeBoundary(self, x0): #FIXME: use self.trialSolution?
        """ensure that initial guess is set within bounds
        - x0: must be a sequence of length self.nDim"""
       #if len(x0) != self.nDim: #XXX: unnecessary w/ self.trialSolution
       #    raise ValueError, "initial guess must be length %s" % self.nDim
        x0 = asarray(x0)
        lo = self._strictMin
        hi = self._strictMax
        # crop x0 at bounds
        x0[x0<lo] = lo[x0<lo]
        x0[x0>hi] = hi[x0>hi]
        return x0

    def SetInitialPoints(self, x0, radius=0.05):
        """Set Initial Points with Guess (x0)
        - x0: must be a sequence of length self.nDim
        - radius: generate random points within [-radius*x0, radius*x0]
                  for i!=0 when a simplex-type initial guess in required"""
        x0 = asfarray(x0)
        rank = len(x0.shape)
        if rank is 0:
            x0 = asfarray([x0])
            rank = 1
        if not -1 < rank < 2:
            raise ValueError, "Initial guess must be a scalar or rank-1 sequence."
        if len(x0) != self.nDim:
            raise ValueError, "Initial guess must be length %s" % self.nDim

        #slightly alter initial values for solvers that depend on randomness
        min = x0*(1-radius)
        max = x0*(1+radius)
        numzeros = len(x0[x0==0])
        min[min==0] = asarray([-radius for i in range(numzeros)])
        max[max==0] = asarray([radius for i in range(numzeros)])
        self.SetRandomInitialPoints(min,max)
        #stick initial values in population[i], i=0
        self.population[0] = x0
    
    def SetRandomInitialPoints(self, min=None, max=None):
        """Generate Random Initial Points within given Bounds
        - min, max: must be a sequence of length self.nDim
        - each min[i] should be <= the corresponding max[i]"""
        if min == None: min = [-1e3]*self.nDim #XXX: good default range?
        if max == None: max = [1e3]*self.nDim
       #if numpy.any(( asarray(min) > asarray(max) ),0):
       #    raise ValueError, "each min[i] must be <= the corresponding max[i]"
        if len(min) != self.nDim or len(max) != self.nDim:
            raise ValueError, "bounds array must be length %s" % self.nDim
        import random
        #generate random initial values
        for i in range(self.nPop):
            for j in range(self.nDim):
                self.population[i][j] = random.uniform(min[j],max[j])
            self.popEnergy[i] = self._popEnergy_x0

    def SetMultinormalInitialPoints(self, mean, var = None):
        """Generate Initial Points from Multivariate Normal.
        - mean must be a sequence of length self.nDim
        - var can be None: -> it becomes the identity
                   scalar: -> var becomes scalar * I
                   matrix: -> the variance matrix. better be the right size !
        """
        from numpy.random import multivariate_normal
        assert(len(mean) == self.nDim)
        if var == None:
            var = numpy.eye(self.nDim)
        else:
            try: # scalar ?
                float(var)
            except: # nope. var better be matrix of the right size (no check)
                pass
            else:
                var = var * numpy.eye(self.nDim)
        for i in range(self.nPop):
            self.population[i] = multivariate_normal(mean, var).tolist()
            self.popEnergy[i] = self._popEnergy_x0
        return

    def enable_signal_handler(self):
        """enable workflow interrupt handler while solver is running"""
        self._handle_sigint = True

    def disable_signal_handler(self):
        """disable workflow interrupt handler while solver is running"""
        self._handle_sigint = False

    def SetEvaluationLimits(self,*args,**kwds):
        """set limits for maxiter and/or maxfun
  - maxiter = maximum number of solver iterations (i.e. steps)
  - maxfun  = maximum number of function evaluations"""
       #self._maxiter,self._maxfun = None,None
        if len(args) == 2:
            self._maxiter,self._maxfun = args[0],args[1]
        elif len(args) == 1:
            self._maxiter = args[0]
        if kwds.has_key('maxiter'): self._maxiter = kwds['maxiter']
        if kwds.has_key('maxfun'): self._maxfun = kwds['maxfun']

    def Solve(self, func, termination, sigint_callback=None,
              EvaluationMonitor=Null, StepMonitor=Null, ExtraArgs=(), **kwds):
        """solve function 'func' with given termination conditions
        *** this method must be overwritten ***"""
        raise NotImplementedError, "must be overwritten..."



if __name__=='__main__':
    help(__name__)

# end of file
