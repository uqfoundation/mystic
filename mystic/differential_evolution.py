#!/usr/bin/env python
#
## Differential Evolution Solver Class
## Based on algorithms developed by Dr. Rainer Storn & Kenneth Price
## Original C++ code written by: Lester E. Godwin
##                               PushCorp, Inc.
##                               Dallas, Texas
##                               972-840-0208 x102
##                               godwin@pushcorp.com
## Created: 6/8/98
## Last Modified: 6/8/98         Revision: 1.0
##
## Solver code ported to Python from C++ July 2002
## by: James R. Phillips
##     Birmingham, Alabama USA
##     zunzun@zunzun.com
##
## DE Solver modified and cleaned by Patrick Hung, May 2006.
## additional DE Solver (DESolver2) added by Patrick Hung.
##
## bounds (and minimal interface) added by mmckerns@caltech.edu
## adapted to AbstractSolver interface by mmckerns@caltech.edu
##
## modified for AbstractMapSolver interface by mmckerns@caltech.edu 

"""
Solvers
=======

This module contains a collection of optimization routines based on
Storn and Price's differential evolution algorithm.  The core solver
algorithm was adapted from Phillips's DETest.py.  An alternate solver
is provided that follows the logic in Price, Storn, and Lampen -- in that
both a current generation and a trial generation are maintained, and all
vectors for creating difference vectors and mutations draw from the
current generation... which remains invariant until the end of the
iteration.

A minimal interface that mimics a scipy.optimize interface has also been
implemented, and functionality from the mystic solver API has been added
with reasonable defaults. 

Minimal function interface to optimization routines::
    diffev      -- Differential Evolution (DE) solver
    diffev2     -- Price & Storn's Differential Evolution solver

The corresponding solvers built on mystic's AbstractSolver are::
    DifferentialEvolutionSolver  -- a DE solver
    DifferentialEvolutionSolver2 -- Storn & Price's DE solver

Mystic solver behavior activated in diffev and diffev2::
    - EvaluationMonitor = Sow()
    - StepMonitor = Sow()
    - enable_signal_handler()
    - strategy = Best1Exp
    - termination = ChangeOverGeneration(ftol,gtol), if gtol provided
          ''      = VTR(ftol), otherwise

Storn & Price's DE Solver has also been implemented to use the "map"
interface. Mystic enables the user to override the standard python
map function with their own 'map' function, or one of the map functions
provided by the pathos package (see http://dev.danse.us/trac/pathos)
for distributed and high-performance computing.


Usage
=====

Practical advice for how to configure the Differential Evolution
Solver for your own objective function can be found on R. Storn's
web page (http://www.icsi.berkeley.edu/~storn/code.html), and is
reproduced here::

    First try the following classical settings for the solver configuration:
    Choose a crossover strategy (e.g. Rand1Bin), set the number of parents
    NP to 10 times the number of parameters, select ScalingFactor=0.8, and
    CrossProbability=0.9.

    It has been found recently that selecting ScalingFactor from the interval
    [0.5, 1.0] randomly for each generation or for each difference vector,
    a technique called dither, improves convergence behaviour significantly,
    especially for noisy objective functions.

    It has also been found that setting CrossProbability to a low value,
    e.g. CrossProbability=0.2 helps optimizing separable functions since
    it fosters the search along the coordinate axes. On the contrary,
    this choice is not effective if parameter dependence is encountered,
    something which is frequently occuring in real-world optimization
    problems rather than artificial test functions. So for parameter
    dependence the choice of CrossProbability=0.9 is more appropriate.

    Another interesting empirical finding is that rasing NP above, say, 40
    does not substantially improve the convergence, independent of the
    number of parameters. It is worthwhile to experiment with these suggestions.
  
    Make sure that you initialize your parameter vectors by exploiting
    their full numerical range, i.e. if a parameter is allowed to exhibit
    values in the range [-100, 100] it's a good idea to pick the initial
    values from this range instead of unnecessarily restricting diversity.

    Keep in mind that different problems often require different settings
    for NP, ScalingFactor and CrossProbability (see Ref 1, 2). If you
    experience misconvergence, you typically can increase the value for NP,
    but often you only have to adjust ScalingFactor to be a little lower or
    higher than 0.8. If you increase NP and simultaneously lower ScalingFactor
    a little, convergence is more likely to occur but generally takes longer,
    i.e. DE is getting more robust (a convergence speed/robustness tradeoff).

    If you still get misconvergence you might want to instead try a different
    crossover strategy. The most commonly used are Rand1Bin, Rand1Exp,
    Best1Bin, and Best1Exp. The crossover strategy is not so important a
    choice, although K. Price claims that binomial (Bin) is never worse than
    exponential (Exp).

    In case of continued misconvergence, check the choice of objective function.
    There might be a better one to describe your problem. Any knowledge that
    you have about the problem should be worked into the objective function.
    A good objective function can make all the difference.

See `mystic.examples.test_rosenbrock` for an example of using
DifferentialEvolutionSolver. DifferentialEvolutionSolver2 has
the identical interface and usage.

All solvers included in this module provide the standard signal handling.
For more information, see `mystic.mystic.abstract_solver`.


References
==========

[1] Storn, R. and Price, K. Differential Evolution - A Simple and Efficient
Heuristic for Global Optimization over Continuous Spaces. Journal of Global
Optimization 11: 341-359, 1997.

[2] Price, K., Storn, R., and Lampinen, J. - Differential Evolution,
A Practical Approach to Global Optimization. Springer, 1st Edition, 2005

"""
__all__ = ['DifferentialEvolutionSolver','DifferentialEvolutionSolver2',\
           'diffev','diffev2']

from mystic.tools import Null, wrap_function, unpair
from mystic.tools import wrap_bounds

from abstract_solver import AbstractSolver
from abstract_map_solver import AbstractMapSolver

class DifferentialEvolutionSolver(AbstractSolver):
    """
Differential Evolution optimization.
    """
    
    def __init__(self, dim, NP):
        """
Takes two initial inputs: 
    dim  -- dimensionality of the problem
    NP   -- size of the trial solution population. [requires: NP >= 4]

All important class members are inherited from AbstractSolver.
        """
        #XXX: raise Error if npop <= 4?
        AbstractSolver.__init__(self,dim,npop=NP)
        self.genealogy     = [ [] for j in range(NP)]
        self.scale         = 0.8
        self.probability   = 0.9
        
### XXX: OBSOLETED by wrap_bounds ###
#   def _keepSolutionWithinRangeBoundary(self, base):
#       """scale trialSolution to be between base value and range boundary"""
#       if not self._useStrictRange:
#           return
#       min = self._strictMin
#       max = self._strictMax
#       import random
#       for i in range(self.nDim):
#           if base[i] < min[i] or base[i] > max[i]:
#               self.trialSolution[i] = random.uniform(min[i],max[i])
#           elif self.trialSolution[i] < min[i]:
#               self.trialSolution[i] = random.uniform(min[i],base[i])
#           elif self.trialSolution[i] > max[i]:
#               self.trialSolution[i] = random.uniform(base[i],max[i])
#       return

    def UpdateGenealogyRecords(self, id, newchild):
        """
Override me for more refined behavior. Currently all changes
are logged.
        """
        self.genealogy[id].append(newchild)
        return

    def Solve(self, costfunction, termination, sigint_callback=None,
              EvaluationMonitor=Null, StepMonitor=Null, ExtraArgs=(), **kwds):
        """Minimize a function using differential evolution.

Description:

    Uses a differential evolution algorith to find the minimum of
    a function of one or more variables.

Inputs:

    costfunction -- the Python function or method to be minimized.
    termination -- callable object providing termination conditions.

Additional Inputs:

    sigint_callback -- callback function for signal handler.
    EvaluationMonitor -- a callable object that will be passed x, fval
        whenever the cost function is evaluated.
    StepMonitor -- a callable object that will be passed x, fval
        after the end of a simplex iteration.
    ExtraArgs -- extra arguments for func.

Further Inputs:

    strategy -- the mutation strategy for generating new trial
        solutions [default = Best1Bin]
    CrossProbability -- the probability of cross-parameter mutations
        [default = 0.9]
    ScalingFactor -- multiplier for the impact of mutations on the
        trial solution [default = 0.8]
    callback -- an optional user-supplied function to call after each
        iteration.  It is called as callback(xk), where xk is
        the current parameter vector.  [default = None]
    disp -- non-zero to print convergence messages.

        """
        #allow for inputs that don't conform to AbstractSolver interface
        from mystic.strategy import Best1Bin
        strategy=Best1Bin    #mutation strategy (see mystic.strategy)
        CrossProbability=0.9 #potential for parameter cross-mutation
        ScalingFactor=0.8    #multiplier for mutation impact
        callback=None        #user-supplied function, called after each step
        disp=0               #non-zero to print convergence messages
        if kwds.has_key('strategy'): strategy = kwds['strategy']
        if kwds.has_key('CrossProbability'): \
           CrossProbability = kwds['CrossProbability']
        if kwds.has_key('ScalingFactor'): ScalingFactor = kwds['ScalingFactor']
        if kwds.has_key('callback'): callback = kwds['callback']
        if kwds.has_key('disp'): disp = kwds['disp']
        #-------------------------------------------------------------

        import signal
        self._EARLYEXIT = False

        self._fcalls, costfunction = wrap_function(costfunction, ExtraArgs, EvaluationMonitor)
        if self._useStrictRange:
            for i in range(self.nPop):
                self.population[i] = self._clipGuessWithinRangeBoundary(self.population[i])
            costfunction = wrap_bounds(costfunction, self._strictMin, self._strictMax)

        #generate signal_handler
        self._generateHandler(sigint_callback) 
        if self._handle_sigint: signal.signal(signal.SIGINT, self.signal_handler)

        id = self.id
        self.probability = CrossProbability
        self.scale = ScalingFactor

        #set initial solution and energy by running a single iteration
        self.bestSolution = self.population[0][:]
        self.bestEnergy = self.popEnergy[0]
        for candidate in range(self.nPop):
            # generate trialSolution (within valid range)
            self.trialSolution[:] = self.population[candidate][:]
            trialEnergy = costfunction(self.trialSolution)

            if trialEnergy < self.popEnergy[candidate]:
                # New low for this candidate
                self.popEnergy[candidate] = trialEnergy
                self.population[candidate][:] = self.trialSolution[:]
                self.UpdateGenealogyRecords(candidate, self.trialSolution[:])

                # Check if all-time low
                if trialEnergy < self.bestEnergy:
                    self.bestEnergy = trialEnergy
                    self.bestSolution[:] = self.trialSolution[:]

        self.energy_history.append(self.bestEnergy)
        StepMonitor(self.bestSolution[:], self.bestEnergy, id)
        self.generations = 0  #XXX: above currently *not* counted as an iteration
        if callback is not None:
            callback(self.bestSolution)
         
        if self._maxiter is None:
            self._maxiter = self.nDim * self.nPop * 10  #XXX: set better defaults?
        if self._maxfun is None:
            self._maxfun = self.nDim * self.nPop * 1000 #XXX: set better defaults?

        #run for generations <= maxiter
        for generation in range(self._maxiter - self.generations):
            if self._fcalls[0] >= self._maxfun: break
            for candidate in range(self.nPop):
                # generate trialSolution (within valid range)
                strategy(self, candidate)
                trialEnergy = costfunction(self.trialSolution)

                if trialEnergy < self.popEnergy[candidate]:
                    # New low for this candidate
                    self.popEnergy[candidate] = trialEnergy
                    self.population[candidate][:] = self.trialSolution[:]
                    self.UpdateGenealogyRecords(candidate, self.trialSolution[:])

                    # Check if all-time low
                    if trialEnergy < self.bestEnergy:
                        self.bestEnergy = trialEnergy
                        self.bestSolution[:] = self.trialSolution[:]

            self.energy_history.append(self.bestEnergy)
            StepMonitor(self.bestSolution[:], self.bestEnergy, id)
            self.generations += 1
            if callback is not None:
                callback(self.bestSolution)
            
            if self._EARLYEXIT or termination(self):
                break

        signal.signal(signal.SIGINT,signal.default_int_handler)

        # code below here pushes output to scipy.optimize.fmin interface
        fval = self.bestEnergy
        warnflag = 0

        if self._fcalls[0] >= self._maxfun:
            warnflag = 1
            if disp:
                print "Warning: Maximum number of function evaluations has "\
                      "been exceeded."
        elif self.generations >= self._maxiter:
            warnflag = 2
            if disp:
                print "Warning: Maximum number of iterations has been exceeded"
        else:
            if disp:
                print "Optimization terminated successfully."
                print "         Current function value: %f" % fval
                print "         Iterations: %d" % self.generations
                print "         Function evaluations: %d" % self._fcalls[0]

        return 



class DifferentialEvolutionSolver2(AbstractMapSolver):
    """
Differential Evolution optimization, using Storn and Price's algorithm.

Alternate implementation: 
    - utilizes a map-reduce interface, extensible to parallel computing
    - both a current and a next generation are kept, while the current
      generation is invariant during the main DE logic
    """
    def __init__(self, dim, NP):
        """
Takes two initial inputs: 
    dim  -- dimensionality of the problem
    NP   -- size of the trial solution population. [requires: NP >= 4]

All important class members are inherited from AbstractSolver.
        """
        #XXX: raise Error if npop <= 4?
        super(DifferentialEvolutionSolver2, self).__init__(dim, npop=NP)
        self.genealogy     = [ [] for j in range(NP)]
        self.scale         = 0.8
        self.probability   = 0.9
        
    def UpdateGenealogyRecords(self, id, newchild):
        """
Override me for more refined behavior. Currently all changes
are logged.
        """
        self.genealogy[id].append(newchild)
        return

    def Solve(self, costfunction, termination, sigint_callback=None,
              EvaluationMonitor=Null, StepMonitor=Null, ExtraArgs=(), **kwds):
        """Minimize a function using differential evolution.

Description:

    Uses a differential evolution algorith to find the minimum of
    a function of one or more variables. This implementation holds
    the current generation invariant until the end of each iteration.

Inputs:

    costfunction -- the Python function or method to be minimized.
    termination -- callable object providing termination conditions.

Additional Inputs:

    sigint_callback -- callback function for signal handler.
    EvaluationMonitor -- a callable object that will be passed x, fval
        whenever the cost function is evaluated.
    StepMonitor -- a callable object that will be passed x, fval
        after the end of a simplex iteration.
    ExtraArgs -- extra arguments for func.

Further Inputs:

    strategy -- the mutation strategy for generating new trial
        solutions [default = Best1Bin]
    CrossProbability -- the probability of cross-parameter mutations
        [default = 0.9]
    ScalingFactor -- multiplier for the impact of mutations on the
        trial solution [default = 0.8]
    callback -- an optional user-supplied function to call after each
        iteration.  It is called as callback(xk), where xk is
        the current parameter vector.  [default = None]
    constraints -- an optional user-supplied function to call just
        prior to each function evaluation.  It is called as
        xk' = constraints(xk), where xk is the current parameter vector.
        Ideally, this function is used to ensure each function evaulation
        encounters a parameter set that satisfies user-supplied constraints.
    disp -- non-zero to print convergence messages.
        """
        #allow for inputs that don't conform to AbstractSolver interface
        from mystic.strategy import Best1Bin
        strategy=Best1Bin    #mutation strategy (see mystic.strategy)
        CrossProbability=0.9 #potential for parameter cross-mutation
        ScalingFactor=0.8    #multiplier for mutation impact
        callback=None        #user-supplied function, called after each step
        disp=0               #non-zero to print convergence messages
        if kwds.has_key('strategy'): strategy = kwds['strategy']
        if kwds.has_key('CrossProbability'): \
           CrossProbability = kwds['CrossProbability']
        if kwds.has_key('ScalingFactor'): ScalingFactor = kwds['ScalingFactor']
        if kwds.has_key('callback'): callback = kwds['callback']
        if kwds.has_key('disp'): disp = kwds['disp']
        #XXX:[HACK] accept constraints
        constraints = lambda x: x
        if kwds.has_key('constraints'): constraints = kwds['constraints']
        if not constraints: constraints = lambda x: x
        #XXX:[HACK] -end-
        #-------------------------------------------------------------

        import signal
        self._EARLYEXIT = False

       #FIXME: EvaluationMonitor fails for MPI, throws error for 'pp'
        from python_map import python_map
        if self._map != python_map:
            self._fcalls = [0] #FIXME: temporary patch for removing the following line
        else:
            self._fcalls, costfunction = wrap_function(costfunction, ExtraArgs, EvaluationMonitor)
        if self._useStrictRange:
            for i in range(self.nPop):
                self.population[i] = self._clipGuessWithinRangeBoundary(self.population[i])
            costfunction = wrap_bounds(costfunction, self._strictMin, self._strictMax)

        #generate signal_handler
        self._generateHandler(sigint_callback) 
        if self._handle_sigint: signal.signal(signal.SIGINT, self.signal_handler)

        id = self.id
        self.probability = CrossProbability
        self.scale = ScalingFactor

        #set initial solution and energy by running a single iteration
        self.bestSolution = self.population[0][:]
        self.bestEnergy = self.popEnergy[0]
        trialPop = [[0.0 for i in range(self.nDim)] for j in range(self.nPop)]
        for candidate in range(self.nPop):
            # generate trialSolution (within valid range)
            self.trialSolution[:] = self.population[candidate][:]
            #XXX:[HACK] apply constraints
            self.trialSolution[:] = constraints(self.trialSolution[:])
            #XXX:[HACK] -end-
            trialPop[candidate][:] = self.trialSolution[:]

        mapconfig = dict(nnodes=self._nnodes, launcher=self._launcher, \
                         mapper=self._mapper, queue=self._queue, \
                         timelimit=self._timelimit, scheduler=self._scheduler, \
                         ncpus=self._ncpus, servers=self._servers)
        trialEnergy = self._map(costfunction, trialPop, **mapconfig)

        for candidate in range(self.nPop):
            if trialEnergy[candidate] < self.popEnergy[candidate]:
                # New low for this candidate
                self.popEnergy[candidate] = trialEnergy[candidate]
                self.population[candidate][:] = trialPop[candidate][:]
                self.UpdateGenealogyRecords(candidate, trialPop[candidate][:])

                # Check if all-time low
                if trialEnergy[candidate] < self.bestEnergy:
                    self.bestEnergy = trialEnergy[candidate]
                    self.bestSolution[:] = trialPop[candidate][:]

        self.energy_history.append(self.bestEnergy)
       #FIXME: StepMonitor works for 'pp'?
        StepMonitor(self.bestSolution[:], self.bestEnergy, id)
        self.generations = 0  #XXX: above currently *not* counted as an iteration
        if callback is not None:
            callback(self.bestSolution)
         
        if self._maxiter is None:
            self._maxiter = self.nDim * self.nPop * 10  #XXX: set better defaults?
        if self._maxfun is None:
            self._maxfun = self.nDim * self.nPop * 1000 #XXX: set better defaults?

        #run for generations <= maxiter
        for generation in range(self._maxiter - self.generations):
            if self._fcalls[0] >= self._maxfun: break
            for candidate in range(self.nPop):
                # generate trialSolution (within valid range)
                strategy(self, candidate)
                #XXX:[HACK] apply constraints
                self.trialSolution[:] = constraints(self.trialSolution[:])
                #XXX:[HACK] -end-
                trialPop[candidate][:] = self.trialSolution[:]

            trialEnergy = self._map(costfunction, trialPop, **mapconfig)
           ##XXX:[HACK] return trialPop b/c may be altered by constraints
           #trials = map(costfunction, trialPop)
           #trialEnergy = [i for i,j in trials]
           #trialPop = [j for i,j in trials] #FIXME: breaks on inf (out of bounds)
           ##XXX:[HACK] -end-

            for candidate in range(self.nPop):
                if trialEnergy[candidate] < self.popEnergy[candidate]:
                    # New low for this candidate
                    self.popEnergy[candidate] = trialEnergy[candidate]
                    self.population[candidate][:] = trialPop[candidate][:]
                    self.UpdateGenealogyRecords(candidate, trialPop[candidate][:])

                    # Check if all-time low
                    if trialEnergy[candidate] < self.bestEnergy:
                        self.bestEnergy = trialEnergy[candidate]
                        self.bestSolution[:] = trialPop[candidate][:]

            self.energy_history.append(self.bestEnergy)
           #FIXME: StepMonitor works for 'pp'?
            StepMonitor(self.bestSolution[:], self.bestEnergy, id)
            self.generations += 1
            if callback is not None:
                callback(self.bestSolution)
            
            if self._EARLYEXIT or termination(self):
                break

        signal.signal(signal.SIGINT,signal.default_int_handler)

        # code below here pushes output to scipy.optimize.fmin interface
        fval = self.bestEnergy
        warnflag = 0

        if self._fcalls[0] >= self._maxfun:
            warnflag = 1
            if disp:
                print "Warning: Maximum number of function evaluations has "\
                      "been exceeded."
        elif self.generations >= self._maxiter:
            warnflag = 2
            if disp:
                print "Warning: Maximum number of iterations has been exceeded"
        else:
            if disp:
                print "Optimization terminated successfully."
                print "         Current function value: %f" % fval
                print "         Iterations: %d" % self.generations
                print "         Function evaluations: %d" % self._fcalls[0]

        return 


def diffev2(func,x0,npop,args=(),bounds=None,ftol=5e-3,gtol=None,
            maxiter=None,maxfun=None,cross=0.9,scale=0.8,
            full_output=0,disp=1,retall=0,callback=None):
    """Minimize a function using Storn & Price's differential evolution.

Description:

    Uses Storn & Prices's differential evolution algorith to find the minimum
    of a function of one or more variables. Mimics a scipy.optimize style
    interface.

Inputs:

    func -- the Python function or method to be minimized.
    x0 -- the initial guess (ndarray), if desired to start from a
        set point; otherwise takes an array of (min,max) bounds,
        for when random initial points are desired
    npop -- size of the trial solution population.

Additional Inputs:

    args -- extra arguments for func.
    bounds -- list - n pairs of bounds (min,max), one pair for each parameter.
    ftol -- number - acceptable relative error in func(xopt) for convergence.
    gtol -- number - maximum number of iterations to run without improvement.
    maxiter -- number - the maximum number of iterations to perform.
    maxfun -- number - the maximum number of function evaluations.
    cross -- number - the probability of cross-parameter mutations
    scale -- number - multiplier for impact of mutations on trial solution.
    full_output -- number - non-zero if fval and warnflag outputs are desired.
    disp -- number - non-zero to print convergence messages.
    retall -- number - non-zero to return list of solutions at each iteration.
    callback -- an optional user-supplied function to call after each
        iteration.  It is called as callback(xk), where xk is the
        current parameter vector.

Returns: (xopt, {fopt, iter, funcalls, warnflag}, {allvecs})

    xopt -- ndarray - minimizer of function
    fopt -- number - value of function at minimum: fopt = func(xopt)
    iter -- number - number of iterations
    funcalls -- number - number of function calls
    warnflag -- number - Integer warning flag:
        1 : 'Maximum number of function evaluations.'
        2 : 'Maximum number of iterations.'
    allvecs -- list - a list of solutions at each iteration

    """
    return diffev(func,x0,npop,args=(),bounds=None,ftol=5e-3,gtol=None,
                  maxiter=None,maxfun=None,cross=0.9,scale=0.8,
                  full_output=0,disp=1,retall=0,callback=None,
                  invariant_current=True)


def diffev(func,x0,npop,args=(),bounds=None,ftol=5e-3,gtol=None,
           maxiter=None,maxfun=None,cross=0.9,scale=0.8,
           full_output=0,disp=1,retall=0,callback=None,**kwds):
    """Minimize a function using differential evolution.

Description:

    Uses a differential evolution algorith to find the minimum of
    a function of one or more variables. Mimics a scipy.optimize style
    interface.

Inputs:

    func -- the Python function or method to be minimized.
    x0 -- the initial guess (ndarray), if desired to start from a
        set point; otherwise takes an array of (min,max) bounds,
        for when random initial points are desired
    npop -- size of the trial solution population.

Additional Inputs:

    args -- extra arguments for func.
    bounds -- list - n pairs of bounds (min,max), one pair for each parameter.
    ftol -- number - acceptable relative error in func(xopt) for convergence.
    gtol -- number - maximum number of iterations to run without improvement.
    maxiter -- number - the maximum number of iterations to perform.
    maxfun -- number - the maximum number of function evaluations.
    cross -- number - the probability of cross-parameter mutations
    scale -- number - multiplier for impact of mutations on trial solution.
    full_output -- number - non-zero if fval and warnflag outputs are desired.
    disp -- number - non-zero to print convergence messages.
    retall -- number - non-zero to return list of solutions at each iteration.
    callback -- an optional user-supplied function to call after each
        iteration.  It is called as callback(xk), where xk is the
        current parameter vector.

Returns: (xopt, {fopt, iter, funcalls, warnflag}, {allvecs})

    xopt -- ndarray - minimizer of function
    fopt -- number - value of function at minimum: fopt = func(xopt)
    iter -- number - number of iterations
    funcalls -- number - number of function calls
    warnflag -- number - Integer warning flag:
        1 : 'Maximum number of function evaluations.'
        2 : 'Maximum number of iterations.'
    allvecs -- list - a list of solutions at each iteration

    """
    invariant_current = False
    if kwds.has_key('invariant_current'):
        invariant_current = kwds['invariant_current']

    from mystic.tools import Sow
    stepmon = Sow()
    evalmon = Sow()
   #from mystic.strategy import Best1Exp #, Best1Bin, Rand1Exp
   #strategy = Best1Exp
    if gtol: #if number of generations provided, use ChangeOverGeneration 
        from mystic.termination import ChangeOverGeneration
        termination = ChangeOverGeneration(ftol,gtol)
    else:
        from mystic.termination import VTR
        termination = VTR(ftol)

    ND = len(x0)
    if invariant_current: #use Solver2, not Solver1
        solver = DifferentialEvolutionSolver2(ND,npop)
    else:
        solver = DifferentialEvolutionSolver(ND,npop)
    solver.SetEvaluationLimits(maxiter,maxfun)
    if bounds:
        minb,maxb = unpair(bounds)
        solver.SetStrictRanges(minb,maxb)

    try: #x0 passed as 1D array of (min,max) pairs
        minb,maxb = unpair(x0)
        solver.SetRandomInitialPoints(minb,maxb)
    except: #x0 passed as 1D array of initial parameter values
        solver.SetInitialPoints(x0)

    solver.enable_signal_handler()
    #TODO: allow sigint_callbacks for all minimal interfaces ?
    solver.Solve(func,termination=termination,\
                #strategy=strategy,sigint_callback=other_callback,\
                 CrossProbability=cross,ScalingFactor=scale,\
                 EvaluationMonitor=evalmon,StepMonitor=stepmon,\
                 ExtraArgs=args,callback=callback)
    solution = solver.Solution()

    # code below here pushes output to scipy.optimize.fmin interface
   #x = list(solver.bestSolution)
    x = solver.bestSolution
    fval = solver.bestEnergy
    warnflag = 0
    fcalls = len(evalmon.x)
    iterations = len(stepmon.x)
    allvecs = stepmon.x

    if fcalls >= solver._maxfun:
        warnflag = 1
        if disp:
            print "Warning: Maximum number of function evaluations has "\
                  "been exceeded."
    elif iterations >= solver._maxiter:
        warnflag = 2
        if disp:
            print "Warning: Maximum number of iterations has been exceeded"
    else:
        if disp:
            print "Optimization terminated successfully."
            print "         Current function value: %f" % fval
            print "         Iterations: %d" % iterations
            print "         Function evaluations: %d" % fcalls

    if full_output:
        retlist = x, fval, iterations, fcalls, warnflag
        if retall:
            retlist += (allvecs,)
    else:
        retlist = x
        if retall:
            retlist = (x, allvecs)

    return retlist


if __name__=='__main__':
    help(__name__)

# end of file
