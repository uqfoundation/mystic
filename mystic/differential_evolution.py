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
    - EvaluationMonitor = Monitor()
    - StepMonitor = Monitor()
    - strategy = Best1Exp
    - termination = ChangeOverGeneration(ftol,gtol), if gtol provided
          ''      = VTRChangeOverGenerations(ftol), otherwise

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

from mystic.tools import wrap_function, unpair
from mystic.tools import wrap_bounds, wrap_penalty

from abstract_solver import AbstractSolver
from abstract_map_solver import AbstractMapSolver

from numpy import asfarray

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

    def _Decorate(self, cost, ExtraArgs=None):
        """decorate cost function with bounds, penalties, monitors, etc"""
        self._fcalls, cost = wrap_function(cost, ExtraArgs, self._evalmon)
        if self._useStrictRange:
            for i in range(self.nPop):
                self.population[i] = self._clipGuessWithinRangeBoundary(self.population[i])
            cost = wrap_bounds(cost, self._strictMin, self._strictMax)
        cost = wrap_penalty(cost, self._penalty)
        return cost

    def Step(self, cost, strategy=None, **kwds):
        """perform a single optimization iteration"""
        if not len(self._stepmon): # do generation = 0
            self.population[0] = asfarray(self.population[0])
            # decouple bestSolution from population and bestEnergy from popEnergy
            self.bestSolution = self.population[0]
            self.bestEnergy = self.popEnergy[0]

        for candidate in range(self.nPop):
            if not len(self._stepmon):
                # generate trialSolution (within valid range)
                self.trialSolution[:] = self.population[candidate]
            if strategy:
                # generate trialSolution (within valid range)
                strategy(self, candidate)
            # apply constraints
            self.trialSolution[:] = self._constraints(self.trialSolution)
            # apply penalty
           #trialEnergy = self._penalty(self.trialSolution)
            # calculate cost
            trialEnergy = cost(self.trialSolution)

            if trialEnergy < self.popEnergy[candidate]:
                # New low for this candidate
                self.popEnergy[candidate] = trialEnergy
                self.population[candidate][:] = self.trialSolution
                self.UpdateGenealogyRecords(candidate, self.trialSolution[:])

                # Check if all-time low
                if trialEnergy < self.bestEnergy:
                    self.bestEnergy = trialEnergy
                    self.bestSolution[:] = self.trialSolution

        # log bestSolution and bestEnergy (includes penalty)
        self._stepmon(self.bestSolution[:], self.bestEnergy, self.id)
        return

    def _process_inputs(self, kwds):
        """process and activate input settings"""
        #allow for inputs that don't conform to AbstractSolver interface
        settings = super(DifferentialEvolutionSolver, self)._process_inputs(kwds)
        from mystic.strategy import Best1Bin
        settings.update({\
        'strategy':Best1Bin})#mutation strategy (see mystic.strategy)
        probability=0.9      #potential for parameter cross-mutation
        scale=0.8            #multiplier for mutation impact
        [settings.update({i:j}) for (i,j) in kwds.items() if i in settings]
        self.probability = kwds.get('CrossProbability', probability)
        self.scale = kwds.get('ScalingFactor', scale)
        return settings

    def Solve(self, cost, termination, sigint_callback=None,
                                               ExtraArgs=(), **kwds):
        """Minimize a function using differential evolution.

Description:

    Uses a differential evolution algorith to find the minimum of
    a function of one or more variables.

Inputs:

    cost -- the Python function or method to be minimized.
    termination -- callable object providing termination conditions.

Additional Inputs:

    sigint_callback -- callback function for signal handler.
    ExtraArgs -- extra arguments for cost.

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
        super(DifferentialEvolutionSolver, self).Solve(cost, termination,\
                                      sigint_callback, ExtraArgs, **kwds)
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

    def _Decorate(self, cost, ExtraArgs=None):
        """decorate cost function with bounds, penalties, monitors, etc"""
       #FIXME: EvaluationMonitor fails for MPI, throws error for 'pp'
        from python_map import python_map
        if self._map != python_map:
            self._fcalls = [0] #FIXME: temporary patch for removing the following line
        else:
            self._fcalls, cost = wrap_function(cost, ExtraArgs, self._evalmon)
        if self._useStrictRange:
            for i in range(self.nPop):
                self.population[i] = self._clipGuessWithinRangeBoundary(self.population[i])
            cost = wrap_bounds(cost, self._strictMin, self._strictMax)
        cost = wrap_penalty(cost, self._penalty)
        return cost

    def Step(self, cost, strategy=None, **kwds):
        """perform a single optimization iteration"""
        if not len(self._stepmon): # do generation = 0
            self.population[0] = asfarray(self.population[0])
            # decouple bestSolution from population and bestEnergy from popEnergy
            self.bestSolution = self.population[0]
            self.bestEnergy = self.popEnergy[0]

        for candidate in range(self.nPop):
            if not len(self._stepmon):
                # generate trialSolution (within valid range)
                self.trialSolution[candidate][:] = self.population[candidate]
            if strategy:
                # generate trialSolution (within valid range)
                strategy(self, candidate)
            # apply constraints
            self.trialSolution[candidate][:] = self._constraints(self.trialSolution[candidate])

        mapconfig = dict(nnodes=self._nnodes, launcher=self._launcher, \
                         mapper=self._mapper, queue=self._queue, \
                         timelimit=self._timelimit, scheduler=self._scheduler, \
                         ncpus=self._ncpus, servers=self._servers)

        # apply penalty
       #trialEnergy = map(self._penalty, self.trialSolution)#, **mapconfig)
        # calculate cost
        trialEnergy = self._map(cost, self.trialSolution, **mapconfig)

        for candidate in range(self.nPop):
            if trialEnergy[candidate] < self.popEnergy[candidate]:
                # New low for this candidate
                self.popEnergy[candidate] = trialEnergy[candidate]
                self.population[candidate][:] = self.trialSolution[candidate]
                self.UpdateGenealogyRecords(candidate, self.trialSolution[candidate][:])

                # Check if all-time low
                if trialEnergy[candidate] < self.bestEnergy:
                    self.bestEnergy = trialEnergy[candidate]
                    self.bestSolution[:] = self.trialSolution[candidate]

        # log bestSolution and bestEnergy (includes penalty)
       #FIXME: StepMonitor works for 'pp'?
        self._stepmon(self.bestSolution[:], self.bestEnergy, self.id)
        return

    def _process_inputs(self, kwds):
        """process and activate input settings"""
        #allow for inputs that don't conform to AbstractSolver interface
        settings = super(DifferentialEvolutionSolver2, self)._process_inputs(kwds)
        from mystic.strategy import Best1Bin
        settings.update({\
        'strategy':Best1Bin})#mutation strategy (see mystic.strategy)
        probability=0.9      #potential for parameter cross-mutation
        scale=0.8            #multiplier for mutation impact
        [settings.update({i:j}) for (i,j) in kwds.items() if i in settings]
        self.probability = kwds.get('CrossProbability', probability)
        self.scale = kwds.get('ScalingFactor', scale)
        return settings

    def Solve(self, cost, termination, sigint_callback=None,
                                               ExtraArgs=(), **kwds):
        """Minimize a function using differential evolution.

Description:

    Uses a differential evolution algorith to find the minimum of
    a function of one or more variables. This implementation holds
    the current generation invariant until the end of each iteration.

Inputs:

    cost -- the Python function or method to be minimized.
    termination -- callable object providing termination conditions.

Additional Inputs:

    sigint_callback -- callback function for signal handler.
    ExtraArgs -- extra arguments for cost.

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
        super(DifferentialEvolutionSolver2, self).Solve(cost, termination,\
                                      sigint_callback, ExtraArgs, **kwds)
        return 


def diffev2(cost,x0,npop,args=(),bounds=None,ftol=5e-3,gtol=None,
            maxiter=None,maxfun=None,cross=0.9,scale=0.8,
            full_output=0,disp=1,retall=0,callback=None,**kwds):
    """Minimize a function using Storn & Price's differential evolution.

Description:

    Uses Storn & Prices's differential evolution algorith to find the minimum
    of a function of one or more variables. Mimics a scipy.optimize style
    interface.

Inputs:

    cost -- the Python function or method to be minimized.
    x0 -- the initial guess (ndarray), if desired to start from a
        set point; otherwise takes an array of (min,max) bounds,
        for when random initial points are desired
    npop -- size of the trial solution population.

Additional Inputs:

    args -- extra arguments for cost.
    bounds -- list - n pairs of bounds (min,max), one pair for each parameter.
    ftol -- number - acceptable relative error in cost(xopt) for convergence.
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
    handler -- boolean - enable/disable handling of interrupt signal
    strategy -- strategy - override the default mutation strategy
    itermon -- monitor - override the default GenerationMonitor
    evalmon -- monitor - override the default EvaluationMonitor
    constraints -- an optional user-supplied function.  It is called as
        constraints(xk), where xk is the current parameter vector.
        This function must return xk', a parameter vector that satisfies
        the encoded constraints.
    penalty -- an optional user-supplied function.  It is called as
        penalty(xk), where xk is the current parameter vector.
        This function should return y', with y' == 0 when the encoded
        constraints are satisfied, and y' > 0 otherwise.

Returns: (xopt, {fopt, iter, funcalls, warnflag}, {allvecs})

    xopt -- ndarray - minimizer of function
    fopt -- number - value of function at minimum: fopt = cost(xopt)
    iter -- number - number of iterations
    funcalls -- number - number of function calls
    warnflag -- number - Integer warning flag:
        1 : 'Maximum number of function evaluations.'
        2 : 'Maximum number of iterations.'
    allvecs -- list - a list of solutions at each iteration

    """
    invariant_current = True
    if kwds.has_key('invariant_current'):
        invariant_current = kwds['invariant_current']
    kwds['invariant_current'] = invariant_current
    return diffev(cost,x0,npop,args=args,bounds=bounds,ftol=ftol,gtol=gtol,
                  maxiter=maxiter,maxfun=maxfun,cross=cross,scale=scale,
                  full_output=full_output,disp=disp,retall=retall,
                  callback=callback,**kwds)


def diffev(cost,x0,npop,args=(),bounds=None,ftol=5e-3,gtol=None,
           maxiter=None,maxfun=None,cross=0.9,scale=0.8,
           full_output=0,disp=1,retall=0,callback=None,**kwds):
    """Minimize a function using differential evolution.

Description:

    Uses a differential evolution algorith to find the minimum of
    a function of one or more variables. Mimics a scipy.optimize style
    interface.

Inputs:

    cost -- the Python function or method to be minimized.
    x0 -- the initial guess (ndarray), if desired to start from a
        set point; otherwise takes an array of (min,max) bounds,
        for when random initial points are desired
    npop -- size of the trial solution population.

Additional Inputs:

    args -- extra arguments for cost.
    bounds -- list - n pairs of bounds (min,max), one pair for each parameter.
    ftol -- number - acceptable relative error in cost(xopt) for convergence.
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
    handler -- boolean - enable/disable handling of interrupt signal
    strategy -- strategy - override the default mutation strategy
    itermon -- monitor - override the default GenerationMonitor
    evalmon -- monitor - override the default EvaluationMonitor
    constraints -- an optional user-supplied function.  It is called as
        constraints(xk), where xk is the current parameter vector.
        This function must return xk', a parameter vector that satisfies
        the encoded constraints.
    penalty -- an optional user-supplied function.  It is called as
        penalty(xk), where xk is the current parameter vector.
        This function should return y', with y' == 0 when the encoded
        constraints are satisfied, and y' > 0 otherwise.

Returns: (xopt, {fopt, iter, funcalls, warnflag}, {allvecs})

    xopt -- ndarray - minimizer of function
    fopt -- number - value of function at minimum: fopt = cost(xopt)
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
    handler = False
    if kwds.has_key('handler'):
        handler = kwds['handler']

    from mystic.strategy import Best1Bin
    strategy = Best1Bin
    if kwds.has_key('strategy'):
        strategy = kwds['strategy']
    from mystic.monitors import Monitor
    stepmon = Monitor()
    evalmon = Monitor()
    if kwds.has_key('itermon'):
        stepmon = kwds['itermon']
    if kwds.has_key('evalmon'):
        evalmon = kwds['evalmon']
    if gtol: #if number of generations provided, use ChangeOverGeneration 
        from mystic.termination import ChangeOverGeneration
        termination = ChangeOverGeneration(ftol,gtol)
    else:
        from mystic.termination import VTRChangeOverGeneration
        termination = VTRChangeOverGeneration(ftol)

    ND = len(x0)
    if invariant_current: #use Solver2, not Solver1
        solver = DifferentialEvolutionSolver2(ND,npop)
    else:
        solver = DifferentialEvolutionSolver(ND,npop)
    solver.SetEvaluationLimits(maxiter,maxfun)
    solver.SetEvaluationMonitor(evalmon)
    solver.SetGenerationMonitor(stepmon)
    if kwds.has_key('penalty'):
        penalty = kwds['penalty']
        solver.SetPenalty(penalty)
    if kwds.has_key('constraints'):
        constraints = kwds['constraints']
        solver.SetConstraints(constraints)
    if bounds is not None:
        minb,maxb = unpair(bounds)
        solver.SetStrictRanges(minb,maxb)

    try: #x0 passed as 1D array of (min,max) pairs
        minb,maxb = unpair(x0)
        solver.SetRandomInitialPoints(minb,maxb)
    except: #x0 passed as 1D array of initial parameter values
        solver.SetInitialPoints(x0)

    if handler: solver.enable_signal_handler()
    #TODO: allow sigint_callbacks for all minimal interfaces ?
    solver.Solve(cost,termination=termination,strategy=strategy,\
                #sigint_callback=other_callback,\
                 CrossProbability=cross,ScalingFactor=scale,\
                 ExtraArgs=args,callback=callback)
    solution = solver.Solution()

    # code below here pushes output to scipy.optimize.fmin interface
   #x = list(solver.bestSolution)
    x = solver.bestSolution
    fval = solver.bestEnergy
    warnflag = 0
    fcalls = solver.evaluations
    iterations = solver.generations
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
