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
## bounds (and minimal interface) added by Mike McKerns
## adapted to AbstractSolver interface by Mike McKerns
##
## modified for AbstractMapSolver interface by Mike McKerns
#
# Author: Patrick Hung (patrickh @caltech)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2006-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

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
    - strategy = Best1Bin
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


References:
    1. Storn, R. and Price, K. Differential Evolution - A Simple and Efficient
       Heuristic for Global Optimization over Continuous Spaces. Journal of
       Global Optimization 11: 341-359, 1997.
    2. Price, K., Storn, R., and Lampinen, J. - Differential Evolution, A
       Practical Approach to Global Optimization. Springer, 1st Edition, 2005.
"""
__all__ = ['DifferentialEvolutionSolver','DifferentialEvolutionSolver2',\
           'diffev','diffev2']

from mystic.tools import wrap_function, unpair, isiterable
from mystic.tools import wrap_bounds, wrap_penalty, reduced

from mystic.abstract_solver import AbstractSolver
from mystic.abstract_map_solver import AbstractMapSolver

from numpy import asfarray, ravel
import collections
_Callable = getattr(collections, 'Callable', None) or getattr(collections.abc, 'Callable')

class DifferentialEvolutionSolver(AbstractSolver):
    """
Differential Evolution optimization.
    """
    
    def __init__(self, dim, NP=4):
        """
Takes two initial inputs: 
    dim  -- dimensionality of the problem
    NP   -- size of the trial solution population. [requires: NP >= 4]

All important class members are inherited from AbstractSolver.
        """
        NP = max(NP, dim, 4) #XXX: raise Error if npop <= 4?
        AbstractSolver.__init__(self,dim,npop=NP)
        self.genealogy     = [ [] for j in range(NP)]
        self.scale         = 0.8
        self.probability   = 0.9
        self.strategy      = 'Best1Bin'
        ftol = 5e-3
        from mystic.termination import VTRChangeOverGeneration
        self._termination = VTRChangeOverGeneration(ftol)
        
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
        """create an in-memory log of the genealogy of the population

Input::
    - id: (int) the index of the candidate in the population matrix
    - newchild: (list[float]) a new trialSolution
        """
        self.genealogy[id].append(newchild)
        return

    def SetConstraints(self, constraints):
        """apply a constraints function to the optimization

input::
    - a constraints function of the form: xk' = constraints(xk),
      where xk is the current parameter vector. Ideally, this function
      is constructed so the parameter vector it passes to the cost function
      will satisfy the desired (i.e. encoded) constraints."""
        if not constraints:
            self._constraints = lambda x: x
        elif not isinstance(constraints, _Callable):
            raise TypeError("'%s' is not a callable function" % constraints)
        else: #XXX: check for format: x' = constraints(x) ?
            self._constraints = constraints
        return # doesn't use wrap_nested

    def _decorate_objective(self, cost, ExtraArgs=None):
        """decorate the cost function with bounds, penalties, monitors, etc

input::
    - cost is the objective function, of the form y = cost(x, *ExtraArgs),
      where x is a candidate solution, and ExtraArgs is the tuple of positional
      arguments required to evaluate the objective."""
        #print("@%r %r %r" % (cost, ExtraArgs, max))
        evalmon = self._evalmon
        raw = cost
        if ExtraArgs is None: ExtraArgs = ()
        self._fcalls, cost = wrap_function(cost, ExtraArgs, evalmon)
        if self._useStrictRange:
            indx = list(self.popEnergy).index(self.bestEnergy)
            ngen = self.generations #XXX: no random if generations=0 ?
            for i in range(self.nPop):
                self.population[i] = self._clipGuessWithinRangeBoundary(self.population[i], (not ngen) or (i is indx))
            cost = wrap_bounds(cost, self._strictMin, self._strictMax) #XXX: remove?
        cost = wrap_penalty(cost, self._penalty)
        if self._reducer:
           #cost = reduced(*self._reducer)(cost) # was self._reducer = (f,bool)
            cost = reduced(self._reducer, arraylike=True)(cost)
        # hold on to the 'wrapped' and 'raw' cost function
        self._cost = (cost, raw, ExtraArgs)
        self._live = True
        return cost

    def _Step(self, cost=None, ExtraArgs=None, **kwds):
        """perform a single optimization iteration

input::
    - cost is the objective function, of the form y = cost(x, *ExtraArgs),
      where x is a candidate solution, and ExtraArgs is the tuple of positional
      arguments required to evaluate the objective.

note::
    ExtraArgs needs to be a *tuple* of extra arguments.

    This method accepts additional args that are specific for the current
    solver, as detailed in the `_process_inputs` method.
        """
        # process and activate input settings
        settings = self._process_inputs(kwds)
        #(hardwired: due to python3.x exec'ing to locals())
        callback = settings['callback'] if 'callback' in settings else None
        disp = settings['disp'] if 'disp' in settings else False
        strategy = settings['strategy'] if 'strategy' in settings else self.strategy

        # HACK to enable not explicitly calling _decorate_objective
        cost = self._bootstrap_objective(cost, ExtraArgs)

        init = False  # flag to do 0th iteration 'post-initialization'

        if not len(self._stepmon): # do generation = 0
            init = True
            strategy = None
            self.population[0] = asfarray(self.population[0])
            # decouple bestSolution from population and bestEnergy from popEnergy
            bs = self.population[0]
            self.bestSolution = bs.copy() if hasattr(bs, 'copy') else bs[:]
            self.bestEnergy = self.popEnergy[0]
            del bs

        if self._useStrictRange:
            from mystic.constraints import and_
            constraints = and_(self._constraints, self._strictbounds, onfail=self._strictbounds)
        else: constraints = self._constraints

        for candidate in range(self.nPop):
            if not len(self._stepmon):
                # generate trialSolution (within valid range)
                self.trialSolution[:] = self.population[candidate]
            if strategy:
                # generate trialSolution (within valid range)
                strategy(self, candidate)
            # apply constraints
            self.trialSolution[:] = constraints(self.trialSolution)
            # apply penalty
           #trialEnergy = self._penalty(self.trialSolution)
            # calculate cost
            trialEnergy = cost(self.trialSolution)

            # trialEnergy should be a scalar
            if isiterable(trialEnergy) and len(trialEnergy) == 1:
                trialEnergy = trialEnergy[0]
                # for len(trialEnergy) > 1, will throw ValueError below

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
        # if savefrequency matches, then save state
        self._AbstractSolver__save_state()

        # do callback
        if callback is not None: callback(self.bestSolution)
        # initialize termination conditions, if needed
        if init: self._termination(self) #XXX: at generation 0 or always?
        return #XXX: call Terminated ?

    def _process_inputs(self, kwds):
        """process and activate input settings

Args:
    callback (func, default=None): function to call after each iteration. The
        interface is ``callback(xk)``, with xk the current parameter vector.
    disp (bool, default=False): if True, print convergence messages.

Additional Args:
    EvaluationMonitor: a monitor instance to capture each evaluation of cost.
    StepMonitor: a monitor instance to capture each iteration's best results.
    penalty: a function of the form: y' = penalty(xk), with y = cost(xk) + y',
        where xk is the current parameter vector.
    constraints: a function of the form: xk' = constraints(xk), where xk is
        the current parameter vector.
    strategy (strategy, default=Best1Bin): the mutation strategy for generating        new trial solutions.
    CrossProbability (float, default=0.9): the probability of cross-parameter
        mutations.
    ScalingFactor (float, default=0.8): multiplier for mutations on the trial
        solution.

Note:
   The additional args are 'sticky', in that once they are given, they remain
   set until they are explicitly changed. Conversely, the args are not sticky,
   and are thus set for a one-time use.
        """
        #allow for inputs that don't conform to AbstractSolver interface
        #NOTE: not sticky: callback, disp
        #NOTE: sticky: EvaluationMonitor, StepMonitor, penalty, constraints
        #NOTE: sticky: strategy, CrossProbability, ScalingFactor
        settings = super(DifferentialEvolutionSolver, self)._process_inputs(kwds)
        from mystic import strategy
        strategy = getattr(strategy,self.strategy,strategy.Best1Bin) #XXX: None?
        settings.update({\
        'strategy': strategy})       #mutation strategy (see mystic.strategy)
        probability=self.probability #potential for parameter cross-mutation
        scale=self.scale             #multiplier for mutation impact
        [settings.update({i:j}) for (i,j) in getattr(kwds, 'iteritems', kwds.items)() if i in settings]
        word = 'CrossProbability'
        self.probability = kwds[word] if word in kwds else probability
        word = 'ScalingFactor'
        self.scale = kwds[word] if word in kwds else scale
        self.strategy = getattr(settings['strategy'],'__name__','Best1Bin')
        return settings

    def Solve(self, cost=None, termination=None, ExtraArgs=None, **kwds):
        """Minimize a function using differential evolution.

Uses a differential evolution algorithm to find the minimum of a function of
one or more variables.

Args:
    cost (func, default=None): the function to be minimized: ``y = cost(x)``.
    termination (termination, default=None): termination conditions.
    ExtraArgs (tuple, default=None): extra arguments for cost.
    strategy (strategy, default=Best1Bin): the mutation strategy for generating        new trial solutions.
    CrossProbability (float, default=0.9): the probability of cross-parameter
        mutations.
    ScalingFactor (float, default=0.8): multiplier for mutations on the trial
        solution.
    sigint_callback (func, default=None): callback function for signal handler.
    callback (func, default=None): function to call after each iteration. The
        interface is ``callback(xk)``, with xk the current parameter vector.
    disp (bool, default=False): if True, print convergence messages.

Returns:
    None
        """
        super(DifferentialEvolutionSolver, self).Solve(cost, termination,\
                                                       ExtraArgs, **kwds)
        return



class DifferentialEvolutionSolver2(AbstractMapSolver):
    """
Differential Evolution optimization, using Storn and Price's algorithm.

Alternate implementation: 
    - utilizes a map-reduce interface, extensible to parallel computing
    - both a current and a next generation are kept, while the current
      generation is invariant during the main DE logic
    """
    def __init__(self, dim, NP=4):
        """
Takes two initial inputs: 
    dim  -- dimensionality of the problem
    NP   -- size of the trial solution population. [requires: NP >= 4]

All important class members are inherited from AbstractSolver.
        """
        NP = max(NP, dim, 4) #XXX: raise Error if npop <= 4?
        super(DifferentialEvolutionSolver2, self).__init__(dim, npop=NP)
        self.genealogy     = [ [] for j in range(NP)]
        self.scale         = 0.8
        self.probability   = 0.9
        self.strategy      = 'Best1Bin'
        ftol = 5e-3
        from mystic.termination import VTRChangeOverGeneration
        self._termination = VTRChangeOverGeneration(ftol)
        
    def UpdateGenealogyRecords(self, id, newchild):
        """create an in-memory log of the genealogy of the population

Input::
    - id: (int) the index of the candidate in the population matrix
    - newchild: (list[float]) a new trialSolution
        """
        self.genealogy[id].append(newchild)
        return

    def SetConstraints(self, constraints):
        """apply a constraints function to the optimization

input::
    - a constraints function of the form: xk' = constraints(xk),
      where xk is the current parameter vector. Ideally, this function
      is constructed so the parameter vector it passes to the cost function
      will satisfy the desired (i.e. encoded) constraints."""
        if not constraints:
            self._constraints = lambda x: x
        elif not isinstance(constraints, _Callable):
            raise TypeError("'%s' is not a callable function" % constraints)
        else: #XXX: check for format: x' = constraints(x) ?
            self._constraints = constraints
        return # doesn't use wrap_nested

    def _decorate_objective(self, cost, ExtraArgs=None):
        """decorate the cost function with bounds, penalties, monitors, etc

input::
    - cost is the objective function, of the form y = cost(x, *ExtraArgs),
      where x is a candidate solution, and ExtraArgs is the tuple of positional
      arguments required to evaluate the objective."""
        #print("@%r %r %r" % (cost, ExtraArgs, max))
        raw = cost
        if ExtraArgs is None: ExtraArgs = ()
        from mystic.python_map import python_map
        if self._map != python_map:
            #FIXME: EvaluationMonitor fails for MPI, throws error for 'pp'
            from mystic.monitors import Null
            evalmon = Null()
        else: evalmon = self._evalmon
        fcalls, cost = wrap_function(cost, ExtraArgs, evalmon)
        if self._useStrictRange:
            indx = list(self.popEnergy).index(self.bestEnergy)
            ngen = self.generations #XXX: no random if generations=0 ?
            for i in range(self.nPop):
                self.population[i] = self._clipGuessWithinRangeBoundary(self.population[i], (not ngen) or (i is indx))
            cost = wrap_bounds(cost, self._strictMin, self._strictMax) #XXX: remove?
        cost = wrap_penalty(cost, self._penalty)
        if self._reducer:
           #cost = reduced(*self._reducer)(cost) # was self._reducer = (f,bool)
            cost = reduced(self._reducer, arraylike=True)(cost)
        # hold on to the 'wrapped' and 'raw' cost function
        self._cost = (cost, raw, ExtraArgs)
        self._live = True
        return cost

    def _Step(self, cost=None, ExtraArgs=None, **kwds):
        """perform a single optimization iteration

input::
    - cost is the objective function, of the form y = cost(x, *ExtraArgs),
      where x is a candidate solution, and ExtraArgs is the tuple of positional
      arguments required to evaluate the objective.

note::
    ExtraArgs needs to be a *tuple* of extra arguments.

    This method accepts additional args that are specific for the current
    solver, as detailed in the `_process_inputs` method.
        """
        # process and activate input settings
        settings = self._process_inputs(kwds)
        #(hardwired: due to python3.x exec'ing to locals())
        callback = settings['callback'] if 'callback' in settings else None
        disp = settings['disp'] if 'disp' in settings else False
        strategy = settings['strategy'] if 'strategy' in settings else self.strategy

        # HACK to enable not explicitly calling _decorate_objective
        cost = self._bootstrap_objective(cost, ExtraArgs)

        init = False  # flag to do 0th iteration 'post-initialization'

        if not len(self._stepmon): # do generation = 0
            init = True
            strategy = None
            self.population[0] = asfarray(self.population[0])
            # decouple bestSolution from population and bestEnergy from popEnergy
            bs = self.population[0]
            self.bestSolution = bs.copy() if hasattr(bs, 'copy') else bs[:]
            self.bestEnergy = self.popEnergy[0]
            del bs

        if self._useStrictRange:
            from mystic.constraints import and_
            constraints = and_(self._constraints, self._strictbounds, onfail=self._strictbounds)
        else: constraints = self._constraints

        for candidate in range(self.nPop):
            if not len(self._stepmon):
                # generate trialSolution (within valid range)
                self.trialSolution[candidate][:] = self.population[candidate]
            if strategy:
                # generate trialSolution (within valid range)
                strategy(self, candidate)
            # apply constraints
            self.trialSolution[candidate][:] = constraints(self.trialSolution[candidate])
        # bind constraints to cost #XXX: apparently imposes constraints poorly
       #concost = wrap_nested(cost, constraints)

        # apply penalty
       #trialEnergy = map(self._penalty, self.trialSolution)#,**self._mapconfig)
        # calculate cost
        trialEnergy = self._map(cost, self.trialSolution, **self._mapconfig)
        self._fcalls[0] += len(self.trialSolution) #FIXME: manually increment

        # each trialEnergy should be a scalar
        if isiterable(trialEnergy[0]) and len(trialEnergy[0]) == 1:
            trialEnergy = ravel(trialEnergy)
            # for len(trialEnergy) > 1, will throw ValueError below

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
        # if savefrequency matches, then save state
        self._AbstractSolver__save_state()

        # do callback
        if callback is not None: callback(self.bestSolution)
        # initialize termination conditions, if needed
        if init: self._termination(self) #XXX: at generation 0 or always?
        return #XXX: call Terminated ?

    def _process_inputs(self, kwds):
        """process and activate input settings

Args:
    callback (func, default=None): function to call after each iteration. The
        interface is ``callback(xk)``, with xk the current parameter vector.
    disp (bool, default=False): if True, print convergence messages.

Additional Args:
    EvaluationMonitor: a monitor instance to capture each evaluation of cost.
    StepMonitor: a monitor instance to capture each iteration's best results.
    penalty: a function of the form: y' = penalty(xk), with y = cost(xk) + y',
        where xk is the current parameter vector.
    constraints: a function of the form: xk' = constraints(xk), where xk is
        the current parameter vector.
    strategy (strategy, default=Best1Bin): the mutation strategy for generating        new trial solutions.
    CrossProbability (float, default=0.9): the probability of cross-parameter
        mutations.
    ScalingFactor (float, default=0.8): multiplier for mutations on the trial
        solution.

Note:
   The additional args are 'sticky', in that once they are given, they remain
   set until they are explicitly changed. Conversely, the args are not sticky,
   and are thus set for a one-time use.
        """
        #allow for inputs that don't conform to AbstractSolver interface
        #NOTE: not sticky: callback, disp
        #NOTE: sticky: EvaluationMonitor, StepMonitor, penalty, constraints
        #NOTE: sticky: strategy, CrossProbability, ScalingFactor
        settings = super(DifferentialEvolutionSolver2, self)._process_inputs(kwds)
        from mystic import strategy
        strategy = getattr(strategy,self.strategy,strategy.Best1Bin) #XXX: None?
        settings.update({\
        'strategy': strategy})       #mutation strategy (see mystic.strategy)
        probability=self.probability #potential for parameter cross-mutation
        scale=self.scale             #multiplier for mutation impact
        [settings.update({i:j}) for (i,j) in getattr(kwds, 'iteritems', kwds.items)() if i in settings]
        word = 'CrossProbability'
        self.probability = kwds[word] if word in kwds else probability
        word = 'ScalingFactor'
        self.scale = kwds[word] if word in kwds else scale
        self.strategy = getattr(settings['strategy'],'__name__','Best1Bin')
        return settings

    def Solve(self, cost=None, termination=None, ExtraArgs=None, **kwds):
        """Minimize a function using differential evolution.

Uses a differential evolution algorithm to find the minimum of a function of
one or more variables. This implementation holds the current generation
invariant until the end of each iteration.

Args:
    cost (func, default=None): the function to be minimized: ``y = cost(x)``.
    termination (termination, default=None): termination conditions.
    ExtraArgs (tuple, default=None): extra arguments for cost.
    strategy (strategy, default=Best1Bin): the mutation strategy for generating        new trial solutions.
    CrossProbability (float, default=0.9): the probability of cross-parameter
        mutations.
    ScalingFactor (float, default=0.8): multiplier for mutations on the trial
        solution.
    sigint_callback (func, default=None): callback function for signal handler.
    callback (func, default=None): function to call after each iteration. The
        interface is ``callback(xk)``, with xk the current parameter vector.
    disp (bool, default=False): if True, print convergence messages.

Returns:
    None
        """
        super(DifferentialEvolutionSolver2, self).Solve(cost, termination,\
                                                        ExtraArgs, **kwds)
        return 


def diffev2(cost,x0,npop=4,args=(),bounds=None,ftol=5e-3,gtol=None,
            maxiter=None,maxfun=None,cross=0.9,scale=0.8,
            full_output=0,disp=1,retall=0,callback=None,**kwds):
    """Minimize a function using Storn & Price's differential evolution.

Uses Storn & Prices's differential evolution algorithm to find the minimum of a
function of one or more variables. Mimics a ``scipy.optimize`` style interface.

Args:
    cost (func): the function or method to be minimized: ``y = cost(x)``.
    x0 (ndarray): the initial guess parameter vector ``x`` if desired start
        is a single point, otherwise takes a list of (min,max) bounds that
        define a region from which random initial points are drawn.
    npop (int, default=4): size of the trial solution population.
    args (tuple, default=()): extra arguments for cost.
    bounds (list(tuple), default=None): list of pairs of bounds (min,max),
        one for each parameter.
    ftol (float, default=5e-3): acceptable relative error in ``cost(xopt)``
        for convergence.
    gtol (float, default=None): maximum iterations to run without improvement.
    maxiter (int, default=None): the maximum number of iterations to perform.
    maxfun (int, default=None): the maximum number of function evaluations.
    cross (float, default=0.9): the probability of cross-parameter mutations.
    scale (float, default=0.8): multiplier for mutations on the trial solution.
    full_output (bool, default=False): True if fval and warnflag are desired.
    disp (bool, default=True): if True, print convergence messages.
    retall (bool, default=False): True if allvecs is desired.
    callback (func, default=None): function to call after each iteration. The
        interface is ``callback(xk)``, with xk the current parameter vector.
    handler (bool, default=False): if True, enable handling interrupt signals.
    id (int, default=None): the ``id`` of the solver used in logging.
    strategy (strategy, default=None): override the default mutation strategy.
    itermon (monitor, default=None): override the default GenerationMonitor.
    evalmon (monitor, default=None): override the default EvaluationMonitor.
    constraints (func, default=None): a function ``xk' = constraints(xk)``,
        where xk is the current parameter vector, and xk' is a parameter
        vector that satisfies the encoded constraints.
    penalty (func, default=None): a function ``y = penalty(xk)``, where xk is
        the current parameter vector, and ``y' == 0`` when the encoded
        constraints are satisfied (and ``y' > 0`` otherwise).
    tightrange (bool, default=None): impose bounds and constraints concurrently.
    cliprange (bool, default=None): bounding constraints clip exterior values.
    map (func, default=None): a (parallel) map function ``y = map(f, x)``.

Returns:
    ``(xopt, {fopt, iter, funcalls, warnflag}, {allvecs})``

Notes:
    - xopt (*ndarray*): the minimizer of the cost function
    - fopt (*float*): value of cost function at minimum: ``fopt = cost(xopt)``
    - iter (*int*): number of iterations
    - funcalls (*int*): number of function calls
    - warnflag (*int*): warning flag:
        - ``1 : Maximum number of function evaluations``
        - ``2 : Maximum number of iterations``
    - allvecs (*list*): a list of solutions at each iteration
    """
    invariant_current = kwds['invariant_current'] if 'invariant_current' in kwds else True
    kwds['invariant_current'] = invariant_current
    return diffev(cost,x0,npop,args=args,bounds=bounds,ftol=ftol,gtol=gtol,
                  maxiter=maxiter,maxfun=maxfun,cross=cross,scale=scale,
                  full_output=full_output,disp=disp,retall=retall,
                  callback=callback,**kwds)


def diffev(cost,x0,npop=4,args=(),bounds=None,ftol=5e-3,gtol=None,
           maxiter=None,maxfun=None,cross=0.9,scale=0.8,
           full_output=0,disp=1,retall=0,callback=None,**kwds):
    """Minimize a function using differential evolution.

Uses a differential evolution algorithm to find the minimum of a function of
one or more variables. Mimics a ``scipy.optimize`` style interface.

Args:
    cost (func): the function or method to be minimized: ``y = cost(x)``.
    x0 (ndarray): the initial guess parameter vector ``x`` if desired start
        is a single point, otherwise takes a list of (min,max) bounds that
        define a region from which random initial points are drawn.
    npop (int, default=4): size of the trial solution population.
    args (tuple, default=()): extra arguments for cost.
    bounds (list(tuple), default=None): list of pairs of bounds (min,max),
        one for each parameter.
    ftol (float, default=5e-3): acceptable relative error in ``cost(xopt)``
        for convergence.
    gtol (float, default=None): maximum iterations to run without improvement.
    maxiter (int, default=None): the maximum number of iterations to perform.
    maxfun (int, default=None): the maximum number of function evaluations.
    cross (float, default=0.9): the probability of cross-parameter mutations.
    scale (float, default=0.8): multiplier for mutations on the trial solution.
    full_output (bool, default=False): True if fval and warnflag are desired.
    disp (bool, default=True): if True, print convergence messages.
    retall (bool, default=False): True if allvecs is desired.
    callback (func, default=None): function to call after each iteration. The
        interface is ``callback(xk)``, with xk the current parameter vector.
    handler (bool, default=False): if True, enable handling interrupt signals.
    id (int, default=None): the ``id`` of the solver used in logging.
    strategy (strategy, default=None): override the default mutation strategy.
    itermon (monitor, default=None): override the default GenerationMonitor.
    evalmon (monitor, default=None): override the default EvaluationMonitor.
    constraints (func, default=None): a function ``xk' = constraints(xk)``,
        where xk is the current parameter vector, and xk' is a parameter
        vector that satisfies the encoded constraints.
    penalty (func, default=None): a function ``y = penalty(xk)``, where xk is
        the current parameter vector, and ``y' == 0`` when the encoded
        constraints are satisfied (and ``y' > 0`` otherwise).
    tightrange (bool, default=None): impose bounds and constraints concurrently.
    cliprange (bool, default=None): bounding constraints clip exterior values.
    map (func, default=None): a (parallel) map function ``y = map(f, x)``.

Returns:
    ``(xopt, {fopt, iter, funcalls, warnflag}, {allvecs})``

Notes:
    - xopt (*ndarray*): the minimizer of the cost function
    - fopt (*float*): value of cost function at minimum: ``fopt = cost(xopt)``
    - iter (*int*): number of iterations
    - funcalls (*int*): number of function calls
    - warnflag (*int*): warning flag:
        - ``1 : Maximum number of function evaluations``
        - ``2 : Maximum number of iterations``
    - allvecs (*list*): a list of solutions at each iteration
    """
    invariant_current = kwds['invariant_current'] if 'invariant_current' in kwds else False
    handler = kwds['handler'] if 'handler' in kwds else False

    from mystic.strategy import Best1Bin
    strategy = kwds['strategy'] if 'strategy' in kwds else Best1Bin
    from mystic.monitors import Monitor
    stepmon = kwds['itermon'] if 'itermon' in kwds else Monitor()
    evalmon = kwds['evalmon'] if 'evalmon' in kwds else Monitor()

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
    if 'id' in kwds:
        solver.id = int(kwds['id'])
    if 'penalty' in kwds:
        solver.SetPenalty(kwds['penalty'])
    if 'constraints' in kwds:
        solver.SetConstraints(kwds['constraints'])
    if bounds is not None:
        minb,maxb = unpair(bounds)
        tight = kwds['tightrange'] if 'tightrange' in kwds else None
        clip = kwds['cliprange'] if 'cliprange' in kwds else None
        solver.SetStrictRanges(minb,maxb,tight=tight,clip=clip)

    try: #x0 passed as 1D array of (min,max) pairs
        minb,maxb = unpair(x0)
        solver.SetRandomInitialPoints(minb,maxb)
    except: #x0 passed as 1D array of initial parameter values
        solver.SetInitialPoints(x0)

    _map = kwds['map'] if 'map' in kwds else None
    if _map: solver.SetMapper(_map)

    if handler: solver.enable_signal_handler()
    #TODO: allow sigint_callbacks for all minimal interfaces ?
    solver.Solve(cost, termination=termination, strategy=strategy, \
                #sigint_callback=other_callback,\
                 CrossProbability=cross, ScalingFactor=scale, \
                 ExtraArgs=args, callback=callback)
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
            print("Warning: Maximum number of function evaluations has "\
                  "been exceeded.")
    elif iterations >= solver._maxiter:
        warnflag = 2
        if disp:
            print("Warning: Maximum number of iterations has been exceeded")
    else:
        if disp:
            print("Optimization terminated successfully.")
            print("         Current function value: %f" % fval)
            print("         Iterations: %d" % iterations)
            print("         Function evaluations: %d" % fcalls)

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
