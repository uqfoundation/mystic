#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2017 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE
#
# Abstract Ensemble Solver Class
"""
This module contains the base class for launching several mystic solvers
instances -- utilizing a parallel "map" function to enable parallel
computing.  This module describes the ensemble solver interface.  As with
the AbstractSolver, the "Solve" method must be overwritten with the derived
solver's optimization algorithm. Similar to AbstractMapSolver, a call to
self.map is required.  In many cases, a minimal function call interface for a
derived solver is provided along with the derived class.  See the following
for an example.

The default map API settings are provided within mystic, while
distributed and high-performance computing mappers and launchers
can be obtained within the "pathos" package, found here::
    - http://dev.danse.us/trac/pathos


Usage
=====

A typical call to a 'ensemble' solver will roughly follow this example:

    >>> # the function to be minimized and the initial values
    >>> from mystic.models import rosen
    >>> lb = [0.0, 0.0, 0.0]
    >>> ub = [2.0, 2.0, 2.0]
    >>> 
    >>> # get monitors and termination condition objects
    >>> from mystic.monitors import Monitor
    >>> stepmon = Monitor()
    >>> from mystic.termination import CandidateRelativeTolerance as CRT
    >>> 
    >>> # select the parallel launch configuration
    >>> from pyina.launchers import Mpi as Pool
    >>> NNODES = 4
    >>> nbins = [4,4,4]
    >>>
    >>> # instantiate and configure the solver
    >>> from mystic.solvers import NelderMeadSimplexSolver
    >>> from mystic.solvers import LatticeSolver
    >>> solver = LatticeSolver(len(nbins), nbins)
    >>> solver.SetNestedSolver(NelderMeadSimplexSolver)
    >>> solver.SetStrictRanges(lb, ub)
    >>> solver.SetMapper(Pool(NNODES).map)
    >>> solver.SetGenerationMonitor(stepmon)
    >>> solver.SetTermination(CRT())
    >>> solver.Solve(rosen)
    >>> 
    >>> # obtain the solution
    >>> solution = solver.Solution()


Handler
=======

All solvers packaged with mystic include a signal handler that
provides the following options::
    sol: Print current best solution.
    cont: Continue calculation.
    call: Executes sigint_callback, if provided.
    exit: Exits with current best solution.

Handlers are enabled with the 'enable_signal_handler' method,
and are configured through the solver's 'Solve' method.  Handlers
trigger when a signal interrupt (usually, Ctrl-C) is given while
the solver is running.  ***NOTE: The handler currently is disabled
when the solver has been launched in parallel.*** 

"""
__all__ = ['AbstractEnsembleSolver']


from mystic.monitors import Null
from mystic.abstract_map_solver import AbstractMapSolver
from mystic.tools import wrap_function
from functools import reduce


class AbstractEnsembleSolver(AbstractMapSolver):
    """
AbstractEnsembleSolver base class for mystic optimizers that are called within
a parallel map.  This allows pseudo-global coverage of parameter space using
non-global optimizers.
    """

    def __init__(self, dim, **kwds):
        """
Takes one initial input:
    dim      -- dimensionality of the problem.

Additional inputs:
    npop     -- size of the trial solution population.      [default = 1]
    nbins    -- tuple of number of bins in each dimension.  [default = [1]*dim]
    npts     -- number of solver instances.                 [default = 1]

Important class members:
    nDim, nPop       = dim, npop
    generations      - an iteration counter.
    evaluations      - an evaluation counter.
    bestEnergy       - current best energy.
    bestSolution     - current best parameter set.           [size = dim]
    popEnergy        - set of all trial energy solutions.    [size = npop]
    population       - set of all trial parameter solutions. [size = dim*npop]
    solution_history - history of bestSolution status.       [StepMonitor.x]
    energy_history   - history of bestEnergy status.         [StepMonitor.y]
    signal_handler   - catches the interrupt signal.         [***disabled***]
        """
        super(AbstractEnsembleSolver, self).__init__(dim, **kwds)
       #self.signal_handler   = None
       #self._handle_sigint   = False

        # default settings for nested optimization
        #XXX: move nbins and npts to _InitialPoints?
        self._dist = None #kwds['dist'] if 'dist' in kwds else None
        nbins = kwds['nbins'] if 'nbins' in kwds else [1]*dim
        if isinstance(nbins, int):
            from mystic.math.grid import randomly_bin
            nbins = randomly_bin(nbins, dim, ones=True, exact=True)
        self._nbins           = nbins
        npts = kwds['npts'] if 'npts' in kwds else 1
        self._npts            = npts
        from mystic.solvers import NelderMeadSimplexSolver
        self._solver          = NelderMeadSimplexSolver
        self._bestSolver      = None # 'best' solver (after Solve)
        self._total_evals     = 0 # total function calls (after Solve)
        NP = reduce(lambda x,y:x*y, nbins) if 'nbins' in kwds else npts
        self._allSolvers      = [None for j in range(NP)]
        return

    def SetNestedSolver(self, solver):
        """set the nested solver

input::
    - solver: a mystic solver instance (e.g. NelderMeadSimplexSolver(3) )"""
        self._solver = solver
        return

    def __get_solver_instance(self):
        """ensure the solver is a solver instance"""
        solver = self._solver

        # if a configured solver is not given, then build one of the given type
        from mystic.abstract_solver import AbstractSolver
        if isinstance(solver, AbstractSolver): # is a configured solver instance
            return solver
        if not hasattr(solver, "Solve"):       # is an Error...
            raise TypeError("%s is not a valid solver" % solver)

        # otherwise, this is a solver class and needs configuring
       #from mystic.monitors import Monitor
       #stepmon = Monitor()
       #evalmon = Monitor()
       #maxiter = 1000
       #maxfun = 1e+6
        solver = solver(self.nDim)
        solver.SetRandomInitialPoints() #FIXME: set population; will override
        if self._useStrictRange: #XXX: always, settable, or sync'd ?
            solver.SetStrictRanges(min=self._strictMin, max=self._strictMax)
        solver.SetEvaluationLimits(self._maxiter, self._maxfun)
        solver.SetEvaluationMonitor(self._evalmon) #XXX: or copy or set?
        solver.SetGenerationMonitor(self._stepmon) #XXX: or copy or set?
        solver.SetTermination(self._termination)
        solver.SetConstraints(self._constraints)
        solver.SetPenalty(self._penalty)
        if self._reducer: #XXX: always, settable, or sync'd ?
            solver.SetReducer(self._reducer, arraylike=True)
        return solver

    def SetInitialPoints(self, x0, radius=0.05):
        """Set Initial Points with Guess (x0)

input::
    - x0: must be a sequence of length self.nDim
    - radius: generate random points within [-radius*x0, radius*x0]
        for i!=0 when a simplex-type initial guess in required

*** this method must be overwritten ***"""
        raise NotImplementedError("must be overwritten...")
    
    def SetRandomInitialPoints(self, min=None, max=None):
        """Generate Random Initial Points within given Bounds

input::
    - min, max: must be a sequence of length self.nDim
    - each min[i] should be <= the corresponding max[i]

*** this method must be overwritten ***"""
        raise NotImplementedError("must be overwritten...")

    def SetMultinormalInitialPoints(self, mean, var=None):
        """Generate Initial Points from Multivariate Normal.

input::
    - mean must be a sequence of length self.nDim
    - var can be...
        None: -> it becomes the identity
        scalar: -> var becomes scalar * I
        matrix: -> the variance matrix. must be the right size!

*** this method must be overwritten ***"""
        raise NotImplementedError("must be overwritten...")

    def SetSampledInitialPoints(self, dist=None):
        """Generate Random Initial Points from Distribution (dist)

input::
    - dist: a mystic.math.Distribution instance

*** this method must be overwritten ***"""
        raise NotImplementedError("must be overwritten...")

    def Terminated(self, disp=False, info=False, termination=None):
        """check if the solver meets the given termination conditions

Input::
    - disp = if True, print termination statistics and/or warnings
    - info = if True, return termination message (instead of boolean)
    - termination = termination conditions to check against

Note::
    If no termination conditions are given, the solver's stored
    termination conditions will be used.
        """
        if self._bestSolver:
            solver = self._bestSolver
        else:
            solver = self
        if termination is None:
            termination = solver._termination
        # ensure evaluation limits have been imposed
        self._SetEvaluationLimits()
        # check for termination messages
        msg = termination(solver, info=True)
        sig = "SolverInterrupt with %s" % {}
        lim = "EvaluationLimits with %s" % {'evaluations':solver._maxfun,
                                            'generations':solver._maxiter}

        # push solver internals to scipy.optimize.fmin interface
        if solver._fcalls[0] >= solver._maxfun and solver._maxfun is not None:
            msg = lim #XXX: prefer the default stop ?
            if disp:
                print("Warning: Maximum number of function evaluations has "\
                      "been exceeded.")
        elif solver.generations >= solver._maxiter and solver._maxiter is not None:
            msg = lim #XXX: prefer the default stop ?
            if disp:
                print("Warning: Maximum number of iterations has been exceeded")
        elif solver._EARLYEXIT: #XXX: self or solver ?
            msg = sig
            if disp:
                print("Warning: Optimization terminated with signal interrupt.")
        elif msg and disp:
            print("Optimization terminated successfully.")
            print("         Current function value: %f" % solver.bestEnergy)
            print("         Iterations: %d" % solver.generations)
            print("         Function evaluations: %d" % solver._fcalls[0])
            print("         Total function evaluations: %d" % self._total_evals)

        if info:
            return msg
        return bool(msg)

    def _update_objective(self):
        """decorate the cost function with bounds, penalties, monitors, etc"""
        # rewrap the cost if the solver has been run
        self.Finalize()
        return

    def SetDistribution(self, dist=None):
        """Set the distribution used for determining solver starting points

Inputs:
    - dist: a mystic.math.Distribution instance
"""
        from mystic.math import Distribution
        if dist and Distribution not in dist.__class__.mro():
            dist = Distribution(dist) #XXX: or throw error?
        self._dist = dist
        return

    def _InitialPoints(self):
        """Generate a grid of starting points for the ensemble of optimizers

*** this method must be overwritten ***"""
        raise NotImplementedError("a sampling algorithm was not provided")

    #FIXME: should take cost=None, ExtraArgs=None... and utilize Step
    def Solve(self, cost, termination=None, ExtraArgs=(), **kwds):
        """Minimize a 'cost' function with given termination conditions.

Description:

    Uses an ensemble of optimizers to find the minimum of
    a function of one or more variables.

Inputs:

    cost -- the Python function or method to be minimized.

Additional Inputs:

    termination -- callable object providing termination conditions.
    ExtraArgs -- extra arguments for cost.

Further Inputs:

    sigint_callback -- callback function for signal handler.
    callback -- an optional user-supplied function to call after each
        iteration.  It is called as callback(xk), where xk is the
        current parameter vector.                           [default = None]
    disp -- non-zero to print convergence messages.         [default = 0]
        """
        # process and activate input settings
        if 'sigint_callback' in kwds:
            self.sigint_callback = kwds['sigint_callback']
            del kwds['sigint_callback']
        else: self.sigint_callback = None
        settings = self._process_inputs(kwds)
        disp = settings['disp'] if 'disp' in settings else False
        echo = settings['callback'] if 'callback' in settings else None
#       for key in settings:
#           exec "%s = settings['%s']" % (key,key)
        if disp in ['verbose', 'all']: verbose = True
        else: verbose = False
        #-------------------------------------------------------------

        from mystic.python_map import python_map
        if self._map != python_map:
            #FIXME: EvaluationMonitor fails for MPI, throws error for 'pp'
            from mystic.monitors import Null
            evalmon = Null()
        else: evalmon = self._evalmon
        fcalls, cost = wrap_function(cost, ExtraArgs, evalmon)

        # set up signal handler
       #self._EARLYEXIT = False

        # activate signal_handler
       #import threading as thread
       #mainthread = isinstance(thread.current_thread(), thread._MainThread)
       #if mainthread: #XXX: if not mainthread, signal will raise ValueError
        import mystic._signal as signal
        if self._handle_sigint:
            signal.signal(signal.SIGINT, signal.Handler(self))

        # register termination function
        if termination is not None: self.SetTermination(termination)

        # get the nested solver instance
        solver = self._AbstractEnsembleSolver__get_solver_instance()
        #-------------------------------------------------------------

        # generate starting points
        initial_values = self._InitialPoints()

        # run optimizer for each grid point
        from copy import deepcopy as _copy
        op = [_copy(solver) for i in range(len(initial_values))]
       #cf = [cost for i in range(len(initial_values))]
        vb = [verbose for i in range(len(initial_values))]
        cb = [echo for i in range(len(initial_values))] #XXX: remove?
        at = self.id if self.id else 0  # start at self.id
        id = range(at,at+len(initial_values))

        # generate the local_optimize function
        def local_optimize(solver, x0, rank=None, disp=False, callback=None):
            from copy import deepcopy as _copy
            from mystic.tools import isNull
            solver.id = rank
            solver.SetInitialPoints(x0)
            if solver._useStrictRange: #XXX: always, settable, or sync'd ?
                solver.SetStrictRanges(min=solver._strictMin, \
                                       max=solver._strictMax) # or lower,upper ?
            solver.Solve(cost, disp=disp, callback=callback)
            sm = solver._stepmon
            em = solver._evalmon
            if isNull(sm): sm = ([],[],[],[])
            else: sm = (_copy(sm._x),_copy(sm._y),_copy(sm._id),_copy(sm._info))
            if isNull(em): em = ([],[],[],[])
            else: em = (_copy(em._x),_copy(em._y),_copy(em._id),_copy(em._info))
            return solver, sm, em

        # map:: solver = local_optimize(solver, x0, id, verbose)
        results = list(self._map(local_optimize, op, initial_values, id, \
                                                 vb, cb, **self._mapconfig))

        # save initial state
        self._AbstractSolver__save_state()
        #XXX: HACK TO GET CONTENT OF ALL MONITORS
        # reconnect monitors; save all solvers
        from mystic.monitors import Monitor
        while results: #XXX: option to not save allSolvers? skip this and _copy
            _solver, _stepmon, _evalmon = results.pop()
            sm = Monitor()
            sm._x,sm._y,sm._id,sm._info = _stepmon
            _solver._stepmon.extend(sm)
            del sm
            em = Monitor()
            em._x,em._y,em._id,em._info = _evalmon
            _solver._evalmon.extend(em)
            del em
            self._allSolvers[len(results)] = _solver
        del results, _solver, _stepmon, _evalmon
        #XXX: END HACK

        # get the results with the lowest energy
        self._bestSolver = self._allSolvers[0]
        bestpath = self._bestSolver._stepmon
        besteval = self._bestSolver._evalmon
        self._total_evals = self._bestSolver.evaluations
        for solver in self._allSolvers[1:]:
            self._total_evals += solver.evaluations # add func evals
            if solver.bestEnergy < self._bestSolver.bestEnergy:
                self._bestSolver = solver
                bestpath = solver._stepmon
                besteval = solver._evalmon

        # return results to internals
        self.population = self._bestSolver.population #XXX: pointer? copy?
        self.popEnergy = self._bestSolver.popEnergy #XXX: pointer? copy?
        self.bestSolution = self._bestSolver.bestSolution #XXX: pointer? copy?
        self.bestEnergy = self._bestSolver.bestEnergy
        self.trialSolution = self._bestSolver.trialSolution #XXX: pointer? copy?
        self._fcalls = self._bestSolver._fcalls #XXX: pointer? copy?
        self._maxiter = self._bestSolver._maxiter
        self._maxfun = self._bestSolver._maxfun

        # write 'bests' to monitors  #XXX: non-best monitors may be useful too
        self._stepmon = bestpath #XXX: pointer? copy?
        self._evalmon = besteval #XXX: pointer? copy?
        self.energy_history = None
        self.solution_history = None
       #from mystic.tools import isNull
       #if isNull(bestpath):
       #    self._stepmon = bestpath
       #else:
       #    for i in range(len(bestpath.y)):
       #        self._stepmon(bestpath.x[i], bestpath.y[i], self.id)
       #        #XXX: could apply callback here, or in exec'd code
       #if isNull(besteval):
       #    self._evalmon = besteval
       #else:
       #    for i in range(len(besteval.y)):
       #        self._evalmon(besteval.x[i], besteval.y[i])
        #-------------------------------------------------------------

        # restore default handler for signal interrupts
        if self._handle_sigint:
            signal.signal(signal.SIGINT, signal.default_int_handler)

        # log any termination messages
        msg = self.Terminated(disp=disp, info=True)
        if msg: self._stepmon.info('STOP("%s")' % msg)
        # save final state
        self._AbstractSolver__save_state(force=True)
        return 


if __name__=='__main__':
    help(__name__)

# end of file
