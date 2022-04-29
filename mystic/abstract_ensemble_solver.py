#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
#
# Abstract Ensemble Solver Class
"""
This module contains the base class for launching several mystic solvers
instances -- utilizing a parallel ``map`` function to enable parallel
computing.  This module describes the ensemble solver interface.  As with
the ``AbstractSolver``, the ``_Step`` method must be overwritten with the
derived solver's optimization algorithm. Similar to ``AbstractMapSolver``,
a call to ``map`` is required.  In addition to the class interface, a
simple function interface for a derived solver class is often provided.
For an example, see the following.

The default map API settings are provided within mystic, while distributed
and parallel computing maps can be obtained from the ``pathos`` package 
(http://dev.danse.us/trac/pathos).

Examples:

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

All solvers packaged with mystic include a signal handler that provides
the following options::

    sol: Print current best solution.
    cont: Continue calculation.
    call: Executes sigint_callback, if provided.
    exit: Exits with current best solution.

Handlers are enabled with the ``enable_signal_handler`` method,
and are configured through the solver's ``Solve`` method.  Handlers
trigger when a signal interrupt (usually, ``Ctrl-C``) is given while
the solver is running.

Notes:

    The handler is currently disabled when the solver is run in parallel.
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
Takes one initial input::

    dim      -- dimensionality of the problem.

Additional inputs::

    npop     -- size of the trial solution population.      [default = 1]
    nbins    -- tuple of number of bins in each dimension.  [default = [1]*dim]
    npts     -- number of solver instances.                 [default = 1]
    rtol     -- size of radial tolerance for sparsity.      [default = None]

Important class members::

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
        npts = kwds['npts'] if 'npts' in kwds else 1
        if npts is None: npts = 1
        self._npts            = npts
        nbins = kwds['nbins'] if 'nbins' in kwds else None
        self._nbins           = nbins
        if isinstance(nbins, int):
            self._npts = nbins
        elif nbins is None:
            pass # nbins = [1]*dim
        else:
            self._npts = reduce(lambda i,j:i*j, nbins) # nbins
        rtol = kwds['rtol'] if 'rtol' in kwds else None
        self._rtol            = rtol
        from mystic.solvers import NelderMeadSimplexSolver
        self._solver          = NelderMeadSimplexSolver
        self._bestSolver      = None # 'best' solver (after Solve)
        self._allSolvers      = [None for j in range(self._npts)]
        self._step            = False
        return

    def __all_evals(self):
        """count of all function calls"""
        return [getattr(i, 'evaluations', 0) for i in self._allSolvers]

    def __all_iters(self):
        """count of all iterations"""
        return [getattr(i, 'generations', 0) for i in self._allSolvers]

    def __all_bestEnergy(self): #XXX: default = None?
        """get bestEnergy from all solvers"""
        return [getattr(i, 'bestEnergy', None) for i in self._allSolvers]

    def __all_bestSolution(self): #XXX: default = None?
        """get bestSolution from all solvers"""
        return [getattr(i, 'bestSolution', None) for i in self._allSolvers]

    def __total_evals(self):
        """total number of function calls"""
        return sum(self._all_evals)

    def __total_iters(self):
        """total number of iterations"""
        return sum(self._all_iters)

    def SetNestedSolver(self, solver):
        """set the nested solver

input::
    - solver: a mystic solver instance (e.g. NelderMeadSimplexSolver(3) )"""
        self._solver = solver
        return

    def __get_solver_instance(self, reset=False):
        """ensure the solver is a solver instance

input::
    - reset: if reset is True, reset the monitor instances"""
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
            solver.SetStrictRanges(min=self._strictMin, \
                                   max=self._strictMax, \
                                   tight=self._useTightRange, \
                                   clip=self._useClipRange)
        if reset:
            solver.SetEvaluationMonitor(self._evalmon[:0], new=True)
            solver.SetGenerationMonitor(self._stepmon[:0], new=True)
        else:
            solver.SetEvaluationMonitor(self._evalmon) #XXX: copy? new?
            solver.SetGenerationMonitor(self._stepmon) #XXX: copy? new?
        solver.SetEvaluationLimits(self._maxiter, self._maxfun) #XXX: new?
        solver.SetTermination(self._termination)
        solver.SetConstraints(self._constraints)
        solver.SetPenalty(self._penalty)
        if self._reducer: #XXX: always, settable, or sync'd ?
            solver.SetReducer(self._reducer, arraylike=True)
        solver.SetObjective(self._cost[1], self._cost[2])
        solver.SetSaveFrequency(self._saveiter, self._state)
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

    def Terminated(self, disp=False, info=False, termination=None, all=None):
        """check if the solver meets the given termination conditions

Input::
    - disp = if True, print termination statistics and/or warnings
    - info = if True, return termination message (instead of boolean)
    - termination = termination conditions to check against
    - all = if True, get results for all solvers; if False, only check 'best'

Notes::
    If no termination conditions are given, the solver's stored
    termination conditions will be used.
        """
        if disp in ['verbose', 'all']: verbose = True
        else: verbose = False
        no = '' if info else False
        if all is True:
            end = [no if s is None else s.Terminated(verbose, info, termination) for s in self._allSolvers]
            return end
        elif all is None:
            end = [False if s is None else s.Terminated(termination=termination) for s in self._allSolvers]
            if False in end: return no
            #else: get info from bestSolver
        self._AbstractEnsembleSolver__update_state()
        solver = self._bestSolver or self
        if termination is None:
            termination = solver._termination
        # ensure evaluation limits have been imposed
        self._SetEvaluationLimits() #XXX: always?
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

    def SetDistribution(self, dist=None):
        """Set the distribution used for determining solver starting points

Inputs:
    - dist: a mystic.math.Distribution instance
        """
        from mystic.math import Distribution
        if dist and Distribution not in dist.__class__.mro():
            dist = Distribution(dist) #XXX: or throw error?
        self._dist = dist #FIXME: accept a list of Distributions
        return

    def _InitialPoints(self):
        """Generate a grid of starting points for the ensemble of optimizers

*** this method must be overwritten ***"""
        raise NotImplementedError("a sampling algorithm was not provided")

    def _is_new(self):
        """determine if solver has been run or not"""
        return not any(self._allSolvers)

    def _is_best(self):
        """get the id of the bestSolver"""
        return getattr(self._bestSolver, 'id', None)

    def __init_allSolvers(self, reset=False):
        """populate NestedSolver state to allSolvers

input::
    - reset: if reset is True, reset the monitor instances"""
        # get the nested solver instance
        solver = self._AbstractEnsembleSolver__get_solver_instance(reset)

        # configure inputs for each solver
        from copy import deepcopy as _copy
        at = self.id if self.id else 0  #XXX start at self.id?
        #at = max((getattr(i, 'id', self.id) or 0) for i in self._allSolvers)
        for i,op in enumerate(self._allSolvers):
            if op is None: #XXX: don't reset existing solvers?
                op = _copy(solver)
                op.id = i
                self._allSolvers[i] = op
        return self._allSolvers

    def __update_allSolvers(self, results):
        """replace allSolvers with solvers found in results"""
        #NOTE: apparently, monitors internal to the solver don't work as well
        # reconnect monitors; save all solvers
        fcalls = [getattr(s, '_fcalls', [0])[0] for s in self._allSolvers]
        from mystic.monitors import Monitor
        while results: #XXX: option to not save allSolvers? skip this and _copy
            _solver, _stepmon, _evalmon = results.pop()
            lr = len(results)
            sm, em = Monitor(), Monitor()
            s = self._allSolvers[lr]
            ls, le = len(s._stepmon), len(s._evalmon)
            # gather old and new results in monitors
            _solver._stepmon[:] = s._stepmon
            sm._x,sm._y,sm._id,sm._info = _stepmon
            _solver._stepmon[ls:] = sm[ls:]
            del sm
            _solver._evalmon[:] = s._evalmon
            em._x,em._y,em._id,em._info = _evalmon
            _solver._evalmon[le:] = em[le:]
            del em
            if not _solver._fcalls[0]: #FIXME: HACK workaround dropped _fcalls
                lm = len(_solver._evalmon)   # ...but only works if has evalmon
                _solver._fcalls[0] = fcalls[lr] or \
                                     (lm if _solver.Terminated() else 0)
                                    #(le if s.Terminated() else 0)
            self._allSolvers[lr] = _solver #XXX: update not replace?
        return

    def __update_bestSolver(self):
        """update _bestSolver from _allSolvers"""
        if self._bestSolver is None:
            self._bestSolver = self._allSolvers[0]
        bestpath = besteval = None
        # get the results with the lowest energy
        for solver in self._allSolvers[:]: #XXX: slice needed?
            if solver is None: continue
            energy = getattr(self._bestSolver,'bestEnergy',self.bestEnergy)
            if solver.bestEnergy <= energy:
                self._bestSolver = solver
                bestpath = solver._stepmon
                besteval = solver._evalmon
        return bestpath, besteval

    def __update_state(self):
        """update solver state from _bestSolver"""
        bestpath, besteval = self._AbstractEnsembleSolver__update_bestSolver()
        if bestpath is besteval is None: return

        # return results to internals
        self.population = self._bestSolver.population #XXX: pointer? copy?
        self.popEnergy = self._bestSolver.popEnergy #XXX: pointer? copy?
        self.bestSolution = self._bestSolver.bestSolution #XXX: pointer? copy?
        self.bestEnergy = self._bestSolver.bestEnergy
        self.trialSolution = self._bestSolver.trialSolution #XXX: pointer? copy?
        self._fcalls = self._bestSolver._fcalls #XXX: pointer? copy?
        self._maxiter = self._bestSolver._maxiter
        self._maxfun = self._bestSolver._maxfun
        # write state relevant for collapse
        self._constraints = self._bestSolver._constraints
        self._termination = self._bestSolver._termination

        # write 'bests' to monitors  #XXX: non-best monitors may be useful too
        self._stepmon = bestpath #XXX: pointer? copy?
        self._evalmon = besteval #XXX: pointer? copy?
        self.energy_history = None
        self.solution_history = None
        return

    def Collapsed(self, disp=False, info=False, all=None):
        """check if the solver meets the given collapse conditions

Input::
    - disp = if True, print details about the solver state at collapse
    - info = if True, return collapsed state (instead of boolean)
    - all = if True, get results for all solvers; if False, only check 'best'
""" #FIXME: how handle collapse of base solver, mask of base solver...?
        _all = __builtins__['all']
        if all is None: all = any #FIXME: what should default be???
        elif all not in (any, _all, True, False): # may be np.all or np.any
            name = getattr(all, '__name__', '')
            if name == 'any': all = any
            elif name == 'all': all = _all
        if disp in ['verbose', 'all']: verbose = True
        else: verbose = False
        no = {} if info else False
        if all is True: # check for 'all' collapses
            end = [no if s is None else s.Collapsed(verbose, info) for s in self._allSolvers]
            return end
        elif all is any: # check for 'any' collapse
            end = {}
            for s in self._allSolvers:
                if s is None: continue
                s = s.Collapsed(disp, info=True)
                if not s: continue
                for k,v in s.items():
                    r = end.get(k,None)
                    if r is not None:
                        if k.startswith(('CollapseCost','CollapseGrad')):
                            from mystic.tools import interval_overlap
                            v = interval_overlap(v,r, union=False)
                        elif k.startswith(('CollapsePosition','CollapseWeight')):
                            from mystic.tools import indicator_overlap
                            v = indicator_overlap(v,r, union=False)
                        else: # CollapseAt, CollapseAs
                            v = v.union(r)
                    end[k] = v
                    if not end[k]: del end[k]
            #NOTE: ignores prior collapses in 'mask'
            return end if info else bool(end)
        elif all is _all: # check for the same collapse
            terminate = [] # get collapses/terminate for all active solvers
            collapses = [s.Collapsed(disp, info=True) for s in self._allSolvers if s is not None and not terminate.append(s._termination)]
            if not any(collapses):
                return {} if info else False
            from mystic.tools import masked_collapse, no_mask, _no_mask
            xlt = {} # dict of {masked: non-masked}
            for i in terminate: # xlt has all prior collapses
                xlt.update(no_mask(i))
            end = {} # dict of {non-masked: collapse}
            for i,s in enumerate(collapses): #XXX: check: overlap of new & old
                s = masked_collapse(terminate[i], s, union=False)
                if not s:
                    end.clear(); break
                for k,v in s.items():
                    # get non-masked from masked
                    x = xlt.setdefault(k,_no_mask(k))
                    r = end.get(x,None) # get value from non-masked
                    if r is not None:
                        if k.startswith(('CollapseCost','CollapseGrad')):
                            from mystic.tools import interval_overlap
                            v = interval_overlap(v,r, union=True)
                        elif k.startswith(('CollapsePosition','CollapseWeight')):
                            from mystic.tools import indicator_overlap
                            v = indicator_overlap(v,r, union=True)
                        else: # CollapseAt, CollapseAs
                            v = v.intersection(r)
                    end[x] = v
                    if not end[x] or not x: del end[x]
            _end = {} # dict of {masked: collapse}
            for k,x in xlt.items():
                x = end.get(x, None)
                if x is None: continue
                _end[k] = x
            return _end if info else bool(_end)
        self._AbstractEnsembleSolver__update_state()
        if self._bestSolver:
            return self._bestSolver.Collapsed(disp, info)
        return super(AbstractEnsembleSolver, self).Collapsed(disp, info)

    def Collapse(self, disp=False): #FIXME #TODO
        """if solver has terminated by collapse, apply the collapse
        (unless both collapse and "stop" are simultaneously satisfied)

input::
    - disp: if True, print details about the solver state at collapse

note::
    updates the solver's termination conditions and constraints
        """
        return False

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
        kwds['step'] = True  #XXX: once Step is taken, step=True thereafter
        settings = self._process_inputs(kwds)
        #(hardwired: due to python3.x exec'ing to locals())
        disp = settings['disp'] if 'disp' in settings else False
        echo = settings['callback'] if 'callback' in settings else None
        if disp in ['verbose', 'all']: verbose = True
        else: verbose = False

        # generate starting points
        if self._is_new(): iv = self._InitialPoints()
        else: iv = [None] * len(self._allSolvers)
        op = self._AbstractEnsembleSolver__init_allSolvers()
        vb = [verbose if not s.Terminated() else False for s in self._allSolvers]
        cb = [echo] * len(op) #XXX: remove?

        # generate the _step function
        def _step(solver, x0, disp=False, callback=None):
            from copy import deepcopy as _copy
            from mystic.tools import isNull
            #ns = len(solver._stepmon)
            #ne = len(solver._evalmon)
            if x0 is not None:
                solver.SetInitialPoints(x0)
                if solver._useStrictRange: #XXX: always, settable, or sync'd ?
                    solver.SetStrictRanges(solver._strictMin, \
                                           solver._strictMax, \
                                           tight=solver._useTightRange, \
                                           clip=self._useClipRange)
            _term = (solver._live is False) and solver.Terminated()
            if _term is True: solver._live = True #XXX: HACK don't reset _fcalls
            solver.Step(cost,ExtraArgs=ExtraArgs,disp=disp,callback=callback)
            if _term is True: solver._live = False
            sm = solver._stepmon
            em = solver._evalmon
            if isNull(sm): sm = ([],[],[],[])
            else:
                sm = _copy(sm)#[ns:]
                sm = (sm._x,sm._y,sm._id,sm._info)
            if isNull(em): em = ([],[],[],[])
            else:
                em = _copy(em)#[ne:]
                em = (em._x,em._y,em._id,em._info)
            return solver, sm, em

        # map:: solver = _step(solver, x0, id, verbose)
        results = list(self._map(_step, op, iv, vb, cb, **self._mapconfig))
        del op, iv, vb, cb

        # save initial state
        #self._AbstractSolver__save_state()

        # save results to allSolvers
        self._AbstractEnsembleSolver__update_allSolvers(results)
        del results

        # update state from bestSolver
        self._AbstractEnsembleSolver__update_state()
        return

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
    step (bool, default=False): if True, enable Step within the ensemble.

Note:
   The additional args are 'sticky', in that once they are given, they remain
   set until they are explicitly changed. Conversely, the args are not sticky,
   and are thus set for a one-time use.
        """
        #allow for inputs that don't conform to AbstractSolver interface
        #NOTE: not sticky: callback, disp
        #NOTE: sticky: EvaluationMonitor, StepMonitor, penalty, constraints
        #NOTE: sticky: step
        settings = super(AbstractEnsembleSolver, self)._process_inputs(kwds)
        settings.update({
        'step':self._step}) #run Solve with (or without) Step
        [settings.update({i:j}) for (i,j) in getattr(kwds, 'iteritems', kwds.items)() if i in settings]
        self._step = settings['step']
        return settings

    def _Solve(self, cost, ExtraArgs, **settings): #XXX: self._cost?
        """Run the optimizer to termination, using the given settings.

Args:
    cost (func): the function to be minimized: ``y = cost(x)``.
    ExtraArgs (tuple): tuple of extra arguments for ``cost``.
    settings (dict): optimizer settings (produced by _process_inputs)

Returns:
    None

Note:
    Optimizer settings for ensemble solvers include the `step` keyword,
    which enables the Step menthod within the ensemble. By default, ensemble
    solvers run to completion (i.e. Solve), for efficiency. Using `Step`
    enables the ensemble to communicate state between members of the ensemble
    at each iteration.
        """
        #FIXME: 'step' is undocumented (in Solve)
        #NOTE: if Step once, will ensure always uses step=True, unless...
        #TODO: if step=False passed after Step taken... this is still TODO!
        step = settings['step'] if 'step' in settings else False
        if step: #FIXME: use abstract_solver _Solve
            super(AbstractEnsembleSolver, self)._Solve(cost, ExtraArgs, **settings)
            return
            #FIXME: evalmon wrapped around _cost doesn't work (__class__()?)
        #FIXME: fix the above... remove HACK below (and previous evalmon HACKs)
        #FIXME: HACK evalmon wrapped around _cost is disabled (bc it works!)
        evalmon, self._evalmon = self._evalmon, Null() #XXX: fcalls start=0
        self._live = False #XXX: redo wrap objective with dummy evalmon
        cost = self._bootstrap_objective(self._cost[1], ExtraArgs)
        self._evalmon = evalmon
        #FIXME: HACK (end) to disable evalmon wrapped around cost

        disp = settings['disp'] if 'disp' in settings else False
        echo = settings['callback'] if 'callback' in settings else None
        if disp in ['verbose', 'all']: verbose = True
        else: verbose = False

        # generate starting points
        if self._is_new(): iv = self._InitialPoints()
        else: iv = [None] * len(self._allSolvers)
        op = self._AbstractEnsembleSolver__init_allSolvers()
        vb = [verbose] * len(op)
        cb = [echo] * len(op) #XXX: remove?

        # generate the _solve function
        def _solve(solver, x0, disp=False, callback=None):
            from copy import deepcopy as _copy
            from mystic.tools import isNull
            if x0 is not None:
                solver.SetInitialPoints(x0)
                if solver._useStrictRange: #XXX: always, settable, or sync'd ?
                    solver.SetStrictRanges(solver._strictMin, \
                                           solver._strictMax, \
                                           tight=solver._useTightRange, \
                                           clip=self._useClipRange)
            _term = (solver._live is False) and solver.Terminated()
            if _term is True: solver._live = True #XXX: HACK don't reset _fcalls
            if solver._cost[1] is None: #XXX: HACK for configured NestedSolver
                solver.SetObjective(cost, ExtraArgs=ExtraArgs)
            solver.Solve(disp=disp,callback=callback)
            if _term is True: solver._live = False
            sm = solver._stepmon
            em = solver._evalmon
            if isNull(sm): sm = ([],[],[],[])
            else:
                sm = _copy(sm)
                sm = (sm._x,sm._y,sm._id,sm._info)
            if isNull(em): em = ([],[],[],[])
            else:
                em = _copy(em)
                em = (em._x,em._y,em._id,em._info)
            return solver, sm, em

        # map:: solver = _solve(solver, x0, id, verbose)
        results = list(self._map(_solve, op, iv, vb, cb, **self._mapconfig))
        del op, iv, vb, cb

        # save initial state
        self._AbstractSolver__save_state()
        # save results to allSolvers
        self._AbstractEnsembleSolver__update_allSolvers(results)
        del results
        # update state from bestSolver
        self._AbstractEnsembleSolver__update_state()

        # log any termination messages
        msg = self.Terminated(disp=disp, info=True)
        if msg: self._stepmon.info('STOP("%s")' % msg)
        # save final state
        self._AbstractSolver__save_state(force=True)
        return

    # extensions to the solver interface
    _total_evals = property(__total_evals )
    _total_iters = property(__total_iters )
    _all_evals = property(__all_evals )
    _all_iters = property(__all_iters )
    _all_bestSolution = property(__all_bestSolution )
    _all_bestEnergy = property(__all_bestEnergy )
    pass


if __name__=='__main__':
    help(__name__)

# end of file
