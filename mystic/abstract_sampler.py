#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2019-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
This module contains the base class for mystic samplers, and describes
the mystic sampler interface.
"""
__all__ = ['AbstractSampler']


class AbstractSampler(object): # derived class per sampling algorithm
    """
AbstractSampler base class for optimizer-directed samplers
    """
    def __init__(self, bounds, model, npts=None, **kwds):
        """
Args:
  bounds (list[tuples]): (lower,upper) bound for each model input
  model (function): ``y = model(x)``, where ``x`` is an input parameter vector
  npts (int, default=None): number of points to sample the model
  **kwds (dict, default={}): keywords for the underlying ensemble of solvers;
    ``(evalmon, stepmon, maxiter, maxfun, dist, saveiter, state, termination,
    constraints, penalty, reducer, solver, id, map, tightrange, cliprange)``
    are available for use. See ``mystic.ensemble`` for more details.
        """ #FIXME: added 'id' and 'map'
        self._bounds = bounds
        self._model = model #FIXME: if None, interpolate
        self._npts = npts

        self._evalmon = kwds.pop('evalmon', None)
        if self._evalmon is None:
            from mystic.monitors import Null
            self._evalmon = Null()
        from collections import defaultdict
        self._kwds = defaultdict(lambda :None)
        self._kwds.update(kwds)

        s = self._init_solver()
        kwd = dict(tight=self._kwds['tightrange'], clip=self._kwds['cliprange'])
        s.SetStrictRanges(*zip(*bounds), **kwd)
        #s.SetObjective(memo) #model) #XXX: ExtraArgs: axis ???
        s.SetObjective(model) #FIXME: ensure cached model

        # apply additional kwds
        solver = self._kwds['solver']
        if solver is not None: s.SetNestedSolver(solver)
        s.id = self._kwds['id']
        s.SetDistribution(self._kwds['dist'])
        s.SetEvaluationLimits(self._kwds['maxiter'], self._kwds['maxfun'])
        s.SetSaveFrequency(self._kwds['saveiter'], self._kwds['state'])
        termination = self._kwds['termination']
        if termination is not None: s.SetTermination(termination) #XXX:?
        s.SetConstraints(self._kwds['constraints'])
        s.SetPenalty(self._kwds['penalty'])
        s.SetReducer(self._kwds['reducer'], arraylike=True)
        s.SetGenerationMonitor(self._kwds['stepmon']) #XXX: use self._stepmon?
        s.SetMapper(self._kwds['map']) #TODO: close/join/clear after Solve?

        # pass a copy of the monitor to all instances
        import copy
        m = copy.deepcopy(self._evalmon) #XXX: no change with direct reference
        s.SetEvaluationMonitor(m)
        self._sampler = s
        self._npts = s._npts

        self._evals = [0] * s._npts #XXX: collect count or monitor?
        self._iters = [0] * s._npts #XXX: collect count or monitor?
        return


    def _init_solver(self):
        """initialize the ensemble solver"""
        from mystic.abstract_ensemble_solver import AbstractEnsembleSolver
        return AbstractEnsembleSolver(len(self._bounds), npts=self._npts)


    def _reset_sampler(self):
        """reinitialize the ensemble solver/sampler"""
        s = self._sampler
        # the following "seems" to work; all/more necessary?
        #XXX: add this method as internal method for ensemble solver?
        s.popEnergy = [s._init_popEnergy] * s.nPop
        s.population = [[0.0 for i in range(s.nDim)] for j in range(s.nPop)]
        # s.trialSolution = [0.0] * s.nDim
        s._bestEnergy = None
        s._bestSolution = None
        s._fcalls = [0] #XXX:?
        s._allSolvers = [None]*s._npts
        s._bestSolver = None
        s.SetGenerationMonitor(s._stepmon[:0], new=True) #XXX:?
        s.SetEvaluationMonitor(s._evalmon[:0], new=True) #XXX:?
        s._live = False
        s._bootstrap_objective()
        return


    def _reset_solved(self):
        """reinitialize all terminated solvers in the ensemble"""
        s = self._sampler
        # the following "seems" to work; all/more necessary?
        #XXX: add this method as internal method for ensemble solver?
        from copy import copy as _copy
        solver = s._AbstractEnsembleSolver__get_solver_instance(reset=True)
        #solver.SetEvaluationMonitor(solver._evalmon[:0], new=True)
        #solver.SetGenerationMonitor(solver._stepmon[:0], new=True)
        [s._allSolvers.__setitem__(i, _copy(solver)) for i,j in enumerate(s.Terminated(all=True)) if j is True]
        s._bestSolver = None
        s._AbstractEnsembleSolver__update_state()
        return


    def _sample(self, reset=False):#, **kwds):
        """collect a sample for each member in the ensemble

Args:
  reset (bool, default=False): reset all solvers before sampling; alternately,
    if ``reset`` is None, then only reset the terminated solvers

Returns:
  None
        """
        #NOTE: sample(reset_all) is like _sample(reset), with None <=> False
        s = self._sampler
        m = self._evalmon

        _eval = s._all_evals
        _iter = s._all_iters
        if reset: self._reset_sampler()
        elif reset is None: self._reset_solved()

        # get evals before taking a step (needed for backfill)
        evals = [len(getattr(i, '_evalmon', ())) for i in s._allSolvers]
        neval = len(m)

        # take a step
        s.Step()
        termination = None #s.Terminated(**kwds) #XXX:?
        #NOTE: i._evalmon._id for i in _allSolvers is [None]

        # backfill original monitor instance #NOTE: not in chronological order
        map(m.extend, [i._evalmon[e:] for e,i in zip(evals, s._allSolvers)])

        _eval = (max(j-i,1) if j>=i else i for (i,j) in zip(_eval,s._all_evals))
        _iter = (max(j-i,1) if j>=i else i for (i,j) in zip(_iter,s._all_iters))
        for i,(e,t) in enumerate(zip(_eval,_iter)):
            self._evals[i] += e
            self._iters[i] += t
        return termination


    def sample(self, if_terminated=None, reset_all=True):#, **kwds):
        """sample npts using vectorized solvers

Args:
  if_terminated (bool, default=None): the amount of termination; must be one
    of ``{all, True, any, False, None}``
  reset_all (bool, default=True): action to take when ``if_terminated`` is met;
    must be one of ``{True, False, None}``

Returns:
  None

Notes:
  - When ``if_terminated`` is None, reset regardless of termination.
  - When ``if_terminated`` is True, reset if the best solver is terminated.
  - When ``if_terminated`` is False, reset if no solvers are terminated.
  - When ``if_terminated`` is all, reset if all solvers are terminated.
  - When ``if_terminated`` is any, reset if any solvers are terminated.
  - If ``reset_all`` is None, never reset.
  - If ``reset_all`` is True, reset all solvers if ``if_terminated`` is met.
  - If ``reset_all`` is False, similarly reset only the terminated solvers.
        """
        if type(reset_all) is bool:
            s = self._sampler
            if if_terminated is None: # reset regardless
                return self._sample(reset=(True if reset_all else None))#, **kwds)
            if not if_terminated in (all, True, any, False):
                msg = "%s not in (all, True, any, False, None)" % if_terminated
                raise ValueError(msg)
            stop = s.Terminated(all=True)
            if all(stop) \
            or (stop[s._is_best() or 0] and if_terminated is not all) \
            or (any(stop) and if_terminated in (any, False)) \
            or (not any(stop) and if_terminated is False):
                return self._sample(reset=(True if reset_all else None))#, **kwds)
            # othewise just take a step
            return self._sample(reset=False)#, **kwds)
        return self._sample(reset=False)#, **kwds)


    def sample_until(self, iters=None, evals=None, terminated=None, **kwds):
        """sample until one of the stop conditions are met

Possible stop conditions are:
  - solver iterations ``iters()`` equals or exceeds ``iters``
  - solver evaluations ``evals()`` equals or exceeds ``evals``
  - number of terminated solvers equals or exceeds ``terminated``

Args:
  iters (int, default=inf): maximum number of iterations
  evals (int, default=inf): maximum number of evaluations
  terminated (int, default=inf): maximum number of terminated solvers
  if_terminated (bool, default=None): the amount of termination; must be one
    of ``{all, True, any, False, None}``
  reset_all (bool, default=True): action to take when ``if_terminated`` is met;
    must be one of ``{True, False, None}``

Notes:
  - The default sampler configuration is to always reset (``reset_all=True``)
  - If ``termination != None``, the default is never reset (``reset_all=None``)
  - A limit for at least one of ``(iters, evals, termination)`` must be set.
  - ``terminated`` may also be one of ``{all, True, any, False, None}``, where
    ``{all: 'all', True: 'best', any: '1', False: '0', None: 'inf', N: 'N'}``
  - ``if_terminated`` may be one of ``{all, True, any, False, None}``, where
    ``{all: 'all', True: 'best', any: '1', False: '0', None: 'always'}``
  - ``reset_all`` may be one of ``{True, False, None}``, where ``{True: 'reset
    all', False: 'reset solved', None: 'never reset'}``
        """
        from numpy import inf, all as all_, any as any_
        if evals is None:
            evals = inf
        if iters is None:
            iters = inf
        if terminated is None:
            terminated = self._npts + 1
        else: #NOTE: if termination is specified, don't reset by default
            kwds['reset_all'] = kwds.get('reset_all', None)
            if terminated is all or terminated is all_:
                terminated = self._npts
            elif terminated is any or terminated is any_:
                terminated = 1
            elif terminated is True: # best
                pass
            elif terminated is False:
                terminated = 0
            elif terminated > self._npts:
                msg = 'terminated = %s is greater than npts = %s' % (terminated, self._npts)
                raise ValueError(msg)
        if_stop = kwds.pop('if_terminated', None)
        quit = inf
        if if_stop is all or if_stop is all_:
            quit = self._npts
        elif if_stop is any or if_stop is any_:
            quit = 1
        elif if_stop is True: # best
            quit = True
        elif if_stop is False:
            quit = 0
        elif if_stop is None:
            quit = 0 #XXX: -1 ?
        reset = kwds.pop('reset_all', True)
        # require iters or evals if: (term > npts) OR (reset!=None & quit<term)
        if iters is inf and evals is inf and (terminated > self._npts or
            (reset is not None and quit < terminated)):
            msg = 'either iters = %s or evals = %s must be less than inf' % (iters,evals)
            raise ValueError(msg)
        ###stop = self.terminated(**kwds)
        while self.iters() < iters and self.evals() < evals and (not self.terminated(all=False) if (terminated is True) else (sum(self.terminated(all=True)) < terminated)): #FIXME: sometimes limiting iters stops very late
            stop = self.sample(if_stop, reset)#, **kwds)
        return### stop


    def evals(self, all=False): #XXX: None?
        """get the total number of function evaluations

Args:
  all (bool, default=False): report the ``evals`` for each ensemble member
        """
        if all:
            return self._evals
        return sum(self._evals)


    def iters(self, all=False): #XXX: None?
        """get the total number of solver iterations

Args:
  all (bool, default=False): report the ``iters`` for each ensemble member
        """
        if all:
            return self._iters
        return sum(self._iters)


    def terminated(self, *args, **kwds): #FIXME: confusing wrt if_terminated?
        """check if termination conditions have been met

Args:
  disp (bool, default=False): print termination statistics and/or warnings
  info (bool, default=False): return termination message (instead of boolean)
  all (bool, default=None): get results for all solvers, else get the 'best'

Notes:
  - ``disp`` expects a bool, but can also take ``'verbose'`` for more verbosity 
  - ``all``, by default (i.e. None), will show only the terminated solvers
        """
        _all = kwds.get('all', None)
        if _all not in (True, False, None):
            msg = 'all = %s, must be one of (True, False, None)' % getattr(_all, '__name__', _all)
            raise ValueError(msg)
        kwds['all'] = _all
        return self._sampler.Terminated(*args, **kwds)


"""
### Notes ###

# utility functions
_reset_sampler: reset all
_reset_solved: reset all solved
_sample(reset): reset all/stop/none, then step

# use cases to manage evals/iters
_restart_all_without_step: _reset_sampler
_restart_solved_without_step: _reset_solved
_restart_all_then_step: _sample(reset=True)
_restart_solved_then_step: _sample(reset=None)
_step_without_restart: _sample(reset=False)

# options
_restart_if_solved(solved): (* = 'all' if all else 'solved')
. all: if all stop, _restart_*_then_step
. True: if best stop, _restart_*_then_step
. any: if any stop, _restart_*_then_step 
. False: if none stop, _restart_*_then_step
. None: _step_without_restart
"""
