#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2020-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
OUQ classes for calculating bounds on statistical quantities
"""
import mystic.cache
from mystic.cache import cached
from mystic.cache.archive import dict_archive, file_archive, read
from mystic.math import almostEqual
from mystic.math.discrete import product_measure
from mystic.math.samples import random_samples
from mystic.solvers import DifferentialEvolutionSolver2
from mystic.monitors import Monitor
from ouq_models import WrapModel


class BaseOUQ(object): #XXX: redo with a "Solver" interface, like ensemble?

    def __init__(self, model, bounds, **kwds):
        """OUQ model for a statistical quantity

    Input:
        model: function of the form y = model(x, axis=None)
        bounds: mystic.bounds.MeasureBounds instance

    Additional Input:
        samples: int, number of samples (used for non-deterministic models)
        penalty: function of the form y' = penalty(x)
        constraint: function of the form x' = constraint(x)
        xvalid: function returning True if x == x', given constraint
        cvalid: function similar to xvalid, but with product_measure input
        map: map instance, to evaluate model on the product_measure in parallel

    NOTE: when y is multivalued models must be a UQModel with ny != None
        """
        self.npts = bounds.n  # (2,1,1)
        self.lb = bounds.lower
        self.ub = bounds.upper
        #NOTE: for *_bound(axis=None) to work, requires ny != None
        #if not isinstance(model, WrapModel.mro()[-2]):
        #    model = WrapModel(model=model)
        self.model = model
        self.axes = getattr(model, 'ny', None) #FIXME: in kwds?, ny
        rnd = getattr(model, 'rnd', True) #FIXME: ?, rnd
        self.samples = kwds.get('samples', None) if rnd else None
        self.map = None if self.samples is None else kwds.get('map', None)
        self.penalty = kwds.get('penalty', lambda rv:0.)
        self.constraint = kwds.get('constraint', lambda rv:rv)
        self.xvalid = kwds.get('xvalid', lambda rv:True)
        self.cvalid = kwds.get('cvalid', lambda c:True)
        self._invalid = float('inf') #XXX: need to save state?
        self.kwds = {} #XXX: good idea???
        self._upper = {} # saved solver instances for upper bounds
        self._lower = {} # saved solver instances for lower bounds
        self._expect = {} # saved solver instances for expected value
        self._ave = {} # saved expected value from sampling
        self._var = {} # saved expected variance from sampling
        self._err = {} # saved misfit to sampled expected value
        self._cost = {} # saved cache of the most recent cost (for penalty)
        self._pts = {} # saved sampled {in:out} objective #XXX: out only? ''?
        return

    # --- extrema ---
    def expected(self, axis=None, **kwds):
        """find the expected value of the statistical quantity

    Input:
        axis: int, the index of y on which to find value (all, by default)
        axmap: map instance, to execute each axis in parallel (None, by default)
        instance: bool, if True, return the solver instance (False, by default)

    Additional Input:
        solver: mystic.solver instance [default: DifferentialEvolutionSolver2]
        npop: population size [default: None]
        id: a unique identifier for the solver [default: None]
        nested: mystic.solver instance [default: None], for ensemble solvers
        x0: initial parameter guess [default: use RandomInitialPoints]
        maxiter: max number of iterations [default: defined in solver]
        maxfun: max number of objective evaluations [default: defined in solver]
        evalmon: mystic.monitor instance [default: Monitor], for evaluations
        stepmon: mystic.monitor instance [default: Monitor], for iterations
        map: pathos map instance for solver.SetMapper [default: None]
        save: iteration frequency to save solver [default: None]
        opts: dict of configuration options for solver.Solve [default: {}]

    Further Input:
        archive: the archive (str name for a new archive, or archive instance)
        dist: a mystic.tools.Distribution instance (or list of Distributions)
        tight: if True, apply bounds concurrent with other constraints
        clip: if True, clip at bounds, else resample [default = False]
        smap: map instance, to sample the quantity in parallel [default = None]
        npts: maximum number of sample points [default = 10000]
        ipts: number of sample points per iteration [default = npts]
        iter: number of iterations to stop after if change < itol [default = 1]
        itol: stop if change in value < itol over iter [default = 1e-15]

    Returns:
        expected value of the statistical quantity, for the specified axis

    NOTE: by default, npts samplings of the expected value will be drawn,
        where npts is large. When the randomness of the model is expected to
        be small, the expected value can often be found quickly by setting
        ipts=1 and iter to a small number. Note however, that iterations are
        serial, while samplings within an iteration can utilize a parallel map.

    NOTE: an optimization is used to solve for inputs that yield the expected
        value, where the objective includes a penalty targeted to stop at the
        sampled mean. The additional penalty is 'linear_equality' with a
        strength, k, set equal to the sampled mean. The additional termination
        condition is 'VTR(ftol, target)', where ftol=1e-16 for models with
        no randomness, and ftol is the sampled variance for models with
        randomness. Target is the sampled mean. If only the expected value is
        of interest, setting `instance=None` will skip the optimization.
        """
        #NOTE: kwds(verbose, reducer) undocumented
        full = kwds.pop('instance', False)
        #self._expect.clear() #XXX: good idea?
        # use __call__ to get expected value [float or tuple(float)]
        from mystic.monitors import VerboseMonitor, Null
        verbose = kwds.pop('verbose', False) #NOTE: used for debug
        label = self.__class__.__name__
        mon = VerboseMonitor(1,None,label=label) if verbose else Null()
        db = kwds.pop('archive', None)
        if db is None: db = file_archive()
        elif type(db) in (str, (u''.__class__)):
            db = read(db, type=file_archive)
        # populate monitor with existing archive data
        m = Monitor()
        xs = list(db.values())
        N = size = ys = len(xs)
        if size:
            xtype = type(xs[-1]) # is tuple or float
            multi = hasattr(xtype, '__len__')
            import numpy as np
            xs = np.array(xs)
            var = xtype(xs.var(axis=0)) if multi else xs.var()
            if xs.ndim == 1: xs = np.atleast_2d(xs).T
            ys = np.arange(1, size+1)
            m._x = xs = (xs.cumsum(axis=0).T/ys).T.tolist()
            m._y = ys = ys.tolist()
            ave = xtype(xs[-1]) if multi else xs[-1][0]
            if verbose: print("%s: %s @ %s" % (label, ave, N))
        else:
            var = ave = float('nan') #XXX: good idea?
            multi = self.axes is not None
            if multi:
                var = ave = (ave,) * self.axes
        # sample expected value iteratively until termination met
        #reducer = kwds.pop('reducer', None) #XXX: throw error if provided?
        npts = kwds.pop('npts', 10000)
        ipts = kwds.pop('ipts', None)
        niter = kwds.pop('iter', None)
        itol = kwds.pop('itol', None)
        if npts is None: npts = float('inf')
        if ipts is None: ipts = npts
        if niter is None: niter = 1
        if itol is None: itol = 1e-15
        if ipts == float('inf') or ipts < 1:
            msg = 'ipts must be a positive integer'
            raise ValueError(msg)
        if npts < ipts:
            msg = 'ipts must be less than npts'
            raise ValueError(msg)
        smap = kwds.pop('smap', None)
        kwds_ = {} # kwds for _expected
        keys = ('reducer','dist','tight','clip') # keys for __call__
        kwds = {k:v for k,v in kwds.items() if k in keys or kwds_.update({k:v})}

        import numpy as np
        from mystic.abstract_solver import AbstractSolver
        from mystic.termination import CollapseAt
        solver = AbstractSolver(self.axes or 1) #XXX: axis?
        solver.SetTermination(CollapseAt(tolerance=itol, generations=niter))
        solver.SetEvaluationLimits(maxiter=npts-1) # max num of samples
        solver.SetGenerationMonitor(m)
        while not solver.Terminated() and N != npts:
            N += ipts                       #XXX: or self.axes?
            N = min(npts, N)
            ave = self.__call__(archive=db, axis=axis, npts=N, map=smap, **kwds)
            # update stepmon with the sampled values
            xs = list(db.values())
            xtype = type(xs[-1]) # is tuple or float
            multi = hasattr(xtype, '__len__')
            xs = np.array(xs)
            var = xtype(np.nanvar(xs, axis=0)) if multi else np.nanvar(xs)
            if xs.ndim == 1: xs = np.atleast_2d(xs).T
            ys = np.arange(1, xs.shape[0]+1)
            y_ = ys - np.cumsum(np.isnan(xs), axis=0).T # nan-adjusted
            solver._stepmon._x = xs = (np.nancumsum(xs, axis=0).T/y_).T.tolist()
            solver._stepmon._y = ys = ys.tolist()
            del y_; ave = xtype(xs[-1]) if multi else xs[-1][0]
            mon(N, ave) #XXX: or use (ave, N)?

        if verbose and len(m) > size:
            print(solver.Terminated(info=True)) #XXX: also disp=True?
            print("%s: %s @ %s" % (label, ave, N))

        # downselect ave to the specified axis, where relevant
        if isinstance(ave, tuple): # self.axes is not None
            for i,me in enumerate(ave):
                self._ave[i] = me
                self._var[i] = var[i]
                self._expect[i] = None
                self._err[i] = None
        else:
            ax = None if self.axes is None else axis
            self._ave[ax] = ave
            self._var[ax] = var
            self._expect[ax] = None
            self._err[ax] = None

        if full is None: # short-circuit the solver
            return ave
        # solve for params that yield expected value
        if self.axes is None or axis is not None:
            # solve for expected value of objective (in measure space)
            solver = self._expected(axis, **kwds_)
            ax = None if self.axes is None else axis
            self._expect[ax] = solver
            self._err[ax] = me = abs(self._ave[ax] - solver.bestEnergy)
            if verbose:
                print("%s: misfit = %s, var = %s" % (ax, me, self._var[ax]))
            if full: return solver
            return ave #NOTE: within misfit of solver.bestEnergy
        # else axis is None
        solvers = self._expected(axis, **kwds_)
        for ax,solver in enumerate(solvers):
            self._expect[ax] = solver
            self._err[ax] = me = abs(self._ave[ax] - solver.bestEnergy)
            if verbose:
                print("%s: misfit = %s, var = %s" % (ax, me, self._var[ax]))
        if full: return solvers
        return ave #NOTE: within misfit of solver.bestEnergy

    def _expected(self, axis=None, **kwds):
        """find the expected value of the statistical quantity

    Input:
        axis: int, the index of y on which to find value (all, by default)
        axmap: map instance, to execute each axis in parallel (None, by default)

    Additional Input:
        solver: mystic.solver instance [default: DifferentialEvolutionSolver2]
        npop: population size [default: None]
        id: a unique identifier for the solver [default: None]
        nested: mystic.solver instance [default: None], for ensemble solvers
        x0: initial parameter guess [default: use RandomInitialPoints]
        maxiter: max number of iterations [default: defined in solver]
        maxfun: max number of objective evaluations [default: defined in solver]
        evalmon: mystic.monitor instance [default: Monitor], for evaluations
        stepmon: mystic.monitor instance [default: Monitor], for iterations
        map: pathos map instance for solver.SetMapper [default: None]
        save: iteration frequency to save solver [default: None]
        opts: dict of configuration options for solver.Solve [default: {}]

    Returns:
        solver instance with solved expected value of the statistical quantity
        """
        axmap = kwds.pop('axmap', map) #NOTE: was _ThreadPool.map w/ join
        if axmap is None: axmap = map
        self.kwds.update(**kwds) #FIXME: good idea???
        if self.axes is None or axis is not None:
            #FIXME: enable user-provided (kpen,ftol)?
            # set penalty at same scale as expected value
            kpen = self._ave[axis]
            from mystic.penalty import linear_equality
            penalty = linear_equality(lambda rv, ave: abs(ave - self._cost[axis]), kwds={'ave':self._ave[axis]}, k=kpen)(lambda rv: 0.)
            # stop at exact cost, however if noisy stop within variance
            ftol = 1e-16 if self.samples is None else self._var[axis]
            from mystic.termination import VTR
            stop = VTR(ftol, self._ave[axis])
            # solve for expected value of objective (in measure space)
            def objective(rv):
                cost = self._cost[axis] = self.objective(rv, axis)
                return cost
            #self._invalid, invalid = float('nan'), self._invalid
            solver = self.solve(objective, penalty=penalty, stop=stop)
            #self._invalid = invalid
            return solver
        # else axis is None
        expected = tuple(axmap(self._expected, range(self.axes)))
        return expected #FIXME: don't accept "uphill" moves?

    def upper_bound(self, axis=None, **kwds):
        """find the upper bound on the statistical quantity

    Input:
        axis: int, the index of y on which to find bound (all, by default)
        axmap: map instance, to execute each axis in parallel (None, by default)
        instance: bool, if True, return the solver instance (False, by default)

    Additional Input:
        solver: mystic.solver instance [default: DifferentialEvolutionSolver2]
        npop: population size [default: None]
        id: a unique identifier for the solver [default: None]
        nested: mystic.solver instance [default: None], for ensemble solvers
        x0: initial parameter guess [default: use RandomInitialPoints]
        maxiter: max number of iterations [default: defined in solver]
        maxfun: max number of objective evaluations [default: defined in solver]
        evalmon: mystic.monitor instance [default: Monitor], for evaluations
        stepmon: mystic.monitor instance [default: Monitor], for iterations
        map: pathos map instance for solver.SetMapper [default: None]
        save: iteration frequency to save solver [default: None]
        opts: dict of configuration options for solver.Solve [default: {}]

    Returns:
        upper bound on the statistical quantity, for the specified axis
        """
        full = kwds.pop('instance', False)
        #self._upper.clear() #XXX: good idea?
        if self.axes is None or axis is not None:
            # solve for upper bound of objective (in measure space)
            solver = self._upper_bound(axis, **kwds)
            self._upper[None if self.axes is None else axis] = solver
            if full: return solver
            return solver.bestEnergy
        # else axis is None
        solvers = self._upper_bound(axis, **kwds)
        for i,solver in enumerate(solvers):
            self._upper[i] = solver
        if full: return solvers
        return tuple(i.bestEnergy for i in solvers)

    def _upper_bound(self, axis=None, **kwds):
        """find the upper bound on the statistical quantity

    Input:
        axis: int, the index of y on which to find bound (all, by default)
        axmap: map instance, to execute each axis in parallel (None, by default)

    Additional Input:
        solver: mystic.solver instance [default: DifferentialEvolutionSolver2]
        npop: population size [default: None]
        id: a unique identifier for the solver [default: None]
        nested: mystic.solver instance [default: None], for ensemble solvers
        x0: initial parameter guess [default: use RandomInitialPoints]
        maxiter: max number of iterations [default: defined in solver]
        maxfun: max number of objective evaluations [default: defined in solver]
        evalmon: mystic.monitor instance [default: Monitor], for evaluations
        stepmon: mystic.monitor instance [default: Monitor], for iterations
        map: pathos map instance for solver.SetMapper [default: None]
        save: iteration frequency to save solver [default: None]
        opts: dict of configuration options for solver.Solve [default: {}]

    Returns:
        solver instance with solved upper bound on the statistical quantity
        """
        axmap = kwds.pop('axmap', map) #NOTE: was _ThreadPool.map w/ join
        if axmap is None: axmap = map
        self.kwds.update(**kwds) #FIXME: good idea???
        if self.axes is None or axis is not None:
            # solve for upper bound of objective (in measure space)
            self._invalid *= -1
            solver = self.solve(lambda rv: -self.objective(rv, axis))
            self._invalid *= -1
            return solver
        # else axis is None
        upper = tuple(axmap(self._upper_bound, range(self.axes)))
        return upper #FIXME: don't accept "uphill" moves?

    def lower_bound(self, axis=None, **kwds):
        """find the lower bound on the statistical quantity

    Input:
        axis: int, the index of y on which to find bound (all, by default)
        axmap: map instance, to execute each axis in parallel (None, by default)
        instance: bool, if True, return the solver instance (False, by default)

    Additional Input:
        solver: mystic.solver instance [default: DifferentialEvolutionSolver2]
        npop: population size [default: None]
        id: a unique identifier for the solver [default: None]
        nested: mystic.solver instance [default: None], for ensemble solvers
        x0: initial parameter guess [default: use RandomInitialPoints]
        maxiter: max number of iterations [default: defined in solver]
        maxfun: max number of objective evaluations [default: defined in solver]
        evalmon: mystic.monitor instance [default: Monitor], for evaluations
        stepmon: mystic.monitor instance [default: Monitor], for iterations
        map: pathos map instance for solver.SetMapper [default: None]
        save: iteration frequency to save solver [default: None]
        opts: dict of configuration options for solver.Solve [default: {}]

    Returns:
        lower bound on the statistical quantity, for the specified axis
        """
        full = kwds.pop('instance', False)
        #self._lower.clear() #XXX: good idea?
        if self.axes is None or axis is not None:
            # solve for lower bound of objective (in measure space)
            solver = self._lower_bound(axis, **kwds)
            self._lower[None if self.axes is None else axis] = solver
            if full: return solver
            return solver.bestEnergy
        # else axis is None
        solvers = self._lower_bound(axis, **kwds)
        for i,solver in enumerate(solvers):
            self._lower[i] = solver
        if full: return solvers
        return tuple(i.bestEnergy for i in solvers)

    def _lower_bound(self, axis=None, **kwds):
        """find the lower bound on the statistical quantity

    Input:
        axis: int, the index of y on which to find bound (all, by default)
        axmap: map instance, to execute each axis in parallel (None, by default)

    Additional Input:
        solver: mystic.solver instance [default: DifferentialEvolutionSolver2]
        npop: population size [default: None]
        id: a unique identifier for the solver [default: None]
        nested: mystic.solver instance [default: None], for ensemble solvers
        x0: initial parameter guess [default: use RandomInitialPoints]
        maxiter: max number of iterations [default: defined in solver]
        maxfun: max number of objective evaluations [default: defined in solver]
        evalmon: mystic.monitor instance [default: Monitor], for evaluations
        stepmon: mystic.monitor instance [default: Monitor], for iterations
        map: pathos map instance for solver.SetMapper [default: None]
        save: iteration frequency to save solver [default: None]
        opts: dict of configuration options for solver.Solve [default: {}]

    Returns:
        solver instance with solved lower bound on the statistical quantity
        """
        axmap = kwds.pop('axmap', map) #NOTE: was _ThreadPool.map w/ join
        if axmap is None: axmap = map
        self.kwds.update(**kwds) #FIXME: good idea???
        if self.axes is None or axis is not None:
            # solve for lower bound of objective (in measure space)
            return self.solve(lambda rv: self.objective(rv, axis))
        # else axis is None
        lower = tuple(axmap(self._lower_bound, range(self.axes)))
        return lower #FIXME: don't accept "uphill" moves?

    # --- func ---
    def objective(self, rv, axis=None):
        """calculate the statistical quantity, under uncertainty

    Input:
        rv: list of input parameters
        axis: int, the index of output to calculate (all, by default)

    Returns:
        the statistical quantity for the specified axis

    NOTE:
        respects constraints on input parameters and product measure
        """
        return NotImplemented

    def __call__(self, axis=None, reducer=None, **kwds):
        """apply the reducer to the sampled statistical quantity

    Input:
        axis: int, the index of y on which to find quantity (all, by default)
        reducer: function, reduces a list to a single value (mean, by default)

    Further Input:
        archive: the archive (str name for a new archive, or archive instance)
        dist: a mystic.tools.Distribution instance (or list of Distributions)
        tight: if True, apply bounds concurrent with other constraints
        clip: if True, clip at bounds, else resample [default = False]
        smap: map instance, to sample the quantity in parallel [default = None]
        npts: number of sample points [default = 10000]

    Returns:
        sampled statistical quantity, for the specified axis, reduced to a float
        """
        #XXX: return what? "energy and solution?" reduced?
        if 'map' in kwds and 'smap' in kwds:
            msg = "__call__() can either accept 'smap' or 'map', not both"
            raise TypeError(msg)
        tight = kwds.pop('tight', True) #XXX: better False?
        archive = kwds.pop('archive', None)
        if archive is None: archive = dict_archive()
        elif type(archive) in (str, (u''.__class__)):
            archive = read(archive, type=file_archive)
        smap = kwds.pop('smap', kwds.pop('map', None))
        if smap is None: smap = map #NOTE: was _ThreadPool.map w/ join
        npts = kwds.get('npts', None)
        if npts is None: kwds.pop('npts', None)
        fobj = cached(archive=archive)(self.objective) #XXX: bad idea?
        self._pts = fobj.__cache__() #XXX: also bad idea? include from *_bounds?
        if tight:
            from mystic.constraints import and_, boundsconstrain
            bounds = boundsconstrain(self.lb, self.ub) #XXX: symbolic?
            constraint = and_(self.constraint, bounds, onfail=bounds)
        else:
            constraint = self.constraint
        objective = lambda rv: fobj(constraint(rv), axis=axis) # penalty?
        _pts = len(self._pts)
        pts = npts - _pts
        self._invalid, invalid = float('nan'), self._invalid
        if pts > 0: # need to sample some new points
            kwds['npts'] = pts
            s = random_samples(self.lb, self.ub, **kwds).T
            if _pts: #NOTE: s = [(...),(...)] or [...]
                s = list(self._pts.values()) + list(smap(objective, s))
            else:
                s = list(smap(objective, s))
        elif pts == 0:
            s = list(self._pts.values())
        else: # randomly choose points from archive
            import numpy as np
            s = np.random.choice(range(_pts), size=npts, replace=False)
            s = np.array(list(self._pts.values()))[s].tolist()
        self._invalid = invalid
        '''
        print(s[-1])
        # mark error (nan/inf) in keys with nan in values
        import numpy as np
        keys = list(self._pts.keys())
        bad = np.where(np.isfinite(np.array(keys).sum(axis=-1)) == False)[0]
        for k in bad:
            self._pts[keys[k]] = float('nan')
        # convert errors in values to nan
        bad = np.isfinite(s)
        multi = bad.ndim > 1
        if multi:
            bad = bad.all(axis=-1)
        bad = np.where(bad == False)[0]
        for k in bad:
            s[k] = tuple(float('nan') for i in s[k]) if multi else float('nan')
        '''
        # calculate expected value
        if axis is None and self.axes is not None: # apply per axis
            if reducer is None: #XXX: nanreducer?
                import numpy as np
                return tuple(np.nanmean(s, axis=0).tolist())
            #XXX: better tuple(reducer(s, axis=0).tolist()) if numpy ufunc?
            return tuple(reducer(si) for si in zip(*s))
        if axis is None:
            s = tuple(s)
        else:
            s = list(zip(*s))[axis]
        if reducer is None:
            import numpy as np
            return np.nanmean(s).tolist()
        return reducer(s)

    # --- solve ---
    def solve(self, objective, **kwds): #NOTE: single axis only
        """solve (in measure space) for bound on given objective

    Input:
        objective: cost function of the form y = objective(x)

    Additional Input:
        solver: mystic.solver instance [default: DifferentialEvolutionSolver2]
        npop: population size [default: None]
        id: a unique identifier for the solver [default: None]
        nested: mystic.solver instance [default: None], for ensemble solvers
        x0: initial parameter guess [default: use RandomInitialPoints]
        maxiter: max number of iterations [default: defined in solver]
        maxfun: max number of objective evaluations [default: defined in solver]
        evalmon: mystic.monitor instance [default: Monitor], for evaluations
        stepmon: mystic.monitor instance [default: Monitor], for iterations
        map: pathos map instance for solver.SetMapper [default: None]
        save: iteration frequency to save solver [default: None]
        opts: dict of configuration options for solver.Solve [default: {}]

    Returns:
        solver instance, after Solve has been called
        """
        #NOTE: kwds(penalty, stop) undocumented
        penalty = kwds.pop('penalty', None) # special case 'and(penalty)'
        stop = kwds.pop('stop', None) # special case 'Or(stop)'
        k = self.kwds.copy(); k.update(kwds) # overrides self.kwds
        kwds.update(k) #FIXME: good idea??? [bad in parallel???]
        lb, ub = self.lb, self.ub
        solver = kwds.get('solver', DifferentialEvolutionSolver2)
        npop = kwds.get('npop', None)
        if npop is not None:
            solver = solver(len(lb),npop)
        else:
            solver = solver(len(lb))
        solver.id = kwds.pop('id', None)
        nested = kwds.get('nested', None)
        x0 = kwds.get('x0', None)
        if nested is not None: # Buckshot/Sparsity
            solver.SetNestedSolver(nested)
        else: # DiffEv/Nelder/Powell
            if x0 is None: solver.SetRandomInitialPoints(min=lb,max=ub)
            else: solver.SetInitialPoints(x0)
        save = kwds.get('save', None)
        if save is not None:
            solver.SetSaveFrequency(save, 'Solver.pkl') #FIXME: set name
        mapper = kwds.get('map', None)
        if mapper is not None:
            solver.SetMapper(mapper) #NOTE: not Nelder/Powell
        maxiter = kwds.get('maxiter', None)
        maxfun = kwds.get('maxfun', None)
        solver.SetEvaluationLimits(maxiter,maxfun)
        evalmon = kwds.get('evalmon', None)
        evalmon = Monitor() if evalmon is None else evalmon
        solver.SetEvaluationMonitor(evalmon[:0])
        stepmon = kwds.get('stepmon', None)
        stepmon = Monitor() if stepmon is None else stepmon[:0]
        solver.SetGenerationMonitor(stepmon)
        solver.SetStrictRanges(min=lb,max=ub)#,tight=True) #XXX: tight?
        solver.SetConstraints(self.constraint)
        if penalty is not None: # add the special-case penalty
            from mystic.coupler import and_
            penalty = and_(self.penalty, penalty)
        else:
            penalty = self.penalty
        solver.SetPenalty(penalty)
        opts = kwds.get('opts', {}) #XXX: copy is necessary?
        if stop is not None: # add the special-case termination
            term = opts.get('termination', solver._termination)
            from mystic.termination import Or
            opts['termination'] = Or(stop, term)
        # solve
        solver.Solve(objective, **opts)
        if mapper is not None:
            mapper.close()
            mapper.join()
            mapper.clear() #NOTE: if used, then shut down pool
        #NOTE: debugging code
        #print("solved: %s" % solver.Solution())
        #func_bound = solver.bestEnergy
        #func_evals = solver.evaluations
        #from mystic.munge import write_support_file
        #write_support_file(solver._stepmon)
        #print("func_bound: %s" % func_bound) #NOTE: may be inverted
        #print("func_evals: %s" % func_evals)
        return solver


class ExpectedValue(BaseOUQ):

    def objective(self, rv, axis=None):
        """calculate expected value of model, under uncertainty

    Input:
        rv: list of input parameters
        axis: int, the index of output to calculate (all, by default)

    Returns:
        the expected value for the specified axis

    NOTE:
        respects constraints on input parameters and product measure

    NOTE:
        for product_measure, use sampled_expect if samples, else expect
        """
        # check constraints
        c = product_measure().load(rv, self.npts)
        if not self.cvalid(c) or not self.xvalid(rv):
            if axis is None and self.axes is not None:
                return (self._invalid,) * (self.axes or 1) #XXX:?
            return self._invalid
        # get expected value
        if axis is None and self.axes is not None:
            model = (lambda x: self.model(x, axis=i) for i in range(self.axes))
            if self.samples is None:
                return tuple(c.expect(m) for m in model)
            # else use sampled support
            return tuple(c.sampled_expect(m, self.samples, map=self.map) for m in model)
        # else, get expected value for the given axis
        if axis is None:
            model = lambda x: self.model(x)
        else:
            model = lambda x: self.model(x, axis=axis)
        if self.samples is None:
            return c.expect(model)
        return c.sampled_expect(model, self.samples, map=self.map)


class MaximumValue(BaseOUQ):

    def objective(self, rv, axis=None):
        """calculate maximum value of model, under uncertainty

    Input:
        rv: list of input parameters
        axis: int, the index of output to calculate (all, by default)

    Returns:
        the maximum value for the specified axis

    NOTE:
        respects constraints on input parameters and product measure

    NOTE:
        for product_measure, use sampled_maximum if samples, else ess_maximum
        """
        # check constraints
        c = product_measure().load(rv, self.npts)
        if not self.cvalid(c) or not self.xvalid(rv):
            if axis is None and self.axes is not None:
                return (self._invalid,) * (self.axes or 1) #XXX:?
            return self._invalid
        # get maximum value
        if axis is None and self.axes is not None:
            model = (lambda x: self.model(x, axis=i) for i in range(self.axes))
            if self.samples is None:
                return tuple(c.ess_maximum(m) for m in model)
            # else use sampled support
            return tuple(c.sampled_maximum(m, self.samples, map=self.map) for m in model)
        # else, get maximum value for the given axis
        if axis is None:
            model = lambda x: self.model(x)
        else:
            model = lambda x: self.model(x, axis=axis)
        if self.samples is None:
            from mystic.math.measures import ess_maximum #TODO: c.ess_maximum
            return ess_maximum(model, c.positions, c.weights)
        return c.sampled_maximum(model, self.samples, map=self.map)


class MinimumValue(BaseOUQ):

    def objective(self, rv, axis=None):
        """calculate minimum value of model, under uncertainty

    Input:
        rv: list of input parameters
        axis: int, the index of output to calculate (all, by default)

    Returns:
        the minimum value for the specified axis

    NOTE:
        respects constraints on input parameters and product measure

    NOTE:
        for product_measure, use sampled_minimum if samples, else ess_minimum
        """
        # check constraints
        c = product_measure().load(rv, self.npts)
        if not self.cvalid(c) or not self.xvalid(rv):
            if axis is None and self.axes is not None:
                return (self._invalid,) * (self.axes or 1) #XXX:?
            return self._invalid
        # get minimum value
        if axis is None and self.axes is not None:
            model = (lambda x: self.model(x, axis=i) for i in range(self.axes))
            if self.samples is None:
                return tuple(c.ess_minimum(m) for m in model)
            # else use sampled support
            return tuple(c.sampled_minimum(m, self.samples, map=self.map) for m in model)
        # else, get minimum value for the given axis
        if axis is None:
            model = lambda x: self.model(x)
        else:
            model = lambda x: self.model(x, axis=axis)
        if self.samples is None:
            from mystic.math.measures import ess_minimum #TODO: c.ess_minimum
            return ess_minimum(model, c.positions, c.weights)
        return c.sampled_minimum(model, self.samples, map=self.map)


class ValueAtRisk(BaseOUQ):

    def objective(self, rv, axis=None):
        """calculate value at risk of model, under uncertainty

    Input:
        rv: list of input parameters
        axis: int, the index of output to calculate (all, by default)

    Returns:
        the value at risk for the specified axis

    NOTE:
        respects constraints on input parameters and product measure

    NOTE:
        for product_measure, use sampled_ptp if samples, else ess_ptp
        """
        # check constraints
        c = product_measure().load(rv, self.npts)
        if not self.cvalid(c) or not self.xvalid(rv):
            if axis is None and self.axes is not None:
                return (self._invalid,) * (self.axes or 1) #XXX:?
            return self._invalid
        # get value at risk
        if axis is None and self.axes is not None:
            model = (lambda x: self.model(x, axis=i) for i in range(self.axes))
            if self.samples is None:
                return tuple(c.ess_ptp(m) for m in model)
            # else use sampled support
            return tuple(c.sampled_ptp(m, self.samples, map=self.map) for m in model)
        # else, get value at risk for the given axis
        if axis is None:
            model = lambda x: self.model(x)
        else:
            model = lambda x: self.model(x, axis=axis)
        if self.samples is None:
            from mystic.math.measures import ess_ptp #TODO: c.ess_ptp
            return ess_ptp(model, c.positions, c.weights)
        return c.sampled_ptp(model, self.samples, map=self.map)


class ProbOfFailure(BaseOUQ):

    def objective(self, rv, axis=None, iter=True):
        """calculate probability of failure for model, under uncertainty

    Input:
        rv: list of input parameters
        axis: int, the index of output to calculate (all, by default)
        iter: bool, if True, calculate per axis, else calculate for all axes

    Returns:
        the probability of failure for the specified axis (or all axes)

    NOTE:
        respects constraints on input parameters and product measure
        model is a function returning a boolean (True for success)

    NOTE:
        for product_measure, use sampled_pof(model) if samples, else pof(model)
        """
        # check constraints
        c = product_measure().load(rv, self.npts)
        if not self.cvalid(c) or not self.xvalid(rv):
            if axis is None and self.axes is not None:
                return (self._invalid,) * (self.axes or 1) #XXX:?
            return self._invalid
        # get probability of failure
        if iter and axis is None and self.axes is not None:
            model = (lambda x: self.model(x,axis=i) for i in range(self.axes))
            if self.samples is None:
                return tuple(c.pof(m) for m in model)
            # else use sampled support
            return tuple(c.sampled_pof(m, self.samples, map=self.map) for m in model)
        # else, get probability of failure for the given axis
        if axis is None:
            model = lambda x: self.model(x)
        else:
            model = lambda x: self.model(x, axis=axis)
        if self.samples is None:
            return c.pof(model)
        return c.sampled_pof(model, self.samples, map=self.map)

