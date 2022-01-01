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
from mystic.cache.archive import dict_archive
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
        constraint: function of the form x' = constraint(x)
        xvalid: function returning True if x == x', given constraint
        cvalid: function similar to xvalid, but with product_measure input

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
        self.constraint = kwds.get('constraint', lambda rv:rv)
        self.xvalid = kwds.get('xvalid', lambda rv:True)
        self.cvalid = kwds.get('cvalid', lambda c:True)
        self.kwds = {} #FIXME: good idea???
        self._upper = {} # saved solver instances for upper bounds #XXX: don't?
        self._lower = {} # saved solver instances for lower bounds #XXX: don't?
        self._pts = {} # saved sampled {in:out} objective #XXX: out only? ''?
        return

    # --- extrema ---
    #XXX: expected?

    def upper_bound(self, axis=None, **kwds):
        """find the upper bound on the statistical quantity

    Input:
        axis: int, the index of y on which to find bound (all, by default)
        instance: bool, if True, return the solver instance (False, by default)

    Additional Input:
        kwds: dict, with updates to the instance's stored kwds

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

    Additional Input:
        kwds: dict, with updates to the instance's stored kwds

    Returns:
        solver instance with solved upper bound on the statistical quantity
        """
        self.kwds.update(**kwds) #FIXME: good idea???
        if self.axes is None or axis is not None:
            # solve for upper bound of objective (in measure space)
            return self.solve(lambda rv: -self.objective(rv, axis))
        # else axis is None
        import multiprocess.dummy as mp #FIXME: process pickle/recursion Error
        pool = mp.Pool(self.axes)
        map = pool.map #TODO: don't hardwire map
        upper = tuple(map(self._upper_bound, range(self.axes)))
        pool.close(); pool.join()
        return upper #FIXME: don't accept "uphill" moves?

    def lower_bound(self, axis=None, **kwds):
        """find the lower bound on the statistical quantity

    Input:
        axis: int, the index of y on which to find bound (all, by default)
        instance: bool, if True, return the solver instance (False, by default)

    Additional Input:
        kwds: dict, with updates to the instance's stored kwds

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

    Additional Input:
        kwds: dict, with updates to the instance's stored kwds

    Returns:
        solver instance with solved lower bound on the statistical quantity
        """
        self.kwds.update(**kwds) #FIXME: good idea???
        if self.axes is None or axis is not None:
            # solve for lower bound of objective (in measure space)
            return self.solve(lambda rv: self.objective(rv, axis))
        # else axis is None
        import multiprocess.dummy as mp #FIXME: process pickle/recursion Error
        pool = mp.Pool(self.axes)
        map = pool.map #TODO: don't hardwire map
        lower = tuple(map(self._lower_bound, range(self.axes)))
        pool.close(); pool.join()
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

    def __call__(self, axis=None, **kwds):
        """apply the reducer to the sampled statistical quantity

    Input:
        axis: int, the index of y on which to find bound (all, by default)

    Additional Input:
        reducer: function to reduce a list to a single value (e.g. mean, max)
        dist: a mystic.tools.Distribution instance (or list of Distributions)
        npts: number of sample points [default = 10000]
        clip: if True, clip at bounds, else resample [default = False]

    Returns:
        sampled statistical quantity, for the specified axis, reduced to a float
        """
        #XXX: return what? "energy and solution?" reduced?
        reducer = kwds.pop('reducer', None)
        if kwds.get('npts', None) is None: kwds.pop('npts', None)
        s = random_samples(self.lb, self.ub, **kwds).T
        fobj = cached(archive=dict_archive())(self.objective) #XXX: bad idea?
        self._pts = fobj.__cache__() #XXX: also bad idea? include from *_bounds?
        objective = lambda rv: fobj(self.constraint(rv), axis=axis)
        import multiprocess.dummy as mp #FIXME: process pickle/recursion Error
        pool = mp.Pool() # len(s)
        map = pool.map #TODO: don't hardwire map
        s = map(objective, s) #NOTE: s = [(...),(...)] or [...]
        pool.close(); pool.join()
        if axis is None and self.axes is not None: # apply per axis
            if reducer is None:
                return tuple(sum(si)/len(si) for si in zip(*s))
            #XXX: better tuple(reducer(s, axis=0).tolist()) if numpy ufunc?
            return tuple(reducer(si) for si in zip(*s))
        s = tuple(s)
        return sum(s)/len(s) if reducer is None else reducer(s)

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
        pool: pathos.pool instance [default: None]
        maxiter: max number of iterations [default: defined in solver]
        maxfun: max number of objective evaluations [default: defined in solver]
        evalmon: mystic.monitor instance [default: Monitor], for evaluations
        stepmon: mystic.monitor instance [default: Monitor], for iterations
        opts: dict of configuration options for solver.Solve [default: {}]

    Returns:
        solver instance, after Solve has been called
        """
        kwds.update(self.kwds) #FIXME: good idea??? [bad in parallel???]
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
        mapper = kwds.get('pool', None)
        if mapper is not None:
            pool = mapper() #XXX: ThreadPool, ProcessPool, etc
            solver.SetMapper(pool.map) #NOTE: not Nelder/Powell
        maxiter = kwds.get('maxiter', None)
        maxfun = kwds.get('maxfun', None)
        solver.SetEvaluationLimits(maxiter,maxfun)
        evalmon = kwds.get('evalmon', None)
        evalmon = Monitor() if evalmon is None else evalmon
        solver.SetEvaluationMonitor(evalmon[:0])
        stepmon = kwds.get('stepmon', None)
        stepmon = Monitor() if stepmon is None else stepmon
        solver.SetGenerationMonitor(stepmon[:0])
        solver.SetStrictRanges(min=lb,max=ub)
        solver.SetConstraints(self.constraint)
        opts = kwds.get('opts', {})
        # solve
        solver.Solve(objective, **opts)
        if mapper is not None:
            pool.close()
            pool.join()
            pool.clear() #NOTE: if used, then shut down pool
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
            import numpy as np
            return np.inf
        # get expected value
        if axis is None and self.axes is not None:
            model = (lambda x: self.model(x, axis=i) for i in range(self.axes))
            if self.samples is None:
                return tuple(c.expect(m) for m in model)
            # else use sampled support
            return tuple(c.sampled_expect(m, self.samples) for m in model)
        # else, get expected value for the given axis
        if axis is None:
            model = lambda x: self.model(x)
        else:
            model = lambda x: self.model(x, axis=axis)
        if self.samples is None:
            return c.expect(model)
        return c.sampled_expect(model, self.samples)


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
            import numpy as np
            return np.inf
        # get maximum value
        if axis is None and self.axes is not None:
            model = (lambda x: self.model(x, axis=i) for i in range(self.axes))
            if self.samples is None:
                return tuple(c.ess_maximum(m) for m in model)
            # else use sampled support
            return tuple(c.sampled_maximum(m, self.samples) for m in model)
        # else, get maximum value for the given axis
        if axis is None:
            model = lambda x: self.model(x)
        else:
            model = lambda x: self.model(x, axis=axis)
        if self.samples is None:
            from mystic.math.measures import ess_maximum #TODO: c.ess_maximum
            return ess_maximum(model, c.positions, c.weights)
        return c.sampled_maximum(model, self.samples)


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
            import numpy as np
            return np.inf
        # get minimum value
        if axis is None and self.axes is not None:
            model = (lambda x: self.model(x, axis=i) for i in range(self.axes))
            if self.samples is None:
                return tuple(c.ess_minimum(m) for m in model)
            # else use sampled support
            return tuple(c.sampled_minimum(m, self.samples) for m in model)
        # else, get minimum value for the given axis
        if axis is None:
            model = lambda x: self.model(x)
        else:
            model = lambda x: self.model(x, axis=axis)
        if self.samples is None:
            from mystic.math.measures import ess_minimum #TODO: c.ess_minimum
            return ess_minimum(model, c.positions, c.weights)
        return c.sampled_minimum(model, self.samples)


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
            import numpy as np
            return np.inf
        # get value at risk
        if axis is None and self.axes is not None:
            model = (lambda x: self.model(x, axis=i) for i in range(self.axes))
            if self.samples is None:
                return tuple(c.ess_ptp(m) for m in model)
            # else use sampled support
            return tuple(c.sampled_ptp(m, self.samples) for m in model)
        # else, get value at risk for the given axis
        if axis is None:
            model = lambda x: self.model(x)
        else:
            model = lambda x: self.model(x, axis=axis)
        if self.samples is None:
            from mystic.math.measures import ess_ptp #TODO: c.ess_ptp
            return ess_ptp(model, c.positions, c.weights)
        return c.sampled_ptp(model, self.samples)


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
            import numpy as np
            return np.inf
        # get probability of failure
        if iter and axis is None and self.axes is not None:
            model = (lambda x: self.model(x,axis=i) for i in range(self.axes))
            if self.samples is None:
                return tuple(c.pof(m) for m in model)
            # else use sampled support
            return tuple(c.sampled_pof(m, self.samples) for m in model)
        # else, get probability of failure for the given axis
        if axis is None:
            model = lambda x: self.model(x)
        else:
            model = lambda x: self.model(x, axis=axis)
        if self.samples is None:
            return c.pof(model)
        return c.sampled_pof(model, self.samples)

