#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2020 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
OUQ classes for calculating bounds on statistical quantities
"""
from mystic.math import almostEqual
from mystic.math.discrete import product_measure
from mystic.solvers import DifferentialEvolutionSolver2
from mystic.monitors import Monitor


class BaseOUQ(object): #XXX: redo with a "Solver" interface, like ensemble?

    def __init__(self, model, bounds, **kwds):
        self.npts = bounds.n  # (2,1,1)
        self.lb = bounds.lower
        self.ub = bounds.upper
        self.model = model
        self.axes = getattr(model, 'ny', None) #FIXME: in kwds?, ny
        rnd = getattr(model, 'rnd', True) #FIXME: ?, rnd
        self.samples = kwds.get('samples', None) if rnd else None
        self.constraint = kwds.get('constraint', lambda rv:rv)
        self.xvalid = kwds.get('xvalid', lambda rv:True)
        self.cvalid = kwds.get('cvalid', lambda c:True)
        self.kwds = {} #FIXME: good idea???
        return

    # --- extrema ---
    #XXX: expected?

    def upper_bound(self, axis=None, **kwds):
        "find the maximum upper bound"
        self.kwds.update(**kwds) #FIXME: good idea???
        if self.axes is None or axis is not None:
            # solve for upper bound of objective (in measure space)
            return self.solve(lambda rv: -self.objective(rv, axis))
        # else axis is None
        import multiprocess.dummy as mp #FIXME: process pickle/recursion Error
        pool = mp.Pool(self.axes)
        map = pool.map #TODO: don't hardwire map
        upper = tuple(map(self.upper_bound, range(self.axes)))
        pool.close(); pool.join()
        return upper #FIXME: don't accept "uphill" moves?

    def lower_bound(self, axis=None, **kwds):
        "find the minimum lower bound"
        self.kwds.update(**kwds) #FIXME: good idea???
        if self.axes is None or axis is not None:
            # solve for lower bound of objective (in measure space)
            return self.solve(lambda rv: self.objective(rv, axis))
        # else axis is None
        import multiprocess.dummy as mp #FIXME: process pickle/recursion Error
        pool = mp.Pool(self.axes)
        map = pool.map #TODO: don't hardwire map
        lower = tuple(map(self.lower_bound, range(self.axes)))
        pool.close(); pool.join()
        return lower #FIXME: don't accept "uphill" moves?

    # --- func ---
    def objective(self, rv, axis=None):
        return NotImplemented

    # --- solve ---
    def solve(self, objective, **kwds): #NOTE: single axis only
        "solve for optimal solution (in measure space), given an objective"
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
        """get expected value from model under uncertainty for npts

        respects constraints on expected mean and variance of inputs
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
        model = lambda x: self.model(x, axis=axis)
        if self.samples is None:
            return c.expect(model)
        return c.sampled_expect(model, self.samples)


class MaximumValue(BaseOUQ):
    def objective(self, rv, axis=None):
        """get maximum value from model under uncertainty for npts

        respects constraints on expected mean and variance of inputs
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
        model = lambda x: self.model(x, axis=axis)
        if self.samples is None:
            from mystic.math.measures import ess_maximum #TODO: c.ess_maximum
            return ess_maximum(model, c.positions, c.weights)
        return c.sampled_maximum(model, self.samples)


class MinimumValue(BaseOUQ):
    def objective(self, rv, axis=None):
        """get minimum value from model under uncertainty for npts

        respects constraints on expected mean and variance of inputs
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
        model = lambda x: self.model(x, axis=axis)
        if self.samples is None:
            from mystic.math.measures import ess_minimum #TODO: c.ess_minimum
            return ess_minimum(model, c.positions, c.weights)
        return c.sampled_minimum(model, self.samples)


class ValueAtRisk(BaseOUQ):
    def objective(self, rv, axis=None):
        """get value at risk from model under uncertainty for npts

        respects constraints on expected mean and variance of T
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
        model = lambda x: self.model(x, axis=axis)
        if self.samples is None:
            from mystic.math.measures import ess_ptp #TODO: c.ess_ptp
            return ess_ptp(model, c.positions, c.weights)
        return c.sampled_ptp(model, self.samples)


class ProbOfFailure(BaseOUQ):
    def objective(self, rv, axis=None, iter=True):
        """get probability of failure from model under uncertainty for npts

        NOTE: model is a function returning a boolean (True for success)
        respects constraints on expected mean and variance of T
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
        model = lambda x: self.model(x, axis=axis)
        if self.samples is None:
            return c.pof(model)
        return c.sampled_pof(model, self.samples)

