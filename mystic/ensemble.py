#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2026 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Solvers
=======

This module contains a collection of optimization routines that use "map"
to distribute several optimizer instances over parameter space. Each
solver accepts a imported solver object as the "nested" solver, which
becomes the target of the callable map instance.

The set of solvers built on mystic's AbstractEnsembleSolver are::
   LatticeSolver -- start from center of N grid points
   BuckshotSolver -- start from N random points in parameter space
   SparsitySolver -- start from N points sampled in sparse regions of space
   ResidualSolver -- start from N points near the largest misfit to legacy data
   MixedSolver -- start from N points using a mixture of ensemble solvers

parallel mapped optimization starting from N points sampled near largest error

Usage
=====

See `mystic.examples.buckshot_example06` for an example of using
BuckshotSolver. See `mystic.examples.lattice_example06`
or an example of using LatticeSolver.

All solvers included in this module provide the standard signal handling.
For more information, see `mystic.mystic.abstract_solver`.
"""
__all__ = ['LatticeSolver','BuckshotSolver','SparsitySolver','ResidualSolver', \
           'MixedSolver', 'lattice','buckshot','sparsity','residual']

from mystic.tools import unpair

from mystic.abstract_ensemble_solver import AbstractEnsembleSolver


class LatticeSolver(AbstractEnsembleSolver):
    """
parallel mapped optimization starting from the centers of N grid points
    """
    def __init__(self, dim, nbins=8, **kwds):
        """
Takes one initial input:
    dim   -- dimensionality of the problem

Additional inputs:
    nbins -- tuple of number of bins in each dimension
    step  -- sync the ensemble after every iteration     [default = False]

All important class members are inherited from AbstractEnsembleSolver.
        """
        super(LatticeSolver, self).__init__(dim, nbins=nbins, **kwds)
        from mystic.termination import NormalizedChangeOverGeneration
        convergence_tol = 1e-4
        self._termination = NormalizedChangeOverGeneration(convergence_tol)

    def _InitialPoints(self):
        """Generate a grid of starting points for the ensemble of optimizers"""
        #XXX: depends on SetStrictRange & SetDistribution
        nbins = self._nbins or self._npts
        if len(self._strictMax): upper = list(self._strictMax)
        else:
            upper = list(self._defaultMax)
        if len(self._strictMin): lower = list(self._strictMin)
        else:
            lower = list(self._defaultMin)

        # build a grid of starting points
        from mystic.math import binnedpts
        solution = binnedpts(lower, upper, nbins, self._dist)
        self._init_solution = solution
        #for i,j in enumerate(solution): #NOTE: loop enables partial reset
        #    if self._init_solution[i] is None: self._init_solution[i] = j
        return self._init_solution


class BuckshotSolver(AbstractEnsembleSolver):
    """
parallel mapped optimization starting from N uniform randomly sampled points
    """
    def __init__(self, dim, npts=8, **kwds):
        """
Takes one initial input:
    dim   -- dimensionality of the problem

Additional inputs:
    npts  -- number of parallel solver instances
    step  -- sync the ensemble after every iteration     [default = False]

All important class members are inherited from AbstractEnsembleSolver.
        """
        super(BuckshotSolver, self).__init__(dim, npts=npts, **kwds)
        from mystic.termination import NormalizedChangeOverGeneration
        convergence_tol = 1e-4
        self._termination = NormalizedChangeOverGeneration(convergence_tol)

    def _InitialPoints(self):
        """Generate a grid of starting points for the ensemble of optimizers"""
        #XXX: depends on SetStrictRange & SetDistribution
        npts = self._npts
        if len(self._strictMax): upper = list(self._strictMax)
        else:
            upper = list(self._defaultMax)
        if len(self._strictMin): lower = list(self._strictMin)
        else:
            lower = list(self._defaultMin)

        # build a grid of starting points
        from mystic.math import samplepts
        solution = samplepts(lower, upper, npts, self._dist)
        self._init_solution = solution
        #for i,j in enumerate(solution): #NOTE: loop enables partial reset
        #    if self._init_solution[i] is None: self._init_solution[i] = j
        return self._init_solution


class SparsitySolver(AbstractEnsembleSolver):
    """
parallel mapped optimization starting from N points sampled from sparse regions
    """
    def __init__(self, dim, npts=8, rtol=None, **kwds):
        """
Takes one initial input:
    dim   -- dimensionality of the problem

Additional inputs:
    npts  -- number of parallel solver instances
    rtol  -- size of radial tolerance for sparsity
    step  -- sync the ensemble after every iteration     [default = False]

All important class members are inherited from AbstractEnsembleSolver.
        """
        super(SparsitySolver, self).__init__(dim, npts=npts, **kwds)
        from mystic.termination import NormalizedChangeOverGeneration
        convergence_tol = 1e-4
        self._termination = NormalizedChangeOverGeneration(convergence_tol)
        self._rtol = rtol

    def _InitialPoints(self): #XXX: user can provide legacy data?
        """Generate a grid of starting points for the ensemble of optimizers"""
        #XXX: depends on SetStrictRange & SetDistribution & Set*Monitor
        npts = self._npts
        if len(self._strictMax): upper = list(self._strictMax)
        else:
            upper = list(self._defaultMax)
        if len(self._strictMin): lower = list(self._strictMin)
        else:
            lower = list(self._defaultMin)

        # build a grid of starting points
        from mystic.math import fillpts
        data = self._evalmon._x if self._evalmon else []
        data += self._stepmon._x if self._stepmon else []
        #print('fillpts: %s' % data)
        solution = fillpts(lower, upper, npts, data, self._rtol, self._dist)
        self._init_solution = solution
        #for i,j in enumerate(solution): #NOTE: loop enables partial reset
        #    if self._init_solution[i] is None: self._init_solution[i] = j
        return self._init_solution


class ResidualSolver(AbstractEnsembleSolver):
    """
parallel mapped optimization starting from N points sampled near largest misfit
    """
    def __init__(self, dim, npts=8, mtol=None, func=None, **kwds):
        """
Takes one initial input:
    dim   -- dimensionality of the problem

Additional inputs:
    npts  -- number of parallel solver instances
    mtol  -- iteration tolerance solving for maximum error
    func  -- function approximating the cost function
    step  -- sync the ensemble after every iteration     [default = False]

All important class members are inherited from AbstractEnsembleSolver.
        """
        super(ResidualSolver, self).__init__(dim, npts=npts, **kwds)
        from mystic.termination import NormalizedChangeOverGeneration
        convergence_tol = 1e-4
        self._termination = NormalizedChangeOverGeneration(convergence_tol)
        self._mtol = mtol
        self._model = func

    def _InitialPoints(self): #XXX: user can provide legacy data and error?
        """Generate a grid of starting points for the ensemble of optimizers"""
        #XXX: depends on SetStrictRange & SetDistribution & Set*Monitor & SetObjective
        npts = self._npts
        if len(self._strictMax): upper = list(self._strictMax)
        else:
            upper = list(self._defaultMax)
        if len(self._strictMin): lower = list(self._strictMin)
        else:
            lower = list(self._defaultMin)

        # build a grid of starting points
        from mystic.math import errorpts
        data = self._evalmon._x if self._evalmon else []
        data += self._stepmon._x if self._stepmon else []
        #print('errorpts: %s' % data)
        if self._model is None and self._cost[1] is None:
            yval = None
        else: # yval = | model(data) - vals |
            model = self._model or self._cost[1]
            vals = self._evalmon._y if self._evalmon else []
            vals += self._stepmon._y if self._stepmon else []
            yval = list(map(model, data)) #FIXME: write to the evalmon
            from mystic.math.distance import euclidean as metric
            yval = metric(vals, yval, pair=True, dmin=2, axis=0)
            #print('vals: %s' % vals)
        #print('yval: %s' % yval)
        solution = errorpts(lower, upper, npts, data, yval, self._mtol, self._dist)
        self._init_solution = solution
        #for i,j in enumerate(solution): #NOTE: loop enables partial reset
        #    if self._init_solution[i] is None: self._init_solution[i] = j
        return self._init_solution


class MixedSolver(AbstractEnsembleSolver):
    """
parallel mapped optimization starting from N points sampled with mixed solvers
    """
    def __init__(self, dim, samp=8, **kwds):
        """
Takes one initial input:
    dim   -- dimensionality of the problem

Additional inputs:
    samp  -- dict of {ensemble solver: npts}
    step  -- sync the ensemble after every iteration     [default = False]

All important class members are inherited from AbstractEnsembleSolver.
        """
        # get the names of the available solvers
        types = [s.capitalize()+'Solver' for s in __all__ if not s.endswith('Solver')]
        npts = None
        # if samp is an int, randomly select samp from available solvers
        if not hasattr(samp, '__len__'):
            npts = samp
            from numpy import random
            samp = random.choice(range(len(types)), samp, replace=True).tolist()
            samp = [(t,(samp.count(i),)) for i,t in enumerate(types) if samp.count(i)]
        elif isinstance(samp, dict): # convert dict to list of tuples
            samp = list(samp.items())
        else: samp = list(samp)
        # get tuple of (__name__, args)
        for i,(name,args) in enumerate(samp): #XXX: what if dim in args?
            name = getattr(name,'__name__',name) + ('' if name in types else 'Solver')
            name = name[:1].upper() + name[1:]
            if not hasattr(args,'__len__'): args = (args,)
            if name == 'LatticeSolver' and not hasattr(args[0], '__len__') and len(args) > 1: args = (args,) #NOTE: handles args=[2,1,1]
            samp[i] = (name,args) #XXX: does not catch pts <= 0
        if npts is None:
            from mystic import ensemble #XXX: THIS file
            npts = sum([getattr(ensemble, name)(dim, *args)._npts for name,args in samp if hasattr(ensemble, name)])
        super(MixedSolver, self).__init__(dim, npts=npts, **kwds)
        from mystic.termination import NormalizedChangeOverGeneration
        convergence_tol = 1e-4
        self._termination = NormalizedChangeOverGeneration(convergence_tol)
        self._samp = samp #XXX: should user provide a list of solver instances?
        #XXX: can solver state be correctly captured without self._samp?

    def _InitialPoints(self):
        """Generate a grid of starting points for the ensemble of optimizers"""
        from mystic import ensemble #XXX: THIS file
        from mystic.monitors import Monitor
        from copy import deepcopy
        data = Monitor()
        for name,args in self._samp:
            solver = getattr(ensemble, name)(self.nDim, *args)
            solver._strictMax = self._strictMax
            solver._strictMin = self._strictMin
            solver._dist = self._dist
            if self._evalmon: solver._evalmon = deepcopy(self._evalmon)
            if self._stepmon: solver._stepmon = deepcopy(self._stepmon)
            #solver.SetEvaluationMonitor(deepcopy(data))  # use prior loop data
            solver._cost = self._cost #XXX: SetObjective? # copy unlinks evalmon
            data._x += solver._InitialPoints()
            data._y = [float('inf')]*len(data._x) #FIXME: ????? nan?
        #print('data: %s' % data._x)
        solution = data._x
        self._init_solution = solution
        #for i,j in enumerate(solution): #NOTE: loop enables partial reset
        #    if self._init_solution[i] is None: self._init_solution[i] = j
        return self._init_solution


def lattice(cost,ndim,nbins=8,args=(),bounds=None,ftol=1e-4,maxiter=None, \
            maxfun=None,full_output=0,disp=1,retall=0,callback=None,**kwds):
    """Minimize a function using the lattice ensemble solver.
    
Uses a lattice ensemble algorithm to find the minimum of a function of one or
more variables. Mimics the ``scipy.optimize.fmin`` interface.  Starts N solver
instances at regular intervals in parameter space, determined by *nbins*
``(N = numpy.prod(nbins); len(nbins) == ndim)``.

Args:
    cost (func): the function or method to be minimized: ``y = cost(x)``.
    ndim (int): dimensionality of the problem.
    nbins (tuple(int), default=8): total bins, or # of bins in each dimension.
    args (tuple, default=()): extra arguments for cost.
    bounds (bounds, default=None): the bounds for each parameter, given as
        a mystic.bounds instance or a list of ``(lower, upper)`` tuples.
    ftol (float, default=1e-4): acceptable relative error in ``cost(xopt)``
        for convergence.
    gtol (int, default=10): maximum iterations to run without improvement.
    maxiter (int, default=None): the maximum number of iterations to perform.
    maxfun (int, default=None): the maximum number of function evaluations.
    full_output (bool, default=False): True if fval and warnflag are desired.
    disp (bool, default=True): if True, print convergence messages.
    retall (bool, default=False): True if allvecs is desired.
    callback (func, default=None): function to call after each iteration. The
        interface is ``callback(xk)``, with xk the current parameter vector.
    solver (solver, default=None): override the default nested Solver instance.
    handler (bool, default=False): if True, enable handling interrupt signals.
    id (int, default=None): the ``id`` of the solver used in logging.
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
    map (func, default=None): a (parallel) map instance ``y = map(f, x)``.
    dist (mystic.math.Distribution, default=None): generate randomness in
        ensemble starting position using the given distribution.
    step (bool, default=False): sync the ensemble after every iteration.

Returns:
    ``(xopt, {fopt, iter, funcalls, warnflag, allfuncalls}, {allvecs})``

Notes:
    - xopt (*ndarray*): the minimizer of the cost function
    - fopt (*float*): value of cost function at minimum: ``fopt = cost(xopt)``
    - iter (*int*): number of iterations
    - funcalls (*int*): number of function calls
    - warnflag (*int*): warning flag:
        - ``1 : Maximum number of function evaluations``
        - ``2 : Maximum number of iterations``
    - allfuncalls (*int*): total function calls (for all solver instances)
    - allvecs (*list*): a list of solutions at each iteration
    """
    handler = kwds['handler'] if 'handler' in kwds else False
    from mystic.solvers import NelderMeadSimplexSolver as _solver
    if 'solver' in kwds: _solver = kwds['solver']

    from mystic.monitors import Monitor
    stepmon = kwds['itermon'] if 'itermon' in kwds else Monitor()
    evalmon = kwds['evalmon'] if 'evalmon' in kwds else Monitor()

    gtol = 10 # termination generations (scipy: 2, default: 10)
    if 'gtol' in kwds: gtol = kwds['gtol']
    if gtol: #if number of generations is provided, use NCOG
        from mystic.termination import NormalizedChangeOverGeneration
        termination = NormalizedChangeOverGeneration(ftol,gtol)
    else:
        from mystic.termination import VTRChangeOverGeneration
        termination = VTRChangeOverGeneration(ftol)

    solver = LatticeSolver(ndim,nbins)
    solver.SetNestedSolver(_solver) #XXX: skip settings for configured solver?
    solver.SetEvaluationLimits(maxiter,maxfun)
    solver.SetEvaluationMonitor(evalmon)
    solver.SetGenerationMonitor(stepmon)
    if 'id' in kwds:
        solver.id = int(kwds['id'])
    if 'dist' in kwds:
        solver.SetDistribution(kwds['dist'])
    if 'penalty' in kwds:
        solver.SetPenalty(kwds['penalty'])
    if 'constraints' in kwds:
        solver.SetConstraints(kwds['constraints'])
    if bounds is not None:
        minb,maxb = bounds.T if hasattr(bounds, 'T') else unpair(bounds)
        tight = kwds['tightrange'] if 'tightrange' in kwds else None
        clip = kwds['cliprange'] if 'cliprange' in kwds else None
        solver.SetStrictRanges(minb,maxb,tight=tight,clip=clip)

    _map = kwds['map'] if 'map' in kwds else None
    if _map: solver.SetMapper(_map)

    if handler: solver.enable_signal_handler()
    solver.Solve(cost,termination=termination,disp=disp, \
                 ExtraArgs=args,callback=callback)
    solution = solver.Solution()

    # code below here pushes output to scipy.optimize.fmin interface
    msg = solver.Terminated(disp=False, info=True)

    x = solver.bestSolution
    fval = solver.bestEnergy
    warnflag = 0
    fcalls = solver.evaluations
    all_fcalls = solver._total_evals
    iterations = solver.generations
    allvecs = solver._stepmon.x

    if fcalls >= solver._maxfun: #XXX: check against total or individual?
        warnflag = 1
    elif iterations >= solver._maxiter: #XXX: check against total or individual?
        warnflag = 2
    else: pass

    if full_output:
        retlist = x, fval, iterations, fcalls, warnflag, all_fcalls
        if retall:
            retlist += (allvecs,)
    else:
        retlist = x
        if retall:
            retlist = (x, allvecs)

    return retlist


def buckshot(cost,ndim,npts=8,args=(),bounds=None,ftol=1e-4,maxiter=None, \
             maxfun=None,full_output=0,disp=1,retall=0,callback=None,**kwds):
    """Minimize a function using the buckshot ensemble solver.
    
Uses a buckshot ensemble algorithm to find the minimum of a function of one or
more variables. Mimics the ``scipy.optimize.fmin`` interface. Starts *npts*
solver instances at random points in parameter space. 

Args:
    cost (func): the function or method to be minimized: ``y = cost(x)``.
    ndim (int): dimensionality of the problem.
    npts (int, default=8): number of solver instances.
    args (tuple, default=()): extra arguments for cost.
    bounds (bounds, default=None): the bounds for each parameter, given as
        a mystic.bounds instance or a list of ``(lower, upper)`` tuples.
    ftol (float, default=1e-4): acceptable relative error in ``cost(xopt)``
        for convergence.
    gtol (int, default=10): maximum iterations to run without improvement.
    maxiter (int, default=None): the maximum number of iterations to perform.
    maxfun (int, default=None): the maximum number of function evaluations.
    full_output (bool, default=False): True if fval and warnflag are desired.
    disp (bool, default=True): if True, print convergence messages.
    retall (bool, default=False): True if allvecs is desired.
    callback (func, default=None): function to call after each iteration. The
        interface is ``callback(xk)``, with xk the current parameter vector.
    solver (solver, default=None): override the default nested Solver instance.
    handler (bool, default=False): if True, enable handling interrupt signals.
    id (int, default=None): the ``id`` of the solver used in logging.
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
    map (func, default=None): a (parallel) map instance ``y = map(f, x)``.
    dist (mystic.math.Distribution, default=None): generate randomness in
        ensemble starting position using the given distribution.
    step (bool, default=False): sync the ensemble after every iteration.

Returns:
    ``(xopt, {fopt, iter, funcalls, warnflag, allfuncalls}, {allvecs})``

Notes:
    - xopt (*ndarray*): the minimizer of the cost function
    - fopt (*float*): value of cost function at minimum: ``fopt = cost(xopt)``
    - iter (*int*): number of iterations
    - funcalls (*int*): number of function calls
    - warnflag (*int*): warning flag:
        - ``1 : Maximum number of function evaluations``
        - ``2 : Maximum number of iterations``
    - allfuncalls (*int*): total function calls (for all solver instances)
    - allvecs (*list*): a list of solutions at each iteration
    """
    handler = kwds['handler'] if 'handler' in kwds else False
    from mystic.solvers import NelderMeadSimplexSolver as _solver
    if 'solver' in kwds: _solver = kwds['solver']

    from mystic.monitors import Monitor
    stepmon = kwds['itermon'] if 'itermon' in kwds else Monitor()
    evalmon = kwds['evalmon'] if 'evalmon' in kwds else Monitor()

    gtol = 10 # termination generations (scipy: 2, default: 10)
    if 'gtol' in kwds: gtol = kwds['gtol']
    if gtol: #if number of generations is provided, use NCOG
        from mystic.termination import NormalizedChangeOverGeneration
        termination = NormalizedChangeOverGeneration(ftol,gtol)
    else:
        from mystic.termination import VTRChangeOverGeneration
        termination = VTRChangeOverGeneration(ftol)

    solver = BuckshotSolver(ndim,npts)
    solver.SetNestedSolver(_solver) #XXX: skip settings for configured solver?
    solver.SetEvaluationLimits(maxiter,maxfun)
    solver.SetEvaluationMonitor(evalmon)
    solver.SetGenerationMonitor(stepmon)
    if 'id' in kwds:
        solver.id = int(kwds['id'])
    if 'dist' in kwds:
        solver.SetDistribution(kwds['dist'])
    if 'penalty' in kwds:
        solver.SetPenalty(kwds['penalty'])
    if 'constraints' in kwds:
        solver.SetConstraints(kwds['constraints'])
    if bounds is not None:
        minb,maxb = bounds.T if hasattr(bounds, 'T') else unpair(bounds)
        tight = kwds['tightrange'] if 'tightrange' in kwds else None
        clip = kwds['cliprange'] if 'cliprange' in kwds else None
        solver.SetStrictRanges(minb,maxb,tight=tight,clip=clip)

    _map = kwds['map'] if 'map' in kwds else None
    if _map: solver.SetMapper(_map)

    if handler: solver.enable_signal_handler()
    solver.Solve(cost,termination=termination,disp=disp, \
                 ExtraArgs=args,callback=callback)
    solution = solver.Solution()

    # code below here pushes output to scipy.optimize.fmin interface
    msg = solver.Terminated(disp=False, info=True)

    x = solver.bestSolution
    fval = solver.bestEnergy
    warnflag = 0
    fcalls = solver.evaluations
    all_fcalls = solver._total_evals
    iterations = solver.generations
    allvecs = solver._stepmon.x

    if fcalls >= solver._maxfun: #XXX: check against total or individual?
        warnflag = 1
    elif iterations >= solver._maxiter: #XXX: check against total or individual?
        warnflag = 2
    else: pass

    if full_output:
        retlist = x, fval, iterations, fcalls, warnflag, all_fcalls
        if retall:
            retlist += (allvecs,)
    else:
        retlist = x
        if retall:
            retlist = (x, allvecs)

    return retlist


def sparsity(cost,ndim,npts=8,args=(),bounds=None,ftol=1e-4,maxiter=None, \
             maxfun=None,full_output=0,disp=1,retall=0,callback=None,**kwds):
    """Minimize a function using the sparsity ensemble solver.
    
Uses a sparsity ensemble algorithm to find the minimum of a function of one or
more variables. Mimics the ``scipy.optimize.fmin`` interface. Starts *npts*
solver instances at points in parameter space where existing points are sparse. 

Args:
    cost (func): the function or method to be minimized: ``y = cost(x)``.
    ndim (int): dimensionality of the problem.
    npts (int, default=8): number of solver instances.
    args (tuple, default=()): extra arguments for cost.
    bounds (bounds, default=None): the bounds for each parameter, given as
        a mystic.bounds instance or a list of ``(lower, upper)`` tuples.
    ftol (float, default=1e-4): acceptable relative error in ``cost(xopt)``
        for convergence.
    gtol (int, default=10): maximum iterations to run without improvement.
    rtol (float, default=None): minimum acceptable distance from other points.
    maxiter (int, default=None): the maximum number of iterations to perform.
    maxfun (int, default=None): the maximum number of function evaluations.
    full_output (bool, default=False): True if fval and warnflag are desired.
    disp (bool, default=True): if True, print convergence messages.
    retall (bool, default=False): True if allvecs is desired.
    callback (func, default=None): function to call after each iteration. The
        interface is ``callback(xk)``, with xk the current parameter vector.
    solver (solver, default=None): override the default nested Solver instance.
    handler (bool, default=False): if True, enable handling interrupt signals.
    id (int, default=None): the ``id`` of the solver used in logging.
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
    map (func, default=None): a (parallel) map instance ``y = map(f, x)``.
    dist (mystic.math.Distribution, default=None): generate randomness in
        ensemble starting position using the given distribution.
    step (bool, default=False): sync the ensemble after every iteration.

Returns:
    ``(xopt, {fopt, iter, funcalls, warnflag, allfuncalls}, {allvecs})``

Notes:
    - xopt (*ndarray*): the minimizer of the cost function
    - fopt (*float*): value of cost function at minimum: ``fopt = cost(xopt)``
    - iter (*int*): number of iterations
    - funcalls (*int*): number of function calls
    - warnflag (*int*): warning flag:
        - ``1 : Maximum number of function evaluations``
        - ``2 : Maximum number of iterations``
    - allfuncalls (*int*): total function calls (for all solver instances)
    - allvecs (*list*): a list of solutions at each iteration
    """
    handler = kwds['handler'] if 'handler' in kwds else False
    from mystic.solvers import NelderMeadSimplexSolver as _solver
    if 'solver' in kwds: _solver = kwds['solver']

    from mystic.monitors import Monitor
    stepmon = kwds['itermon'] if 'itermon' in kwds else Monitor()
    evalmon = kwds['evalmon'] if 'evalmon' in kwds else Monitor()

    gtol = 10 # termination generations (scipy: 2, default: 10)
    if 'gtol' in kwds: gtol = kwds['gtol']
    if gtol: #if number of generations is provided, use NCOG
        from mystic.termination import NormalizedChangeOverGeneration
        termination = NormalizedChangeOverGeneration(ftol,gtol)
    else:
        from mystic.termination import VTRChangeOverGeneration
        termination = VTRChangeOverGeneration(ftol)
    rtol = kwds['rtol'] if 'rtol' in kwds else None #NOTE: 'data' set w/monitors

    solver = SparsitySolver(ndim,npts,rtol)
    solver.SetNestedSolver(_solver) #XXX: skip settings for configured solver?
    solver.SetEvaluationLimits(maxiter,maxfun)
    solver.SetEvaluationMonitor(evalmon)
    solver.SetGenerationMonitor(stepmon)
    if 'id' in kwds:
        solver.id = int(kwds['id'])
    if 'dist' in kwds:
        solver.SetDistribution(kwds['dist'])
    if 'penalty' in kwds:
        solver.SetPenalty(kwds['penalty'])
    if 'constraints' in kwds:
        solver.SetConstraints(kwds['constraints'])
    if bounds is not None:
        minb,maxb = bounds.T if hasattr(bounds, 'T') else unpair(bounds)
        tight = kwds['tightrange'] if 'tightrange' in kwds else None
        clip = kwds['cliprange'] if 'cliprange' in kwds else None
        solver.SetStrictRanges(minb,maxb,tight=tight,clip=clip)

    _map = kwds['map'] if 'map' in kwds else None
    if _map: solver.SetMapper(_map)

    if handler: solver.enable_signal_handler()
    solver.Solve(cost,termination=termination,disp=disp, \
                 ExtraArgs=args,callback=callback)
    solution = solver.Solution()

    # code below here pushes output to scipy.optimize.fmin interface
    msg = solver.Terminated(disp=False, info=True)

    x = solver.bestSolution
    fval = solver.bestEnergy
    warnflag = 0
    fcalls = solver.evaluations
    all_fcalls = solver._total_evals
    iterations = solver.generations
    allvecs = solver._stepmon.x

    if fcalls >= solver._maxfun: #XXX: check against total or individual?
        warnflag = 1
    elif iterations >= solver._maxiter: #XXX: check against total or individual?
        warnflag = 2
    else: pass

    if full_output:
        retlist = x, fval, iterations, fcalls, warnflag, all_fcalls
        if retall:
            retlist += (allvecs,)
    else:
        retlist = x
        if retall:
            retlist = (x, allvecs)

    return retlist


def residual(cost,ndim,npts=8,args=(),bounds=None,ftol=1e-4,maxiter=None, \
             maxfun=None,full_output=0,disp=1,retall=0,callback=None,**kwds):
    """Minimize a function using the residual ensemble solver.

Uses a residual ensemble algorithm to find the minimum of a function of one or
more variables. Mimics the ``scipy.optimize.fmin`` interface. Starts *npts*
solver instances near the maximum of an interpolated error surface, where
error is the absolute difference between a given function and existing points.

Args:
    cost (func): the function or method to be minimized: ``y = cost(x)``.
    ndim (int): dimensionality of the problem.
    npts (int, default=8): number of solver instances.
    args (tuple, default=()): extra arguments for cost.
    bounds (bounds, default=None): the bounds for each parameter, given as
        a mystic.bounds instance or a list of ``(lower, upper)`` tuples.
    ftol (float, default=1e-4): acceptable relative error in ``cost(xopt)``
        for convergence.
    gtol (int, default=10): maximum iterations to run without improvement.
    mtol (float, default=None): iteration tolerance solving for maximum error.
    maxiter (int, default=None): the maximum number of iterations to perform.
    maxfun (int, default=None): the maximum number of function evaluations.
    full_output (bool, default=False): True if fval and warnflag are desired.
    disp (bool, default=True): if True, print convergence messages.
    retall (bool, default=False): True if allvecs is desired.
    callback (func, default=None): function to call after each iteration. The
        interface is ``callback(xk)``, with xk the current parameter vector.
    solver (solver, default=None): override the default nested Solver instance.
    handler (bool, default=False): if True, enable handling interrupt signals.
    id (int, default=None): the ``id`` of the solver used in logging.
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
    map (func, default=None): a (parallel) map instance ``y = map(f, x)``.
    dist (mystic.math.Distribution, default=None): generate randomness in
        ensemble starting position using the given distribution.
    step (bool, default=False): sync the ensemble after every iteration.

Returns:
    ``(xopt, {fopt, iter, funcalls, warnflag, allfuncalls}, {allvecs})``

Notes:
    - xopt (*ndarray*): the minimizer of the cost function
    - fopt (*float*): value of cost function at minimum: ``fopt = cost(xopt)``
    - iter (*int*): number of iterations
    - funcalls (*int*): number of function calls
    - warnflag (*int*): warning flag:
        - ``1 : Maximum number of function evaluations``
        - ``2 : Maximum number of iterations``
    - allfuncalls (*int*): total function calls (for all solver instances)
    - allvecs (*list*): a list of solutions at each iteration
    """
    handler = kwds['handler'] if 'handler' in kwds else False
    from mystic.solvers import NelderMeadSimplexSolver as _solver
    if 'solver' in kwds: _solver = kwds['solver']

    from mystic.monitors import Monitor
    stepmon = kwds['itermon'] if 'itermon' in kwds else Monitor()
    evalmon = kwds['evalmon'] if 'evalmon' in kwds else Monitor()

    gtol = 10 # termination generations (scipy: 2, default: 10)
    if 'gtol' in kwds: gtol = kwds['gtol']
    if gtol: #if number of generations is provided, use NCOG
        from mystic.termination import NormalizedChangeOverGeneration
        termination = NormalizedChangeOverGeneration(ftol,gtol)
    else:
        from mystic.termination import VTRChangeOverGeneration
        termination = VTRChangeOverGeneration(ftol)
    mtol = kwds['mtol'] if 'mtol' in kwds else None #NOTE: 'data' set w/monitors

    solver = ResidualSolver(ndim,npts,mtol)
    solver.SetNestedSolver(_solver) #XXX: skip settings for configured solver?
    solver.SetEvaluationLimits(maxiter,maxfun)
    solver.SetEvaluationMonitor(evalmon)
    solver.SetGenerationMonitor(stepmon)
    if 'id' in kwds:
        solver.id = int(kwds['id'])
    if 'dist' in kwds:
        solver.SetDistribution(kwds['dist'])
    if 'penalty' in kwds:
        solver.SetPenalty(kwds['penalty'])
    if 'constraints' in kwds:
        solver.SetConstraints(kwds['constraints'])
    if bounds is not None:
        minb,maxb = bounds.T if hasattr(bounds, 'T') else unpair(bounds)
        tight = kwds['tightrange'] if 'tightrange' in kwds else None
        clip = kwds['cliprange'] if 'cliprange' in kwds else None
        solver.SetStrictRanges(minb,maxb,tight=tight,clip=clip)

    _map = kwds['map'] if 'map' in kwds else None
    if _map: solver.SetMapper(_map)

    if handler: solver.enable_signal_handler()
    solver.Solve(cost,termination=termination,disp=disp, \
                 ExtraArgs=args,callback=callback)
    solution = solver.Solution()

    # code below here pushes output to scipy.optimize.fmin interface
    msg = solver.Terminated(disp=False, info=True)

    x = solver.bestSolution
    fval = solver.bestEnergy
    warnflag = 0
    fcalls = solver.evaluations
    all_fcalls = solver._total_evals
    iterations = solver.generations
    allvecs = solver._stepmon.x

    if fcalls >= solver._maxfun: #XXX: check against total or individual?
        warnflag = 1
    elif iterations >= solver._maxiter: #XXX: check against total or individual?
        warnflag = 2
    else: pass

    if full_output:
        retlist = x, fval, iterations, fcalls, warnflag, all_fcalls
        if retall:
            retlist += (allvecs,)
    else:
        retlist = x
        if retall:
            retlist = (x, allvecs)

    return retlist


# backward compatibility
ScattershotSolver = BuckshotSolver
BatchGridSolver = LatticeSolver


if __name__=='__main__':
    help(__name__)

# end of file
