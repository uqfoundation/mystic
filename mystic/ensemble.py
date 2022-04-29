#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Solvers
=======

This module contains a collection of optimization routines that use "map"
to distribute several optimizer instances over parameter space. Each
solver accepts a imported solver object as the "nested" solver, which
becomes the target of the map function.

The set of solvers built on mystic's AbstractEnsembleSolver are::
   LatticeSolver -- start from center of N grid points
   BuckshotSolver -- start from N random points in parameter space
   SparsitySolver -- start from N points sampled in sparse regions of space


Usage
=====

See `mystic.examples.buckshot_example06` for an example of using
BuckshotSolver. See `mystic.examples.lattice_example06`
or an example of using LatticeSolver.

All solvers included in this module provide the standard signal handling.
For more information, see `mystic.mystic.abstract_solver`.
"""
__all__ = ['LatticeSolver','BuckshotSolver','SparsitySolver', \
           'lattice','buckshot','sparsity']

from mystic.tools import unpair

from mystic.abstract_ensemble_solver import AbstractEnsembleSolver


class LatticeSolver(AbstractEnsembleSolver):
    """
parallel mapped optimization starting from the centers of N grid points
    """
    def __init__(self, dim, nbins=8):
        """
Takes two initial inputs: 
    dim   -- dimensionality of the problem
    nbins -- tuple of number of bins in each dimension

All important class members are inherited from AbstractEnsembleSolver.
        """
        super(LatticeSolver, self).__init__(dim, nbins=nbins)
        from mystic.termination import NormalizedChangeOverGeneration
        convergence_tol = 1e-4
        self._termination = NormalizedChangeOverGeneration(convergence_tol)

    def _InitialPoints(self):
        """Generate a grid of starting points for the ensemble of optimizers"""
        nbins = self._nbins or self._npts
        if isinstance(nbins, int):
            from mystic.math.grid import randomly_bin
            nbins = randomly_bin(nbins, self.nDim, ones=True, exact=True)
        if len(self._strictMax): upper = list(self._strictMax)
        else:
            upper = list(self._defaultMax)
        if len(self._strictMin): lower = list(self._strictMin)
        else:
            lower = list(self._defaultMin)

        # generate arrays of points defining a grid in parameter space
        grid_dimensions = self.nDim
        bins = []
        for i in range(grid_dimensions):
            step = 1. * abs(upper[i] - lower[i])/nbins[i]
            bins.append( [lower[i] + (j+0.5)*step for j in range(nbins[i])] )

        # build a grid of starting points
        from mystic.math import gridpts
        return gridpts(bins, self._dist)


class BuckshotSolver(AbstractEnsembleSolver):
    """
parallel mapped optimization starting from N uniform randomly sampled points
    """
    def __init__(self, dim, npts=8):
        """
Takes two initial inputs: 
    dim   -- dimensionality of the problem
    npts  -- number of parallel solver instances

All important class members are inherited from AbstractEnsembleSolver.
        """
        super(BuckshotSolver, self).__init__(dim, npts=npts)
        from mystic.termination import NormalizedChangeOverGeneration
        convergence_tol = 1e-4
        self._termination = NormalizedChangeOverGeneration(convergence_tol)

    def _InitialPoints(self):
        """Generate a grid of starting points for the ensemble of optimizers"""
        npts = self._npts
        if len(self._strictMax): upper = list(self._strictMax)
        else:
            upper = list(self._defaultMax)
        if len(self._strictMin): lower = list(self._strictMin)
        else:
            lower = list(self._defaultMin)

        # build a grid of starting points
        from mystic.math import samplepts
        return samplepts(lower,upper,npts, self._dist)


class SparsitySolver(AbstractEnsembleSolver):
    """
parallel mapped optimization starting from N points sampled from sparse regions
    """
    def __init__(self, dim, npts=8, rtol=None):
        """
Takes three initial inputs: 
    dim   -- dimensionality of the problem
    npts  -- number of parallel solver instances
    rtol  -- size of radial tolerance for sparsity

All important class members are inherited from AbstractEnsembleSolver.
        """
        super(SparsitySolver, self).__init__(dim, npts=npts, rtol=rtol)
        from mystic.termination import NormalizedChangeOverGeneration
        convergence_tol = 1e-4
        self._termination = NormalizedChangeOverGeneration(convergence_tol)

    def _InitialPoints(self): #XXX: user can provide legacy data?
        """Generate a grid of starting points for the ensemble of optimizers"""
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
        return fillpts(lower,upper,npts, data, self._rtol, self._dist)


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
    bounds (list(tuple), default=None): list of pairs of bounds (min,max),
        one for each parameter.
    ftol (float, default=1e-4): acceptable relative error in ``cost(xopt)``
        for convergence.
    gtol (float, default=10): maximum iterations to run without improvement.
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
    map (func, default=None): a (parallel) map function ``y = map(f, x)``.
    dist (mystic.math.Distribution, default=None): generate randomness in
        ensemble starting position using the given distribution.
    step (bool, default=False): if True, enable Step within the ensemble.

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
        minb,maxb = unpair(bounds)
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
    bounds (list(tuple), default=None): list of pairs of bounds (min,max),
        one for each parameter.
    ftol (float, default=1e-4): acceptable relative error in ``cost(xopt)``
        for convergence.
    gtol (float, default=10): maximum iterations to run without improvement.
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
    map (func, default=None): a (parallel) map function ``y = map(f, x)``.
    dist (mystic.math.Distribution, default=None): generate randomness in
        ensemble starting position using the given distribution.
    step (bool, default=False): if True, enable Step within the ensemble.

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
        minb,maxb = unpair(bounds)
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
    bounds (list(tuple), default=None): list of pairs of bounds (min,max),
        one for each parameter.
    ftol (float, default=1e-4): acceptable relative error in ``cost(xopt)``
        for convergence.
    gtol (float, default=10): maximum iterations to run without improvement.
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
    map (func, default=None): a (parallel) map function ``y = map(f, x)``.
    dist (mystic.math.Distribution, default=None): generate randomness in
        ensemble starting position using the given distribution.
    step (bool, default=False): if True, enable Step within the ensemble.

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
        minb,maxb = unpair(bounds)
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
