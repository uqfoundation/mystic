#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2017 The Uncertainty Quantification Foundation.
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


Usage
=====

See `mystic.examples.buckshot_example06` for an example of using
BuckshotSolver. See `mystic.examples.lattice_example06`
or an example of using LatticeSolver.

All solvers included in this module provide the standard signal handling.
For more information, see `mystic.mystic.abstract_solver`.
"""
__all__ = ['LatticeSolver','BuckshotSolver']

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
        nbins = self._nbins
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
            step = abs(upper[i] - lower[i])/nbins[i]
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


def lattice(cost,ndim,nbins=8,args=(),bounds=None,ftol=1e-4,maxiter=None, \
            maxfun=None,full_output=0,disp=1,retall=0,callback=None,**kwds):
    """Minimize a function using the lattice ensemble solver.
    
Description:

    Uses a lattice ensemble algorithm to find the minimum of
    a function of one or more variables. Mimics the scipy.optimize.fmin
    interface. Starts N solver instances at regular intervals in parameter
    space, determined by 'nbins' (N = numpy.prod(nbins); len(nbins) == ndim). 

Inputs:

    cost -- the Python function or method to be minimized.
    ndim -- dimensionality of the problem.
    nbins -- tuple of number of bins in each dimension. [default = 8 bins total]

Additional Inputs:

    args -- extra arguments for cost.
    bounds -- list - n pairs of bounds (min,max), one pair for each parameter.
    ftol -- number - acceptable relative error in cost(xopt) for convergence.
    gtol -- number - maximum number of iterations to run without improvement.
    maxiter -- number - the maximum number of iterations to perform.
    maxfun -- number - the maximum number of function evaluations.
    full_output -- number - non-zero if fval and warnflag outputs are desired.
    disp -- number - non-zero to print convergence messages.
    retall -- number - non-zero to return list of solutions at each iteration.
    callback -- an optional user-supplied function to call after each
        iteration.  It is called as callback(xk), where xk is the
        current parameter vector.
    solver -- solver - override the default nested Solver instance.
    handler -- boolean - enable/disable handling of interrupt signal.
    itermon -- monitor - override the default GenerationMonitor.
    evalmon -- monitor - override the default EvaluationMonitor.
    constraints -- an optional user-supplied function.  It is called as
        constraints(xk), where xk is the current parameter vector.
        This function must return xk', a parameter vector that satisfies
        the encoded constraints.
    penalty -- an optional user-supplied function.  It is called as
        penalty(xk), where xk is the current parameter vector.
        This function should return y', with y' == 0 when the encoded
        constraints are satisfied, and y' > 0 otherwise.
    dist -- an optional mystic.math.Distribution instance.  If provided,
        this distribution generates randomness in ensemble starting position.

Returns: (xopt, {fopt, iter, funcalls, warnflag, allfuncalls}, {allvecs})

    xopt -- ndarray - minimizer of function
    fopt -- number - value of function at minimum: fopt = cost(xopt)
    iter -- number - number of iterations
    funcalls -- number - number of function calls
    warnflag -- number - Integer warning flag:
        1 : 'Maximum number of function evaluations.'
        2 : 'Maximum number of iterations.'
    allfuncalls -- number - total function calls (for all solver instances)
    allvecs -- list - a list of solutions at each iteration

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
    if 'dist' in kwds:
        solver.SetDistribution(kwds['dist'])
    if 'penalty' in kwds:
        solver.SetPenalty(kwds['penalty'])
    if 'constraints' in kwds:
        solver.SetConstraints(kwds['constraints'])
    if bounds is not None:
        minb,maxb = unpair(bounds)
        solver.SetStrictRanges(minb,maxb)

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
    
Description:

    Uses a buckshot ensemble algorithm to find the minimum of
    a function of one or more variables. Mimics the scipy.optimize.fmin
    interface. Starts 'npts' solver instances at random points in parameter
    space. 

Inputs:

    cost -- the Python function or method to be minimized.
    ndim -- dimensionality of the problem.
    npts -- number of solver instances.

Additional Inputs:

    args -- extra arguments for cost.
    bounds -- list - n pairs of bounds (min,max), one pair for each parameter.
    ftol -- number - acceptable relative error in cost(xopt) for convergence.
    gtol -- number - maximum number of iterations to run without improvement.
    maxiter -- number - the maximum number of iterations to perform.
    maxfun -- number - the maximum number of function evaluations.
    full_output -- number - non-zero if fval and warnflag outputs are desired.
    disp -- number - non-zero to print convergence messages.
    retall -- number - non-zero to return list of solutions at each iteration.
    callback -- an optional user-supplied function to call after each
        iteration.  It is called as callback(xk), where xk is the
        current parameter vector.
    solver -- solver - override the default nested Solver instance.
    handler -- boolean - enable/disable handling of interrupt signal.
    itermon -- monitor - override the default GenerationMonitor.
    evalmon -- monitor - override the default EvaluationMonitor.
    constraints -- an optional user-supplied function.  It is called as
        constraints(xk), where xk is the current parameter vector.
        This function must return xk', a parameter vector that satisfies
        the encoded constraints.
    penalty -- an optional user-supplied function.  It is called as
        penalty(xk), where xk is the current parameter vector.
        This function should return y', with y' == 0 when the encoded
        constraints are satisfied, and y' > 0 otherwise.
    dist -- an optional mystic.math.Distribution instance.  If provided,
        this distribution generates randomness in ensemble starting position.

Returns: (xopt, {fopt, iter, funcalls, warnflag, allfuncalls}, {allvecs})

    xopt -- ndarray - minimizer of function
    fopt -- number - value of function at minimum: fopt = cost(xopt)
    iter -- number - number of iterations
    funcalls -- number - number of function calls
    warnflag -- number - Integer warning flag:
        1 : 'Maximum number of function evaluations.'
        2 : 'Maximum number of iterations.'
    allfuncalls -- number - total function calls (for all solver instances)
    allvecs -- list - a list of solutions at each iteration

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
    if 'dist' in kwds:
        solver.SetDistribution(kwds['dist'])
    if 'penalty' in kwds:
        solver.SetPenalty(kwds['penalty'])
    if 'constraints' in kwds:
        solver.SetConstraints(kwds['constraints'])
    if bounds is not None:
        minb,maxb = unpair(bounds)
        solver.SetStrictRanges(minb,maxb)

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
