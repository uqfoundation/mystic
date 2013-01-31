#!/usr/bin/env python
#

"""
Solvers
=======

This module contains a collection of optimization that use map-reduce
to distribute several optimizer instances over parameter space. Each
solver accepts a imported solver object as the "nested" solver, which
becomes the target of the map function.

The set of solvers built on mystic's AbstractNestdSolver are::
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

from mystic.tools import wrap_function

from abstract_nested_solver import AbstractNestedSolver


class LatticeSolver(AbstractNestedSolver):
    """
parallel mapped optimization starting from the center of N grid points
    """
    def __init__(self, dim, nbins):
        """
Takes two initial inputs: 
    dim   -- dimensionality of the problem
    nbins -- tuple of number of bins in each dimension

All important class members are inherited from AbstractNestedSolver.
        """
        super(LatticeSolver, self).__init__(dim, nbins=nbins)

#   def SetGridBins(self, nbins):
#   def ConfigureNestedSolver(self, **kwds):
#       """Override me for more refined behavior.
#       """
#       return

    def Solve(self, cost, termination, sigint_callback=None,
                                       ExtraArgs=(), **kwds):
        """Minimize a function using batch grid optimization.

Description:

    Uses parallel mapping of solvers on a regular grid to find the
    minimum of a function of one or more variables.

Inputs:

    cost -- the Python function or method to be minimized.
    termination -- callable object providing termination conditions.

Additional Inputs:

    sigint_callback -- callback function for signal handler.
    ExtraArgs -- extra arguments for cost.

Further Inputs:

    callback -- an optional user-supplied function to call after each
        iteration.  It is called as callback(xk), where xk is the
        current parameter vector.                           [default = None]
    disp -- non-zero to print convergence messages.         [default = 0]
        """
        #allow for inputs that don't conform to AbstractSolver interface
#       callback=None        #user-supplied function, called after each step
        disp=0               #non-zero to print convergence messages
#       if kwds.has_key('callback'): callback = kwds['callback']
        if kwds.has_key('disp'): disp = kwds['disp']
        if disp in ['verbose', 'all']: verbose = True
        else: verbose = False
        # backward compatibility
        if kwds.has_key('EvaluationMonitor'): \
           self._evalmon = kwds['EvaluationMonitor']
        if kwds.has_key('StepMonitor'): \
           self._stepmon = kwds['StepMonitor']
       #if kwds.has_key('constraints'): \
       #   self._constraints = kwds['constraints']
       #if not self._constraints: self._constraints = lambda x: x
        #-------------------------------------------------------------

        import signal
       #self._EARLYEXIT = False

       #FIXME: EvaluationMonitor fails for MPI, throws error for 'pp'
        from python_map import python_map
        if self._map != python_map:
            self._fcalls = [0] #FIXME: temporary patch for removing the following line
        else:
            self._fcalls, cost = wrap_function(cost, ExtraArgs, self._evalmon)

        #generate signal_handler
        self._generateHandler(sigint_callback) 
        if self._handle_sigint: signal.signal(signal.SIGINT,self.signal_handler)

        #-------------------------------------------------------------

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
        initial_values = gridpts(bins)

        # run optimizer for each grid point
        cf = [cost for i in range(len(initial_values))]
        tm = [termination for i in range(len(initial_values))]
        id = range(len(initial_values))

        # generate the local_optimize function
        local_opt = """\n
def local_optimize(cost, termination, x0, rank):
    from %s import %s as LocalSolver
    from mystic.monitors import Monitor

    stepmon = Monitor()
    evalmon = Monitor()

    ndim = len(x0)

    solver = LocalSolver(ndim)
    solver.id = rank
    solver.SetInitialPoints(x0)
    solver.SetEvaluationMonitor(evalmon)
    solver.SetGenerationMonitor(stepmon)
""" % (self._solver.__module__, self._solver.__name__)
        if self._useStrictRange:
            local_opt += """\n
    solver.SetStrictRanges(min=%s, max=%s)
""" % (str(lower), str(upper))
#        local_opt += """\n
#    solver.SetConstraints(%s)
#""" % (self._solver._constraints)  #FIXME: needs to take a string
        local_opt += """\n
    solver.SetEvaluationLimits(%s, %s)
    solver.Solve(cost, termination, disp=%s)
    return solver, stepmon, evalmon
""" % (str(self._maxiter), str(self._maxfun), str(verbose))
        exec local_opt

        # map:: params, energy, smon, emon = local_optimize(cost,term,x0,id)
        mapconfig = dict(nnodes=self._nnodes, launcher=self._launcher, \
                         mapper=self._mapper, queue=self._queue, \
                         timelimit=self._timelimit, scheduler=self._scheduler, \
                         ncpus=self._ncpus, servers=self._servers)
        results = self._map(local_optimize, cf, tm, initial_values, id, **mapconfig)

        # save initial state
        self.AbstractSolver__save_state()
        # get the results with the lowest energy
        self._bestSolver = results[0][0]
        bestpath = results[0][1]
        besteval = results[0][2]
        self._total_evals = len(besteval.y)
        for result in results[1:]:
          self._total_evals += len((result[2]).y)  # add function evaluations
          if result[0].bestEnergy < self._bestSolver.bestEnergy:
            self._bestSolver = result[0]
            bestpath = result[1]
            besteval = result[2]

        # return results to internals
        self.population = self._bestSolver.population #XXX: pointer? copy?
        self.popEnergy = self._bestSolver.popEnergy #XXX: pointer? copy?
        self.bestSolution = self._bestSolver.bestSolution #XXX: pointer? copy?
        self.bestEnergy = self._bestSolver.bestEnergy
        self.trialSolution = self._bestSolver.trialSolution #XXX: pointer? copy?
        self._fcalls = [ len(besteval.y) ]
        self._maxiter = self._bestSolver._maxiter
        self._maxfun = self._bestSolver._maxfun

        # write 'bests' to monitors  #XXX: non-best monitors may be useful too
        for i in range(len(bestpath.y)):
            self._stepmon(bestpath.x[i], bestpath.y[i], self.id)
            #XXX: could apply callback here, or in exec'd code
        for i in range(len(besteval.y)):
            self._evalmon(besteval.x[i], besteval.y[i])

        #-------------------------------------------------------------

        signal.signal(signal.SIGINT,signal.default_int_handler)

        # log any termination messages
        msg = self._terminated(termination, disp=disp, info=True)
        if msg: self._stepmon.info('STOP("%s")' % msg)
        # save final state
        self.AbstractSolver__save_state(force=True)
        return 

class BuckshotSolver(AbstractNestedSolver):
    """
parallel mapped optimization starting from the N random points
    """
    def __init__(self, dim, npts):
        """
Takes two initial inputs: 
    dim   -- dimensionality of the problem
    npts  -- number of parallel solver instances

All important class members are inherited from AbstractNestedSolver.
        """
        super(BuckshotSolver, self).__init__(dim, npts=npts)

    def Solve(self, cost, termination, sigint_callback=None,
                                       ExtraArgs=(), **kwds):
        """Minimize a function using buckshot optimization.

Description:

    Uses parallel mapping of solvers on randomly selected points
    to find the minimum of a function of one or more variables.

Inputs:

    cost -- the Python function or method to be minimized.
    termination -- callable object providing termination conditions.

Additional Inputs:

    sigint_callback -- callback function for signal handler.
    ExtraArgs -- extra arguments for cost.

Further Inputs:

    callback -- an optional user-supplied function to call after each
        iteration.  It is called as callback(xk), where xk is the
        current parameter vector.                           [default = None]
    disp -- non-zero to print convergence messages.         [default = 0]
        """
        #allow for inputs that don't conform to AbstractSolver interface
#       callback=None        #user-supplied function, called after each step
        disp=0               #non-zero to print convergence messages
#       if kwds.has_key('callback'): callback = kwds['callback']
        if kwds.has_key('disp'): disp = kwds['disp']
        if disp in ['verbose', 'all']: verbose = True
        else: verbose = False
        # backward compatibility
        if kwds.has_key('EvaluationMonitor'): \
           self._evalmon = kwds['EvaluationMonitor']
        if kwds.has_key('StepMonitor'): \
           self._stepmon = kwds['StepMonitor']
       #if kwds.has_key('constraints'): \
       #   self._constraints = kwds['constraints']
       #if not self._constraints: self._constraints = lambda x: x
        #-------------------------------------------------------------

        import signal
       #self._EARLYEXIT = False

       #FIXME: EvaluationMonitor fails for MPI, throws error for 'pp'
        from python_map import python_map
        if self._map != python_map:
            self._fcalls = [0] #FIXME: temporary patch for removing the following line
        else:
            self._fcalls, cost = wrap_function(cost, ExtraArgs, self._evalmon)

        #generate signal_handler
        self._generateHandler(sigint_callback) 
        if self._handle_sigint: signal.signal(signal.SIGINT,self.signal_handler)

        #-------------------------------------------------------------

        npts = self._npts
        if len(self._strictMax): upper = list(self._strictMax)
        else:
            upper = list(self._defaultMax)
        if len(self._strictMin): lower = list(self._strictMin)
        else:
            lower = list(self._defaultMin)

        # generate a set of starting points
        from mystic.math import samplepts
        initial_values = samplepts(lower,upper,npts)

        # run optimizer for each grid point
        cf = [cost for i in range(len(initial_values))]
        tm = [termination for i in range(len(initial_values))]
        id = range(len(initial_values))

        # generate the local_optimize function
        local_opt = """\n
def local_optimize(cost, termination, x0, rank):
    from %s import %s as LocalSolver
    from mystic.monitors import Monitor

    stepmon = Monitor()
    evalmon = Monitor()

    ndim = len(x0)

    solver = LocalSolver(ndim)
    solver.id = rank
    solver.SetInitialPoints(x0)
    solver.SetEvaluationMonitor(evalmon)
    solver.SetGenerationMonitor(stepmon)
""" % (self._solver.__module__, self._solver.__name__)
        if self._useStrictRange:
            local_opt += """\n
    solver.SetStrictRanges(min=%s, max=%s)
""" % (str(lower), str(upper))
#        local_opt += """\n
#    solver.SetConstraints(%s)
#""" % (self._solver._constraints)  #FIXME: needs to take a string
        local_opt += """\n
    solver.SetEvaluationLimits(%s, %s)
    solver.Solve(cost, termination, disp=%s)
    return solver, stepmon, evalmon
""" % (str(self._maxiter), str(self._maxfun), str(verbose))
        exec local_opt

        # map:: params, energy, smon, emon = local_optimize(cost,term,x0,id)
        mapconfig = dict(nnodes=self._nnodes, launcher=self._launcher, \
                         mapper=self._mapper, queue=self._queue, \
                         timelimit=self._timelimit, scheduler=self._scheduler, \
                         ncpus=self._ncpus, servers=self._servers)
        results = self._map(local_optimize, cf, tm, initial_values, id, **mapconfig)

        # save initial state
        self.AbstractSolver__save_state()
        # get the results with the lowest energy
        self._bestSolver = results[0][0]
        bestpath = results[0][1]
        besteval = results[0][2]
        self._total_evals = len(besteval.y)
        for result in results[1:]:
          self._total_evals += len((result[2]).y)  # add function evaluations
          if result[0].bestEnergy < self._bestSolver.bestEnergy:
            self._bestSolver = result[0]
            bestpath = result[1]
            besteval = result[2]

        # return results to internals
        self.population = self._bestSolver.population #XXX: pointer? copy?
        self.popEnergy = self._bestSolver.popEnergy #XXX: pointer? copy?
        self.bestSolution = self._bestSolver.bestSolution #XXX: pointer? copy?
        self.bestEnergy = self._bestSolver.bestEnergy
        self.trialSolution = self._bestSolver.trialSolution #XXX: pointer? copy?
        self._fcalls = [ len(besteval.y) ]
        self._maxiter = self._bestSolver._maxiter
        self._maxfun = self._bestSolver._maxfun

        # write 'bests' to monitors  #XXX: non-best monitors may be useful too
        for i in range(len(bestpath.y)):
            self._stepmon(bestpath.x[i], bestpath.y[i], self.id)
            #XXX: could apply callback here, or in exec'd code
        for i in range(len(besteval.y)):
            self._evalmon(besteval.x[i], besteval.y[i])

        #-------------------------------------------------------------

        signal.signal(signal.SIGINT,signal.default_int_handler)

        # log any termination messages
        msg = self._terminated(termination, disp=disp, info=True)
        if msg: self._stepmon.info('STOP("%s")' % msg)
        # save final state
        self.AbstractSolver__save_state(force=True)
        return 

# backward compatibility
ScattershotSolver = BuckshotSolver
BatchGridSolver = LatticeSolver


if __name__=='__main__':
    help(__name__)

# end of file
