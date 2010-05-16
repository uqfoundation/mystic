#!/usr/bin/env python
#

"""
...
"""
__all__ = ['BatchGridSolver','ScattershotSolver']

from mystic.tools import Null, wrap_function

from abstract_nested_solver import AbstractNestedSolver


class BatchGridSolver(AbstractNestedSolver):
    """
...
    """
    def __init__(self, dim, nbins):
        """
Takes two initial inputs: 
    dim   -- dimensionality of the problem
    nbins -- tuple of number of bins in each dimension

All important class members are inherited from AbstractNestedSolver.
        """
        super(BatchGridSolver, self).__init__(dim, nbins=nbins)

#   def SetGridBins(self, nbins):
#   def ConfigureNestedSolver(self, **kwds):
#       """Override me for more refined behavior.
#       """
#       return

    def Solve(self, cost, termination, sigint_callback=None,
              EvaluationMonitor=Null, StepMonitor=Null, ExtraArgs=(), **kwds):
        """Minimize a function using batch grid optimization.
        ...
        """
        #allow for inputs that don't conform to AbstractSolver interface
#       callback=None        #user-supplied function, called after each step
        disp=0               #non-zero to print convergence messages
#       if kwds.has_key('callback'): callback = kwds['callback']
        if kwds.has_key('disp'): disp = kwds['disp']
        #-------------------------------------------------------------

        import signal
       #self._EARLYEXIT = False

       #FIXME: EvaluationMonitor fails for MPI, throws error for 'pp'
        from python_map import python_map
        if self._map != python_map:
            fcalls = [0] #FIXME: temporary patch for removing the following line
        else:
            fcalls, cost = wrap_function(cost, ExtraArgs, EvaluationMonitor)

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
       #from mystic.math import gridpts
        from mystic_math import gridpts
        initial_values = gridpts(bins)

        # run optimizer for each grid point
        cf = [cost for i in range(len(initial_values))]
        tm = [termination for i in range(len(initial_values))]

        # generate the local_optimize function
        local_opt = """\n
def local_optimize(cost, termination, x0):
    from %s import %s as LocalSolver
    from mystic import Sow

    stepmon = Sow()
    evalmon = Sow()

    ndim = len(x0)

    solver = LocalSolver(ndim)
    solver.SetInitialPoints(x0)
""" % (self._solver.__module__, self._solver.__name__)
        if self._useStrictRange:
            local_opt += """\n
    solver.SetStrictRanges(min=%s, max=%s)
""" % (str(lower), str(upper))
        local_opt += """\n
    solver.SetEvaluationLimits(%s, %s)
    solver.Solve(cost, termination, StepMonitor=stepmon, EvaluationMonitor=evalmon, disp=%s)
    solved_params = solver.Solution()
    solved_energy = solver.bestEnergy
    return solved_params, solved_energy, stepmon, evalmon
""" % (str(self._maxiter), str(self._maxfun), str(disp))
        exec local_opt

        # map:: params, energy, smon, emon = local_optimize(cost,term,x0)
        mapconfig = dict(nnodes=self._nnodes, launcher=self._launcher, \
                         mapper=self._mapper, queue=self._queue, \
                         timelimit=self._timelimit, \
                         ncpus=self._ncpus, servers=self._servers)
        results = self._map(local_optimize, cf, tm, initial_values, **mapconfig)

        # get the results with the lowest energy
        best = list(results[0][0]), results[0][1]
        bestpath = results[0][2]
        besteval = results[0][3]
        func_evals = len(besteval.y)
        for result in results[1:]:
          func_evals += len((result[3]).y)  # add function evaluations
          if result[1] < best[1]: # compare energy
            best = list(result[0]), result[1]
            bestpath = result[2]
            besteval = result[3]

        # return results to internals
        self.bestSolution = best[0]
        self.bestEnergy = best[1]
        self.generations = len(bestpath.y)
        fcalls = [ len(besteval.y) ]

        # write 'bests' to monitors  #XXX: non-best monitors may be useful too
        for i in range(len(bestpath.y)):
            StepMonitor(bestpath.x[i], bestpath.y[i])
            #XXX: could apply callback here, or in exec'd code
        for i in range(len(besteval.y)):
            EvaluationMonitor(besteval.x[i], besteval.y[i])

        #-------------------------------------------------------------

        signal.signal(signal.SIGINT,signal.default_int_handler)

        # code below here pushes output to scipy.optimize.fmin interface
        fval = self.bestEnergy
        warnflag = 0

        # little hack to not set off Warnings when maxiter/maxfun not set
        if self._maxiter is None: self._maxiter = self.nDim * 1e8
        if self._maxfun is None: self._maxfun = self.nDim * 1e8

        if fcalls[0] >= self._maxfun:
            warnflag = 1
            if disp:
                print "Warning: Maximum number of function evaluations has "\
                      "been exceeded."
        elif self.generations >= self._maxiter:
            warnflag = 2
            if disp:
                print "Warning: Maximum number of iterations has been exceeded"
        else:
            if disp:
                print "Optimization terminated successfully."
                print "         Current function value: %f" % fval
                print "         Iterations: %d" % self.generations
                print "         Function evaluations: %d" % fcalls[0]

        return 

class ScattershotSolver(AbstractNestedSolver):
    """
...
    """
    def __init__(self, dim, npts):
        """
Takes two initial inputs: 
    dim   -- dimensionality of the problem
    npts  -- number of parallel solver instances

All important class members are inherited from AbstractNestedSolver.
        """
        super(ScattershotSolver, self).__init__(dim, npts=npts)

    def Solve(self, cost, termination, sigint_callback=None,
              EvaluationMonitor=Null, StepMonitor=Null, ExtraArgs=(), **kwds):
        """Minimize a function using scattershot optimization.
        ...
        """
        #allow for inputs that don't conform to AbstractSolver interface
#       callback=None        #user-supplied function, called after each step
        disp=0               #non-zero to print convergence messages
#       if kwds.has_key('callback'): callback = kwds['callback']
        if kwds.has_key('disp'): disp = kwds['disp']
        #-------------------------------------------------------------

        import signal
       #self._EARLYEXIT = False

       #FIXME: EvaluationMonitor fails for MPI, throws error for 'pp'
        from python_map import python_map
        if self._map != python_map:
            fcalls = [0] #FIXME: temporary patch for removing the following line
        else:
            fcalls, cost = wrap_function(cost, ExtraArgs, EvaluationMonitor)

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
       #from mystic.math import samplepts
        from mystic_math import samplepts
        initial_values = samplepts(lower,upper,npts)

        # run optimizer for each grid point
        cf = [cost for i in range(len(initial_values))]
        tm = [termination for i in range(len(initial_values))]

        # generate the local_optimize function
        local_opt = """\n
def local_optimize(cost, termination, x0):
    from %s import %s as LocalSolver
    from mystic import Sow

    stepmon = Sow()
    evalmon = Sow()

    ndim = len(x0)

    solver = LocalSolver(ndim)
    solver.SetInitialPoints(x0)
""" % (self._solver.__module__, self._solver.__name__)
        if self._useStrictRange:
            local_opt += """\n
    solver.SetStrictRanges(min=%s, max=%s)
""" % (str(lower), str(upper))
        local_opt += """\n
    solver.SetEvaluationLimits(%s, %s)
    solver.Solve(cost, termination, StepMonitor=stepmon, EvaluationMonitor=evalmon, disp=%s)
    solved_params = solver.Solution()
    solved_energy = solver.bestEnergy
    return solved_params, solved_energy, stepmon, evalmon
""" % (str(self._maxiter), str(self._maxfun), str(disp))
        exec local_opt

        # map:: params, energy, smon, emon = local_optimize(cost,term,x0)
        mapconfig = dict(nnodes=self._nnodes, launcher=self._launcher, \
                         mapper=self._mapper, queue=self._queue, \
                         timelimit=self._timelimit, \
                         ncpus=self._ncpus, servers=self._servers)
        results = self._map(local_optimize, cf, tm, initial_values, **mapconfig)

        # get the results with the lowest energy
        best = list(results[0][0]), results[0][1]
        bestpath = results[0][2]
        besteval = results[0][3]
        func_evals = len(besteval.y)
        for result in results[1:]:
          func_evals += len((result[3]).y)  # add function evaluations
          if result[1] < best[1]: # compare energy
            best = list(result[0]), result[1]
            bestpath = result[2]
            besteval = result[3]

        # return results to internals
        self.bestSolution = best[0]
        self.bestEnergy = best[1]
        self.generations = len(bestpath.y)
        fcalls = [ len(besteval.y) ]

        # write 'bests' to monitors  #XXX: non-best monitors may be useful too
        for i in range(len(bestpath.y)):
            StepMonitor(bestpath.x[i], bestpath.y[i])
            #XXX: could apply callback here, or in exec'd code
        for i in range(len(besteval.y)):
            EvaluationMonitor(besteval.x[i], besteval.y[i])

        #-------------------------------------------------------------

        signal.signal(signal.SIGINT,signal.default_int_handler)

        # code below here pushes output to scipy.optimize.fmin interface
        fval = self.bestEnergy
        warnflag = 0

        # little hack to not set off Warnings when maxiter/maxfun not set
        if self._maxiter is None: self._maxiter = self.nDim * 1e8
        if self._maxfun is None: self._maxfun = self.nDim * 1e8

        if fcalls[0] >= self._maxfun:
            warnflag = 1
            if disp:
                print "Warning: Maximum number of function evaluations has "\
                      "been exceeded."
        elif self.generations >= self._maxiter:
            warnflag = 2
            if disp:
                print "Warning: Maximum number of iterations has been exceeded"
        else:
            if disp:
                print "Optimization terminated successfully."
                print "         Current function value: %f" % fval
                print "         Iterations: %d" % self.generations
                print "         Function evaluations: %d" % fcalls[0]

        return 


if __name__=='__main__':
    help(__name__)

# end of file
