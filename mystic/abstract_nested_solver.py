#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2014 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE
#
# Abstract Nested Solver Class
"""
This module contains the base class for launching several mystic solvers
instances -- utilizing a parallel "map" function to enable parallel
computing.  This module describes the nested solver interface.  As with
the AbstractSolver, the "Solve" method must be overwritten with the derived
solver's optimization algorithm. Similar to AbstractMapSolver, a call to
self.map is required.  In many cases, a minimal function call interface for a
derived solver is provided along with the derived class.  See the following
for an example.

The default map API settings are provided within mystic, while
distributed and high-performance computing mappers and launchers
can be obtained within the "pathos" package, found here::
    - http://dev.danse.us/trac/pathos


Usage
=====

A typical call to a 'nested' solver will roughly follow this example:

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

All solvers packaged with mystic include a signal handler that
provides the following options::
    sol: Print current best solution.
    cont: Continue calculation.
    call: Executes sigint_callback, if provided.
    exit: Exits with current best solution.

Handlers are enabled with the 'enable_signal_handler' method,
and are configured through the solver's 'Solve' method.  Handlers
trigger when a signal interrupt (usually, Ctrl-C) is given while
the solver is running.  ***NOTE: The handler currently is disabled
when the solver has been launched in parallel.*** 

"""
__all__ = ['AbstractNestedSolver']


from mystic.monitors import Null
from mystic.abstract_map_solver import AbstractMapSolver


class AbstractNestedSolver(AbstractMapSolver):
    """
AbstractNestedSolver base class for mystic optimizers that are nested within
a parallel map.  This allows pseudo-global coverage of parameter space using
non-global optimizers.
    """

    def __init__(self, dim, **kwds):
        """
Takes one initial input:
    dim      -- dimensionality of the problem.

Additional inputs:
    npop     -- size of the trial solution population.      [default = 1]
    nbins    -- tuple of number of bins in each dimension.  [default = [1]*dim]
    npts     -- number of solver instances.                 [default = 1]

Important class members:
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
        super(AbstractNestedSolver, self).__init__(dim, **kwds)
       #self.signal_handler   = None
       #self._handle_sigint   = False

        # default settings for nested optimization
        nbins = [1]*dim
        if kwds.has_key('nbins'): nbins = kwds['nbins']
        if isinstance(nbins, int):
            from mystic.math.grid import randomly_bin
            nbins = randomly_bin(nbins, dim)
        self._nbins           = nbins
        npts = 1
        if kwds.has_key('npts'): npts = kwds['npts']
        self._npts            = npts
        from mystic.solvers import NelderMeadSimplexSolver
        self._solver          = NelderMeadSimplexSolver
        self._bestSolver      = None # 'best' solver (after Solve)
        self._total_evals     = 0 # total function calls (after Solve)
        return

    def SetNestedSolver(self, solver):
        """set the nested solver

input::
    - solver: a mystic solver instance (e.g. NelderMeadSimplexSolver(3) )"""
        self._solver = solver
        return

    def __get_solver_instance(self):
        """ensure the solver is a solver instance"""
        solver = self._solver

        # if a configured solver is not given, then build one of the given type
        from mystic.abstract_solver import AbstractSolver
        if isinstance(solver, AbstractSolver): # is a configured solver instance
            return solver
        if not hasattr(solver, "Solve"):       # is an Error...
            raise TypeError, "%s is not a valid solver" % solver

        # otherwise, this is a solver class and needs configuring
       #from mystic.monitors import Monitor
       #stepmon = Monitor()
       #evalmon = Monitor()
       #maxiter = 1000
       #maxfun = 1e+6
        solver = solver(self.nDim)
        solver.SetRandomInitialPoints() #FIXME: set population; will override
        if self._useStrictRange: #XXX: always, settable, or sync'd ?
            solver.SetStrictRanges(min=self._strictMin, max=self._strictMax)
        solver.SetEvaluationLimits(self._maxiter, self._maxfun)
        solver.SetEvaluationMonitor(self._evalmon) #XXX: or copy or set?
        solver.SetGenerationMonitor(self._stepmon) #XXX: or copy or set?
        solver.SetTermination(self._termination)
        solver.SetConstraints(self._constraints)
        solver.SetPenalty(self._penalty)
        return solver

    def SetInitialPoints(self, x0, radius=0.05):
        """Set Initial Points with Guess (x0)

input::
    - x0: must be a sequence of length self.nDim
    - radius: generate random points within [-radius*x0, radius*x0]
        for i!=0 when a simplex-type initial guess in required

*** this method must be overwritten ***"""
        raise NotImplementedError, "must be overwritten..."
    
    def SetRandomInitialPoints(self, min=None, max=None):
        """Generate Random Initial Points within given Bounds

input::
    - min, max: must be a sequence of length self.nDim
    - each min[i] should be <= the corresponding max[i]

*** this method must be overwritten ***"""
        raise NotImplementedError, "must be overwritten..."

    def SetMultinormalInitialPoints(self, mean, var = None):
        """Generate Initial Points from Multivariate Normal.

input::
    - mean must be a sequence of length self.nDim
    - var can be...
        None: -> it becomes the identity
        scalar: -> var becomes scalar * I
        matrix: -> the variance matrix. must be the right size!

*** this method must be overwritten ***"""
        raise NotImplementedError, "must be overwritten..."

    def CheckTermination(self, disp=False, info=False, termination=None):
        """check if the solver meets the given termination conditions

Input::
    - disp = if True, print termination statistics and/or warnings
    - info = if True, return termination message (instead of boolean)
    - termination = termination conditions to check against

Note::
    If no termination conditions are given, the solver's stored
    termination conditions will be used.
        """
        if self._bestSolver:
            solver = self._bestSolver
        else:
            solver = self
        if termination == None:
            termination = solver._termination

        # check for termination messages
        msg = termination(solver, info=True)
        lim = "EvaluationLimits with %s" % {'evaluations':solver._maxfun,
                                            'generations':solver._maxiter}

        # push solver internals to scipy.optimize.fmin interface
        if solver._fcalls[0] >= solver._maxfun and solver._maxfun is not None:
            msg = lim #XXX: prefer the default stop ?
            if disp:
                print "Warning: Maximum number of function evaluations has "\
                      "been exceeded."
        elif solver.generations >= solver._maxiter and solver._maxiter is not None:
            msg = lim #XXX: prefer the default stop ?
            if disp:
                print "Warning: Maximum number of iterations has been exceeded"
        elif msg and disp:
            print "Optimization terminated successfully."
            print "         Current function value: %f" % solver.bestEnergy
            print "         Iterations: %d" % solver.generations
            print "         Function evaluations: %d" % solver._fcalls[0]
            print "         Total Function evaluations: %d" % self._total_evals

        if info:
            return msg
        return bool(msg)


if __name__=='__main__':
    help(__name__)

# end of file
