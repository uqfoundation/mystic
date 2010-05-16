#!/usr/bin/env python
#
## Abstract Nested Solver Class
# by mmckerns@caltech.edu

"""
This module contains the base class for launching several mystic solvers
instances -- utilizing a parallel "map" function to enable parallel
computing.  This module describes the nested solver interface.  As with
the AbstractSolver, the "Solve" method must be overwritte with the derived
solver's optimization algorithm. Similar to AbstractMapSolver, a call to
self.map is required.  In many cases, a minimal function call interface for a
derived solver is provided along with the derived class.  See the following
for an example.

The default map API settings are provided within mystic, while
distributed and high-performance computing mappers and launchers
can be obtained within the "pathos" package, found here:
    - http://dev.danse.us/trac/pathos   (see subpackage = pyina)


Usage
=====

A typical call to a 'nested' solver will roughly follow this example:

    >>> # the function to be minimized and the initial values
    >>> from mystic.models import rosen
    >>> lb = [0.0, 0.0, 0.0]
    >>> ub = [2.0, 2.0, 2.0]
    >>> 
    >>> # get monitors and termination condition objects
    >>> from mystic.tools import Sow
    >>> stepmon = Sow()
    >>> from mystic.termination import CandidateRelativeTolerance as CRT
    >>> 
    >>> # select the parallel launch configuration
    >>> from pyina.launchers import mpirun_launcher
    >>> from pyina.ez_map import ez_map
    >>> NNODES = 4
    >>> nbins = [4,4,4]
    >>>
    >>> # instantiate and configure the solver
    >>> from mystic.scipy_optimize import NelderMeadSimplexSolver
    >>> from mystic.nested import BatchGridSolver
    >>> solver = BatchGridSolver(len(nbins), nbins)
    >>> solver.SetNestedSolver(NelderMeadSimplexSolver)
    >>> solver.SetStrictRanges(lb, ub)
    >>> solver.SetMapper(ez_map)
    >>> solver.SetLauncher(mpirun_launcher, NNODES)
    >>> solver.Solve(rosen, CRT(), StepMonitor=stepmon)
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


from mystic.tools import Null
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
    npop     -- size of the trial solution population.  [default = 1]
    nbins    -- tuple of number of bins in each dimension.  [default = [1]*dim]
    npts     -- number of solver instances.  [default = 1]

Important class members:
    nDim, nPop     = dim, npop
    generations    - an iteration counter.
    bestEnergy     - current best energy.
    bestSolution   - current best parameter set. [size = dim]
    popEnergy      - set of all trial energy solutions. [size = npop]
    population     - set of all trial parameter solutions.
        [size = dim*npop]
    energy_history - history of bestEnergy status.
        [equivalent to StepMonitor]
    signal_handler - catches the interrupt signal.
        [***disabled***]
        """
        super(AbstractNestedSolver, self).__init__(dim, **kwds)
       #self.signal_handler   = None
       #self._handle_sigint   = False

        # default settings for nested optimization
        nbins = [1]*dim
        if kwds.has_key('nbins'): nbins = kwds['nbins']
        self._nbins           = nbins
        npts = 1
        if kwds.has_key('npts'): npts = kwds['npts']
        self._npts            = npts
        from mystic.scipy_optimize import NelderMeadSimplexSolver
        self._solver          = NelderMeadSimplexSolver
        return

    def SetNestedSolver(self, solver):
        """set the nested solver

input::
    - solver: a mystic solver class (e.g. NelderMeadSimplexSolver)"""
        self._solver = solver
        return

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


if __name__=='__main__':
    help(__name__)

# end of file
