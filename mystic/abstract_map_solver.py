#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2025 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
#
# Abstract Solver Class
"""
This module contains the base class for mystic solvers that utilize
a parallel ``map`` function to enable parallel computing.  This module
describes the map solver interface. As with the ``AbstractSolver``, the
``_Step`` method must be overwritten with the derived solver's optimization
algorithm. Additionally, for the ``AbstractMapSolver``, a call to ``map`` is
required. In addition to the class interface, a simple function interface for
a derived solver class is often provided. For an example, see the following.

The default map API settings are provided within mystic, while distributed
and parallel computing maps can be obtained from the ``pathos`` package 
(http://dev.danse.us/trac/pathos).

Examples:

    A typical call to a 'map' solver will roughly follow this example:

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
    >>> npts = 20
    >>>
    >>> # instantiate and configure the solver
    >>> from mystic.solvers import BuckshotSolver
    >>> solver = BuckshotSolver(len(lb), npts)
    >>> solver.SetMapper(Pool(NNODES).map)
    >>> solver.SetGenerationMonitor(stepmon)
    >>> solver.SetTermination(CRT())
    >>> solver.Solve(rosen)
    >>> 
    >>> # obtain the solution
    >>> solution = solver.Solution()


Handler
=======

All solvers packaged with mystic include a signal handler that provides
the following options::

    sol: Print current best solution.
    cont: Continue calculation.
    call: Executes sigint_callback, if provided.
    exit: Exits with current best solution.

Handlers are enabled with the ``enable_signal_handler`` method,
and are configured through the solver's ``Solve`` method.  Handlers
trigger when a signal interrupt (usually, ``Ctrl-C``) is given while
the solver is running.

Notes:

    The handler is currently disabled when the solver is run in parallel.
"""
__all__ = ['AbstractMapSolver']


from mystic.monitors import Null
from mystic.abstract_solver import AbstractSolver


class AbstractMapSolver(AbstractSolver):
    """
AbstractMapSolver base class for mystic optimizers that utilize parallel map.
    """

    def __init__(self, dim, **kwds):
        """
Takes one initial input::

    dim      -- dimensionality of the problem.

Additional inputs::

    npop     -- size of the trial solution population.       [default = 1]

Important class members::

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
        super(AbstractMapSolver, self).__init__(dim, **kwds)
       #AbstractSolver.__init__(self,dim,**kwds)
       #self.signal_handler   = None
       #self._handle_sigint   = False
        trialPop = [[0.0 for i in range(dim)] for j in range(self.nPop)]
        self.trialSolution    = trialPop
        self._map_solver      = True

        # import 'map' defaults
        from mystic.python_map import python_map

        # map and kwds used for parallel and distributed computing
        self._map       = python_map        # map
        self._mapconfig = dict()
        return

    def SetMapper(self, map, **kwds):
        """Set the map and any mapping keyword arguments.

    Sets a callable map to enable parallel and/or distributed evaluations.
    Any kwds given are passed to the map.

Inputs:
    map -- the callable map instance [DEFAULT: python_map]
        """
        self._map = map
        self._mapconfig = kwds
        return


if __name__=='__main__':
    help(__name__)

# end of file
