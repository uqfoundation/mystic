#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
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
        from mystic.python_map import serial_launcher
        from mystic.python_map import python_map
        from mystic.python_map import worker_pool
        from mystic.python_map import defaults

        # default settings for parallel and distributed computing
        launcher        = serial_launcher   # launcher
        mapper          = worker_pool       # map_strategy
        nodes           = int(defaults['nodes'])
        scheduler       = defaults['scheduler'] # scheduler
        queue           = defaults['queue'] # scheduler_queue
        timelimit       = defaults['timelimit']
        servers         = ('*',) #<detect>  # hostname:port
        ncpus           = None   #<detect>  # local processors
       #servers         = ()     #<None>    # hostname:port
       #ncpus           = 0      #<None>    # local processors
        self._mapconfig = dict(nodes=nodes, launcher=launcher, \
                               mapper=mapper, queue=queue, \
                               timelimit=timelimit, scheduler=scheduler, \
                               ncpus=ncpus, servers=servers)
        self._map       = python_map        # map
        return

    def SelectServers(self, servers, ncpus=None): #XXX: needs some thought...
        """Select the compute server.

    Accepts a tuple of ('hostname:port',), listing each available
    computing server.

    If ncpus=None, then 'autodetect'; or if ncpus=0, then 'no local'.
    If servers=('*',), then 'autodetect'; or if servers=(), then 'no remote'.

Inputs:
    servers -- tuple of compute servers  [DEFAULT: autodetect]

Additional inputs:
    ncpus -- number of local processors  [DEFAULT: autodetect]
        """
        self._mapconfig['servers'] = servers
        self._mapconfig['ncpus'] = ncpus  #XXX: merge with nodes, somehow ???
       #print("known servers: %s" % str(servers))
       #print("known # of local processors: %s" % str(ncpus))
        return

    def SetMapper(self, map, strategy=None): #XXX: use strategy+format ?
        """Set the map function and the mapping strategy.

    Sets a mapping function to perform the map-reduce algorithm.
    Uses a mapping strategy to provide the algorithm for distributing
    the work list of optimization jobs across available resources.

Inputs:
    map -- the mapping function [DEFAULT: python_map]
    strategy -- map strategy (see pyina.mappers) [DEFAULT: worker_pool]
        """
        self._map = map
        if strategy:
          self._mapconfig['mapper'] = strategy
        #FIXME: not a true mapping function... just a dummy interface to a str
        # a real mapper has map(func,*args) interface... this expects map().
        # should be...
        #   def xxx_map( f, input, *args, **kwds ):
        #     ...
        #     from yyy.xxx import map
        #     result = map(f, input, *args, **kwds)
        #     ...
        #     return result
        return

    def SetLauncher(self, launcher, nnodes=None): #XXX: use run+scheduler ?
        """Set launcher and (optionally) number of nodes.

    Uses a launcher to provide the solver with the syntax to
    configure and launch optimization jobs on the selected resource.

Inputs:
    launcher -- launcher function (see pyina.launchers)  [DEFAULT: serial_launcher]

Additional inputs:
    nnodes -- number of parallel compute nodes  [DEFAULT: 1]
        """
       #XXX: should be a Launcher class, not a function (see pathos.SSHLauncher)
        self._mapconfig['launcher'] = launcher
       #if launcher != python_map: #FIXME: CANNOT currently change to ez_map!!!
       #    exec("import pyina")   #FIXME: launcher should provide launcher.map
       #    exec("self._map = pyina.ez_map")
        self._mapconfig['nodes'] = nnodes
        return

    def SelectScheduler(self, scheduler, queue, timelimit=None):
        """Select scheduler and queue (and optionally) timelimit.

    Takes a scheduler function and a string queue name to submit
    the optimization job. Additionally takes string time limit
    for scheduled job.

    Example: scheduler, queue='normal', timelimit='00:02'

Inputs:
    scheduler -- scheduler function (see pyina.launchers)  [DEFAULT: None]
    queue -- queue name string (see pyina.launchers)  [DEFAULT: None]

Additional inputs:
    timelimit -- time string HH:MM:SS format  [DEFAULT: '00:05:00']
        """
        self._mapconfig['scheduler'] = scheduler
        self._mapconfig['queue'] = queue
        self._mapconfig['timelimit'] = timelimit
        return


if __name__=='__main__':
    help(__name__)

# end of file
