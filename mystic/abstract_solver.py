#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2014 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE
#
## Abstract Solver Class
# derived from Patrick Hung's original DifferentialEvolutionSolver
"""
This module contains the base class for mystic solvers, and describes
the mystic solver interface.  The "Solve" method must be overwritten
with the derived solver's optimization algorithm.  In many cases, a
minimal function call interface for a derived solver is provided
along with the derived class.  See `mystic.scipy_optimize`, and the
following for an example.


Usage
=====

A typical call to a mystic solver will roughly follow this example:

    >>> # the function to be minimized and the initial values
    >>> from mystic.models import rosen
    >>> x0 = [0.8, 1.2, 0.7]
    >>> 
    >>> # get monitors and termination condition objects
    >>> from mystic.monitors import Monitor
    >>> stepmon = Monitor()
    >>> evalmon = Monitor()
    >>> from mystic.termination import CandidateRelativeTolerance as CRT
    >>> 
    >>> # instantiate and configure the solver
    >>> from mystic.solvers import NelderMeadSimplexSolver
    >>> solver = NelderMeadSimplexSolver(len(x0))
    >>> solver.SetInitialPoints(x0)
    >>> solver.SetEvaluationMonitor(evalmon)
    >>> solver.SetGenerationMonitor(stepmon)
    >>> solver.enable_signal_handler()
    >>> solver.SetTermination(CRT())
    >>> solver.Solve(rosen)
    >>> 
    >>> # obtain the solution
    >>> solution = solver.Solution()


An equivalent, yet less flexible, call using the minimal interface is:

    >>> # the function to be minimized and the initial values
    >>> from mystic.models import rosen
    >>> x0 = [0.8, 1.2, 0.7]
    >>> 
    >>> # configure the solver and obtain the solution
    >>> from mystic.solvers import fmin
    >>> solution = fmin(rosen,x0)


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
the solver is running.

"""
__all__ = ['AbstractSolver']


import numpy
from numpy import inf, shape, asarray, absolute, asfarray
from mystic.tools import wrap_function, wrap_nested, wrap_reducer
from mystic.tools import wrap_bounds, wrap_penalty, reduced

abs = absolute


class AbstractSolver(object):
    """
AbstractSolver base class for mystic optimizers.
    """

    def __init__(self, dim, **kwds):
        """
Takes one initial input:
    dim      -- dimensionality of the problem.

Additional inputs:
    npop     -- size of the trial solution population.       [default = 1]

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
    signal_handler   - catches the interrupt signal.
        """
        NP = 1
        if kwds.has_key('npop'): NP = kwds['npop']

        self.nDim             = dim
        self.nPop             = NP
        self._init_popEnergy  = inf
        self.popEnergy	      = [self._init_popEnergy] * NP
        self.population	      = [[0.0 for i in range(dim)] for j in range(NP)]
        self.trialSolution    = [0.0] * dim
        self._map_solver      = False
        self._bestEnergy      = None
        self._bestSolution    = None
        self._state           = None
        self._type            = self.__class__.__name__

        self.signal_handler   = None
        self._handle_sigint   = False
        self._useStrictRange  = False
        self._defaultMin      = [-1e3] * dim
        self._defaultMax      = [ 1e3] * dim
        self._strictMin       = []
        self._strictMax       = []
        self._maxiter         = None
        self._maxfun          = None
        self._saveiter        = None
       #self._saveeval        = None

        from mystic.monitors import Null, Monitor
        self._evalmon         = Null()
        self._stepmon         = Monitor()
        self._fcalls          = [0]
        self._energy_history  = None
        self._solution_history= None
        self.id               = None     # identifier (use like "rank" for MPI)

        self._constraints     = lambda x: x
        self._penalty         = lambda x: 0.0
        self._reducer         = None
        self._cost            = (None, None)
        self._termination     = lambda x, *ar, **kw: False if len(ar) < 1 or ar[0] is False or kw.get('info',True) == False else '' #XXX: better default ?
        # (get termination details with self._termination.__doc__)

        import mystic.termination
        self._EARLYEXIT       = mystic.termination.EARLYEXIT
        return

    def Solution(self):
        """return the best solution"""
        return self.bestSolution

    def __evaluations(self):
        """get the number of function calls"""
        return self._fcalls[0]

    def __generations(self):
        """get the number of iterations"""
        return max(0,len(self.energy_history)-1)
       #return max(0,len(self._stepmon)-1)

    def __energy_history(self):
        """get the energy_history (default: energy_history = _stepmon.y)"""
        if self._energy_history is None: return self._stepmon.y
        return self._energy_history

    def __set_energy_history(self, energy):
        """set the energy_history (energy=None will sync with _stepmon.y)"""
        self._energy_history = energy
        return

    def __solution_history(self):
        """get the solution_history (default: solution_history = _stepmon.x)"""
        if self._solution_history is None: return self._stepmon.x
        return self._solution_history

    def __set_solution_history(self, params):
        """set the solution_history (params=None will sync with _stepmon.x)"""
        self._solution_history = params
        return

    def __bestSolution(self):
        """get the bestSolution (default: bestSolution = population[0])"""
        if self._bestSolution is None: return self.population[0]
        return self._bestSolution

    def __set_bestSolution(self, params):
        """set the bestSolution (params=None will sync with population[0])"""
        self._bestSolution = params
        return

    def __bestEnergy(self):
        """get the bestEnergy (default: bestEnergy = popEnergy[0])"""
        if self._bestEnergy is None: return self.popEnergy[0]
        return self._bestEnergy

    def __set_bestEnergy(self, energy):
        """set the bestEnergy (energy=None will sync with popEnergy[0])"""
        self._bestEnergy = energy
        return

    def SetReducer(self, reducer, arraylike=False):
        """apply a reducer function to the cost function

input::
    - a reducer function of the form: y' = reducer(yk), where yk is a results
      vector and y' is a single value.  Ideally, this method is applied to
      a cost function with a multi-value return, to reduce the output to a
      single value.  If arraylike, the reducer provided should take a single
      array as input and produce a scalar; otherwise, the reducer provided
      should meet the requirements of the python's builtin 'reduce' method 
      (e.g. lambda x,y: x+y), taking two scalars and producing a scalar."""
        if not reducer:
            self._reducer = None
        elif not callable(reducer):
            raise TypeError, "'%s' is not a callable function" % reducer
        elif not arraylike:
            self._reducer = wrap_reducer(reducer)   
        else: #XXX: check if is arraylike?
            self._reducer = reducer
        return

    def SetPenalty(self, penalty):
        """apply a penalty function to the optimization

input::
    - a penalty function of the form: y' = penalty(xk), with y = cost(xk) + y',
      where xk is the current parameter vector. Ideally, this function
      is constructed so a penalty is applied when the desired (i.e. encoded)
      constraints are violated. Equality constraints should be considered
      satisfied when the penalty condition evaluates to zero, while
      inequality constraints are satisfied when the penalty condition
      evaluates to a non-positive number."""
        if not penalty:
            self._penalty = lambda x: 0.0
        elif not callable(penalty):
            raise TypeError, "'%s' is not a callable function" % penalty
        else: #XXX: check for format: y' = penalty(x) ?
            self._penalty = penalty
        return

    def SetConstraints(self, constraints):
        """apply a constraints function to the optimization

input::
    - a constraints function of the form: xk' = constraints(xk),
      where xk is the current parameter vector. Ideally, this function
      is constructed so the parameter vector it passes to the cost function
      will satisfy the desired (i.e. encoded) constraints."""
        if not constraints:
            self._constraints = lambda x: x
        elif not callable(constraints):
            raise TypeError, "'%s' is not a callable function" % constraints
        else: #XXX: check for format: x' = constraints(x) ?
            self._constraints = constraints
        return

    def SetGenerationMonitor(self, monitor, new=False):
        """select a callable to monitor (x, f(x)) after each solver iteration"""
        from mystic.monitors import Null, Monitor#, CustomMonitor
        current = Null() if new else self._stepmon
        if isinstance(monitor, Monitor):  # is Monitor()
            self._stepmon = monitor
            self._stepmon.prepend(current)
        elif isinstance(monitor, Null) or monitor == Null: # is Null() or Null
            self._stepmon = Monitor()  #XXX: don't allow Null
            self._stepmon.prepend(current)
        elif hasattr(monitor, '__module__'):  # is CustomMonitor()
            if monitor.__module__ in ['mystic._genSow']:
                self._stepmon = monitor #FIXME: need .prepend(current)
        else:
            raise TypeError, "'%s' is not a monitor instance" % monitor
        self.energy_history   = self._stepmon.y
        self.solution_history = self._stepmon.x
        return

    def SetEvaluationMonitor(self, monitor, new=False):
        """select a callable to monitor (x, f(x)) after each cost function evaluation"""
        from mystic.monitors import Null, Monitor#, CustomMonitor
        current = Null() if new else self._evalmon
        if isinstance(monitor, (Null, Monitor) ):  # is Monitor() or Null()
            self._evalmon = monitor
            self._evalmon.prepend(current)
        elif monitor == Null:  # is Null
            self._evalmon = monitor()
            self._evalmon.prepend(current)
        elif hasattr(monitor, '__module__'):  # is CustomMonitor()
            if monitor.__module__ in ['mystic._genSow']:
                self._evalmon = monitor #FIXME: need .prepend(current)
        else:
            raise TypeError, "'%s' is not a monitor instance" % monitor
        return

    def SetStrictRanges(self, min=None, max=None):
        """ensure solution is within bounds

input::
    - min, max: must be a sequence of length self.nDim
    - each min[i] should be <= the corresponding max[i]

note::
    SetStrictRanges(None) will remove strict range constraints"""
        if min is False or max is False:
            self._useStrictRange = False
            return
        #XXX: better to use 'defaultMin,defaultMax' or '-inf,inf' ???
        if min == None: min = self._defaultMin
        if max == None: max = self._defaultMax
        # when 'some' of the bounds are given as 'None', replace with default
        for i in range(len(min)): 
            if min[i] == None: min[i] = self._defaultMin[0]
            if max[i] == None: max[i] = self._defaultMax[0]

        min = asarray(min); max = asarray(max)
        if numpy.any(( min > max ),0):
            raise ValueError, "each min[i] must be <= the corresponding max[i]"
        if len(min) != self.nDim:
            raise ValueError, "bounds array must be length %s" % self.nDim
        self._useStrictRange = True
        self._strictMin = min
        self._strictMax = max
        return

    def _clipGuessWithinRangeBoundary(self, x0): #FIXME: use self.trialSolution?
        """ensure that initial guess is set within bounds

input::
    - x0: must be a sequence of length self.nDim"""
       #if len(x0) != self.nDim: #XXX: unnecessary w/ self.trialSolution
       #    raise ValueError, "initial guess must be length %s" % self.nDim
        x0 = asarray(x0)
        lo = self._strictMin
        hi = self._strictMax
        # crop x0 at bounds
        x0[x0<lo] = lo[x0<lo]
        x0[x0>hi] = hi[x0>hi]
        return x0

    def SetInitialPoints(self, x0, radius=0.05):
        """Set Initial Points with Guess (x0)

input::
    - x0: must be a sequence of length self.nDim
    - radius: generate random points within [-radius*x0, radius*x0]
        for i!=0 when a simplex-type initial guess in required"""
        x0 = asfarray(x0)
        rank = len(x0.shape)
        if rank is 0:
            x0 = asfarray([x0])
            rank = 1
        if not -1 < rank < 2:
            raise ValueError, "Initial guess must be a scalar or rank-1 sequence."
        if len(x0) != self.nDim:
            raise ValueError, "Initial guess must be length %s" % self.nDim

        #slightly alter initial values for solvers that depend on randomness
        min = x0*(1-radius)
        max = x0*(1+radius)
        numzeros = len(x0[x0==0])
        min[min==0] = asarray([-radius for i in range(numzeros)])
        max[max==0] = asarray([radius for i in range(numzeros)])
        self.SetRandomInitialPoints(min,max)
        #stick initial values in population[i], i=0
        self.population[0] = x0.tolist()
    
    def SetRandomInitialPoints(self, min=None, max=None):
        """Generate Random Initial Points within given Bounds

input::
    - min, max: must be a sequence of length self.nDim
    - each min[i] should be <= the corresponding max[i]"""
        if min == None: min = self._defaultMin
        if max == None: max = self._defaultMax
       #if numpy.any(( asarray(min) > asarray(max) ),0):
       #    raise ValueError, "each min[i] must be <= the corresponding max[i]"
        if len(min) != self.nDim or len(max) != self.nDim:
            raise ValueError, "bounds array must be length %s" % self.nDim
        # when 'some' of the bounds are given as 'None', replace with default
        for i in range(len(min)): 
            if min[i] == None: min[i] = self._defaultMin[0]
            if max[i] == None: max[i] = self._defaultMax[0]
        import random
        #generate random initial values
        for i in range(len(self.population)):
            for j in range(self.nDim):
                self.population[i][j] = random.uniform(min[j],max[j])

    def SetMultinormalInitialPoints(self, mean, var = None):
        """Generate Initial Points from Multivariate Normal.

input::
    - mean must be a sequence of length self.nDim
    - var can be...
        None: -> it becomes the identity
        scalar: -> var becomes scalar * I
        matrix: -> the variance matrix. must be the right size!
        """
        from numpy.random import multivariate_normal
        assert(len(mean) == self.nDim)
        if var == None:
            var = numpy.eye(self.nDim)
        else:
            try: # scalar ?
                float(var)
            except: # nope. var better be matrix of the right size (no check)
                pass
            else:
                var = var * numpy.eye(self.nDim)
        for i in range(len(self.population)):
            self.population[i] = multivariate_normal(mean, var).tolist()
        return

    def enable_signal_handler(self):
        """enable workflow interrupt handler while solver is running"""
        self._handle_sigint = True

    def disable_signal_handler(self):
        """disable workflow interrupt handler while solver is running"""
        self._handle_sigint = False

    def _generateHandler(self,sigint_callback):
        """factory to generate signal handler

Available switches::
    - sol  --> Print current best solution.
    - cont --> Continue calculation.
    - call --> Executes sigint_callback, if provided.
    - exit --> Exits with current best solution.
"""
        def handler(signum, frame):
            import inspect
            print inspect.getframeinfo(frame)
            print inspect.trace()
            while 1:
                s = raw_input(\
"""
 
 Enter sense switch.

    sol:  Print current best solution.
    cont: Continue calculation.
    call: Executes sigint_callback [%s].
    exit: Exits with current best solution.

 >>> """ % sigint_callback)
                if s.lower() == 'sol': 
                    print self.bestSolution
                elif s.lower() == 'cont': 
                    return
                elif s.lower() == 'call': 
                    # sigint call_back
                    if sigint_callback is not None:
                        sigint_callback(self.bestSolution)
                elif s.lower() == 'exit': 
                    self._EARLYEXIT = True
                    return
                else:
                    print "unknown option : %s" % s
            return
        self.signal_handler = handler
        return

    def SetSaveFrequency(self, generations=None, filename=None, **kwds):
        """set frequency for saving solver restart file

input::
    - generations = number of solver iterations before next save of state
    - filename = name of file in which to save solver state

note::
    SetSaveFrequency(None) will disable saving solver restart file"""
        self._saveiter = generations
       #self._saveeval = evaluations
        self._state = filename
        return

    def SetEvaluationLimits(self, generations=None, evaluations=None, \
                                                    new=False, **kwds):
        """set limits for generations and/or evaluations

input::
    - generations = maximum number of solver iterations (i.e. steps)
    - evaluations = maximum number of function evaluations"""
        self._maxiter = generations
        self._maxfun = evaluations
        # backward compatibility
        if kwds.has_key('maxiter'):
            self._maxiter = kwds['maxiter']
        if kwds.has_key('maxfun'):
            self._maxfun = kwds['maxfun']
        # handle if new (reset counter, instead of extend counter)
        if new:
            if generations is not None:
                self._maxiter += self.generations
            else:
                self._maxiter = "*" #XXX: better as self._newmax = True ?
            if evaluations is not None:
                self._maxfun += self.evaluations
            else:
                self._maxfun = "*"
        return

    def _SetEvaluationLimits(self, iterscale=None, evalscale=None):
        """set the evaluation limits"""
        if iterscale is None: iterscale = 10
        if evalscale is None: evalscale = 1000
        N = len(self.population[0]) # usually self.nDim
        # if SetEvaluationLimits not applied, use the solver default
        if self._maxiter is None:
            self._maxiter = N * self.nPop * iterscale
        elif self._maxiter == "*": # (i.e. None, but 'reset counter') 
            self._maxiter = (N * self.nPop * iterscale) + self.generations
        if self._maxfun is None:
            self._maxfun = N * self.nPop * evalscale
        elif self._maxiter == "*":
            self._maxfun = (N * self.nPop * evalscale) + self.evaluations
        return

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
        if termination == None:
            termination = self._termination
        # check for termination messages
        msg = termination(self, info=True)
        lim = "EvaluationLimits with %s" % {'evaluations':self._maxfun,
                                            'generations':self._maxiter}

        # push solver internals to scipy.optimize.fmin interface
        if self._fcalls[0] >= self._maxfun and self._maxfun is not None:
            msg = lim #XXX: prefer the default stop ?
            if disp:
                print "Warning: Maximum number of function evaluations has "\
                      "been exceeded."
        elif self.generations >= self._maxiter and self._maxiter is not None:
            msg = lim #XXX: prefer the default stop ?
            if disp:
                print "Warning: Maximum number of iterations has been exceeded"
        elif msg and disp:
            print "Optimization terminated successfully."
            print "         Current function value: %f" % self.bestEnergy
            print "         Iterations: %d" % self.generations
            print "         Function evaluations: %d" % self._fcalls[0]

        if info:
            return msg
        return bool(msg)

    def SetTermination(self, termination):
        """set the termination conditions"""
        #XXX: validate that termination is a 'condition' ?
        self._termination = termination
        return

    def _RegisterObjective(self, cost, ExtraArgs=None):
        """decorate cost function with bounds, penalties, monitors, etc"""
        if ExtraArgs == None: ExtraArgs = ()
        self._fcalls, cost = wrap_function(cost, ExtraArgs, self._evalmon)
        if self._useStrictRange:
            for i in range(self.nPop):
                self.population[i] = self._clipGuessWithinRangeBoundary(self.population[i])
            cost = wrap_bounds(cost, self._strictMin, self._strictMax)
        cost = wrap_penalty(cost, self._penalty)
        cost = wrap_nested(cost, self._constraints)
        if self._reducer:
           #cost = reduced(*self._reducer)(cost) # was self._reducer = (f,bool)
            cost = reduced(self._reducer, arraylike=True)(cost)
        # hold on to the 'wrapped' cost function
        self._cost = (cost, ExtraArgs)
        return cost

    def _bootstrap_decorate(self, cost=None, ExtraArgs=None):
        """HACK to enable not explicitly calling _RegisterObjective"""
        args = None
        if cost == None: # 'use existing cost'
            cost,args = self._cost # use args, unless override with ExtraArgs
        if ExtraArgs != None: args = ExtraArgs
        if self._cost[0] == None: # '_RegisterObjective not yet called'
            if args is None: args = ()
            cost = self._RegisterObjective(cost, args)
        return cost

    def Step(self, cost=None, ExtraArgs=None, **kwds):
        """perform a single optimization iteration

*** this method must be overwritten ***"""
        raise NotImplementedError, "an optimization algorithm was not provided"

    def SaveSolver(self, filename=None, **kwds):
        """save solver state to a restart file"""
        import dill
        if filename == None: # then check if already has registered file
            if self._state == None: # then create a new one
                import tempfile
                self._state = tempfile.mkstemp(suffix='.pkl')[-1]
            filename = self._state
        self._state = filename
        f = file(filename, 'wb')
        try:
            dill.dump(self, f, **kwds)
            self._stepmon.info('DUMPED("%s")' % filename) #XXX: before / after ?
        finally:
            f.close()
        return

    def __save_state(self, force=False):
        """save the solver state, if chosen save frequency is met"""
        # save the last iteration
        if force and bool(self._state):
            self.SaveSolver()
            return
        # save the zeroth iteration
        nonzero = True #XXX: or bool(self.generations) ?
        # after _saveiter generations, then save state
        iters = self._saveiter
        saveiter = bool(iters) and not bool(self.generations % iters)
        if nonzero and saveiter:
            self.SaveSolver()
        #FIXME: if _saveeval (or more) since last check, then save state
       #save = self.evaluations % self._saveeval
        return

    def __load_state(self, solver, **kwds):
        """load solver.__dict__ into self.__dict__; override with kwds"""
        #XXX: should do some filtering on kwds ?
        self.__dict__.update(solver.__dict__, **kwds)
        return

    def _exitMain(self, **kwds):
        """cleanup upon exiting the main optimization loop"""
        pass

    def _process_inputs(self, kwds):
        """process and activate input settings"""
        #allow for inputs that don't conform to AbstractSolver interface
        settings = \
       {'callback':None,     #user-supplied function, called after each step
        'disp':0}            #non-zero to print convergence messages
        [settings.update({i:j}) for (i,j) in kwds.items() if i in settings]
        # backward compatibility
        if kwds.has_key('EvaluationMonitor'): \
           self.SetEvaluationMonitor(kwds.get('EvaluationMonitor'))
        if kwds.has_key('StepMonitor'): \
           self.SetGenerationMonitor(kwds.get('StepMonitor'))
        if kwds.has_key('penalty'): \
           self.SetPenalty(kwds.get('penalty'))
        if kwds.has_key('constraints'): \
           self.SetConstraints(kwds.get('constraints'))
        return settings

    def Solve(self, cost=None, termination=None, sigint_callback=None,
                                                 ExtraArgs=None, **kwds):
        """Minimize a 'cost' function with given termination conditions.

Description:

    Uses an optimization algorith to find the minimum of
    a function of one or more variables.

Inputs:

    cost -- the Python function or method to be minimized.

Additional Inputs:

    termination -- callable object providing termination conditions.
    sigint_callback -- callback function for signal handler.
    ExtraArgs -- extra arguments for cost.

Further Inputs:

    callback -- an optional user-supplied function to call after each
        iteration.  It is called as callback(xk), where xk is
        the current parameter vector.  [default = None]
    disp -- non-zero to print convergence messages.
        """
        # HACK to enable not explicitly calling _RegisterObjective
        cost = self._bootstrap_decorate(cost, ExtraArgs)
        # process and activate input settings
        settings = self._process_inputs(kwds)
        for key in settings:
            exec "%s = settings['%s']" % (key,key)

        # set up signal handler
        import signal
        self._EARLYEXIT = False
        self._generateHandler(sigint_callback) 
        if self._handle_sigint: signal.signal(signal.SIGINT, self.signal_handler)

       ## decorate cost function with bounds, penalties, monitors, etc
       #self._RegisterObjective(cost, ExtraArgs)    #XXX: SetObjective ?
        # register termination function
        if termination is not None:
            self.SetTermination(termination)

        # the initital optimization iteration
        if not len(self._stepmon): # do generation = 0
            self.Step()
            if callback is not None:
                callback(self.bestSolution)
         
            # initialize termination conditions, if needed
            self._termination(self) #XXX: call at generation 0 or always?
        # impose the evaluation limits
        self._SetEvaluationLimits()

        # the main optimization loop
        while not self.CheckTermination() and not self._EARLYEXIT:
            self.Step(**settings)
            if callback is not None:
                callback(self.bestSolution)
        else: self._exitMain()

        # handle signal interrupts
        signal.signal(signal.SIGINT,signal.default_int_handler)

        # log any termination messages
        msg = self.CheckTermination(disp=disp, info=True)
        if msg: self._stepmon.info('STOP("%s")' % msg)
        # save final state
        self.__save_state(force=True)
        return

    # extensions to the solver interface
    evaluations = property(__evaluations )
    generations = property(__generations )
    energy_history = property(__energy_history,__set_energy_history )
    solution_history = property(__solution_history,__set_solution_history )
    bestEnergy = property(__bestEnergy,__set_bestEnergy )
    bestSolution = property(__bestSolution,__set_bestSolution )
    pass


if __name__=='__main__':
    help(__name__)

# end of file
