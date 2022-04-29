#!/usr/bin/env python
#
## Nelder Mead Simplex Solver Class
## Powell Direction Search optimization,
# (derives from optimize.py module by Travis E. Oliphant)
#
# adapted scipy.optimize.fmin (from scipy version 0.4.8)
# by Patrick Hung, Caltech.
#
# adapted from function to class (& added bounds)
# adapted scipy.optimize.fmin_powell
# updated solvers to scipy version 0.9.0
# by Mike McKerns
#
# updated solvers to scipy version 1.1.0
# by Mike McKerns
#
# Author: Patrick Hung (patrickh @caltech)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

"""
Solvers
=======

This module contains a collection of optimization routines adapted
from scipy.optimize.  The minimal scipy interface has been preserved,
and functionality from the mystic solver API has been added with
reasonable defaults.

Minimal function interface to optimization routines::
   fmin        -- Nelder-Mead Simplex algorithm
                    (uses only function calls)
   fmin_powell -- Powell's (modified) level set method
                    (uses only function calls)

The corresponding solvers built on mystic's AbstractSolver are::
   NelderMeadSimplexSolver -- Nelder-Mead Simplex algorithm
   PowellDirectionalSolver -- Powell's (modified) level set method

Mystic solver behavior activated in fmin::
   - EvaluationMonitor = Monitor()
   - StepMonitor = Monitor()
   - termination = CandidateRelativeTolerance(xtol,ftol)

Mystic solver behavior activated in fmin_powell::
   - EvaluationMonitor = Monitor()
   - StepMonitor = Monitor()
   - termination = NormalizedChangeOverGeneration(ftol)


Usage
=====

See `mystic.examples.test_rosenbrock2` for an example of using
NelderMeadSimplexSolver. See `mystic.examples.test_rosenbrock3`
or an example of using PowellDirectionalSolver.

All solvers included in this module provide the standard signal handling.
For more information, see `mystic.mystic.abstract_solver`.


References:
    1. Nelder, J.A. and Mead, R. (1965), "A simplex method for function
       minimization", The Computer Journal, 7, pp. 308-313.
    2. Wright, M.H. (1996), "Direct Search Methods: Once Scorned, Now
       Respectable", in Numerical Analysis 1995, Proceedings of the
       1995 Dundee Biennial Conference in Numerical Analysis, D.F.
       Griffiths and G.A. Watson (Eds.), Addison Wesley Longman,
       Harlow, UK, pp. 191-208.
    3. Gao, F. and Han, L. (2012), "Implementing the Nelder-Mead simplex
       algorithm with adaptive parameters", Computational Optimization and
       Applications. 51:1, pp. 259-277.
    4. Powell M.J.D. (1964) An efficient method for finding the minimum of a
       function of several variables without calculating derivatives,
       Computer Journal, 7 (2):155-162.
    5. Press W., Teukolsky S.A., Vetterling W.T., and Flannery B.P.:
       Numerical Recipes (any edition), Cambridge University Press
"""
__all__ = ['NelderMeadSimplexSolver','PowellDirectionalSolver',
           'fmin','fmin_powell']


from mystic.tools import wrap_function, unpair, wrap_nested
from mystic.tools import wrap_bounds, wrap_penalty, reduced

import numpy
from numpy import eye, zeros, shape, asarray, absolute, asfarray
from numpy import clip, squeeze

abs = absolute

from mystic._scipy060optimize import brent #NOTE: to avoid scipy dependency

from mystic.abstract_solver import AbstractSolver

class NelderMeadSimplexSolver(AbstractSolver):
    """
Nelder Mead Simplex optimization adapted from scipy.optimize.fmin.
    """
    
    def __init__(self, dim):
        """
Takes one initial input: 
    dim      -- dimensionality of the problem

The size of the simplex is dim+1.
        """
        simplex = dim+1
        #XXX: cleaner to set npop=simplex, and use 'population' as simplex
        AbstractSolver.__init__(self,dim) #,npop=simplex)
        for i in range(dim):
            self.popEnergy.append(self._init_popEnergy)
            self.population.append([0.0 for i in range(dim)])
        self.radius = 0.05 #percentage change for initial simplex values
        self.adaptive = False #use adaptive algorithm parameters
        xtol, ftol = 1e-4, 1e-4
        from mystic.termination import CandidateRelativeTolerance as CRT
        self._termination = CRT(xtol,ftol)

    def _setSimplexWithinRangeBoundary(self, radius=None):
        """ensure that initial simplex is set within bounds

Input::
    - radius: size of the initial simplex [default=0.05]"""
        x0 = self.population[0]
        #code modified from park-1.2/park/simplex.py (version 1257)
        if self._useStrictRange:
            x0 = self._clipGuessWithinRangeBoundary(x0)

        if radius is None: radius = 0.05 # nonzdelt=0.05 from scipy-0.9
        val = x0*(1+radius)
        val[val==0] = (radius**2) * 0.1 # zdelt=0.00025 update from scipy-0.9
        if not self._useStrictRange:
            self.population[0] = x0
            return val

        lo = self._strictMin
        hi = self._strictMax
        radius = clip(radius,0,0.5)
        # rescale val by bounded range...
        # (increases fit for tight bounds; makes worse[?] for large bounds)
        bounded = ~numpy.isinf(lo) & ~numpy.isinf(hi)
        val[bounded] = x0[bounded] + (hi[bounded]-lo[bounded])*radius
        # crop val at bounds
        settings = numpy.seterr(all='ignore')
        val[val<lo] = lo[val<lo]
        val[val>hi] = hi[val>hi]
        numpy.seterr(**settings)
        # handle collisions (when val[i] == x0[i])
        collision = val==x0
        if numpy.any(collision):
            rval = x0*(1-radius)
            rval[rval==0] = -radius
            rval[bounded] = x0[bounded] - (hi[bounded]-lo[bounded])*radius
            val[collision] = rval[collision]
        # make tolerance relative for bounded parameters
     #  tol = numpy.ones(x0.shape)*xtol
     #  tol[bounded] = (hi[bounded]-lo[bounded])*xtol
     #  xtol = tol
        self.population[0] = x0
        return val

    def _SetEvaluationLimits(self, iterscale=200, evalscale=200):
        """set the evaluation limits

input::
    - iterscale and evalscale are integers used to set the maximum iteration
      and evaluation limits, respectively. The new limit is defined as
      limit = (nDim * nPop * scale) + count, where count is the number
      of existing iterations or evaluations, respectively. The default for
      iterscale is 200, and the default for evalscale is also 200.
        """
        super(NelderMeadSimplexSolver, self)._SetEvaluationLimits(iterscale,evalscale)
        return

    def _decorate_objective(self, cost, ExtraArgs=None):
        """decorate the cost function with bounds, penalties, monitors, etc

input::
    - cost is the objective function, of the form y = cost(x, *ExtraArgs),
      where x is a candidate solution, and ExtraArgs is the tuple of positional
      arguments required to evaluate the objective."""
        #print("@%r %r %r" % (cost, ExtraArgs, max))
        evalmon = self._evalmon
        raw = cost
        if ExtraArgs is None: ExtraArgs = ()
        self._fcalls, cost = wrap_function(cost, ExtraArgs, evalmon)
        if self._useStrictRange:
            if self.generations:
                #NOTE: pop[0] was best, may not be after resetting simplex
                for i,j in enumerate(self._setSimplexWithinRangeBoundary()):
                    self.population[i+1] = self.population[0].copy()
                    self.population[i+1][i] = j
            else:
                self.population[0] = self._clipGuessWithinRangeBoundary(self.population[0])
            cost = wrap_bounds(cost, self._strictMin, self._strictMax) #XXX: remove?
            from mystic.constraints import and_
            constraints = and_(self._constraints, self._strictbounds, onfail=self._strictbounds)
        else: constraints = self._constraints
        cost = wrap_penalty(cost, self._penalty)
        cost = wrap_nested(cost, constraints)
        if self._reducer:
           #cost = reduced(*self._reducer)(cost) # was self._reducer = (f,bool)
            cost = reduced(self._reducer, arraylike=True)(cost)
        # hold on to the 'wrapped' and 'raw' cost function
        self._cost = (cost, raw, ExtraArgs)
        self._live = True
        return cost

    def _Step(self, cost=None, ExtraArgs=None, **kwds):
        """perform a single optimization iteration

input::
    - cost is the objective function, of the form y = cost(x, *ExtraArgs),
      where x is a candidate solution, and ExtraArgs is the tuple of positional
      arguments required to evaluate the objective.

note::
    ExtraArgs needs to be a *tuple* of extra arguments.

    This method accepts additional args that are specific for the current
    solver, as detailed in the `_process_inputs` method.
        """
        # process and activate input settings
        settings = self._process_inputs(kwds)
        #(hardwired: due to python3.x exec'ing to locals())
        callback = settings['callback'] if 'callback' in settings else None
        disp = settings['disp'] if 'disp' in settings else False
        radius = settings['radius'] if 'radius' in settings else self.radius
        adaptive = settings['adaptive'] if 'adaptive' in settings else self.adaptive

        # HACK to enable not explicitly calling _decorate_objective
        cost = self._bootstrap_objective(cost, ExtraArgs)

        if self._useStrictRange: #XXX: necessary? or handled by wrap_nested?
            from mystic.constraints import and_
            constraints = and_(self._constraints, self._strictbounds, onfail=self._strictbounds)
        else: constraints = self._constraints

        if adaptive:
            dim = float(len(self.population[0])) # dimensionality of x0
            rho = 1; chi = 1+2/dim; psi = 0.75-1/(2*dim); sigma = 1-1/dim;
        else:
            rho = 1; chi = 2; psi = 0.5; sigma = 0.5;
        init = False  # flag to do 0th iteration 'post-initialization'

        if not len(self._stepmon): # do generation = 0
            init = True
            x0 = self.population[0]
            x0 = asfarray(x0).flatten()
            x0 = asfarray(constraints(x0))
            #####XXX: this blows away __init__, so replace __init__ with this?
            N = len(x0)
            rank = len(x0.shape)
            if not -1 < rank < 2:
                raise ValueError("Initial guess must be a scalar or rank-1 sequence.")
            if rank == 0:
                sim = numpy.zeros((N+1,), dtype=x0.dtype)
            else:
                sim = numpy.zeros((N+1,N), dtype=x0.dtype)
            fsim = numpy.ones((N+1,), float) * self._init_popEnergy
            ####################################################
            sim[0] = x0
            fsim[0] = cost(x0)

        elif not self.generations: # do generations = 1
            #--- ensure initial simplex is within bounds ---
            val = self._setSimplexWithinRangeBoundary(radius)
            #--- end bounds code ---
            sim = self.population
            fsim = self.popEnergy
            x0 = sim[0]
            N = len(x0)
            # populate the simplex
            for k in range(0,N):
                y = numpy.array(x0,copy=True)
                y[k] = val[k]
                sim[k+1] = y
                f = cost(y) #XXX: use self._map?
                fsim[k+1] = f

        else: # do generations > 1
            sim = self.population
            fsim = self.popEnergy
            N = len(sim[0])
            one2np1 = range(1,N+1)

            # apply constraints  #XXX: is this the only appropriate place???
            sim[0] = asfarray(constraints(sim[0]))

            xbar = numpy.add.reduce(sim[:-1],0) / N
            xr = (1+rho)*xbar - rho*sim[-1]
            fxr = cost(xr)
            doshrink = 0

            if fxr < fsim[0]:
                xe = (1+rho*chi)*xbar - rho*chi*sim[-1]
                fxe = cost(xe)

                if fxe < fxr:
                    sim[-1] = xe
                    fsim[-1] = fxe
                else:
                    sim[-1] = xr
                    fsim[-1] = fxr
            else: # fsim[0] <= fxr
                if fxr < fsim[-2]:
                    sim[-1] = xr
                    fsim[-1] = fxr
                else: # fxr >= fsim[-2]
                    # Perform contraction
                    if fxr < fsim[-1]:
                        xc = (1+psi*rho)*xbar - psi*rho*sim[-1]
                        fxc = cost(xc)
    
                        if fxc <= fxr:
                            sim[-1] = xc
                            fsim[-1] = fxc
                        else:
                            doshrink=1
                    else:
                        # Perform an inside contraction
                        xcc = (1-psi)*xbar + psi*sim[-1]
                        fxcc = cost(xcc)

                        if fxcc < fsim[-1]:
                            sim[-1] = xcc
                            fsim[-1] = fxcc
                        else:
                            doshrink = 1

                    if doshrink:
                        for j in one2np1:
                            sim[j] = sim[0] + sigma*(sim[j] - sim[0])
                            fsim[j] = cost(sim[j]) #XXX: use self._map?

        if len(self._stepmon):
            # sort so sim[0,:] has the lowest function value
            ind = numpy.argsort(fsim)
            sim = numpy.take(sim,ind,0)
            fsim = numpy.take(fsim,ind,0)
        self.population = sim # bestSolution = sim[0]
        self.popEnergy = fsim # bestEnergy = fsim[0]
        self._stepmon(sim[0], fsim[0], self.id) # sim = all; "best" is sim[0]
        # if savefrequency matches, then save state
        self._AbstractSolver__save_state()

        # do callback
        if callback is not None: callback(self.bestSolution)
        # initialize termination conditions, if needed
        if init: self._termination(self) #XXX: at generation 0 or always?
        return #XXX: call Terminated ?

    def _process_inputs(self, kwds):
        """process and activate input settings

Args:
    callback (func, default=None): function to call after each iteration. The
        interface is ``callback(xk)``, with xk the current parameter vector.
    disp (bool, default=False): if True, print convergence messages.

Additional Args:
    EvaluationMonitor: a monitor instance to capture each evaluation of cost.
    StepMonitor: a monitor instance to capture each iteration's best results.
    penalty: a function of the form: y' = penalty(xk), with y = cost(xk) + y',
        where xk is the current parameter vector.
    constraints: a function of the form: xk' = constraints(xk), where xk is
        the current parameter vector.
    radius (float, default=0.05): percentage change for initial simplex values.
    adaptive (bool, default=False): adapt algorithm parameters to the
        dimensionality of the initial parameter vector ``x``.

Note:
   The additional args are 'sticky', in that once they are given, they remain
   set until they are explicitly changed. Conversely, the args are not sticky,
   and are thus set for a one-time use.
        """
        #allow for inputs that don't conform to AbstractSolver interface
        #NOTE: not sticky: callback, disp
        #NOTE: sticky: EvaluationMonitor, StepMonitor, penalty, constraints
        #NOTE: sticky: radius, adaptive
        settings = super(NelderMeadSimplexSolver, self)._process_inputs(kwds)
        settings.update({
        'radius':self.radius, #percentage change for initial simplex values
        'adaptive':self.adaptive}) #use adaptive algorithm parameters
        [settings.update({i:j}) for (i,j) in getattr(kwds, 'iteritems', kwds.items)() if i in settings]
        self.radius = settings['radius']
        self.adaptive = settings['adaptive']
        return settings

    def Solve(self, cost=None, termination=None, ExtraArgs=None, **kwds):
        """Minimize a function using the downhill simplex algorithm.

Uses a Nelder-Mead simplex algorithm to find the minimum of a function of one
or more variables.

Args:
    cost (func, default=None): the function to be minimized: ``y = cost(x)``.
    termination (termination, default=None): termination conditions.
    ExtraArgs (tuple, default=None): extra arguments for cost.
    sigint_callback (func, default=None): callback function for signal handler.
    callback (func, default=None): function to call after each iteration. The
        interface is ``callback(xk)``, with xk the current parameter vector.
    disp (bool, default=False): if True, print convergence messages.
    radius (float, default=0.05): percentage change for initial simplex values.
    adaptive (bool, default=False): adapt algorithm parameters to the
        dimensionality of the initial parameter vector ``x``.

Returns:
    None
"""
        super(NelderMeadSimplexSolver, self).Solve(cost, termination,\
                                                   ExtraArgs, **kwds)
        return


def fmin(cost, x0, args=(), bounds=None, xtol=1e-4, ftol=1e-4,
         maxiter=None, maxfun=None, full_output=0, disp=1, retall=0,
         callback=None, **kwds):
    """Minimize a function using the downhill simplex algorithm.
    
Uses a Nelder-Mead simplex algorithm to find the minimum of a function of one
or more variables. This algorithm only uses function values, not derivatives or second derivatives. Mimics the ``scipy.optimize.fmin`` interface.

This algorithm has a long history of successful use in applications. It will
usually be slower than an algorithm that uses first or second derivative
information. In practice it can have poor performance in high-dimensional
problems and is not robust to minimizing complicated functions. Additionally,
there currently is no complete theory describing when the algorithm will
successfully converge to the minimum, or how fast it will if it does. Both the
ftol and xtol criteria must be met for convergence.

Args:
    cost (func): the function or method to be minimized: ``y = cost(x)``.
    x0 (ndarray): the initial guess parameter vector ``x``.
    args (tuple, default=()): extra arguments for cost.
    bounds (list(tuple), default=None): list of pairs of bounds (min,max),
        one for each parameter.
    xtol (float, default=1e-4): acceptable absolute error in ``xopt`` for
        convergence.
    ftol (float, default=1e-4): acceptable absolute error in ``cost(xopt)``
        for convergence.
    maxiter (int, default=None): the maximum number of iterations to perform.
    maxfun (int, default=None): the maximum number of function evaluations.
    full_output (bool, default=False): True if fval and warnflag are desired.
    disp (bool, default=True): if True, print convergence messages.
    retall (bool, default=False): True if allvecs is desired.
    callback (func, default=None): function to call after each iteration. The
        interface is ``callback(xk)``, with xk the current parameter vector.
    id (int, default=None): the ``id`` of the solver used in logging.
    handler (bool, default=False): if True, enable handling interrupt signals.
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

Returns:
    ``(xopt, {fopt, iter, funcalls, warnflag}, {allvecs})``

Notes:
    - xopt (*ndarray*): the minimizer of the cost function
    - fopt (*float*): value of cost function at minimum: ``fopt = cost(xopt)``
    - iter (*int*): number of iterations
    - funcalls (*int*): number of function calls
    - warnflag (*int*): warning flag:
        - ``1 : Maximum number of function evaluations``
        - ``2 : Maximum number of iterations``
    - allvecs (*list*): a list of solutions at each iteration
    """
    handler = kwds['handler'] if 'handler' in kwds else False

    from mystic.monitors import Monitor
    stepmon = kwds['itermon'] if 'itermon' in kwds else Monitor()
    evalmon = kwds['evalmon'] if 'evalmon' in kwds else Monitor()

    if xtol: #if tolerance in x is provided, use CandidateRelativeTolerance
        from mystic.termination import CandidateRelativeTolerance as CRT
        termination = CRT(xtol,ftol)
    else:
        from mystic.termination import VTRChangeOverGeneration
        termination = VTRChangeOverGeneration(ftol)
    solver = NelderMeadSimplexSolver(len(x0))
    solver.SetInitialPoints(x0)
    solver.SetEvaluationLimits(maxiter,maxfun)
    solver.SetEvaluationMonitor(evalmon)
    solver.SetGenerationMonitor(stepmon)
    if 'id' in kwds:
        solver.id = int(kwds['id'])
    if 'penalty' in kwds:
        solver.SetPenalty(kwds['penalty'])
    if 'constraints' in kwds:
        solver.SetConstraints(kwds['constraints'])
    if bounds is not None:
        minb,maxb = unpair(bounds)
        tight = kwds['tightrange'] if 'tightrange' in kwds else None
        clip = kwds['cliprange'] if 'cliprange' in kwds else None
        solver.SetStrictRanges(minb,maxb,tight=tight,clip=clip)

    if handler: solver.enable_signal_handler()
    solver.Solve(cost, termination=termination, \
                 disp=disp, ExtraArgs=args, callback=callback)
    solution = solver.Solution()

    # code below here pushes output to scipy.optimize.fmin interface
   #x = list(solver.bestSolution)
    x = solver.bestSolution
    fval = solver.bestEnergy
    warnflag = 0
    fcalls = solver.evaluations
    iterations = solver.generations
    allvecs = stepmon.x

    if fcalls >= solver._maxfun:
        warnflag = 1
    elif iterations >= solver._maxiter:
        warnflag = 2

    if full_output:
        retlist = x, fval, iterations, fcalls, warnflag
        if retall:
            retlist += (allvecs,)
    else:
        retlist = x
        if retall:
            retlist = (x, allvecs)

    return retlist

############################################################################

def _linesearch_powell(func, p, xi, tol=1e-3, maxiter=500):
    # line-search algorithm using fminbound
    #  find the minimium of the function
    #  func(x0+ alpha*direc)
    def myfunc(alpha):
        return func(p + alpha * xi)
    settings = numpy.seterr(all='ignore')
    alpha_min, fret, iter, num = brent(myfunc, full_output=1, tol=tol, maxiter=maxiter)
    numpy.seterr(**settings)
    xi = alpha_min*xi
    return squeeze(fret), p+xi, xi


class PowellDirectionalSolver(AbstractSolver):
    """
Powell Direction Search optimization,
adapted from scipy.optimize.fmin_powell.
    """
    
    def __init__(self, dim):
        """
Takes one initial input: 
    dim      -- dimensionality of the problem
        """
        AbstractSolver.__init__(self,dim)
        self._direc = None # this is the easy way to return 'direc'...
        x1 = self.population[0]
        fx = self.popEnergy[0]
        #                  [x1, fx, bigind, delta]
        self.__internals = [x1, fx,      0,   0.0]
        self.imax  = 500   #line-search maximum iterations
        self.xtol  = 1e-4  #line-search error tolerance
        ftol, gtol = 1e-4, 2
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        self._termination = NCOG(ftol,gtol)

    def __generations(self):
        """get the number of iterations"""
        return max(0,len(self.energy_history)-1)  #XXX: slower for k=-1 ?

    def _SetEvaluationLimits(self, iterscale=1000, evalscale=1000):
        """set the evaluation limits

input::
    - iterscale and evalscale are integers used to set the maximum iteration
      and evaluation limits, respectively. The new limit is defined as
      limit = (nDim * nPop * scale) + count, where count is the number
      of existing iterations or evaluations, respectively. The default for
      iterscale is 1000, and the default for evalscale is also 1000.
        """
        super(PowellDirectionalSolver, self)._SetEvaluationLimits(iterscale,evalscale)
        return

    def _Step(self, cost=None, ExtraArgs=None, **kwds):
        """perform a single optimization iteration

input::
    - cost is the objective function, of the form y = cost(x, *ExtraArgs),
      where x is a candidate solution, and ExtraArgs is the tuple of positional
      arguments required to evaluate the objective.

note::
    ExtraArgs needs to be a *tuple* of extra arguments.

    This method accepts additional args that are specific for the current
    solver, as detailed in the `_process_inputs` method.
        """
        # process and activate input settings
        settings = self._process_inputs(kwds)
        #(hardwired: due to python3.x exec'ing to locals())
        callback = settings['callback'] if 'callback' in settings else None
        disp = settings['disp'] if 'disp' in settings else False
        xtol = settings['xtol'] if 'xtol' in settings else self.xtol
        imax = settings['imax'] if 'imax' in settings else self.imax

        # HACK to enable not explicitly calling _decorate_objective
        cost = self._bootstrap_objective(cost, ExtraArgs)

        if self._useStrictRange: #XXX: necessary? or handled by wrap_nested?
            from mystic.constraints import and_
            constraints = and_(self._constraints, self._strictbounds, onfail=self._strictbounds)
        else: constraints = self._constraints

        direc = self._direc #XXX: throws Error if direc=None after generation=0
        x = self.population[0][:]   # bestSolution
        fval = self.popEnergy[0] # bestEnergy
        self.__internals[0] = self.__internals[0][:] # decouple x1 from x
        x1, fx, bigind, delta = self.__internals
        init = False  # flag to do 0th iteration 'post-initialization'

        if not len(self._stepmon): # do generation = 0
            init = True
            x = asfarray(x).flatten()
            x = asfarray(constraints(x))
            N = len(x) #XXX: this should be equal to self.nDim
            rank = len(x.shape)
            if not -1 < rank < 2:
                raise ValueError("Initial guess must be a scalar or rank-1 sequence.")

            if direc is None:
                direc = eye(N, dtype=float)
            else:
                direc = asarray(direc, dtype=float)
            fval = squeeze(cost(x))
            if self._maxiter != 0:
                self._stepmon(x, fval, self.id) # get initial values
                # if savefrequency matches, then save state
                self._AbstractSolver__save_state()

        elif not self.generations: # do generations = 1
            ilist = range(len(x))
            x1 = x.copy()
            # do initial "second half" of solver step 
            fx = fval
            bigind = 0
            delta = 0.0
            for i in ilist:
                direc1 = self._direc[i]
                fx2 = fval
                fval, x, direc1 = _linesearch_powell(cost, x, direc1, tol=xtol*100, maxiter=imax)
                isnan = numpy.isinf(fx2) & numpy.isinf(fval)
                if not isnan and (fx2 - fval) > delta:
                    delta = fx2 - fval
                    bigind = i

                # apply constraints
                x = asfarray(constraints(x)) #XXX: use self._map?
            # decouple from 'best' energy
            self.energy_history = self.energy_history + [fval]

        else: # do generations > 1
            # Construct the extrapolated point
            direc1 = x - x1
            x2 = 2*x - x1
            x1 = x.copy()
            fx2 = squeeze(cost(x2))

            if (fx > fx2):
                t = 2.0*(fx+fx2-2.0*fval)
                temp = (fx-fval-delta)
                t *= temp*temp
                temp = fx-fx2
                t -= delta*temp*temp
                if t < 0.0:
                    fval, x, direc1 = _linesearch_powell(cost, x, direc1, tol=xtol*100, maxiter=imax)
                    direc[bigind] = direc[-1]
                    direc[-1] = direc1

           #        x = asfarray(constraints(x))

            self._direc = direc
            self.population[0] = x   # bestSolution
            self.popEnergy[0] = fval # bestEnergy
            self.energy_history = None # resync with 'best' energy
            self._stepmon(x, fval, self.id) # get ith values
            # if savefrequency matches, then save state
            self._AbstractSolver__save_state()

            fx = fval
            bigind = 0
            delta = 0.0
            ilist = range(len(x))
            for i in ilist:
                direc1 = direc[i]
                fx2 = fval
                fval, x, direc1 = _linesearch_powell(cost, x, direc1, tol=xtol*100, maxiter=imax)
                isnan = numpy.isinf(fx2) & numpy.isinf(fval)
                if not isnan and (fx2 - fval) > delta:
                    delta = fx2 - fval
                    bigind = i

                # apply constraints
                x = asfarray(constraints(x)) #XXX: use self._map?

            # decouple from 'best' energy
            self.energy_history = self.energy_history + [fval]

        self.__internals = [x1, fx, bigind, delta]
        self._direc = direc
        self.population[0] = x   # bestSolution
        self.popEnergy[0] = fval # bestEnergy

        # do callback
        if callback is not None: callback(self.bestSolution)
        # initialize termination conditions, if needed
        if init: self._termination(self) #XXX: at generation 0 or always?
        return #XXX: call Terminated ?

    def Finalize(self):
        """cleanup upon exiting the main optimization loop"""
        if self.energy_history != None and self._live:
            self.energy_history = None # resync with 'best' energy
            self._stepmon(self.bestSolution, self.bestEnergy, self.id)
            # if savefrequency matches, then save state
            self._AbstractSolver__save_state()
        self._live = False
        return

    def _process_inputs(self, kwds):
        """process and activate input settings

Args:
    callback (func, default=None): function to call after each iteration. The
        interface is ``callback(xk)``, with xk the current parameter vector.
    disp (bool, default=False): if True, print convergence messages.

Additional Args:
    EvaluationMonitor: a monitor instance to capture each evaluation of cost.
    StepMonitor: a monitor instance to capture each iteration's best results.
    penalty: a function of the form: y' = penalty(xk), with y = cost(xk) + y',
        where xk is the current parameter vector.
    constraints: a function of the form: xk' = constraints(xk), where xk is
        the current parameter vector.
    direc (tuple, default=None): the initial direction set.
    xtol (float, default=1e-4): line-search error tolerance.
    imax (float, default=500): line-search maximum iterations.

Note:
   The additional args are 'sticky', in that once they are given, they remain
   set until they are explicitly changed. Conversely, the args are not sticky,
   and are thus set for a one-time use.
        """
        #allow for inputs that don't conform to AbstractSolver interface
        #NOTE: not sticky: callback, disp
        #NOTE: sticky: EvaluationMonitor, StepMonitor, penalty, constraints
        #NOTE: sticky: imax, xtol, direc
        settings = super(PowellDirectionalSolver, self)._process_inputs(kwds)
        settings.update({\
        'xtol':self.xtol,    #line-search error tolerance
        'imax':self.imax})   #line-search maximum iterations
        direc=self._direc    #initial direction set
        [settings.update({i:j}) for (i,j) in getattr(kwds, 'iteritems', kwds.items)() if i in settings]
        self._direc = kwds['direc'] if 'direc' in kwds else direc
        self.xtol = settings['xtol']
        self.imax = settings['imax']
        return settings

    def Solve(self, cost=None, termination=None, ExtraArgs=None, **kwds):
        """Minimize a function using modified Powell's method.

Uses a modified Powell Directional Search algorithm to find the minimum of a
function of one or more variables.

Args:
    cost (func, default=None): the function to be minimized: ``y = cost(x)``.
    termination (termination, default=None): termination conditions.
    ExtraArgs (tuple, default=None): extra arguments for cost.
    sigint_callback (func, default=None): callback function for signal handler.
    callback (func, default=None): function to call after each iteration. The
        interface is ``callback(xk)``, with xk the current parameter vector.
    direc (tuple, default=None): the initial direction set.
    xtol (float, default=1e-4): line-search error tolerance.
    imax (float, default=500): line-search maximum iterations.
    disp (bool, default=False): if True, print convergence messages.

Returns:
    None
"""
        super(PowellDirectionalSolver, self).Solve(cost, termination,\
                                                   ExtraArgs, **kwds)
        return

    # extensions to the solver interface
    generations = property(__generations )
    pass


def fmin_powell(cost, x0, args=(), bounds=None, xtol=1e-4, ftol=1e-4,
                maxiter=None, maxfun=None, full_output=0, disp=1, retall=0,
                callback=None, direc=None, **kwds):
    """Minimize a function using modified Powell's method.
    
Uses a modified Powell Directional Search algorithm to find the minimum of a
function of one or more variables. This method only uses function values,
not derivatives. Mimics the ``scipy.optimize.fmin_powell`` interface.

Powell's method is a conjugate direction method that has two loops. The outer
loop simply iterates over the inner loop, while the inner loop minimizes over
each current direction in the direction set. At the end of the inner loop,
if certain conditions are met, the direction that gave the largest decrease
is dropped and replaced with the difference between the current estimated x
and the estimated x from the beginning of the inner-loop. The conditions for
replacing the direction of largest increase is that: (a) no further gain can
be made along the direction of greatest increase in the iteration, and (b) the
direction of greatest increase accounted for a large sufficient fraction of
the decrease in the function value from the current iteration of the inner loop.

Args:
    cost (func): the function or method to be minimized: ``y = cost(x)``.
    x0 (ndarray): the initial guess parameter vector ``x``.
    args (tuple, default=()): extra arguments for cost.
    bounds (list(tuple), default=None): list of pairs of bounds (min,max),
        one for each parameter.
    xtol (float, default=1e-4): acceptable relative error in ``xopt`` for
        convergence.
    ftol (float, default=1e-4): acceptable relative error in ``cost(xopt)``
        for convergence.
    gtol (float, default=2): maximum iterations to run without improvement.
    maxiter (int, default=None): the maximum number of iterations to perform.
    maxfun (int, default=None): the maximum number of function evaluations.
    full_output (bool, default=False): True if fval and warnflag are desired.
    disp (bool, default=True): if True, print convergence messages.
    retall (bool, default=False): True if allvecs is desired.
    callback (func, default=None): function to call after each iteration. The
        interface is ``callback(xk)``, with xk the current parameter vector.
    direc (tuple, default=None): the initial direction set.
    id (int, default=None): the ``id`` of the solver used in logging.
    handler (bool, default=False): if True, enable handling interrupt signals.
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

Returns:
    ``(xopt, {fopt, iter, funcalls, warnflag, direc}, {allvecs})``

Notes:
    - xopt (*ndarray*): the minimizer of the cost function
    - fopt (*float*): value of cost function at minimum: ``fopt = cost(xopt)``
    - iter (*int*): number of iterations
    - funcalls (*int*): number of function calls
    - warnflag (*int*): warning flag:
        - ``1 : Maximum number of function evaluations``
        - ``2 : Maximum number of iterations``
    - direc (*tuple*): the current direction set
    - allvecs (*list*): a list of solutions at each iteration
    """
    #FIXME: need to resolve "direc"
    #        - should just pass 'direc', and then hands-off ?  How return it ?
    #XXX: enable use of imax?

    handler = kwds['handler'] if 'handler' in kwds else False

    from mystic.monitors import Monitor
    stepmon = kwds['itermon'] if 'itermon' in kwds else Monitor()
    evalmon = kwds['evalmon'] if 'evalmon' in kwds else Monitor()

    gtol = 2 # termination generations (scipy: 2, default: 10)
    if 'gtol' in kwds: gtol = kwds['gtol']
    if gtol: #if number of generations is provided, use NCOG
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        termination = NCOG(ftol,gtol)
    else:
        from mystic.termination import VTRChangeOverGeneration
        termination = VTRChangeOverGeneration(ftol)

    solver = PowellDirectionalSolver(len(x0))
    solver.SetInitialPoints(x0)
    solver.SetEvaluationLimits(maxiter,maxfun)
    solver.SetEvaluationMonitor(evalmon)
    solver.SetGenerationMonitor(stepmon)
    if 'id' in kwds:
        solver.id = int(kwds['id'])
    if 'penalty' in kwds:
        solver.SetPenalty(kwds['penalty'])
    if 'constraints' in kwds:
        solver.SetConstraints(kwds['constraints'])
    if bounds is not None:
        minb,maxb = unpair(bounds)
        tight = kwds['tightrange'] if 'tightrange' in kwds else None
        clip = kwds['cliprange'] if 'cliprange' in kwds else None
        solver.SetStrictRanges(minb,maxb,tight=tight,clip=clip)

    if handler: solver.enable_signal_handler()
    solver.Solve(cost, termination=termination, \
                 xtol=xtol, ExtraArgs=args, callback=callback, \
                 disp=disp, direc=direc)   #XXX: last two lines use **kwds
    solution = solver.Solution()

    # code below here pushes output to scipy.optimize.fmin_powell interface
   #x = list(solver.bestSolution)
    x = solver.bestSolution
    fval = solver.bestEnergy
    warnflag = 0
    fcalls = solver.evaluations
    iterations = solver.generations
    allvecs = stepmon.x
    direc = solver._direc

    if fcalls >= solver._maxfun:
        warnflag = 1
    elif iterations >= solver._maxiter:
        warnflag = 2

    x = squeeze(x) #FIXME: write squeezed x to stepmon instead?

    if full_output:
        retlist = x, fval, iterations, fcalls, warnflag, direc
        if retall:
            retlist += (allvecs,)
    else:
        retlist = x
        if retall:
            retlist = (x, allvecs)

    return retlist


if __name__=='__main__':
    help(__name__)

# end of file
