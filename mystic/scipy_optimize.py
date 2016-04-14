#!/usr/bin/env python
#
## Nelder Mead Simplex Solver Class
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
# Author: Patrick Hung (patrickh @caltech)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE

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
   fmin_powell -- Powell's (modified) level set method (uses only
                    function calls)

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

"""
from __future__ import division
from __future__ import absolute_import
from builtins import range
from past.utils import old_div
__all__ = ['NelderMeadSimplexSolver','PowellDirectionalSolver',
           'fmin','fmin_powell']


from mystic.tools import wrap_function, unpair, wrap_nested
from mystic.tools import wrap_bounds, wrap_penalty, reduced

import numpy
from numpy import eye, zeros, shape, asarray, absolute, asfarray
from numpy import clip, squeeze

abs = absolute

from ._scipy060optimize import brent #XXX: local copy to avoid dependency!

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
        self.popEnergy.append(self._init_popEnergy)
        self.population.append([0.0 for i in range(dim)])
        self.radius= 0.05 #percentage change for initial simplex values
        xtol, ftol = 1e-4, 1e-4
        from mystic.termination import CandidateRelativeTolerance as CRT
        self._termination = CRT(xtol,ftol)

    def _setSimplexWithinRangeBoundary(self, radius=None):
        """ensure that initial simplex is set within bounds
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
        super(NelderMeadSimplexSolver, self)._SetEvaluationLimits(iterscale,evalscale)
        return

    def _decorate_objective(self, cost, ExtraArgs=None):
        """decorate the cost function with bounds, penalties, monitors, etc"""
        #print ("@", cost, ExtraArgs, max)
        raw = cost
        if ExtraArgs is None: ExtraArgs = ()
        self._fcalls, cost = wrap_function(cost, ExtraArgs, self._evalmon)
        if self._useStrictRange:
            if self.generations:
                #NOTE: pop[0] was best, may not be after resetting simplex
                for i,j in enumerate(self._setSimplexWithinRangeBoundary()):
                    self.population[i+1] = self.population[0].copy()
                    self.population[i+1][i] = j
            else:
                self.population[0] = self._clipGuessWithinRangeBoundary(self.population[0])
            cost = wrap_bounds(cost, self._strictMin, self._strictMax)
        cost = wrap_penalty(cost, self._penalty)
        cost = wrap_nested(cost, self._constraints)
        if self._reducer:
           #cost = reduced(*self._reducer)(cost) # was self._reducer = (f,bool)
            cost = reduced(self._reducer, arraylike=True)(cost)
        # hold on to the 'wrapped' and 'raw' cost function
        self._cost = (cost, raw, ExtraArgs)
        self._live = True
        return cost

    def _Step(self, cost=None, ExtraArgs=None, **kwds):
        """perform a single optimization iteration
        Note that ExtraArgs should be a *tuple* of extra arguments"""
        # process and activate input settings
        settings = self._process_inputs(kwds)

        # HACK to enable not explicitly calling _decorate_objective
        cost = self._bootstrap_objective(cost, ExtraArgs)

        rho = 1; chi = 2; psi = 0.5; sigma = 0.5;
        init = False  # flag to do 0th iteration 'post-initialization'

        if not len(self._stepmon): # do generation = 0
            init = True
            x0 = self.population[0]
            x0 = asfarray(x0).flatten()
            x0 = asfarray(self._constraints(x0))
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
            val = self._setSimplexWithinRangeBoundary(settings['radius'])
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
            one2np1 = list(range(1,N+1))

            # apply constraints  #XXX: is this the only appropriate place???
            sim[0] = asfarray(self._constraints(sim[0]))

            xbar = old_div(numpy.add.reduce(sim[:-1],0), N)
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
        if settings['callback'] is not None:
            settings['callback'](self.bestSolution)
        # initialize termination conditions, if needed
        if init: self._termination(self) #XXX: at generation 0 or always?
        return #XXX: call Terminated ?

    def _process_inputs(self, kwds):
        """process and activate input settings"""
        #allow for inputs that don't conform to AbstractSolver interface
        #NOTE: not sticky: callback, disp
        #NOTE: sticky: EvaluationMonitor, StepMonitor, penalty, constraints
        #NOTE: sticky: radius
        settings = super(NelderMeadSimplexSolver, self)._process_inputs(kwds)
        settings.update({\
        'radius':self.radius}) #percentage change for initial simplex values
        [settings.update({i:j}) for (i,j) in list(kwds.items()) if i in settings]
        self.radius = settings['radius']
        return settings

    def Solve(self, cost=None, termination=None, ExtraArgs=None,
              callback=None, **kwds):
        """Minimize a function using the downhill simplex algorithm.

Description:

    Uses a Nelder-Mead simplex algorithm to find the minimum of
    a function of one or more variables.

Inputs:

    cost -- the Python function or method to be minimized.

Additional Inputs:

    termination -- callable object providing termination conditions.
    ExtraArgs -- extra arguments for cost.

Further Inputs:

    sigint_callback -- callback function for signal handler.
    callback -- an optional user-supplied function to call after each
        iteration.  It is called as callback(xk), where xk is the
        current parameter vector.                           [default = None]
    disp -- non-zero to print convergence messages.         [default = 0]
    radius -- percentage change for initial simplex values. [default = 0.05]
"""
        super(NelderMeadSimplexSolver, self).Solve(cost, termination,\
                                                   ExtraArgs, **kwds)
        return


def fmin(cost, x0, args=(), bounds=None, xtol=1e-4, ftol=1e-4,
         maxiter=None, maxfun=None, full_output=0, disp=1, retall=0,
         callback=None, **kwds):
    """Minimize a function using the downhill simplex algorithm.

Description:

    Uses a Nelder-Mead simplex algorithm to find the minimum of
    a function of one or more variables. Mimics the scipy.optimize.fmin
    interface.

Inputs:

    cost -- the Python function or method to be minimized.
    x0 -- ndarray - the initial guess.

Additional Inputs:

    args -- extra arguments for cost.
    bounds -- list - n pairs of bounds (min,max), one pair for each parameter.
    xtol -- number - acceptable relative error in xopt for convergence.
    ftol -- number - acceptable relative error in cost(xopt) for
        convergence.
    maxiter -- number - the maximum number of iterations to perform.
    maxfun -- number - the maximum number of function evaluations.
    full_output -- number - non-zero if fval and warnflag outputs are
        desired.
    disp -- number - non-zero to print convergence messages.
    retall -- number - non-zero to return list of solutions at each
        iteration.
    callback -- an optional user-supplied function to call after each
        iteration.  It is called as callback(xk), where xk is the
        current parameter vector.
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

Returns: (xopt, {fopt, iter, funcalls, warnflag}, {allvecs})

    xopt -- ndarray - minimizer of function
    fopt -- number - value of function at minimum: fopt = cost(xopt)
    iter -- number - number of iterations
    funcalls -- number - number of function calls
    warnflag -- number - Integer warning flag:
        1 : 'Maximum number of function evaluations.'
        2 : 'Maximum number of iterations.'
    allvecs -- list - a list of solutions at each iteration

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
    if 'penalty' in kwds:
        solver.SetPenalty(kwds['penalty'])
    if 'constraints' in kwds:
        solver.SetConstraints(kwds['constraints'])
    if bounds is not None:
        minb,maxb = unpair(bounds)
        solver.SetStrictRanges(minb,maxb)

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

def _linesearch_powell(func, p, xi, tol=1e-3):
    # line-search algorithm using fminbound
    #  find the minimium of the function
    #  func(x0+ alpha*direc)
    def myfunc(alpha):
        return func(p + alpha * xi)
    settings = numpy.seterr(all='ignore')
    alpha_min, fret, iter, num = brent(myfunc, full_output=1, tol=tol)
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
        self.xtol  = 1e-4  #line-search error tolerance
        ftol, gtol = 1e-4, 2
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        self._termination = NCOG(ftol,gtol)

    def __generations(self):
        """get the number of iterations"""
        return max(0,len(self.energy_history)-1)  #XXX: slower for k=-1 ?

    def _SetEvaluationLimits(self, iterscale=1000, evalscale=1000):
        super(PowellDirectionalSolver, self)._SetEvaluationLimits(iterscale,evalscale)
        return

    def _Step(self, cost=None, ExtraArgs=None, **kwds):
        """perform a single optimization iteration
        Note that ExtraArgs should be a *tuple* of extra arguments"""
        # process and activate input settings
        settings = self._process_inputs(kwds)

        # HACK to enable not explicitly calling _decorate_objective
        cost = self._bootstrap_objective(cost, ExtraArgs)

        direc = self._direc #XXX: throws Error if direc=None after generation=0
        x = self.population[0]   # bestSolution
        fval = self.popEnergy[0] # bestEnergy
        x1, fx, bigind, delta = self.__internals
        init = False  # flag to do 0th iteration 'post-initialization'

        if not len(self._stepmon): # do generation = 0
            init = True
            x = asfarray(x).flatten()
            x = asfarray(self._constraints(x))
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
            ilist = list(range(len(x)))
            x1 = x.copy()
            # do initial "second half" of solver step
            fx = fval
            bigind = 0
            delta = 0.0
            for i in ilist:
                direc1 = self._direc[i]
                fx2 = fval
                fval, x, direc1 = _linesearch_powell(cost, x, direc1,
                                                     tol=settings['xtol']*100)
                if (fx2 - fval) > delta:
                    delta = fx2 - fval
                    bigind = i

                # apply constraints
                x = asfarray(self._constraints(x)) #XXX: use self._map?
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
                    fval, x, direc1 = _linesearch_powell(cost, x,
                                                         direc1,
                                                         tol=settings['xtol']*100)
                    direc[bigind] = direc[-1]
                    direc[-1] = direc1

           #        x = asfarray(self._constraints(x))

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
            ilist = list(range(len(x)))
            for i in ilist:
                direc1 = direc[i]
                fx2 = fval
                fval, x, direc1 = _linesearch_powell(cost, x, direc1,
                                                     tol=settings['xtol']*100)
                if (fx2 - fval) > delta:
                    delta = fx2 - fval
                    bigind = i

                # apply constraints
                x = asfarray(self._constraints(x)) #XXX: use self._map?

            # decouple from 'best' energy
            self.energy_history = self.energy_history + [fval]

        self.__internals = [x1, fx, bigind, delta]
        self._direc = direc
        self.population[0] = x   # bestSolution
        self.popEnergy[0] = fval # bestEnergy

        # do callback
        if settings['callback'] is not None:
            settings['callback'](self.bestSolution)
        # initialize termination conditions, if needed
        if init: self._termination(self) #XXX: at generation 0 or always?
        return #XXX: call Terminated ?

    def Finalize(self, **kwds):
        """cleanup upon exiting the main optimization loop"""
        if self.energy_history != None and self._live:
            self.energy_history = None # resync with 'best' energy
            self._stepmon(self.bestSolution, self.bestEnergy, self.id)
            # if savefrequency matches, then save state
            self._AbstractSolver__save_state()
        self._live = False
        return

    def _process_inputs(self, kwds):
        """process and activate input settings"""
        #allow for inputs that don't conform to AbstractSolver interface
        #NOTE: not sticky: callback, disp
        #NOTE: sticky: EvaluationMonitor, StepMonitor, penalty, constraints
        #NOTE: sticky: xtol, direc
        settings = super(PowellDirectionalSolver, self)._process_inputs(kwds)
        settings.update({\
        'xtol':self.xtol})   #line-search error tolerance
        direc=self._direc    #initial direction set
        [settings.update({i:j}) for (i,j) in list(kwds.items()) if i in settings]
        self._direc = kwds['direc'] if 'direc' in kwds else direc
        self.xtol = settings['xtol']
        return settings

    def Solve(self, cost=None, termination=None, ExtraArgs=None, **kwds):
        """Minimize a function using modified Powell's method.

Description:

    Uses a modified Powell Directional Search algorithm to find
    the minimum of function of one or more variables.

Inputs:

    cost -- the Python function or method to be minimized.

Additional Inputs:

    termination -- callable object providing termination conditions.
    ExtraArgs -- extra arguments for cost.

Further Inputs:

    sigint_callback -- callback function for signal handler.
    callback -- an optional user-supplied function to call after each
        iteration.  It is called as callback(xk), where xk is the
        current parameter vector
    direc -- initial direction set
    xtol -- line-search error tolerance.
    disp -- non-zero to print convergence messages.
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

Description:

    Uses a modified Powell Directional Search algorithm to find
    the minimum of function of one or more variables.  Mimics the
    scipy.optimize.fmin_powell interface.

Inputs:

    cost -- the Python function or method to be minimized.
    x0 -- ndarray - the initial guess.

Additional Inputs:

    args -- extra arguments for cost.
    bounds -- list - n pairs of bounds (min,max), one pair for each parameter.
    xtol -- number - acceptable relative error in xopt for
        convergence.
    ftol -- number - acceptable relative error in cost(xopt) for
        convergence.
    gtol -- number - maximum number of iterations to run without improvement.
    maxiter -- number - the maximum number of iterations to perform.
    maxfun -- number - the maximum number of function evaluations.
    full_output -- number - non-zero if fval and warnflag outputs
        are desired.
    disp -- number - non-zero to print convergence messages.
    retall -- number - non-zero to return list of solutions at each
        iteration.
    callback -- an optional user-supplied function to call after each
        iteration.  It is called as callback(xk), where xk is the
        current parameter vector.
    direc -- initial direction set.
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

Returns: (xopt, {fopt, iter, funcalls, warnflag, direc}, {allvecs})

    xopt -- ndarray - minimizer of function
    fopt -- number - value of function at minimum: fopt = cost(xopt)
    iter -- number - number of iterations
    funcalls -- number - number of function calls
    warnflag -- number - Integer warning flag:
        1 : 'Maximum number of function evaluations.'
        2 : 'Maximum number of iterations.'
    direc -- current direction set
    allvecs -- list - a list of solutions at each iteration

    """
    #FIXME: need to resolve "direc"
    #        - should just pass 'direc', and then hands-off ?  How return it ?

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
    if 'penalty' in kwds:
        solver.SetPenalty(kwds['penalty'])
    if 'constraints' in kwds:
        solver.SetConstraints(kwds['constraints'])
    if bounds is not None:
        minb,maxb = unpair(bounds)
        solver.SetStrictRanges(minb,maxb)

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
