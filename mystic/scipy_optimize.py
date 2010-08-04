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
# updated solvers to scipy version 0.6.0
# by mmckerns@caltech.edu

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
   - EvaluationMonitor = Sow()
   - StepMonitor = Sow()
   - termination = CandidateRelativeTolerance(xtol,ftol)
   - enable_signal_handler()

Mystic solver behavior activated in fmin_powell::
   - EvaluationMonitor = Sow()
   - StepMonitor = Sow()
   - termination = NormalizedChangeOverGeneration(ftol)
   - enable_signal_handler()


Usage
=====

See `mystic.examples.test_rosenbrock2` for an example of using
NelderMeadSimplexSolver. See `mystic.examples.test_rosenbrock3`
or an example of using PowellDirectionalSolver.

All solvers included in this module provide the standard signal handling.
For more information, see `mystic.mystic.abstract_solver`.

"""
__all__ = ['NelderMeadSimplexSolver','PowellDirectionalSolver',
           'fmin','fmin_powell']


from mystic.tools import Null, wrap_function
from mystic.tools import wrap_bounds

import numpy
from numpy import eye, zeros, shape, asarray, absolute, asfarray
from numpy import clip, squeeze

abs = absolute

from _scipy060optimize import brent #XXX: local copy to avoid dependency!

from abstract_solver import AbstractSolver

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
        self.popEnergy	   = [0.0] * simplex
        self.population	   = [[0.0 for i in range(dim)] for j in range(simplex)]

    def _setSimplexWithinRangeBoundary(self, x0, radius): #XXX: use population?
        """ensure that initial simplex is set within bounds
    - x0: must be a sequence of length self.nDim
    - radius: size of the initial simplex"""
        #code modified from park-1.2/park/simplex.py (version 1257)
        if self._useStrictRange:
            x0 = self._clipGuessWithinRangeBoundary(x0)

        val = x0*(1+radius)
        val[val==0] = radius
        if not self._useStrictRange:
            return x0, val

        lo = self._strictMin
        hi = self._strictMax
        radius = clip(radius,0,0.5)
        # rescale val by bounded range...
        # (increases fit for tight bounds; makes worse[?] for large bounds)
        bounded = ~numpy.isinf(lo) & ~numpy.isinf(hi)
        val[bounded] = x0[bounded] + (hi[bounded]-lo[bounded])*radius
        # crop val at bounds
        val[val<lo] = lo[val<lo]
        val[val>hi] = hi[val>hi]
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
        return x0, val

    def Solve(self, func, termination, sigint_callback=None,
              EvaluationMonitor=Null, StepMonitor=Null, ExtraArgs=(), **kwds):
        """Minimize a function using the downhill simplex algorithm.

Description:

    Uses a Nelder-Mead simplex algorithm to find the minimum of
    a function of one or more variables.

Inputs:

    func -- the Python function or method to be minimized.
    termination -- callable object providing termination conditions.

Additional Inputs:

    sigint_callback -- callback function for signal handler.
    EvaluationMonitor -- a callable object that will be passed x, fval
        whenever the cost function is evaluated.
    StepMonitor -- a callable object that will be passed x, fval
        after the end of a simplex iteration.
    ExtraArgs -- extra arguments for func.

Further Inputs:

    callback -- an optional user-supplied function to call after each
        iteration.  It is called as callback(xk), where xk is the
        current parameter vector.                           [default = None]
    disp -- non-zero to print convergence messages.         [default = 0]
    radius -- percentage change for initial simplex values. [default = 0.05]

"""
        # set arg names to scipy.optimize.fmin names; set fixed inputs
        x0 = self.population[0]
        args = ExtraArgs
        full_output=1  #non-zero if fval and warnflag outputs are desired.
        disp=0         #non-zero to print convergence messages.
        retall=0       #non-zero to return all steps
        callback=None  #user-supplied function, called after each step
        radius=0.05    #percentage change for initial simplex values
        if kwds.has_key('callback'): callback = kwds['callback']
        if kwds.has_key('disp'): disp = kwds['disp']
        if kwds.has_key('radius'): radius = kwds['radius']
        #-------------------------------------------------------------

        import signal
        self._EARLYEXIT = False

        fcalls, func = wrap_function(func, args, EvaluationMonitor)
        if self._useStrictRange:
            x0 = self._clipGuessWithinRangeBoundary(x0)
            func = wrap_bounds(func, self._strictMin, self._strictMax)

        #generate signal_handler
        self._generateHandler(sigint_callback) 
        if self._handle_sigint: signal.signal(signal.SIGINT, self.signal_handler)
        #-------------------------------------------------------------

        id = self.id
        x0 = asfarray(x0).flatten()
        N = len(x0) #XXX: this should be equal to self.nDim
        rank = len(x0.shape)
        if not -1 < rank < 2:
            raise ValueError, "Initial guess must be a scalar or rank-1 sequence."
        if self._maxiter is None:
            self._maxiter = N * 200
        if self._maxfun is None:
            self._maxfun = N * 200

        rho = 1; chi = 2; psi = 0.5; sigma = 0.5;
        one2np1 = range(1,N+1)

        if rank == 0:
            sim = numpy.zeros((N+1,), dtype=x0.dtype)
        else:
            sim = numpy.zeros((N+1,N), dtype=x0.dtype)
        fsim = numpy.zeros((N+1,), float)
        sim[0] = x0
        if retall:
            allvecs = [sim[0]]
        fsim[0] = func(x0)
        StepMonitor(sim[0], fsim[0], id) # sim = all; "best" is sim[0]

        #--- ensure initial simplex is within bounds ---
        x0,val = self._setSimplexWithinRangeBoundary(x0,radius)
        #--- end bounds code ---
        for k in range(0,N):
            y = numpy.array(x0,copy=True)
            y[k] = val[k]
            sim[k+1] = y
            f = func(y)
            fsim[k+1] = f
    
        ind = numpy.argsort(fsim)
        fsim = numpy.take(fsim,ind,0)
        # sort so sim[0,:] has the lowest function value
        sim = numpy.take(sim,ind,0)
        self.bestSolution = sim[0]
        self.bestEnergy = min(fsim)
        self.population = sim
        self.popEnergy = fsim
        self.energy_history.append(self.bestEnergy)
        StepMonitor(sim[0], fsim[0], id) # sim = all; "best" is sim[0]

        iterations = 1

        while (fcalls[0] < self._maxfun and iterations < self._maxiter):
            if self._EARLYEXIT or termination(self):
                break

            xbar = numpy.add.reduce(sim[:-1],0) / N
            xr = (1+rho)*xbar - rho*sim[-1]
            fxr = func(xr)
            doshrink = 0

            if fxr < fsim[0]:
                xe = (1+rho*chi)*xbar - rho*chi*sim[-1]
                fxe = func(xe)

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
                        fxc = func(xc)
    
                        if fxc <= fxr:
                            sim[-1] = xc
                            fsim[-1] = fxc
                        else:
                            doshrink=1
                    else:
                        # Perform an inside contraction
                        xcc = (1-psi)*xbar + psi*sim[-1]
                        fxcc = func(xcc)

                        if fxcc < fsim[-1]:
                            sim[-1] = xcc
                            fsim[-1] = fxcc
                        else:
                            doshrink = 1

                    if doshrink:
                        for j in one2np1:
                            sim[j] = sim[0] + sigma*(sim[j] - sim[0])
                            fsim[j] = func(sim[j])

            ind = numpy.argsort(fsim)
            sim = numpy.take(sim,ind,0)
            fsim = numpy.take(fsim,ind,0)
            if callback is not None:
                callback(sim[0])
            iterations = iterations + 1
            if retall:
                allvecs.append(sim[0])

            self.bestSolution = sim[0]
            self.bestEnergy = min(fsim)
            self.population = sim
            self.popEnergy = fsim
            self.energy_history.append(self.bestEnergy)
            StepMonitor(sim[0], fsim[0],id) # sim = all; "best" is sim[0]

        self.generations = iterations
        signal.signal(signal.SIGINT,signal.default_int_handler)

        # code below here is dead, unless disp!=0
        x = sim[0]
        fval = min(fsim)
        warnflag = 0

        if fcalls[0] >= self._maxfun:
            warnflag = 1
            if disp:
                print "Warning: Maximum number of function evaluations has "\
                  "been exceeded."
        elif iterations >= self._maxiter:
            warnflag = 2
            if disp:
                print "Warning: Maximum number of iterations has been exceeded"
        else:
            if disp:
                print "Optimization terminated successfully."
                print "         Current function value: %f" % fval
                print "         Iterations: %d" % iterations
                print "         Function evaluations: %d" % fcalls[0]


        if full_output:
            retlist = x, fval, iterations, fcalls[0], warnflag
            if retall:
                retlist += (allvecs,)
        else:
            retlist = x
            if retall:
                retlist = (x, allvecs)

        return #retlist


def fmin(func, x0, args=(), xtol=1e-4, ftol=1e-4, maxiter=None, maxfun=None,
         full_output=0, disp=1, retall=0, callback=None):
    """Minimize a function using the downhill simplex algorithm.
    
Description:

    Uses a Nelder-Mead simplex algorithm to find the minimum of
    a function of one or more variables. Mimics the scipy.optimize.fmin
    interface.

Inputs:

    func -- the Python function or method to be minimized.
    x0 -- ndarray - the initial guess.

Additional Inputs:

    args -- extra arguments for func.
    xtol -- number - acceptable relative error in xopt for convergence.
    ftol -- number - acceptable relative error in func(xopt) for
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

Returns: (xopt, {fopt, iter, funcalls, warnflag}, {allvecs})

    xopt -- ndarray - minimizer of function
    fopt -- number - value of function at minimum: fopt = func(xopt)
    iter -- number - number of iterations
    funcalls -- number - number of function calls
    warnflag -- number - Integer warning flag:
        1 : 'Maximum number of function evaluations.'
        2 : 'Maximum number of iterations.'
    allvecs -- list - a list of solutions at each iteration

    """

    from mystic.tools import Sow
    stepmon = Sow()
    evalmon = Sow()
    from mystic.termination import CandidateRelativeTolerance as CRT

    solver = NelderMeadSimplexSolver(len(x0))
    solver.SetInitialPoints(x0)
    solver.enable_signal_handler()
    solver.SetEvaluationLimits(maxiter,maxfun)
    solver.Solve(func,termination=CRT(xtol,ftol),\
                 EvaluationMonitor=evalmon,StepMonitor=stepmon,\
                 disp=disp, ExtraArgs=args, callback=callback)
    solution = solver.Solution()

    # code below here pushes output to scipy.optimize.fmin interface
   #x = list(solver.bestSolution)
    x = solver.bestSolution
    fval = solver.bestEnergy
    warnflag = 0
    fcalls = len(evalmon.x)
    iterations = len(stepmon.x)
    allvecs = []
    for i in range(iterations):
       #allvecs.append(list(stepmon.x[i]))
        allvecs.append(stepmon.x[i])

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
    alpha_min, fret, iter, num = brent(myfunc, full_output=1, tol=tol)
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
        self._direc = None #FIXME: this is the easy way to return 'direc'...


    def Solve(self, func, termination, sigint_callback=None,
              EvaluationMonitor=Null, StepMonitor=Null, ExtraArgs=(), **kwds):
        """Minimize a function using modified Powell's method.

Description:

    Uses a modified Powell Directional Search algorithm to find
    the minimum of function of one or more variables.

Inputs:

    func -- the Python function or method to be minimized.
    termination -- callable object providing termination conditions.

Additional Inputs:

    sigint_callback -- callback function for signal handler.
    EvaluationMonitor -- a callable object that will be passed x, fval
        whenever the cost function is evaluated.
    StepMonitor -- a callable object that will be passed x, fval
        after the end of a simplex iteration.
    ExtraArgs -- extra arguments for func.

Further Inputs:

    callback -- an optional user-supplied function to call after each
        iteration.  It is called as callback(xk), where xk is the
        current parameter vector
    direc -- initial direction set
    xtol -- line-search error tolerance.
    disp -- non-zero to print convergence messages.

"""

        # set arg names to scipy.optimize.fmin_powell names; set fixed inputs
        x0 = self.population[0]
        args = ExtraArgs
        full_output=1  #non-zero if fval and warnflag outputs are desired.
        disp=0         #non-zero to print convergence messages.
        retall=0       #non-zero to return all steps
        direc=None
        callback=None  #user-supplied function, called after each step
        xtol=1e-4      #line-search error tolerance
        if kwds.has_key('callback'): callback = kwds['callback']
        if kwds.has_key('direc'): direc = kwds['direc']  #XXX: best interface?
        if kwds.has_key('xtol'): xtol = kwds['xtol']
        if kwds.has_key('disp'): disp = kwds['disp']
        #-------------------------------------------------------------

        import signal
        self._EARLYEXIT = False

        fcalls, func = wrap_function(func, args, EvaluationMonitor)
        if self._useStrictRange:
            x0 = self._clipGuessWithinRangeBoundary(x0)
            func = wrap_bounds(func, self._strictMin, self._strictMax)

        #generate signal_handler
        self._generateHandler(sigint_callback) 
        if self._handle_sigint: signal.signal(signal.SIGINT, self.signal_handler)
        #-------------------------------------------------------------

        id = self.id
        x = asfarray(x0).flatten()
        if retall:
            allvecs = [x]
        N = len(x) #XXX: this should be equal to self.nDim
        rank = len(x.shape)
        if not -1 < rank < 2:
            raise ValueError, "Initial guess must be a scalar or rank-1 sequence."
        if self._maxiter is None:
            self._maxiter = N * 1000
        if self._maxfun is None:
            self._maxfun = N * 1000

        if direc is None:
            direc = eye(N, dtype=float)
        else:
            direc = asarray(direc, dtype=float)
        fval = squeeze(func(x))
        x1 = x.copy()

        self._direc = direc #XXX: instead, use a monitor?
        self.bestSolution = x
        self.bestEnergy = fval
        self.population[0] = x    #XXX: pointless?
        self.popEnergy[0] = fval  #XXX: pointless?
        self.energy_history.append(self.bestEnergy)
        StepMonitor(x, fval, id) # get initial values

        iter = 0;
        ilist = range(N)

        CONTINUE = True
        while CONTINUE:
            fx = fval
            bigind = 0
            delta = 0.0
            for i in ilist:
                direc1 = direc[i]
                fx2 = fval
                fval, x, direc1 = _linesearch_powell(func, x, direc1, tol=xtol*100)
                if (fx2 - fval) > delta:
                    delta = fx2 - fval
                    bigind = i

            iter += 1
            if callback is not None:
                callback(x)
            if retall:
                allvecs.append(x)

            self.energy_history.append(fval) #XXX: the 'best' for now...
            if self._EARLYEXIT or termination(self): CONTINUE = False #break
            elif fcalls[0] >= self._maxfun: CONTINUE = False #break
            elif iter >= self._maxiter: CONTINUE = False #break

            else: # Construct the extrapolated point
                direc1 = x - x1
                x2 = 2*x - x1
                x1 = x.copy()
                fx2 = squeeze(func(x2))
    
                if (fx > fx2):
                    t = 2.0*(fx+fx2-2.0*fval)
                    temp = (fx-fval-delta)
                    t *= temp*temp
                    temp = fx-fx2
                    t -= delta*temp*temp
                    if t < 0.0:
                        fval, x, direc1 = _linesearch_powell(func, x, direc1, tol=xtol*100)
                        direc[bigind] = direc[-1]
                        direc[-1] = direc1

                self.energy_history[-1] = fval #...update to 'best' energy

            self._direc = direc #XXX: instead, use a monitor?
            self.bestSolution = x
            self.bestEnergy = fval
            self.population[0] = x    #XXX: pointless
            self.popEnergy[0] = fval  #XXX: pointless
            StepMonitor(x, fval, id) # get ith values
    
        self.generations = iter
        signal.signal(signal.SIGINT,signal.default_int_handler)

        # code below here is dead, unless disp!=0
        warnflag = 0

        if fcalls[0] >= self._maxfun:
            warnflag = 1
            if disp:
                print "Warning: Maximum number of function evaluations has "\
                      "been exceeded."
        elif iter >= self._maxiter:
            warnflag = 2
            if disp:
                print "Warning: Maximum number of iterations has been exceeded"
        else:
            if disp:
                print "Optimization terminated successfully."
                print "         Current function value: %f" % fval
                print "         Iterations: %d" % iter
                print "         Function evaluations: %d" % fcalls[0]
    
        x = squeeze(x)

        if full_output:
            retlist = x, fval, direc, iter, fcalls[0], warnflag
            if retall:
                retlist += (allvecs,)
        else:
            retlist = x
            if retall:
                retlist = (x, allvecs)

        return #retlist


def fmin_powell(func, x0, args=(), xtol=1e-4, ftol=1e-4, maxiter=None,
                maxfun=None, full_output=0, disp=1, retall=0, callback=None,
                direc=None):
    """Minimize a function using modified Powell's method.
    
Description:

    Uses a modified Powell Directional Search algorithm to find
    the minimum of function of one or more variables.  Mimics the
    scipy.optimize.fmin_powell interface.

Inputs:

    func -- the Python function or method to be minimized.
    x0 -- ndarray - the initial guess.

Additional Inputs:

    args -- extra arguments for func.
    xtol -- number - acceptable relative error in xopt for
        convergence.
    ftol -- number - acceptable relative error in func(xopt) for
        convergence.
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
    direc -- initial direction set

Returns: (xopt, {fopt, direc, iter, funcalls, warnflag}, {allvecs})

    xopt -- ndarray - minimizer of function
    fopt -- number - value of function at minimum: fopt = func(xopt)
    direc -- current direction set
    iter -- number - number of iterations
    funcalls -- number - number of function calls
    warnflag -- number - Integer warning flag:
        1 : 'Maximum number of function evaluations.'
        2 : 'Maximum number of iterations.'
    allvecs -- list - a list of solutions at each iteration

    """
    #FIXME: need to resolve "direc"
    #        - should just pass 'direc', and then hands-off ?  How return it ?

    from mystic.tools import Sow
    stepmon = Sow()
    evalmon = Sow()
    from mystic.termination import NormalizedChangeOverGeneration as NCOG

    solver = PowellDirectionalSolver(len(x0))
    solver.SetInitialPoints(x0)
    solver.enable_signal_handler()
    solver.SetEvaluationLimits(maxiter,maxfun)
    solver.Solve(func,termination=NCOG(ftol),\
                 EvaluationMonitor=evalmon,StepMonitor=stepmon,\
                 xtol=xtol, ExtraArgs=args, callback=callback, \
                 disp=disp, direc=direc)   #XXX: last two lines use **kwds
    solution = solver.Solution()

    # code below here pushes output to scipy.optimize.fmin_powell interface
   #x = list(solver.bestSolution)
    x = solver.bestSolution
    fval = solver.bestEnergy
    warnflag = 0
    fcalls = len(evalmon.x)
    iterations = len(stepmon.x) - 1
    allvecs = []
    for i in range(len(stepmon.x)):
       #allvecs.append(list(stepmon.x[i]))
        allvecs.append(stepmon.x[i])
    direc = solver._direc #FIXME: better way to get direc from Solve() ?

    if fcalls >= solver._maxfun:
        warnflag = 1
    elif iterations >= solver._maxiter:
        warnflag = 2

    x = squeeze(x) #FIXME: write squeezed x to stepmon instead?

    if full_output:
        retlist = x, fval, direc, iterations, fcalls, warnflag
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
