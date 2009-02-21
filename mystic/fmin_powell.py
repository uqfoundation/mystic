#!/usr/bin/env python

# evolution notes...

"""
Algorithm adapted from scipy.optimize, 

fmin_powell ---      Powell's method, with directional search
                     Takes two additional input args
                        -- EvaluationMonitor
                        -- StepMonitor
                     
"""

from mystic.tools import Null, wrap_function
from mystic.tools import wrap_bounds

import numpy
from numpy import atleast_1d, eye, zeros, shape, \
     asarray, absolute, sqrt, Inf, asfarray
from numpy import clip, squeeze

abs = absolute


from scipy_optimize import NelderMeadSimplexSolver
from scipy.optimize import brent #FIXME: replace w/ my version!

def _linesearch_powell(func, p, xi, tol=1e-3):
    # line-search algorithm using fminbound
    #  find the minimium of the function
    #  func(x0+ alpha*direc)
    def myfunc(alpha):
        return func(p + alpha * xi)
    alpha_min, fret, iter, num = brent(myfunc, full_output=1, tol=tol)
    xi = alpha_min*xi
    return squeeze(fret), p+xi, xi


class PowellDirectionalSolver(NelderMeadSimplexSolver): #FIXME: not a simplex solver
    """
    Downhill optimization with Powell's method adapted from scipy.optimize.fmin_powell.
    """
    
    def __init__(self, dim):
        NelderMeadSimplexSolver.__init__(self,dim)
        self._direc = None #FIXME: this is the easy way to return 'direc'...
       #FIXME: NO SIMPLEX, so maybe best not to inherit __init__?


    def Solve(self, func, termination,
              maxiter=None, maxfun=None, sigint_callback=None,
              EvaluationMonitor=Null, StepMonitor=Null, ExtraArgs=(), **kwds):
        #FIXME: trim the following doc...
        """Minimize a function using modified Powell's method.

    :Parameters:

      func -- the Python function or method to be minimized.
      x0 : ndarray
        the initial guess.
      args -- extra arguments for func
      callback -- an optional user-supplied function to call after each
                  iteration.  It is called as callback(xk), where xk is the
                  current parameter vector
      direc -- initial direction set

    :Returns: (xopt, {fopt, xi, direc, iter, funcalls, warnflag}, {allvecs})

      xopt : ndarray
        minimizer of function

      fopt : number
        value of function at minimum: fopt = func(xopt)
      direc -- current direction set
      iter : number
        number of iterations
      funcalls : number 
        number of function calls
      warnflag : number
        Integer warning flag:
                  1 : 'Maximum number of function evaluations.'
                  2 : 'Maximum number of iterations.'
      allvecs : Python list
        a list of solutions at each iteration

    :OtherParameters:

      xtol : number
        line-search error tolerance.
      ftol : number
        acceptable relative error in func(xopt) for convergence.
      maxiter : number
        the maximum number of iterations to perform.
      maxfun : number
        the maximum number of function evaluations.
      full_output : number
        non-zero if fval and warnflag outputs are desired.
      disp : number
        non-zero to print convergence messages.
      retall : number
        non-zero to return a list of the solution at each iteration

    Notes
    
    -----------------------

      Uses a modification of Powell's method to find the minimum of a function
      of N variables
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
        if kwds.has_key('disp'): disp = kwds['disp']
        if kwds.has_key('xtol'): xtol = kwds['xtol']
        if kwds.has_key('direc'): direc = kwds['direc']  #XXX: best interface?
        if kwds.has_key('callback'): callback = kwds['callback']  #XXX: best interface (or better pushed into stepmon)?
        #-------------------------------------------------------------

        import signal
        import mystic.termination as detools
        detools.EARLYEXIT = False

        fcalls, func = wrap_function(func, args, EvaluationMonitor)
        if self._useStrictRange:
            x0 = self._setGuessWithinRangeBoundary(x0)
            func = wrap_bounds(func, self._strictMin, self._strictMax)

        def handler(signum, frame):
            import inspect
            print inspect.getframeinfo(frame)
            print inspect.trace()
            while 1:
                s = raw_input(\
"""
 
 Enter sense switch.

   sol: Write current best solution.
   cont: Continue calculation.
   call: Executes sigint_callback [%s].
   exit: Exits with current best solution.

 >>> """ % sigint_callback)
                if s.lower() == 'sol': 
                    print "sw1."
                    print self.bestSolution
                elif s.lower() == 'cont': 
                    return
                elif s.lower() == 'call': 
                    # sigint call_back
                    if sigint_callback is not None:
                        sigint_callback(self.bestSolution)
                elif s.lower() == 'exit': 
                    detools.EARLYEXIT = True
                    return
                else:
                    print "unknown option : %s ", s

        self.signal_handler = handler

        if self._handle_sigint: signal.signal(signal.SIGINT, self.signal_handler)
        #-------------------------------------------------------------

        x = asfarray(x0).flatten()
        if retall:
            allvecs = [x]
        N = len(x) #XXX: this should be equal to self.nDim
        rank = len(x.shape)
        if not -1 < rank < 2:
            raise ValueError, "Initial guess must be a scalar or rank-1 sequence."
        if maxiter is None:
            maxiter = N * 1000
        if maxfun is None:
            maxfun = N * 1000
        self._maxiter = maxiter #XXX: better to just copy the code?
        self._maxfun = maxfun   #XXX: better to just copy the code?

        if direc is None:
            direc = eye(N, dtype=float)
        else:
            direc = asarray(direc, dtype=float)
        fval = squeeze(func(x))
        x1 = x.copy()

        self._direc = direc #XXX: instead, use a monitor?
        self.bestSolution = x
        self.bestEnergy = fval
        self.population = [x]    #XXX: pointless, if simplex not used
        self.popEnergy = [fval]  #XXX: pointless, if simplex not used
        self.energy_history.append(self.bestEnergy)
        StepMonitor(x,fval) # get initial values

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
            if detools.EARLYEXIT or termination(self): CONTINUE = False #break
            elif fcalls[0] >= maxfun: CONTINUE = False #break
            elif iter >= maxiter: CONTINUE = False #break

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
            self.population = [x]    #XXX: pointless, if simplex not used
            self.popEnergy = [fval]  #XXX: pointless, if simplex not used
            StepMonitor(x,fval) # get ith values; #XXX: should be [x],[fval] ?
    
        self.generations = iter
        signal.signal(signal.SIGINT,signal.default_int_handler)

        # code below here is dead, unless disp!=0
        warnflag = 0

        if fcalls[0] >= maxfun:
            warnflag = 1
            if disp:
                print "Warning: Maximum number of function evaluations has "\
                      "been exceeded."
        elif iter >= maxiter:
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
    """fmin_powell using the 'original' scipy.optimize.fmin_powell interface"""

    #FIXME: need to resolve "callback" and "direc"
    #        - should just pass 'direc', and then hands-off ?  How return it ?
    #        - 'callback' should get pulled into the stepmon ?

    from mystic.tools import Sow
    stepmon = Sow()
    evalmon = Sow()
    from mystic.termination import NormalizedChangeOverGeneration as NCOG

    solver = PowellDirectionalSolver(len(x0))
    solver.SetInitialPoints(x0)
   #solver.enable_signal_handler()
    solver.Solve(func,termination=NCOG(ftol),\
                 maxiter=maxiter,maxfun=maxfun,\
                 EvaluationMonitor=evalmon,StepMonitor=stepmon,\
                 xtol=xtol, callback=callback, \
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
