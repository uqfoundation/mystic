#!/usr/bin/env python

"""
Example:
    - Solve 8th-order Chebyshev polynomial coefficients with Nelder-Mead.
    - Callable plot of fitting to Chebyshev polynomial.
    - Plot (x2) of convergence to Chebyshev polynomial.
    - Monitor (x2) Chi-Squared for Chebyshev polynomial.

Demonstrates:
    - standard models
    - expanded solver interface
    - parameter bounds constraints
    - solver interactivity
    - customized monitors and termination conditions
"""

# Nelder-Mead Simplex solver
from mystic.scipy_optimize import NelderMeadSimplexSolver

# Chebyshev polynomial and cost function
from mystic.models.poly import chebyshev8, chebyshev8cost

# tools
from mystic.termination import CandidateRelativeTolerance as CRT
from mystic import getch, random_seed, VerboseSow, Sow
from mystic.models.poly import poly1d, chebyshev8coeffs
import pylab
pylab.ion()

# draw the plot
def plot_frame(label=None):
    pylab.close()
    pylab.title("8th-order Chebyshev coefficient convergence")
    pylab.xlabel("Nelder-Mead Simplex Solver %s" % label)
    pylab.ylabel("Chi-Squared")
    return
 
# plot the polynomial trajectories
def plot_params(monitor):
    x = range(len(monitor.y))
    y = monitor.y
    pylab.plot(x,y,'b-')
    pylab.axis([1,0.5*x[-1],0,y[1]],'k-')
    return

# draw the plot
def plot_exact():
    pylab.title("fitting 8th-order Chebyshev polynomial coefficients")
    pylab.xlabel("x")
    pylab.ylabel("f(x)")
    import numpy
    x = numpy.arange(-1.2, 1.2001, 0.01)
    exact = chebyshev8(x)
    pylab.plot(x,exact,'b-')
    pylab.legend(["Exact"])
    pylab.axis([-1.4,1.4,-2,8],'k-')
    return
 
# plot the polynomial
def plot_solution(params):
    import numpy
    x = numpy.arange(-1.2, 1.2001, 0.01)
    f = poly1d(params)
    y = f(x)
    pylab.plot(x,y,'y-')
    pylab.legend(["Exact","Fitted"])
    pylab.axis([-1.4,1.4,-2,8],'k-')
    return

if __name__ == '__main__':

    print "Nelder-Mead Simplex"
    print "==================="

    # initial guess
    import random
    random.seed(123)
    ndim = 9
    x0 = [random.uniform(-5,5) + chebyshev8coeffs[i] for i in range(ndim)]

    # suggest that the user interacts with the solver
    print "NOTE: while solver is running, press 'Ctrl-C' in console window"
    getch()

    # draw frame and exact coefficients
    plot_exact()

    # select parameter bounds constraints
    from numpy import inf
    min_bounds = [  0,-1,-300,-1,  0,-1,-100,-inf,-inf]
    max_bounds = [200, 1,   0, 1,200, 1,   0, inf, inf]

    # configure monitors
    stepmon = VerboseSow(100)
    evalmon = Sow()

    # use Nelder-Mead to solve 8th-order Chebyshev coefficients
    solver = NelderMeadSimplexSolver(ndim)
    solver.SetInitialPoints(x0)
    solver.SetEvaluationLimits(maxiter=999)
    solver.SetStrictRanges(min_bounds,max_bounds)
    solver.enable_signal_handler()
    solver.Solve(chebyshev8cost,termination=CRT(1e-4,1e-4), \
                 StepMonitor=stepmon, EvaluationMonitor=evalmon,
                 sigint_callback=plot_solution)
    solution = solver.Solution()

    # print solved coefficients and Chi-Squared
    iterations = len(stepmon.x)
    cost = solver.bestEnergy
    print "\nGeneration %d has best Chi-Squared: %f" % (iterations, cost)
    print "Solved Coefficients:\n %s\n" % poly1d(solver.bestSolution)

    # compare solution with actual 8th-order Chebyshev coefficients
    print "Actual Coefficients:\n %s\n" % poly1d(chebyshev8coeffs)

    # plot solution versus exact coefficients
    plot_solution(solution) 
    getch()

    # plot convergence of coefficients per iteration
    plot_frame('iterations')
    plot_params(stepmon) 
    getch()

    # plot convergence of coefficients per function call
    plot_frame('function calls')
    plot_params(evalmon) 
    getch()

# end of file
