#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
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
from mystic.solvers import NelderMeadSimplexSolver

# Chebyshev polynomial and cost function
from mystic.models.poly import chebyshev8, chebyshev8cost
from mystic.models.poly import chebyshev8coeffs

# tools
from mystic.termination import CandidateRelativeTolerance as CRT
from mystic.monitors import VerboseMonitor, Monitor
from mystic.tools import getch
from mystic.math import poly1d
import matplotlib.pyplot as plt
plt.ion()

# draw the plot
def plot_frame(label=None):
    plt.close()
    plt.title("8th-order Chebyshev coefficient convergence")
    plt.xlabel("Nelder-Mead Simplex Solver %s" % label)
    plt.ylabel("Chi-Squared")
    plt.draw()
    plt.pause(0.001)
    return
 
# plot the polynomial trajectories
def plot_params(monitor):
    x = list(range(len(monitor)))
    y = monitor.y
    plt.plot(x,y,'b-')
    plt.axis([1,0.5*x[-1],0,y[1]])#,'k-')
    plt.draw()
    plt.pause(0.001)
    return

# draw the plot
def plot_exact():
    plt.title("fitting 8th-order Chebyshev polynomial coefficients")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    import numpy
    x = numpy.arange(-1.2, 1.2001, 0.01)
    exact = chebyshev8(x)
    plt.plot(x,exact,'b-')
    plt.legend(["Exact"])
    plt.axis([-1.4,1.4,-2,8])#,'k-')
    plt.draw()
    plt.pause(0.001)
    return
 
# plot the polynomial
def plot_solution(params,style='y-'):
    import numpy
    x = numpy.arange(-1.2, 1.2001, 0.01)
    f = poly1d(params)
    y = f(x)
    plt.plot(x,y,style)
    plt.legend(["Exact","Fitted"])
    plt.axis([-1.4,1.4,-2,8])#,'k-')
    plt.draw()
    plt.pause(0.001)
    return

if __name__ == '__main__':

    print("Nelder-Mead Simplex")
    print("===================")

    # initial guess
    import random
    from mystic.tools import random_seed
    random_seed(123)
    ndim = 9
    x0 = [random.uniform(-5,5) + chebyshev8coeffs[i] for i in range(ndim)]

    # suggest that the user interacts with the solver
    print("NOTE: while solver is running, press 'Ctrl-C' in console window")
    getch()

    # draw frame and exact coefficients
    plot_exact()

    # select parameter bounds constraints
    from numpy import inf
    min_bounds = [  0,-1,-300,-1,  0,-1,-100,-inf,-inf]
    max_bounds = [200, 1,   0, 1,200, 1,   0, inf, inf]

    # configure monitors
    stepmon = VerboseMonitor(100)
    evalmon = Monitor()

    # use Nelder-Mead to solve 8th-order Chebyshev coefficients
    solver = NelderMeadSimplexSolver(ndim)
    solver.SetInitialPoints(x0)
    solver.SetEvaluationLimits(generations=999)
    solver.SetEvaluationMonitor(evalmon)
    solver.SetGenerationMonitor(stepmon)
    solver.SetStrictRanges(min_bounds,max_bounds)
    solver.enable_signal_handler()
    solver.Solve(chebyshev8cost, termination=CRT(1e-4,1e-4), \
                 sigint_callback=plot_solution)
    solution = solver.bestSolution

    # get solved coefficients and Chi-Squared (from solver members)
    iterations = solver.generations
    cost = solver.bestEnergy
    print("Generation %d has best Chi-Squared: %f" % (iterations, cost))
    print("Solved Coefficients:\n %s\n" % poly1d(solver.bestSolution))

    # compare solution with actual 8th-order Chebyshev coefficients
    print("Actual Coefficients:\n %s\n" % poly1d(chebyshev8coeffs))

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
