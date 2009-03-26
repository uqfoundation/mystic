#!/usr/bin/env python

"""
Example:
    - Solve 8th-order Chebyshev polynomial coefficients with DE.
    - Callable plot of fitting to Chebyshev polynomial.
    - Monitor Chi-Squared for Chebyshev polynomial.
    - Impact of mutation and crossover coefficients

Demonstrates:
    - standard models
    - expanded solver interface
    - built-in random initial guess
    - solver interactivity
    - customized monitors and termination conditions
    - customized DE mutation strategies
"""

# Differential Evolution solver
from mystic.differential_evolution import DifferentialEvolutionSolver2

# Chebyshev polynomial and cost function
from mystic.models.poly import chebyshev8, chebyshev8cost

# tools
from mystic.termination import VTR
from mystic.strategy import Best1Exp
from mystic import getch, random_seed, VerboseSow
from mystic.models.poly import poly1d, chebyshev8coeffs
import pylab
pylab.ion()

# draw the plot
def plot_exact():
    pylab.suptitle("fitting 8th-order Chebyshev polynomial coefficients")
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

    print "Differential Evolution"
    print "======================"

    # set range for random initial guess
    ndim = 9
    x0 = [(-100,100)]*ndim
    random_seed(123)

    # suggest that the user interacts with the solver
    print "NOTE: while solver is running, press 'Ctrl-C' in console window"
    getch()

    # draw frame and exact coefficients
    plot_exact()

    # configure monitor
    stepmon = VerboseSow(50)

    # use DE to solve 8th-order Chebyshev coefficients
    npop = 10*ndim
    solver = DifferentialEvolutionSolver2(ndim,npop)
    solver.SetRandomInitialPoints(min=[-100]*ndim, max=[100]*ndim)
    solver.SetEvaluationLimits(maxiter=999)
    solver.enable_signal_handler()
    solver.Solve(chebyshev8cost,termination=VTR(0.01),strategy=Best1Exp, \
                 CrossProbability=0.8, ScalingFactor=0.5, \
                 StepMonitor=stepmon, sigint_callback=plot_solution)
    solution = solver.Solution()

    # use monitor to retrieve results information
    iterations = len(stepmon.x)
    cost = stepmon.y[-1]
    print "Generation %d has best Chi-Squared: %f" % (iterations, cost)

    # use pretty print for polynomials
    print poly1d(solution)

    # compare solution with actual 8th-order Chebyshev coefficients
    print "\nActual Coefficients:\n %s\n" % poly1d(chebyshev8coeffs)

    # plot solution versus exact coefficients
    plot_solution(solution) 
    getch()

# end of file
