#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Example:
    - Solve 8th-order Chebyshev polynomial coefficients with DE.
    - Callable plot of fitting to Chebyshev polynomial.
    - Monitor Chi-Squared for Chebyshev polynomial.

Demonstrates:
    - standard models
    - expanded solver interface
    - built-in random initial guess
    - customized monitors and termination conditions
    - customized DE mutation strategies
    - use of monitor to retrieve results information
"""

# Differential Evolution solver
from mystic.solvers import DifferentialEvolutionSolver2

# Chebyshev polynomial and cost function
from mystic.models.poly import chebyshev8, chebyshev8cost
from mystic.models.poly import chebyshev8coeffs

# tools
from mystic.termination import VTR
from mystic.strategy import Best1Exp
from mystic.monitors import VerboseMonitor
from mystic.tools import getch, random_seed
from mystic.math import poly1d
import matplotlib.pyplot as plt
plt.ion()

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

    print("Differential Evolution")
    print("======================")

    # set range for random initial guess
    ndim = 9
    x0 = [(-100,100)]*ndim
    random_seed(123)

    # draw frame and exact coefficients
    plot_exact()

    # configure monitor
    stepmon = VerboseMonitor(50)

    # use DE to solve 8th-order Chebyshev coefficients
    npop = 10*ndim
    solver = DifferentialEvolutionSolver2(ndim,npop)
    solver.SetRandomInitialPoints(min=[-100]*ndim, max=[100]*ndim)
    solver.SetGenerationMonitor(stepmon)
    solver.enable_signal_handler()
    solver.Solve(chebyshev8cost, termination=VTR(0.01), strategy=Best1Exp, \
                 CrossProbability=1.0, ScalingFactor=0.9, \
                 sigint_callback=plot_solution)
    solution = solver.Solution()

    # use monitor to retrieve results information
    iterations = len(stepmon)
    cost = stepmon.y[-1]
    print("Generation %d has best Chi-Squared: %f" % (iterations, cost))

    # use pretty print for polynomials
    print(poly1d(solution))

    # compare solution with actual 8th-order Chebyshev coefficients
    print("\nActual Coefficients:\n %s\n" % poly1d(chebyshev8coeffs))

    # plot solution versus exact coefficients
    plot_solution(solution)
    getch()

# end of file
