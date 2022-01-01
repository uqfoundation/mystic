#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Example:
    - Solve 8th-order Chebyshev polynomial coefficients with Powell's method.
    - Uses LatticeSolver to provide 'pseudo-global' optimization
    - Plot of fitting to Chebyshev polynomial.

Demonstrates:
    - standard models
    - minimal solver interface
"""
# the Lattice solver
from mystic.solvers import LatticeSolver

# Powell's Directonal solver
from mystic.solvers import PowellDirectionalSolver

# Chebyshev polynomial and cost function
from mystic.models.poly import chebyshev8, chebyshev8cost
from mystic.models.poly import chebyshev8coeffs

# if available, use a pathos worker pool
try:
    from pathos.pools import ProcessPool as Pool
   #from pathos.pools import ParallelPool as Pool
except ImportError:
    from mystic.pools import SerialPool as Pool

# tools
from mystic.termination import NormalizedChangeOverGeneration as NCOG
from mystic.math import poly1d
from mystic.monitors import VerboseMonitor
from mystic.tools import getch
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
    from pathos.helpers import freeze_support, shutdown
    freeze_support() # help Windows use multiprocessing

    print("Powell's Method")
    print("===============")

    # dimensional information
    from mystic.tools import random_seed
    random_seed(123)
    ndim = 9
    nbins = 8 #[2,1,2,1,2,1,2,1,1]

    # draw frame and exact coefficients
    plot_exact()

    # configure monitor
    stepmon = VerboseMonitor(1)

    # use lattice-Powell to solve 8th-order Chebyshev coefficients
    solver = LatticeSolver(ndim, nbins)
    solver.SetNestedSolver(PowellDirectionalSolver)
    solver.SetMapper(Pool().map)
    solver.SetGenerationMonitor(stepmon)
    solver.SetStrictRanges(min=[-300]*ndim, max=[300]*ndim)
    solver.Solve(chebyshev8cost, NCOG(1e-4), disp='all', step=False)
    solution = solver.Solution()
    shutdown() # help multiprocessing shutdown all workers

    # use pretty print for polynomials
    print(poly1d(solution))

    # compare solution with actual 8th-order Chebyshev coefficients
    print("\nActual Coefficients:\n %s\n" % poly1d(chebyshev8coeffs))

    # plot solution versus exact coefficients
    plot_solution(solution) 
    getch()

# end of file
