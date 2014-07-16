#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2014 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE
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

# if available, use a multiprocessing worker pool
try:
    from pathos.multiprocessing import ProcessingPool as Pool
except ImportError:
    from mystic.python import PythonSerial as Pool

# tools
from mystic.termination import NormalizedChangeOverGeneration as NCOG
from mystic.math import poly1d
from mystic.monitors import VerboseMonitor
from mystic.tools import getch
import pylab
pylab.ion()

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
    pylab.draw()
    return
 
# plot the polynomial
def plot_solution(params,style='y-'):
    import numpy
    x = numpy.arange(-1.2, 1.2001, 0.01)
    f = poly1d(params)
    y = f(x)
    pylab.plot(x,y,style)
    pylab.legend(["Exact","Fitted"])
    pylab.axis([-1.4,1.4,-2,8],'k-')
    pylab.draw()
    return


if __name__ == '__main__':

    print "Powell's Method"
    print "==============="

    # dimensional information
    import random
    random.seed(123)
    ndim = 9
   #nbins = [1]*ndim
    nbins = [2,1,2,1,2,1,1,1,1]

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
    solver.Solve(chebyshev8cost, NCOG(1e-4), disp=1)
    solution = solver.Solution()

    # use pretty print for polynomials
    print poly1d(solution)

    # compare solution with actual 8th-order Chebyshev coefficients
    print "\nActual Coefficients:\n %s\n" % poly1d(chebyshev8coeffs)

    # plot solution versus exact coefficients
    plot_solution(solution) 
    getch()

# end of file
