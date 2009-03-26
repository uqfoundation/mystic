#!/usr/bin/env python

"""
Example:
    - Minimize Rosenbrock's Function with Nelder-Mead.
    - Plot of parameter convergence to function minimum.

Demonstrates:
    - standard models
    - minimal solver interface
    - parameter trajectories using retall
"""

# Nelder-Mead solver
from mystic.scipy_optimize import fmin

# Rosenbrock function
from mystic.models import rosen

# tools
import pylab


if __name__ == '__main__':

    # initial guess
    x0 = [0.8,1.2,0.7]

    # use Nelder-Mead to minimize the Rosenbrock function
    solution = fmin(rosen,x0,disp=0,retall=1)
    allvecs = solution[-1]

    # plot the parameter trajectories
    pylab.plot([i[0] for i in allvecs])
    pylab.plot([i[1] for i in allvecs])
    pylab.plot([i[2] for i in allvecs])

    # draw the plot
    pylab.suptitle("Rosenbrock parameter convergence")
    pylab.xlabel("Nelder-Mead solver iterations")
    pylab.ylabel("parameter value")
    pylab.legend(["x", "y", "z"])
    pylab.show()
 
# end of file
