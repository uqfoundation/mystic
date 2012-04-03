#!/usr/bin/env python

"""
Example:
    - Minimize Rosenbrock's Function with Powell's method.

Demonstrates:
    - standard models
    - minimal solver interface
    - parameter constraints solver
    - customized monitors
"""

# Powell's Directonal solver
from mystic.solvers import fmin_powell

# Rosenbrock function
from mystic.models import rosen

# tools
from mystic.monitors import VerboseMonitor


if __name__ == '__main__':

    print "Powell's Method"
    print "==============="

    # initial guess
    x0 = [0.8,1.2,0.7]

    # define constraints function
    def constraints(x):
        # constrain the last x_i to be the same value as the first x_i
        x[-1] = x[0]
        return x

    # configure monitor
    stepmon = VerboseMonitor(1)

    # use Powell's method to minimize the Rosenbrock function
    solution = fmin_powell(rosen,x0,constraints=constraints,itermon=stepmon)
    print solution
 
# end of file
