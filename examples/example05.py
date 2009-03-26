#!/usr/bin/env python

"""
Example:
    - Minimize Rosenbrock's Function with Powell's method.
    - Dynamic print of parameter convergence to function minimum.

Demonstrates:
    - standard models
    - minimal solver interface
    - parameter trajectories using callback
"""

# Powell's Directonal solver
from mystic.scipy_optimize import fmin_powell

# Rosenbrock function
from mystic.models import rosen

iter = 0
# plot the parameter trajectories
def print_params(params):
    global iter
    from numpy import asarray
    print "Generation %d has best fit parameters: %s" % (iter,asarray(params))
    iter += 1
    return


if __name__ == '__main__':

    # initial guess
    x0 = [0.8,1.2,0.7]
    print_params(x0)

    # use Powell's method to minimize the Rosenbrock function
    solution = fmin_powell(rosen,x0,disp=1,callback=print_params)
    print solution

# end of file
