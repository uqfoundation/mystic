#!/usr/bin/env python

"""
Example:
    - Minimize Rosenbrock's Function with Powell's method.

Demonstrates:
    - standard models
    - minimal solver interface
"""

# Powell's Directonal solver
from mystic.scipy_optimize import fmin_powell

# Rosenbrock function
from mystic.models import rosen


if __name__ == '__main__':

    print "Powell's Method"
    print "==============="

    # initial guess
    x0 = [0.8,1.2,0.7]

    # use Powell's method to minimize the Rosenbrock function
    solution = fmin_powell(rosen,x0)
    print solution
 
# end of file
