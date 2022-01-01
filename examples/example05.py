#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
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
from mystic.solvers import fmin_powell

# Rosenbrock function
from mystic.models import rosen

iter = 0
# plot the parameter trajectories
def print_params(params):
    global iter
    from numpy import asarray
    print("Generation %d has best fit parameters: %s" % (iter,asarray(params)))
    iter += 1
    return


if __name__ == '__main__':

    # initial guess
    x0 = [0.8,1.2,0.7]
    print_params(x0)

    # use Powell's method to minimize the Rosenbrock function
    solution = fmin_powell(rosen,x0,disp=1,callback=print_params,handler=True)
    print(solution)

# end of file
