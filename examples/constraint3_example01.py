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

Demonstrates:
    - standard models
    - minimal solver interface
    - parameter constraints solver and constraints factory decorator
    - statistical parameter constraints
    - customized monitors
"""

# Powell's Directonal solver
from mystic.solvers import fmin_powell

# Rosenbrock function
from mystic.models import rosen

# tools
from mystic.monitors import VerboseMonitor
from mystic.math.measures import mean, impose_mean
from mystic.math import almostEqual


if __name__ == '__main__':

    print("Powell's Method")
    print("===============")

    # initial guess
    x0 = [0.8,1.2,0.7]

    # use the mean constraints factory decorator
    from mystic.constraints import with_mean

    # define constraints function
    @with_mean(1.0)
    def constraints(x):
        # constrain the last x_i to be the same value as the first x_i
        x[-1] = x[0]
        return x

    # configure monitor
    stepmon = VerboseMonitor(1)

    # use Powell's method to minimize the Rosenbrock function
    solution = fmin_powell(rosen,x0,constraints=constraints,itermon=stepmon)
    print(solution)
 
# end of file
