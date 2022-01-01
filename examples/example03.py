#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
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
from mystic.solvers import fmin

# Rosenbrock function
from mystic.models import rosen

# tools
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # initial guess
    x0 = [0.8,1.2,0.7]

    # use Nelder-Mead to minimize the Rosenbrock function
    solution = fmin(rosen,x0,disp=0,retall=1)
    allvecs = solution[-1]

    # plot the parameter trajectories
    plt.plot([i[0] for i in allvecs])
    plt.plot([i[1] for i in allvecs])
    plt.plot([i[2] for i in allvecs])

    # draw the plot
    plt.title("Rosenbrock parameter convergence")
    plt.xlabel("Nelder-Mead solver iterations")
    plt.ylabel("parameter value")
    plt.legend(["x", "y", "z"])
    plt.show()
 
# end of file
