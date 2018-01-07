#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2018 The Uncertainty Quantification Foundation.
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
    pylab.title("Rosenbrock parameter convergence")
    pylab.xlabel("Nelder-Mead solver iterations")
    pylab.ylabel("parameter value")
    pylab.legend(["x", "y", "z"])
    pylab.show()
 
# end of file
