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
    - Dynamic plot of parameter convergence to function minimum.

Demonstrates:
    - standard models
    - minimal solver interface
    - parameter trajectories using callback
    - solver interactivity
"""

# Nelder-Mead solver
from mystic.solvers import fmin

# Rosenbrock function
from mystic.models import rosen

# tools
from mystic.tools import getch
import matplotlib.pyplot as plt
plt.ion()

# draw the plot
def plot_frame():
    plt.title("Rosenbrock parameter convergence")
    plt.xlabel("Nelder-Mead solver iterations")
    plt.ylabel("parameter value")
    plt.draw()
    plt.pause(0.001)
    return
 
iter = 0
step, xval, yval, zval = [], [], [], []
# plot the parameter trajectories
def plot_params(params):
    global iter, step, xval, yval, zval
    step.append(iter)
    xval.append(params[0])
    yval.append(params[1])
    zval.append(params[2])
    plt.plot(step,xval,'b-')
    plt.plot(step,yval,'g-')
    plt.plot(step,zval,'r-')
    plt.legend(["x", "y", "z"])
    plt.draw()
    plt.pause(0.001)
    iter += 1
    return


if __name__ == '__main__':

    # initial guess
    x0 = [0.8,1.2,0.7]

    # suggest that the user interacts with the solver
    print("NOTE: while solver is running, press 'Ctrl-C' in console window")
    getch()
    plot_frame()

    # use Nelder-Mead to minimize the Rosenbrock function
    solution = fmin(rosen,x0,disp=1,callback=plot_params,handler=True)
    print(solution)

    # don't exit until user is ready
    getch()

# end of file
