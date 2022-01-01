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
    - Plot of Rosenbrock's function minimum.

Demonstrates:
    - standard models
    - minimal solver interface
"""

# Nelder-Mead solver
from mystic.solvers import fmin

# Rosenbrock function
from mystic.models import rosen

# tools
import matplotlib.pyplot as plt


if __name__ == '__main__':

    print("Nelder-Mead Simplex")
    print("===================")

    # initial guess
    x0 = [0.8,1.2,0.7]

    # use Nelder-Mead to minimize the Rosenbrock function
    solution = fmin(rosen,x0)
    print(solution)
 
    # plot the Rosenbrock function (one plot per axis)
    x = [0.01*i for i in range(200)]
    plt.plot(x,[rosen([i,1.,1.]) for i in x])
    plt.plot(x,[rosen([1.,i,1.]) for i in x])
    plt.plot(x,[rosen([1.,1.,i]) for i in x])

    # plot the solved minimum (for x)
    plt.plot([solution[0]],[rosen(solution)],'bo')

    # draw the plot
    plt.title("minimium of Rosenbrock's function")
    plt.xlabel("x, y, z")
    plt.ylabel("f(i) = Rosenbrock's function")
    plt.legend(["f(x,1,1)","f(1,y,1)","f(1,1,z)"])
    plt.show()

# end of file
