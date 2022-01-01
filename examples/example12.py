#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Example:
    - Solve 5th-order polynomial coefficients with Powell's method.
    - Plot of fitting to 5th-order polynomial.

Demonstrates:
    - model and cost function construction
    - minimal solver interface
"""

# Powell's Directonal solver
from mystic.solvers import fmin_powell

# cost function factory
from mystic.forward_model import CostFactory

# tools
from mystic.tools import getch, random_seed
random_seed(123)
import matplotlib.pyplot as plt
plt.ion()

# build the forward model
def ForwardPolyFactory(coeff):
    """generate a 1-D polynomial instance from a list of coefficients"""
    from numpy import poly1d
    return poly1d(coeff)

# build the cost function
def PolyCostFactory(evalpts,datapts,ndim):
    """generate a cost function instance from evaluation points and data"""
    from numpy import sum
    F = CostFactory()
    F.addModel(ForwardPolyFactory,ndim,"noisy_polynomial")
    return F.getCostFunction(evalpts=evalpts,observations=datapts, \
                             metric=lambda x: sum(x*x),sigma=1000.)

# 'data' generators
def data(params):
    """generate 'data' from polynomial coefficients"""
    from numpy import array
    x = 0.1*(array([list(range(101))])-50.)[0]
    fwd = ForwardPolyFactory(params)
    return x,fwd(x)

def noisy_data(params):
    """generate noisy data from polynomial coefficients"""
    from numpy import random
    x,y = data(params)
    y = [random.normal(0,1) + i for i in y]
    return x,y

# draw the plot
def plot_frame(label=None):
    plt.close()
    plt.title("fitting noisy 5th-order polynomial coefficients")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.draw()
    plt.pause(0.001)
    return
 
# plot the polynomial
def plot_data(evalpts,datapts,style='k.'):
    plt.plot(evalpts,datapts,'%s' % style)
    plt.axis([0,5,0,50])#,'k-')
    plt.draw()
    plt.pause(0.001)
    return
    
def plot_solution(params,style='b-'):
    x,y = data(params)
    plot_data(x,y,style)
    plt.legend(["Data","Fitted"])
    plt.draw()
    plt.pause(0.001)
    return


if __name__ == '__main__':

    print("Powell's Method")
    print("===============")

    # target and initial guess
    target = [-1.,4.,-5.,20.,5.]
    x0     = [-1.,2.,-3.,10.,5.]

    # generate 'observed' data
    x,datapts = noisy_data(target)

    # plot observed data
    plot_frame()
    plot_data(x,datapts)

    # generate cost function
    costfunction = PolyCostFactory(x,datapts,len(target))

    # use Powell's method to solve 5th-order polynomial coefficients
    solution = fmin_powell(costfunction,x0)

    # compare solution with actual target 5th-order polynomial coefficients
    print("\nSolved Coefficients:\n %s\n" % ForwardPolyFactory(solution))
    print("Target Coefficients:\n %s\n" % ForwardPolyFactory(target))
 
    # plot solution versus target coefficients
    plot_solution(solution)
    getch() 

# end of file
