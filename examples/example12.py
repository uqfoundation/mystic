#!/usr/bin/env python

"""
Example:
    - Solve 5th-order polynomial coefficients with Powell's method.
    - Plot of fitting to 5th-order polynomial.

Demonstrates:
    - model and cost function construction
    - minimal solver interface
"""

# Powell's Directonal solver
from mystic.scipy_optimize import fmin_powell

# cost function factory
from mystic.forward_model import CostFactory

# tools
from mystic import getch, random_seed
random_seed(123)
import pylab
pylab.ion()

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
    F.addModel(ForwardPolyFactory,"noisy_polynomial",ndim)
    return F.getCostFunction(evalpts=evalpts,observations=datapts, \
                             metric=lambda x: sum(x*x),sigma=1000.)

# 'data' generators
def data(params):
    """generate 'data' from polynomial coefficients"""
    from numpy import array
    x = 0.1*(array([range(101)])-50.)[0]
    fwd = ForwardPolyFactory(params)
    return x,fwd(x)

def noisy_data(params):
    """generate noisy data from polynomial coefficients"""
    from numpy import random
    x,y = data(params)
    y = [random.normal(0,1) + i for i in y]
    return x,y

# plot the polynomial
def plot_data(evalpts,datapts,style='k.'):
    pylab.plot(evalpts,datapts,'%s' % style)
    pylab.axis([0,5,0,50],'k-')
    return
    
def plot_solution(params,style='b-'):
    x,y = data(params)
    plot_data(x,y,style)
    return


if __name__ == '__main__':

    print "Powell's Method"
    print "==============="

    # target and initial guess
    target = [-1.,4.,-5.,20.,5.]
    x0     = [-1.,2.,-3.,10.,5.]

    # generate 'observed' data
    x,datapts = noisy_data(target)

    # plot observed and target data
    plot_data(x,datapts)
    plot_solution(target)

    # generate cost function
    costfunction = PolyCostFactory(x,datapts,len(target))

    # use Powell's method to solve 5th-order polynomial coefficients
    solution = fmin_powell(costfunction,x0)

    # compare solution with actual target 5th-order polynomial coefficients
    print "\nSolved Coefficients:\n %s\n" % ForwardPolyFactory(solution)
    print "Target Coefficients:\n %s\n" % ForwardPolyFactory(target)
 
    # plot solution versus target coefficients
    plot_solution(solution,'r-')
    getch() 

# end of file
