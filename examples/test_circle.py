#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Given a set of points in the plane, find the smallest circle
that contains them. (using DE and scipy.fmin)

Requires:
  -- numpy, matplotlib

The matplotlib output will draw 
  -- a set of points inside a circle defined by x0,y0,R0 
  -- the circle (x0,y0) with rad R0
  -- the optimized circle with minimum R enclosing the points
"""

from mystic.models import circle, sparse_circle
import matplotlib.pyplot as plt

# generate training set & define cost function
# CostFactory2 allows costfunction to reuse datapoints from training set
x0, y0, R0 = [10., 20., 3]
npts = 20
xy = sparse_circle(x0, y0, R0, npts)
cost = sparse_circle.CostFactory2(xy)

# function to find the 'support vectors' given R
# SV in quotes because they are found by optimizing the primal,
# as opposed to "directly" via the dual.
def sv(data, xx,yy,rr):
    svl = []
    for i in range(len(data)):
       x,y = data[i]
       if abs((xx-x)*(xx-x)+(yy-y)*(yy-y) - rr*rr) < 0.01:
           svl.append(i)
    return svl

# DEsolver inputs
MAX_GENERATIONS = 2000
ND, NP = 3, 30    # dimension, population size
minrange = [0., 0., 0.]
maxrange = [50., 50., 10.]

# prepare DESolver
from mystic.solvers import DifferentialEvolutionSolver2 \
      as DifferentialEvolutionSolver
from mystic.termination import ChangeOverGeneration, VTR
solver = DifferentialEvolutionSolver(ND, NP)
solver.enable_signal_handler()
solver.SetRandomInitialPoints(min=minrange,max=maxrange)
solver.SetEvaluationLimits(generations=MAX_GENERATIONS)
solver.Solve(cost, termination=ChangeOverGeneration(generations=100))


if __name__ == '__main__':     

    # x0, y0, R0
    #guess = [1,1,1] # bad initial guess
    #guess = [5,5,1] # ok guess
    guess = [10,15,5] # good initial guess

    # plot training set & training set boundary
    plt.plot(xy[:,0],xy[:,1],'k+',markersize=6)
    c = circle(x0, y0, R0)
    plt.plot(c[:,0],c[:,1],'r-',linewidth=2)
    legend = ['random points','generating circle : %f' % R0]
    plt.axis('equal')

    # solve with mystic's differential evolution solver
    solution = solver.Solution()
    sx, sy, sr = solution
    print("DEsol : (%f, %f) @ R = %f" % (sx, sy, sr))

    # plot DEsolver solution
    c = circle(sx, sy, sr)
    plt.plot(c[:,0],c[:,1],'b-',linewidth=2)
    legend.append('DE optimal : %f' % sr)

    # solve with scipy.fmin
    from mystic.solvers import fmin
    sol = fmin(cost, guess)
    print("scipy.fmin sol: %s" % sol)
    ax, ay, ar = sol

    # plot scipy.fmin solution
    c = circle(ax, ay, ar)
    plt.plot(c[:,0],c[:,1],'g-',linewidth=2)
    legend.append('Nelder-Mead : %f' % ar)

    # solve with scipy.brute
   #from mystic._scipyoptimize import brute
   #ranges = tuple(zip(minrange,maxrange))
   #sol = brute(cost, ranges, Ns=NP)
   #print("scipy.brute sol: %s" % sol)
   #bx, by, br = sol

    # plot scipy.brute solution
   #c = circle(bx, by, br)
   #plt.plot(c[:,0],c[:,1],'y-',linewidth=2)
   #legend.append('Brute : %f' % br)

    # find & draw the support vectors from DE
    svl = sv(xy, sx,sy,sr)
    print("DE support vectors: %s" % svl)
    plt.plot(xy[svl,0],xy[svl,1],'bx',markersize=6)

    # find & draw the support vectors from scipy.brute
   #svl = sv(xy, bx,by,br)
   #print("Brute support vectors: %s" % svl)
   #plt.plot(xy[svl,0],xy[svl,1],'yx',markersize=6)

    plt.legend(legend)
    plt.show()

# $Id$
# 
# end of file
