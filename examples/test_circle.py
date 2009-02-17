#!/usr/bin/env python
# 
# Patrick Hung.

"""
Given a set of points in the plane, find the smallest circle
that contains them. (brute force via DE)

The pylab output will draw 
  -- a set of points inside a circle defined by x0,y0,R0 
  -- the circle (x0,y0) with rad R0
  -- the DE optimized circle with minimum R enclosing the points
"""

from mystic.differential_evolution import DifferentialEvolutionSolver2 as DifferentialEvolutionSolver
from mystic.detools import Best1Exp, Best1Bin, Rand1Exp, ChangeOverGeneration, VTR
from mystic import getch
from numpy import random, array, pi, arange, sin, cos, sqrt
import pylab

random.seed(123)

# x0, y0, R0
x0, y0, R0 = [10., 20., 3];

def get_circle(N):
    # generate N random points in a unit circle
    n = 0
    while n < N:
        x = random.random()*2.-1.
        y = random.random()*2.-1.
        if x*x + y*y <= 1:
            n = n+1
            yield [x,y]

# generating training set
npt = 20
xy = array(list(get_circle(npt)))*R0
xy[:,0] += x0
xy[:,1] += y0
theta = arange(0, 2*pi, 0.02)

# define cost function.
def cost(params):
    x,y,r = params
    if r<0:
        return -999. * r
    penalty = 0
    for xx,yy in xy:
       # compute distance to origin
       d = sqrt((xx-x)*(xx-x) + (yy-y)*(yy-y))
       if d > r:
           # each violation adds 1 to the cost plus amount of violation
           penalty += 1+d-r
    return r+penalty
       
MAX_GENERATIONS = 2000
ND, NP = 3, 30
solver = DifferentialEvolutionSolver(ND, NP)
solver.enable_signal_handler()
minrange = [0., 0., 0.]
maxrange = [50., 50., 10.]
solver.SetRandomInitialPoints(min=minrange,max=maxrange)

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

if __name__ == '__main__':     

    solver.Solve(cost, Best1Exp, termination=ChangeOverGeneration(generations=100), \
                 maxiter = MAX_GENERATIONS)
    solution = solver.Solution()
    sx, sy, sr = solution
    svl = sv(xy, sx,sy,sr)
    print "support vectors: ", svl
    print "DEsol : (%f, %f) @ R = %f" % (sx, sy, sr)

    pylab.plot(xy[:,0],xy[:,1],'k+',markersize=6)
    pylab.plot(R0 * cos(theta)+x0, R0*sin(theta)+y0, 'r-',linewidth=2)

    legend = ['random points','generating circle : %f' % R0,'DE optimal : %f' % sr]
    pylab.axis('equal')

    pylab.plot(sr * cos(theta)+sx, sr*sin(theta)+sy, 'b-',linewidth=2)
    # try scipy as well
    try: 
        import scipy.optimize
        xx = [1,1,1] # bad initial guess
        xx = [5,5,1] # ok guess
        xx = [10,15,5] # good initial guess
        sol = scipy.optimize.fmin(cost, xx)
        ax, ay, ar = sol
        pylab.plot(ar * cos(theta)+ax, ar*sin(theta)+ay, 'g-',linewidth=2)
        legend.append('Nelder-Mead : %f' % ar)
        print "scipy sol: ", sol
    except ImportError:
        print "Install scipy for more"

    # draw the support vectors found from DE
    pylab.plot(xy[svl,0],xy[svl,1],'ro',markersize=6)

    pylab.legend(legend)
    pylab.show()

# $Id$
# 
# end of file
