#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2020 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
See test_mogi.py

This uses Nelder-Mead and GenerationMonitor
"""

import sam
from test_mogi import *
from mystic.solvers import NelderMeadSimplexSolver as fmin
from mystic.termination import CandidateRelativeTolerance as CRT
from mystic.monitors import Monitor
from mystic.tools import getch

x0,y0,z0,v0 = actual_params

def draw_contour_xy():
    import numpy
    x, y = mgrid[-40:40:0.5, -40:40:0.5]
    x = x0 + x
    y = y0 + y
    s,t = x.shape
    c = 0*x
    s,t = x.shape
    for i in range(s):
       for j in range(t):
          xx,yy = x[i,j], y[i,j]
          c[i,j] = cost_function([xx,yy, z0, v0])


    sam.putarray('X',x)
    sam.putarray('Y',y)
    sam.putarray('C',c)

    sam.verbose()    
    sam.eval("[c,h]=contourf(X,Y,C,100);set(h,'EdgeColor','none')")
    sam.eval("title('Mogi Fitting')")
    sam.eval('hold on')

def draw_contour_xv():
    import numpy
    x, y = mgrid[-40:40:0.5, -0.1:0.3:.01]
    x = x0 + x
    y = v0 + y
    s,t = x.shape
    c = 0*x
    s,t = x.shape
    for i in range(s):
       for j in range(t):
          xx,yy = x[i,j], y[i,j]
          c[i,j] = cost_function([xx, y0, z0, yy])

    sam.putarray('X',x)
    sam.putarray('Y',y)
    sam.putarray('C',c)

    sam.eval("[c,h]=contourf(X,Y,C,100);set(h,'EdgeColor','none')")
    sam.eval('hold on')

def run_once_xy():
    simplex = Monitor()
    z1 = z0*random.uniform(0.5,1.5)
    v1 = v0*random.uniform(0.5,1.5)
    xinit = [random.uniform(x0-40,x0+40), random.uniform(y0-40,y0+40), z1, v1]

    solver = fmin(len(xinit))
    solver.SetInitialPoints(xinit)
    solver.SetGenerationMonitor(simplex)
    solver.Solve(cost_function, termination=CRT())
    sol = solver.Solution()
    print(sol)
    
    for x in simplex.x:
        sam.putarray('x',x)
        sam.eval("plot(x([1,2,3,1],1),x([1,2,3,1],4),'w-','LineWidth',2)")
    return sol


def run_once_xv():
    simplex = Monitor()
    y1 = y0*random.uniform(0.5,1.5)
    z1 = z0*random.uniform(0.5,1.5)
    xinit = [random.uniform(x0-40,x0+40), y1, z1, random.uniform(v0-0.1,v0+0.1)]

    solver = fmin(len(xinit))
    solver.SetInitialPoints(xinit)
    solver.SetGenerationMonitor(simplex)
    solver.Solve(cost_function, termination=CRT())
    sol = solver.Solution()
    print(sol)

    for x in simplex.x:
        sam.putarray('x',x)
        sam.eval("plot(x([1,2,3,1],1),x([1,2,3,1],2),'w-','LineWidth',2)")
    return sol

draw_contour_xv()
xysol = run_once_xy()

sam.eval("figure(2)")

draw_contour_xy()
xvsol = run_once_xv()


sam.eval("figure(3)")
plot_noisy_data()
sam.eval("hold on")
plot_sol(xysol)
plot_sol(xvsol)

getch()

# end of file
