#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2020 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
See test_rosenbrock.py.

This one uses Nelder-Mead plus matlab viz.

It uses a GenerationMonitor to track all simplices generated during 
the search.
"""

import sam
from test_rosenbrock import *
from mystic.solvers import NelderMeadSimplexSolver as fmin
from mystic.termination import CandidateRelativeTolerance as CRT
from mystic.monitors import Monitor
from mystic.tools import getch

def draw_contour():
    import numpy

    x, y = numpy.mgrid[0:2.1:0.02,0:2.1:0.02]
    c = 0*x
    s,t = x.shape
    for i in range(s):
       for j in range(t):
          xx,yy = x[i,j], y[i,j]
          c[i,j] = rosen([xx,yy])


    sam.putarray('X',x)
    sam.putarray('Y',y)
    sam.putarray('C',c)

    sam.verbose()    
    #sam.eval("[c,h]=contourf(X,Y,C,60);set(h,'EdgeColor','none')")
    sam.eval("[c,h]=contourf(X,Y,log(C*20+1)+2,60);set(h,'EdgeColor','none')")
    sam.eval("title('Rosenbrock''s function in 2D. Min at 1,1')")
    sam.eval('hold on')


def run_once(x0,x1):
    simplex = Monitor()
    xinit = [x0, x1]

    solver = fmin(len(xinit))
    solver.SetInitialPoints(xinit)
    solver.SetGenerationMonitor(simplex)
    solver.Solve(rosen, termination=CRT())
    sol = solver.Solution()
    
    for x in simplex.x:
        sam.putarray('x',x)
        sam.eval("plot(x([1,2,3,1],1),x([1,2,3,1],2),'w-')")

draw_contour()
run_once(0.5,0.1)

sam.eval("figure(2)")
draw_contour()
run_once(1.5,0.1)

sam.eval("figure(3)")
draw_contour()
run_once(0.5,1.8)
#run_once(1.5,1.8)

getch("Press any key to quit")

# end of file
