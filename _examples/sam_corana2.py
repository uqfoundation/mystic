#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2020 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
See test_corana.py.

This one uses Nelder-Mead plus matlab viz.

Corana's parabola in 2D.
"""

import sam
#from test_corana import *
from mystic.solvers import NelderMeadSimplexSolver as fmin
from mystic.termination import CandidateRelativeTolerance as CRT
from mystic.monitors import Monitor
from mystic.tools import getch

from mystic.models.corana import corana2d as Corana2

def draw_contour():
    import numpy

    x, y = numpy.mgrid[0:2.1:0.05,0:2.1:0.05]
    c = 0*x
    s,t = x.shape
    for i in range(s):
       for j in range(t):
          xx,yy = x[i,j], y[i,j]
          c[i,j] = Corana2([xx,yy])


    sam.putarray('X',x)
    sam.putarray('Y',y)
    sam.putarray('C',c)

    sam.verbose()    
    sam.eval("[c,h]=contourf(X,Y,C,100);set(h,'EdgeColor','none')")
    #sam.eval("[c,h]=contourf(X,Y,log(C*20+1)+2,100);set(h,'EdgeColor','none')")
    sam.eval("title('Corana''s Parabola in 2D. Min at 0,0')")
    sam.eval('hold on')


def run_once():
    simplex = Monitor()
    solver = fmin(2)
    solver.SetRandomInitialPoints([0,0],[2,2])
    solver.SetGenerationMonitor(simplex)
    solver.Solve(Corana2, termination=CRT())
    sol = solver.Solution()
    
    for x in simplex.x:
        sam.putarray('x',x)
        sam.eval("plot(x([1,2,3,1],1),x([1,2,3,1],2),'w-')")

draw_contour()
for i in range(8):
    run_once()

getch("Press any key to quit")

# end of file
