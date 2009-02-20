#!/usr/bin/env python

"""
See test_zimmermann.py.

This one uses Nelder-Mead plus matlab viz.

It uses the StepMonitor option to track all simplices generated during 
the search.
"""

import sam
from test_zimmermann import *
from mystic.scipy_optimize import NelderMeadSimplexSolver as fmin
from mystic.termination import CandidateRelativeTolerance as CRT
from mystic import getch, Sow

def draw_contour():
    import numpy

    x, y = numpy.mgrid[0:7.5:0.05,0:7.5:0.05]
    c = 0*x
    s,t = x.shape
    for i in range(s):
       for j in range(t):
          xx,yy = x[i,j], y[i,j]
          c[i,j] = CostFunction([xx,yy])


    sam.putarray('X',x)
    sam.putarray('Y',y)
    sam.putarray('C',c)

    sam.verbose()    
    sam.eval("[c,h]=contourf(X,Y,log(C*20+1)+2,100);set(h,'EdgeColor','none')")
    sam.eval("title('Zimmermann''s Corner. Min at 7,2')")
    sam.eval('hold on')


def run_once():
    simplex = Sow()
    solver = fmin(2)
    solver.SetRandomInitialPoints([0,0],[7,7])
    solver.Solve(CostFunction, termination=CRT(), StepMonitor = simplex)
    sol = solver.Solution()

    for x in simplex.x:
        sam.putarray('x',x)
        sam.eval("plot(x([1,2,3,1],1),x([1,2,3,1],2),'k-')")

draw_contour()
for i in range(8):
    run_once()

getch("Press any key to quit")

# end of file
