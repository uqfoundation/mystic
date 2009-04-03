#!/usr/bin/env python

"""
See test_rosenbrock.py.

This one uses Scipy's CG (Polak-Ribiere) plus viz via matplotlib

cg works well on this problem.
"""

import pylab
from test_rosenbrock import *
from numpy import log
from scipy.optimize import fmin_cg
import numpy
from mystic import getch

def show():
     import pylab, Image
     pylab.savefig('cg_rosenbrock_out',dpi=72)
     im = Image.open('cg_rosenbrock_out.png')
     im.show()
     return


def draw_contour():
    import numpy

    x, y = numpy.mgrid[0:2.1:0.02,0:2.1:0.02]
    c = 0*x
    s,t = x.shape
    for i in range(s):
       for j in range(t):
          xx,yy = x[i,j], y[i,j]
          c[i,j] = rosen([xx,yy])

    pylab.contourf(x,y,log(c*20+1)+2,60)
    show()


def run_once(x0,x1):
    sol = fmin_cg(rosen, [x0, x1], retall = True, full_output=1)
    xy = numpy.asarray(sol[-1])
    pylab.plot(xy[:,0],xy[:,1],'w-',linewidth=2)
    pylab.plot(xy[:,0],xy[:,1],'wo',markersize=6)
    show()
    return sol
    
if __name__ == '__main__':
    draw_contour()
    run_once(0.3,0.3)
    run_once(0.5,1.3)
    run_once(1.8,0.2)

# end of file
