#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
See test_rosenbrock.py.

This one uses Scipy's CG (Polak-Ribiere) plus viz via matplotlib

cg works well on this problem.
"""

import matplotlib.pyplot as plt
from test_rosenbrock import *
from numpy import log
from mystic._scipyoptimize import fmin_cg
import numpy
from mystic.tools import getch

def show():
    import matplotlib.pyplot as plt, Image
    plt.savefig('cg_rosenbrock_out',dpi=72)
    im = Image.open('cg_rosenbrock_out.png')
    im.show()
    return


def draw_contour():
    import numpy

    x, y = numpy.mgrid[-1:2.1:0.02,-0.1:2.1:0.02]
    c = 0*x
    s,t = x.shape
    for i in range(s):
        for j in range(t):
            xx,yy = x[i,j], y[i,j]
            c[i,j] = rosen([xx,yy])

    plt.contourf(x,y,log(c*20+1)+2,60)


def run_once(x0,x1,color='w'):
    sol = fmin_cg(rosen, [x0, x1], retall = True, full_output=1)
    xy = numpy.asarray(sol[-1])
    plt.plot(xy[:,0],xy[:,1],color+'-',linewidth=2)
    plt.plot(xy[:,0],xy[:,1],color+'o',markersize=6)
    return sol

    
if __name__ == '__main__':
    draw_contour()
    run_once(0.3,0.3,'k')
    run_once(0.5,1.3,'y')
    run_once(1.8,0.2,'w')
    try:
        show()
    except ImportError:
        plt.show()


# end of file
