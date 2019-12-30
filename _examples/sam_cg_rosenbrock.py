#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2020 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
See test_rosenbrock.py.

This one uses Scipy's CG (Polak-Ribiere) plus matlab viz.

"""

import sam
from test_rosenbrock import *
from scipy.optimize import fmin_cg
import numpy
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
    sol = fmin_cg(rosen, [x0, x1], retall = True, full_output=1)
    xy = numpy.asarray(sol[-1])
    sam.putarray('xy',xy)
    sam.eval("plot(xy(:,1),xy(:,2),'w-','LineWidth',2)")
    sam.eval("plot(xy(:,1),xy(:,2),'wo','MarkerSize',6)")
    return sol
    
if __name__ == '__main__':
    draw_contour()
    run_once(0.3,0.3)
    run_once(0.5,1.3)

    getch("Press any key to quit")

# end of file
