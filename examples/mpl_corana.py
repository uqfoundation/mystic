#!/usr/bin/env python

"""
Testing the Corana parabola in 1D. Requires matplotlib.
"""

import pylab, numpy, mystic
#from test_corana import *
from mystic.scipy_optimize_fmin import fmin
from mystic import getch

from mystic.models.corana import corana1d as Corana1

x = numpy.arange(-2., 2., 0.01)
y = [Corana1([c]) for c in x]

pylab.plot(x,y,linewidth=1)


for xinit in numpy.arange(0.1,2,0.1):
    sol = fmin(Corana1, [xinit], full_output=1, retall=1)
    xx = mystic.flatten_array(sol[-1])
    yy = [Corana1([c]) for c in xx]
    pylab.plot(xx,yy,'r-',xx,yy,'ko',linewidth=2)

pylab.title("Solution trajectories for scipy's fmin at different ICs")
pylab.show()

# end of file
