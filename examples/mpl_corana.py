#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE
"""
Testing the Corana parabola in 1D. Requires matplotlib.
"""

import pylab, numpy, mystic
#from test_corana import *
from mystic.solvers import fmin
from mystic.tools import getch

from mystic.models.storn import Corana
Corana1 = Corana(1)

x = numpy.arange(-2., 2., 0.01)
y = [Corana1([c]) for c in x]

pylab.plot(x,y,linewidth=1)


for xinit in numpy.arange(0.1,2,0.1):
    sol = fmin(Corana1, [xinit], full_output=1, retall=1)
    xx = mystic.flatten_array(sol[-1])
    yy = [Corana1([c]) for c in xx]
    pylab.plot(xx,yy,'r-',xx,yy,'ko',linewidth=2)

pylab.title("Solution trajectories for fmin at different initial conditions")
pylab.show()

# end of file
