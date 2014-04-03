#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2014 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE

from mystic.tools import getch

import Gnuplot, numpy
g = Gnuplot.Gnuplot(debug = 1)
g.clear()
x = numpy.arange(-4, 4, 0.01)
y = numpy.cos(x)
y2 = numpy.cos(2* x)
g.plot(Gnuplot.Data(x, y, with='line'))
getch('next: any key')
g.plot(Gnuplot.Data(x, y2, with='line'))
getch('any key to quit')

# end of file
