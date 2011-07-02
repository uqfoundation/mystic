#!/usr/bin/env python

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
