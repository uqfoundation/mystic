#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2020 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

from mystic.tools import getch

import Gnuplot, numpy
g = Gnuplot.Gnuplot(debug = 1)
g.clear()
x = numpy.arange(-4, 4, 0.01)
y = numpy.cos(x)
y2 = numpy.cos(2* x)
kwds = {'with':'line'}
g.plot(Gnuplot.Data(x, y, **kwds))
getch('next: any key')
g.plot(Gnuplot.Data(x, y2, **kwds))
getch('any key to quit')

# end of file
