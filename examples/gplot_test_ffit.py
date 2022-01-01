#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Adapted from DETest.py by Patrick Hung

Sets up Storn and Price's Polynomial 'Fitting' Problem.

Exact answer: Chebyshev Polynomial of the first kind. T8(x)

Reference:

[1] Storn, R. and Price, K. Differential Evolution - A Simple and Efficient
Heuristic for Global Optimization over Continuous Spaces. Journal of Global
Optimization 11: 341-359, 1997.
"""
from test_ffit import *

def plot_solution(func):
    try:
        import Gnuplot, numpy
        g = Gnuplot.Gnuplot(debug = 1)
        x = numpy.arange(-1.2, 1.2001, 0.01)
        x2 = numpy.array([-1.0, 1.0])
        p = poly1d(func)
        chebyshev = poly1d(Chebyshev8)
        y = p(x)
        exact = chebyshev(x)
        # g('set style line lw 5')
        g('set xrange[-1.4:1.4]')
        g('set yrange[-2:8]')
        g.plot(Gnuplot.Data(x, y, with_='line', inline=1, title="Storn and Price's Function Fitting problem"), \
               Gnuplot.Data(x, exact, with_='l 4 4'), \
               Gnuplot.Data(x, 0*x-1, with_='l 2 2'), \
               Gnuplot.Data(x2, 0*x2+1, with_='l 2 2'),  \
               Gnuplot.Data([-1.2, -1.2], [-1, 10], with_='l 2 2'),  \
               Gnuplot.Data([1.2, 1.2], [-1, 10], with_='l 2 2'),  \
               Gnuplot.Data([-1.0, -1.0], [1, 10], with_='l 2 2'),  \
               Gnuplot.Data([1.0, 1.0], [1, 10], with_='l 2 2')  \
               )
        getch('Press any key to exit plot')
    except ImportError:
        print("Install Gnuplot and Gnuplot.py for visualization")
        pass


if __name__ == '__main__':
    solution = main()
    print_solution(solution)
    plot_solution(solution)

# end of file
