#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2014 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE
"""
Testing the polynomial fitting problem of [1] using scipy's Nelder-Mead algorithm.

Reference:

[1] Storn, R. and Price, K. Differential Evolution - A Simple and Efficient
Heuristic for Global Optimization over Continuous Spaces. Journal of Global
Optimization 11: 341-359, 1997.
"""

from test_ffit import Chebyshev8, ChebyshevCost, plot_solution, print_solution

if __name__ == '__main__':
    from mystic.solvers import fmin
   #from mystic._scipyoptimize import fmin
    import random
    random.seed(123)

    import pp
    import sys

    if len(sys.argv) > 1:
        tunnelport = sys.argv[1]
        ppservers = ("localhost:%s" % tunnelport,)
    else:
        ppservers = ()

    myserver = pp.Server(ppservers=ppservers)

    trials = []
    for trial in range(8):
        x = tuple([random.uniform(-100,100) + Chebyshev8[i] for i in range(9)])
        trials.append(x)

    results = [myserver.submit(fmin,(ChebyshevCost,x),(),()) for x in trials]

    for solution in results:
        print_solution(solution())

   #plot_solution(solution)

# end of file
