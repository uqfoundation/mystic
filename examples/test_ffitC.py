#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Testing the polynomial fitting problem of [1] using scipy's Nelder-Mead algorithm.

Reference:

[1] Storn, R. and Price, K. Differential Evolution - A Simple and Efficient
Heuristic for Global Optimization over Continuous Spaces. Journal of Global
Optimization 11: 341-359, 1997.
"""

from test_ffit import Chebyshev8, ChebyshevCost, plot_solution, print_solution

if __name__ == '__main__':
    import random
    from mystic.solvers import fmin
   #from mystic._scipyoptimize import fmin
    from mystic.tools import random_seed
    random_seed(123)
    x = [random.uniform(-100,100) + Chebyshev8[i] for i in range(9)]
    solution = fmin(ChebyshevCost, x)
    print_solution(solution)
    plot_solution(solution)

# end of file
