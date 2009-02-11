#!/usr/bin/env python

"""
Testing the polynomial fitting problem of [1] using scipy's Nelder-Mead algorithm.

Reference:

[1] Storn, R. and Price, K. Differential Evolution - A Simple and Efficient
Heuristic for Global Optimization over Continuous Spaces. Journal of Global
Optimization 11: 341-359, 1997.
"""

Chebyshev8 = [128., 0., -256., 0., 160., 0., -32., 0., 1.]

from test_ffit import ChebyshevCost, plot_solution, print_solution

if __name__ == '__main__':
    import scipy.optimize
    import random
    random.seed(123)
    x = [random.uniform(-100,100) + Chebyshev8[i] for i in range(9)]
    solution = scipy.optimize.fmin(ChebyshevCost, x)
    print_solution(solution)
    plot_solution(solution)

# end of file
