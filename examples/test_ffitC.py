#!/usr/bin/env python

"""
Testing the polynomial fitting problem of [1] using scipy's Nelder-Mead algorithm.

Reference:

[1] Storn, R. and Price, K. Differential Evolution - A Simple and Efficient
Heuristic for Global Optimization over Continuous Spaces. Journal of Global
Optimization 11: 341-359, 1997.
"""

from test_ffit import Chebyshev8, ChebyshevCost, plot_solution, print_solution

if __name__ == '__main__':
    from mystic.scipy_optimize import fmin
   #from scipy.optimize import fmin
    import random
    random.seed(123)
    x = [random.uniform(-100,100) + Chebyshev8[i] for i in range(9)]
    solution = fmin(ChebyshevCost, x)
    print_solution(solution)
    plot_solution(solution)

# end of file
