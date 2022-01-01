#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Same as test_ffitB.py
but using the 'one-liner' solver interface.

Reference:

[1] Storn, R. and Price, K. Differential Evolution - A Simple and Efficient
Heuristic for Global Optimization over Continuous Spaces. Journal of Global
Optimization 11: 341-359, 1997.
"""

from mystic.solvers import diffev
from test_ffit import plot_solution, print_solution, Chebyshev8, ChebyshevCost

from mystic.tools import random_seed
random_seed(123)

ND = 9
NP = ND*10
MAX_GENERATIONS = ND*NP

def main():
    range = [(-100.0,100.0)]*ND
    solution = diffev(ChebyshevCost,range,NP,bounds=None,ftol=0.01,\
                      maxiter=MAX_GENERATIONS,cross=1.0,scale=0.9)
    return solution
  

if __name__ == '__main__':
   #plot_solution(Chebyshev8)
    solution = main()
    print_solution(solution)
    plot_solution(solution)

# end of file
