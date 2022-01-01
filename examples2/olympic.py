#!/usr/bin/env python
#
# Problem definition:
# Example in google/or-tools
# https://github.com/google/or-tools/blob/master/examples/python/olympic.py
# with Copyright 2010 Hakan Kjellerstrand hakank@bonetmail.com
# and disclamer as stated at the above reference link.
# 
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
  Olympic puzzle in Google CP Solver.

  Prolog benchmark problem
  '''
  Name   : olympic.pl
  Author : Neng-Fa
  Date   : 1993

  Purpose: solve a puzzle taken from Olympic Arithmetic Contest

  Given ten variables with the following configuration:

                 X7   X8   X9   X10

                    X4   X5   X6

                       X2   X3

                          X1

  We already know that X1 is equal to 3 and want to assign each variable
  with a different integer from {1,2,...,10} such that for any three
  variables
                        Xi   Xj

                           Xk
  the following constraint is satisfied:

                      |Xi-Xj| = Xk
  '''
"""

def objective(x):
    return 0.0

n = 10

bounds = [(1,n)]*n
# with penalty='penalty' applied, solution is:
xs = [[3, 7, 4, 2, 9, 5, 8, 10, 1, 6],
      [3, 7, 4, 2, 8, 5, 10, 9, 1, 6],
      [3, 2, 5, 7, 9, 4, 8, 1, 10, 6],
      [3, 5, 2, 4, 9, 7, 6, 10, 1, 8],
      [3, 4, 7, 5, 1, 8, 6, 10, 9, 2],
      [3, 4, 7, 5, 8, 1, 6, 2, 10, 9],
      [3, 4, 7, 5, 9, 2, 6, 1, 10, 8]]
ys = 0.0

# constraints
def penalty1(x): # == 0
    return x[0] - 3

def penalty2(x): # == 0
    return abs(x[1] - x[2]) - x[0]

def penalty3(x): # == 0
    return abs(x[3] - x[4]) - x[1]

def penalty4(x): # == 0
    return abs(x[4] - x[5]) - x[2]

def penalty5(x): # == 0
    return abs(x[6] - x[7]) - x[3]

def penalty6(x): # == 0
    return abs(x[7] - x[8]) - x[4]

def penalty7(x): # == 0
    return abs(x[8] - x[9]) - x[5]


from mystic.penalty import quadratic_equality
from mystic.constraints import as_constraint

@quadratic_equality(penalty1)
@quadratic_equality(penalty2)
@quadratic_equality(penalty3)
@quadratic_equality(penalty4)
@quadratic_equality(penalty5)
@quadratic_equality(penalty6)
@quadratic_equality(penalty7)
def penalty(x):
    return 0.0

solver = as_constraint(penalty)


from mystic.constraints import unique

from numpy import round, hstack, clip
def constraint(x):
    x = round(x).astype(int) # force round and convert type to int
    x = clip(x, 1,n)         #XXX: impose bounds
    x = unique(x, list(range(1,n+1)))
    return x


if __name__ == '__main__':

    from mystic.solvers import diffev2
    from mystic.math import almostEqual
    from mystic.monitors import Monitor, VerboseMonitor
    mon = VerboseMonitor(10)#,10)

    result = diffev2(objective, x0=bounds, bounds=bounds, penalty=penalty, constraints=constraint, npop=50, ftol=1e-8, gtol=200, disp=True, full_output=True, cross=0.1, scale=0.9, itermon=mon)

    print(result[0])
    assert almostEqual(result[0], xs[0], tol=1e-8) \
        or almostEqual(result[0], xs[1], tol=1e-8) \
        or almostEqual(result[0], xs[2], tol=1e-8) \
        or almostEqual(result[0], xs[3], tol=1e-8) \
        or almostEqual(result[0], xs[4], tol=1e-8) \
        or almostEqual(result[0], xs[-1], tol=1e-8)
    assert almostEqual(result[1], ys, tol=1e-4)


# EOF
