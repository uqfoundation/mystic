#!/usr/bin/env python
#
# Problem definition:
# Example in google/or-tools
# https://github.com/google/or-tools/blob/master/examples/python/crypto.py
# with Copyright 2010 Hakan Kjellerstrand hakank@bonetmail.com
# and disclamer as stated at the above reference link.
# 
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
  Crypto problem in Google CP Solver.

  Prolog benchmark problem
  '''
  Name           : crypto.pl
  Original Source: P. Van Hentenryck's book
  Adapted by     : Daniel Diaz - INRIA France
  Date           : September 1992
  '''
"""

def objective(x):
    return 0.0

nletters = 26

bounds = [(1,nletters)]*nletters
# with penalty='penalty' applied, solution is:
#      A   B   C   D   E   F   G   H   I   J   K   L   M   N   O   P   Q
xs = [ 5, 13,  9, 16, 20,  4, 24, 21, 25, 17, 23,  2,  8, 12, 10, 19,  7, \
#      R   S   T   U   V   W   X   Y   Z
      11, 15,  3,  1, 26,  6, 22, 14, 18]
ys = 0.0

# constraints
equations = """
B + A + L + L + E + T - 45 == 0
C + E + L + L + O - 43 == 0
C + O + N + C + E + R + T - 74 == 0
F + L + U + T + E - 30 == 0
F + U + G + U + E - 50 == 0
G + L + E + E - 66 == 0
J + A + Z + Z - 58 == 0
L + Y + R + E - 47 == 0
O + B + O + E - 53 == 0
O + P + E + R + A - 65 == 0
P + O + L + K + A - 59 == 0
Q + U + A + R + T + E + T - 50 == 0
S + A + X + O + P + H + O + N + E - 134 == 0
S + C + A + L + E - 51 == 0
S + O + L + O - 37 == 0
S + O + N + G - 61 == 0
S + O + P + R + A + N + O - 82 == 0
T + H + E + M + E - 72 == 0
V + I + O + L + I + N - 100 == 0
W + A + L + T + Z - 34 == 0
"""
var = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
#NOTE: FOR A MORE DIFFICULT PROBLEM, COMMENT OUT THE FOLLOWING 5 LINES
bounds[0] = (5,5)    # A
bounds[4] = (20,20)  # E
bounds[8] = (25,25)  # I
bounds[14] = (10,10) # O
bounds[20] = (1,1)   # U

from mystic.constraints import unique, near_integers, has_unique

from mystic.symbolic import generate_penalty, generate_conditions
pf = generate_penalty(generate_conditions(equations,var),k=1)
from mystic.constraints import as_constraint
cf = as_constraint(pf)
from mystic.penalty import quadratic_equality

@quadratic_equality(near_integers)
@quadratic_equality(has_unique)
def penalty(x):
    return pf(x)

from numpy import round, hstack, clip
def constraint(x):
    x = round(x).astype(int) # force round and convert type to int
    x = clip(x, 1,nletters)  #XXX: hack to impose bounds
    x = unique(x, list(range(1,nletters+1)))
    return x


if __name__ == '__main__':

    from mystic.solvers import diffev2
    from mystic.math import almostEqual
    from mystic.monitors import Monitor, VerboseMonitor
    mon = VerboseMonitor(10)#,10)

    result = diffev2(objective, x0=bounds, bounds=bounds, penalty=pf, constraints=constraint, npop=52, ftol=1e-8, gtol=1000, disp=True, full_output=True, cross=0.1, scale=0.9, itermon=mon)
   # FIXME: solves at 0%... but w/ vowels fixed 80%?
   #result = diffev2(objective, x0=bounds, bounds=bounds, penalty=pf, constraints=constraint, npop=52, ftol=1e-8, gtol=2000, disp=True, full_output=True, cross=0.1, scale=0.9, itermon=mon)
   #result = diffev2(objective, x0=bounds, bounds=bounds, penalty=pf, constraints=constraint, npop=130, ftol=1e-8, gtol=1000, disp=True, full_output=True, cross=0.1, scale=0.9, itermon=mon)
   #result = diffev2(objective, x0=bounds, bounds=bounds, penalty=pf, constraints=constraint, npop=260, ftol=1e-8, gtol=500, disp=True, full_output=True, cross=0.1, scale=0.9, itermon=mon)
   #result = diffev2(objective, x0=bounds, bounds=bounds, penalty=pf, constraints=constraint, npop=520, ftol=1e-8, gtol=100, disp=True, full_output=True, cross=0.1, scale=0.9, itermon=mon)

    print(result[0])
    assert almostEqual(result[0], xs, tol=1e-8)
    assert almostEqual(result[1], ys, tol=1e-4)


# EOF
