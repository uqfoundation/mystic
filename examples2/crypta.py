#!/usr/bin/env python
#
# Problem definition:
# Example in google/or-tools
# https://github.com/google/or-tools/blob/master/examples/python/crypta.py
# with Copyright 2010 Hakan Kjellerstrand hakank@bonetmail.com
# and disclamer as stated at the above reference link.
# 
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
  Cryptarithmetic puzzle in Google CP Solver.

  Prolog benchmark problem
  '''
  Name           : crypta.pl
  Title          : crypt-arithmetic
  Original Source: P. Van Hentenryck's book
  Adapted by     : Daniel Diaz - INRIA France
  Date           : September 1992

  Solve the operation:

     B A I J J A J I I A H F C F E B B J E A
   + D H F G A B C D I D B I F F A G F E J E
   -----------------------------------------
   = G J E G A C D D H F A F J B F I H E E F
  '''
"""

def objective(x):
    return 0.0

bounds = [(0,9)]*10 + [(0,1)]*2
# with penalty='penalty' applied, solution is:
xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 0]
ys = 0.0

# constraints
equations = """
A + 10*E + 100*J + 1000*B + 10000*B + 100000*E + 1000000*F + E + 10*J + 100*E + 1000*F + 10000*G + 100000*A + 1000000*F - F - 10*E - 100*E - 1000*H - 10000*I - 100000*F - 1000000*B - 10000000*Y == 0
C + 10*F + 100*H + 1000*A + 10000*I + 100000*I + 1000000*J + F + 10*I + 100*B + 1000*D + 10000*I + 100000*D + 1000000*C + Y - J - 10*F - 100*A - 1000*F - 10000*H - 100000*D - 1000000*D - 10000000*Z == 0
A + 10*J + 100*J + 1000*I + 10000*A + 100000*B + B + 10*A + 100*G + 1000*F + 10000*H + 100000*D + Z - C - 10*A - 100*G - 1000*E - 10000*J - 100000*G == 0
"""
var = list('ABCDEFGHIJYZ')
# correct bounds for:
# B - 1 >= 0
# D - 1 >= 0
# G - 1 >= 0
bounds[1] = bounds[3] = bounds[6] = (1,9)
#NOTE: FOR A MORE DIFFICULT PROBLEM, COMMENT OUT THE FOLLOWING 2 LINES
bounds[-1] = (0,0)
bounds[-2] = (1,1)

from mystic.constraints import unique, near_integers

from mystic.symbolic import generate_constraint, generate_solvers, solve
from mystic.symbolic import generate_penalty, generate_conditions
pf = generate_penalty(generate_conditions(equations,var),k=1e-6)
cf = generate_constraint(generate_solvers(solve(equations,var)))

from mystic.penalty import quadratic_inequality, quadratic_equality

#@quadratic_equality(near_integers)
def penalty(x):
    return pf(x)
   #return 0.0

#XXX: better way to constrain the last 2 (0,1) differently than rest (0,9)?
from numpy import round, hstack, clip
def constraint(x):
    # x[0:10] in range(0,10); x[10:] in range(0,2)
    x = round(x).astype(int) # force round and convert type to int
    x0,x1 = x[:-2],x[-2:]
    x0 = clip(x0, 0,9)       #XXX: hack to impose bounds
    x1 = clip(x1, 0,1)       #XXX: hack to impose bounds
    x0 = unique(x0, list(range(0,10)))
    return hstack([x0, x1])


if __name__ == '__main__':

    from mystic.solvers import diffev2, lattice
    from mystic.math import almostEqual
    from mystic.monitors import Monitor, VerboseMonitor
    mon = VerboseMonitor(10)#,10)

   #result = lattice(objective, 12, 960, bounds=bounds, penalty=penalty, constraints=constraint, ftol=1e-6, xtol=1e-6, disp=True, full_output=True, itermon=mon, maxiter=25)
   #result = diffev2(objective, x0=bounds, bounds=bounds, penalty=pf, constraints=constraint, npop=80, gtol=100, disp=True, full_output=True)
   #result = diffev2(objective, x0=bounds, bounds=bounds, penalty=penalty, constraints=constraint, npop=1000, ftol=1e-8, gtol=50, disp=True, full_output=True, cross=0.9, scale=0.9, itermon=mon)
    #FIXME: SOLVES at about 20%?... but w/ last 2 fixed 90%?
    result = diffev2(objective, x0=bounds, bounds=bounds, penalty=penalty, constraints=constraint, npop=360, ftol=1e-8, gtol=200, disp=True, full_output=True, cross=0.2, scale=0.9, itermon=mon)

    print(result[0])
    assert almostEqual(result[0], xs, tol=1e-8) #XXX: fails b/c rel & zero?
    assert almostEqual(result[1], ys, tol=1e-4)


# EOF
