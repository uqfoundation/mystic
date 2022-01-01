#!/usr/bin/env python
#
# Problem definition:
# A-R Hedar and M Fukushima, "Derivative-Free Filter Simulated Annealing
# Method for Constrained Continuous Global Optimization", Journal of
# Global Optimization, 35(4), 521-549 (2006).
# 
# Original Matlab code written by A. Hedar (Nov. 23, 2005)
# http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/go.htm
# and ported to Python by Mike McKerns (December 2014)
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

def objective(x):
    x0,x1,x2 = x
    return 1.0 - 0.01*((x0 - 5)**2 + (x1 - 5)**2 + (x2 - 5)**2)

bounds = [(0,10)]*3
# with penalty='penalty' applied, solution is:
xs = [5,5,5]
ys = 1.0

from mystic.symbolic import generate_constraint, generate_solvers, solve
from mystic.symbolic import generate_penalty, generate_conditions

equations = """
"""
for i in range(1,10):
  for j in range(1,10):
    for k in range(1,10):
      equations += "(x0 - %s)**2 + (x1 - %s)**2 + (x2 - %s)**2 - 48.0 <= 0.0\n" % (i,j,k)

from mystic.constraints import as_constraint
from mystic.penalty import quadratic_inequality
from mystic.constraints import with_penalty

def penalty(x): #NOTE: not built as a 'penalty function' (loses functionality)
    sum = 0.0
    for i in range(1,10):
      for j in range(1,10):
        for k in range(1,10):
          p = lambda v: ((v[0]-i)**2 + (v[1]-j)**2 + (v[2]-k)**2 - 48.0)
          p = with_penalty(quadratic_inequality, k=1e2)(p)
          sum += p(x)
    return sum

solver = as_constraint(penalty) #XXX: may not work, as not a penalty function



if __name__ == '__main__':

    from mystic.solvers import buckshot
    from mystic.math import almostEqual

    result = buckshot(objective, 3, 10, bounds=bounds, penalty=penalty, disp=False, full_output=True)

    assert almostEqual(result[0], xs, tol=1e-2)
    assert almostEqual(result[1], ys, rel=1e-2)



# EOF
