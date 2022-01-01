#!/usr/bin/env python
#
# Problem definition:
# Example in reference documentation for scipy.optimize.linprog.
# http://docs.scipy.org/doc/scipy-dev/reference/optimize.linprog-simplex.html
# 
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
    Minimize: f = -1*x[0] + 4*x[1]

    Subject to: -3*x[0] + 1*x[1] <= 6
                 1*x[0] + 2*x[1] <= 4
                            x[1] >= -3

    where:  -inf <= x[0] <= inf
"""

def objective(x):
  x0,x1 = x
  return -x0 + 4*x1

equations = """
-3*x0 + x1 - 6.0 <= 0.0
x0 + 2*x1 - 4.0 <= 0.0
"""
bounds = [(None, None),(-3.0, None)]

# with penalty='penalty' applied, solution is:
xs = [10.0, -3.0]
ys = -22.0
# alternately, if solving for the maximum, the solution is:
_xs = [-1.14285714,  2.57142857]
_ys = 11.428571428571429

from mystic.symbolic import generate_conditions, generate_penalty
pf = generate_penalty(generate_conditions(equations))
from mystic.symbolic import generate_constraint, generate_solvers, simplify
cf = generate_constraint(generate_solvers(simplify(equations)))

# inverted objective, used in solving for the maximum
_objective = lambda x: -objective(x)


if __name__ == '__main__':

  from mystic.solvers import diffev2, fmin_powell
  from mystic.math import almostEqual

  result = diffev2(objective, x0=bounds, bounds=bounds, constraint=cf, penalty=pf, npop=40, disp=False, full_output=True)
  assert almostEqual(result[0], xs, rel=1e-2)
  assert almostEqual(result[1], ys, rel=1e-2)

  result = fmin_powell(objective, x0=[0.0,0.0], bounds=bounds, constraint=cf, penalty=pf, disp=False, full_output=True)
  assert almostEqual(result[0], xs, rel=1e-2)
  assert almostEqual(result[1], ys, rel=1e-2)

  # alternately, solving for the maximum
  result = diffev2(_objective, x0=bounds, bounds=bounds, constraint=cf, penalty=pf, npop=40, disp=False, full_output=True)
  assert almostEqual( result[0], _xs, rel=1e-2)
  assert almostEqual(-result[1], _ys, rel=1e-2)

  result = fmin_powell(_objective, x0=[0,0], bounds=bounds, constraint=cf, penalty=pf, npop=40, disp=False, full_output=True)
  assert almostEqual( result[0], _xs, rel=1e-2)
  assert almostEqual(-result[1], _ys, rel=1e-2)


# EOF
