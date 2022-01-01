#!/usr/bin/env python
#
# Problem definition:
# Example in reference documentation for scipy.optimize.slsqp.
# http://docs.scipy.org/doc/scipy-0.10.0/reference/tutorial/optimize.html
# 
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
    Maximize: f = 2*x[0]*x[1] + 2*x[0] - x[0]**2 - 2*x[1]**2

    Subject to: -2*x[0] + 2*x[1] <= -2
                 2*x[0] - 4*x[1] <= 0
                   x[0]**3 -x[1] == 0

    where: 0 <= x[0] <= inf
           1 <= x[1] <= inf
"""

import numpy as np
import mystic.symbolic as ms
import mystic.solvers as my
import mystic.math as mm

# generate constraints and penalty for a nonlinear system of equations 
ieqn = '''
   -2*x0 + 2*x1 <= -2
    2*x0 - 4*x1 <= 0
'''
eqn = '''
     x0**3 - x1 == 0
'''
cons = ms.generate_constraint(ms.generate_solvers(ms.simplify(eqn,target='x1')))
pens = ms.generate_penalty(ms.generate_conditions(ieqn), k=1e3)
bounds = [(0., None), (1., None)]

# get the objective
def objective(x, sign=1):
  x = np.asarray(x)
  return sign * (2*x[0]*x[1] + 2*x[0] - x[0]**2 - 2*x[1]**2)

x0 = np.random.rand(2)

# compare against the exact minimum
xs = np.array([2., 1.])
ys = objective(xs, -1)


sol = my.fmin_powell(objective, x0, constraint=cons, penalty=pens, disp=False,
                     bounds=bounds, gtol=3, ftol=1e-6, full_output=True,
                     args=(-1,))

assert mm.almostEqual(sol[0], xs, tol=1e-2)
assert mm.almostEqual(sol[1], ys, tol=1e-2)


sol = my.diffev(objective, bounds, constraint=cons, penalty=pens, disp=False,
                bounds=bounds, npop=15, gtol=100, ftol=1e-6, full_output=True,
                args=(-1,))

assert mm.almostEqual(sol[0], xs, tol=1e-2)
assert mm.almostEqual(sol[1], ys, tol=1e-2)


# EOF
