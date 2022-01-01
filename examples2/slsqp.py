#!/usr/bin/env python
#
# Problem definition:
# Example from stack overflow.
# http://stackoverflow.com/questions/17009774
# 
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
    Minimize: f = x[0]**2 + 4*x[1]**2 - 32*x[1] + 64

    Subject to: x[0] + 1*x[1] <= 7
               -x[0] + 2*x[1] <= 4

    where:  0 <= x[0] <= inf
            0 <= x[1] <= 4
"""
import numpy as np
import mystic.symbolic as ms
import mystic.solvers as my
import mystic.math as mm

# generate constraints and penalty for a linear system of equations 
A = np.array([[1, 1],[-1, 2]])
b = np.array([7,4])
eqns = ms.linear_symbolic(G=A, h=b)
cons = ms.generate_constraint(ms.generate_solvers(ms.simplify(eqns)))
pens = ms.generate_penalty(ms.generate_conditions(eqns), k=1e3)
bounds = [(0., None), (0., 4.)]

# get the objective
def objective(x):
  x = np.asarray(x)
  return x[0]**2 + 4*x[1]**2 - 32*x[1] + 64

x0 = np.random.rand(2)

# compare against the exact minimum
xs = np.array([2., 3.])
ys = objective(xs)


sol = my.fmin_powell(objective, x0, constraint=cons, penalty=pens, disp=False,
                     bounds=bounds, gtol=3, ftol=1e-6, full_output=True)

assert mm.almostEqual(sol[0], xs, tol=1e-2)
assert mm.almostEqual(sol[1], ys, tol=1e-2)


sol = my.diffev(objective, bounds, constraint=cons, penalty=pens, disp=False,
                bounds=bounds, npop=10, gtol=100, ftol=1e-6, full_output=True)

assert mm.almostEqual(sol[0], xs, tol=1e-2)
assert mm.almostEqual(sol[1], ys, tol=1e-2)


# EOF
