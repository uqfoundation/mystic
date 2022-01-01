#!/usr/bin/env python
#
# Problem definition and original response:
# https://stackoverflow.com/a/17025929/2379433
# https://stackoverflow.com/a/32906273/2379433
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2018-2022 The Uncertainty Quantification Foundation.
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
    2*x0 - 4*x1 <= 0'''
eqn = '''
     x0**3 - x1 == 0'''
cons = ms.generate_constraint(ms.generate_solvers(ms.simplify(eqn,target='x1')))
pens = ms.generate_penalty(ms.generate_conditions(ieqn), k=1e3)
bounds = [(0., None), (1., None)]

# get the objective
def objective(x, sign=1):
  x = np.asarray(x)
  return sign * (2*x[0]*x[1] + 2*x[0] - x[0]**2 - 2*x[1]**2)

# solve    
x0 = np.random.rand(2)
sol = my.fmin_powell(objective, x0, constraint=cons, penalty=pens, disp=True,
                     bounds=bounds, gtol=3, ftol=1e-6, full_output=True,
                     args=(-1,))

print('x* = %s; f(x*) = %s' % (sol[0], -sol[1]))
