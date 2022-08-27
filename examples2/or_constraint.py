#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
'''
find the maximum of y = x0 * f(x1), where:

f(x) = 1 - 2**(-x)/2
x0 * x1 < 100
x0 and x1 in [0,100]
'''
import mystic as my
import numpy as np

# define the objective
def cost(x):
  return x[0]*f(x[1])

def f(x):
  return 1 - .5*2**(-x)

# define and generate the constraint
eqn = "x0 * x1 < 100"
eqns = my.symbolic.simplify(eqn, all=True)
c = my.symbolic.generate_constraint(my.symbolic.generate_solvers(eqns), join=my.constraints.or_)

# test the constraint
assert np.prod(c([5,5])) < 100
assert np.prod(c([5,25])) < 100

# solve for the maximum
result = my.solvers.fmin(lambda x: -cost(x), [1,1], constraints=c, bounds=[(0,100)]*2, full_output=True, disp=True)

print(result[0])
assert np.prod(result[0]) < 100
