#!/usr/bin/env python
#
# Problem definition and original response:
# https://stackoverflow.com/q/67560280/2379433
# https://stackoverflow.com/a/67571398/2379433
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2021-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Objective:
    MIN (1500 * x0) + (625 * x1) + (100 * x2)

Constraints:
    schedule(x0,x1,x2) >= 480
    x2 = round(log(x0)**2)
    x1 > x0
    x1 > x2
    x0, x1, x2 are integers

Bounds:
    (5 <= x0 <= 50), (5 <= x1 <= 100), (1 <= x2 <= 20)
"""
import mystic as my
import mystic.symbolic as ms
import numpy as np
 
# define the cost function
def objective(x):
    x0,x1,x2 = x
    return (1500 * x0) + (625 * x1) + (100 * x2)

# define the bounds
bounds = [(5,50),(5,100),(1,20)]

# define symbolic constraints
eqns = '''
x1 > x0
x2 < x1
'''

# generate constraint operator from symbolic constraints
and_ = my.constraints.and_
cons = ms.generate_constraint(ms.generate_solvers(ms.simplify(eqns)), join=and_)

''' #NOTE: cons ensures eqns are satisfied
>>> cons([1,2,3])
[1, 2, 1.999999999999997]
>>> cons([5,5,1])
[5, 5.000000000000006, 1]
'''

# define an analytical function (i.e. non-symbolic)
def schedule(x0,x1,x2):
    return x0 * x1 - x2 * x2

# define the penalty condition
def penalty1(x): # <= 0.0
    x0,x1,x2 = x
    return 480 - schedule(x0,x1,x2)

# generate penalty from the penalty condition
@my.penalty.linear_inequality(penalty1)
def penalty(x):
    return 0.0

# generate constraint operator from the penalty
lb,ub = zip(*bounds)
c = my.constraints.as_constraint(penalty, lower_bounds=lb, upper_bounds=ub, nvars=3)

''' #NOTE: c ensures penalty1 is satisfied
>>> c([5,5,1])
[13.126545665004528, 44.97820356778645, 1.0138152329128338]
>>> schedule(*_)
589.3806217359323
>>> c([50,50,10])
[50.0, 50.0, 10.0]
>>> schedule(*_)
2400.0
'''

# define a constrait function for the input constraint
def intlog2(x):
    x[2] = np.round(np.log(x[0])**2)
    return x

# generate a constraint operator for all given constraints
ints = np.round
constraint = and_(c, cons, ints, intlog2)

''' #NOTE: constraint ensures c, cons, ints, and intlog2 are satisfied
>>> constraint([5,5,1])
[16.0, 42.0, 8.0]
>>> c(_)
[16.0, 42.0, 8.0]
>>> cons(_)
[16.0, 42.0, 8.0]
>>> intlog2(_)
[16.0, 42.0, 8.0]
>>> objective(_)
51050.0
'''

# solve
from mystic.solvers import diffev2
from mystic.monitors import VerboseMonitor
mon = VerboseMonitor(10)

result = diffev2(objective,x0=bounds, bounds=bounds, constraints=constraint, npop=20, gtol=50, disp=False, full_output=True, itermon=mon)

''' #NOTE: solution
>>> result[0]
array([ 14., 38.,   7.])
>>> result[1]
45450.0
'''
print(result[0])
print(result[1])
