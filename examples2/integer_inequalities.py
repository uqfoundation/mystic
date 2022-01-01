#!/usr/bin/env python
#
# Problem definition and original response:
# https://stackoverflow.com/q/59813985/2379433
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2020-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
'''
solve:
     a system of inequalities (defined below) with unknowns xi,
     where i = range(1,11)

such that: 
     0 < x10 < x9 < x8 < x7 < x6 < x5 < x4 < x3 < x2 < x1,
     xi's are integers
'''

inequalities = '''
x1 > x2
x2 > x3
x3 > x4
x4 > x5
x5 > x6
x6 > x7
x7 > x8
x8 > x9
x9 > x10
x10 > 0
x5 < 3*x1/2 + x2 + x3/2 + x4/2 - x6/2
x5 < 3*x1/2 + 3*x2/2 + x3/2 + x8 - 3*x9/2 - x10/2
x4 > -x1 - 3*x2/2 + x5/2 - x8/2 + x10/2
x3 > -2*x1 - 3*x2/2 - x4 + x5 - x6/2 - x7/2 - x8/2 + x9 + 3*x10
x3 > -3*x1/2 - 3*x2/2 - x4/2 + 3*x5/2 - x6 + 3*x7/2 - x8/2 + x10
x3 > -3*x1/2 + x2 - x4/2 + 3*x5/2 - x6/2 + x7/2 - x8/2 + x10
x2 > -x1 - x3/2 - x4/2 - x6/4 + 3*x9/4 + x10/4
x2 > -x1 - x3/4 - x4/4 - x6/4 - x8/2 + 3*x9/4 + x10/2
x2 > -x1 + x4/4 + x5 - x6/4 - x7/4 - x8/4 + x9/4 + x10/4
x2 > -3*x1/4 - x3/2 - x4/2 - x6/2 - x7/4 - x8/2 + x9/2
x2 > -x1 - x3/2 - x4/4 - x6/4 - x8/4 + x9/4 + x10/4
x2 > -x1/2 + 3*x3/4 + x4/4 - x6/2 - x7/4 - x8/4 + x9/4 + x10/4
x1 > -x2 - x3/4 - x4/2 - x6/2 - x8/4
x1 > -x2 - x3/2 - x4/4 - x6/2 - x7/4 - x8/2 + x9/2
x1 > -x2 - x3/4 - x4/4 + x5/4 + 3*x7/4
x1 > -x2 - x3/2 - x4/2 - x6/2 - x7/4 - x8/4
x1 > -x2 - x3/2 - x4/4 - x6/4 - x7/4 - x8/4 + x9/4
'''

# test solutions
xA = [10,9,8,7,6,5,4,3,2,1]
xB = [1.1,2.3,3.7,4.3,5,6,7,8.932,9.0002,10]
xC = [64.251, 94., 62.123, 0.0, 41.234, 17.4, 0.0, 0.0, 81.341, 1.987]
x = xA, xB, xC


def failures(x):
    'count the number of failures in solvinge the inequalities'
    return tuple(eval(i, dict(zip(var,x))) for i in inequalities.strip().split('\n')).count(False)

def noints(x):
    'count the number of non-integer entries'
    return tuple(i == int(i) for i in x).count(False)


# solving the system of inequalities
import mystic as my
var = my.symbolic.get_variables(inequalities)
solve = my.symbolic.generate_constraint(my.symbolic.generate_solvers(inequalities, var), join=my.constraints.and_)

for xi in x:
    xo = xi.copy()
    y = solve(xo)
    assert not failures(y)


# building an integer constraint
ints = my.constraints.integers(float)(lambda x:x)

for xi in x:
    xo = xi.copy()
    y = ints(xo)
    assert not noints(y) #NOTE: yikes, poor english!


# now combine, with 'unique' and 'discrete' helping to force to integers
helper = my.constraints.and_(my.constraints.discrete(range(100))(lambda x:x), my.constraints.impose_unique(range(100))(lambda x:x))
intsolve = my.constraints.and_(ints, solve, helper, maxiter=1000)

for xi in x:
    xo = xi.copy()
    y = intsolve(xo)
    assert not failures(y) and not noints(y)

