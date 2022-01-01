#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2018-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Minimization with a slack variable and penalty.
"""
from mystic.solvers import diffev2, fmin_powell
from mystic.penalty import quadratic_inequality
from mystic.monitors import VerboseMonitor
import numpy as np

C = 221
Q = 500
production = 1000

x0 = [0.0, 0.0, 0.0, 0.0]
bounds = [(0,5), (0,4), (0,3), (0,None)]
#x[3] is the slack variable

def func_value(d):
    curve_vec=[]
    for val in d:
        curve = (0.3 * val) + ((2 * (val ** (3/2))) / 3)
        curve_vec.append(curve)
    return curve_vec

def func(x):
    curve = func_value(x[0:3])
    return -(sum(np.dot(curve,production))-Q+x[3])

objective = lambda x: sum(np.dot(x[0:3],C))+1000*x[3]     

constraint = lambda x: func(x)

@quadratic_inequality(constraint)
def penalty(x):
    return 0.0


mon = VerboseMonitor(50)
solution = diffev2(objective,x0,penalty=penalty,bounds=bounds,itermon=mon,gtol=100, maxiter=1000, maxfun=10000, npop=40)
print(solution)

mon = VerboseMonitor(50)
solution = fmin_powell(objective,x0,penalty=penalty,bounds=bounds,itermon=mon,gtol=100, maxiter=1000, maxfun=10000)
print(solution)

