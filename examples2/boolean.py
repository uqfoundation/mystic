#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2018-2020 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Maximization with a boolean variable and constraints.
"""
from mystic.solvers import diffev2
from mystic.monitors import VerboseMonitor
from mystic.constraints import impose_sum, discrete, and_
import numpy as np

N = 10
b = 5
bounds = [(0,1)] * N

def objective(x, w):
    s = 0
    for i in range(len(x)-1):
        for j in range(i, len(x)):
            s += w[i,j] * x[i] * x[j]
    return s


w = np.ones((N,N)) #XXX: replace with actual values of wij

cost = lambda x: -objective(x, w)

c = and_(lambda x: impose_sum(b, x), discrete([0,1])(lambda x:x))

mon = VerboseMonitor(10)
solution = diffev2(cost,bounds,constraints=c,bounds=bounds,itermon=mon,gtol=50, maxiter=5000, maxfun=50000, npop=10)
print(solution)

