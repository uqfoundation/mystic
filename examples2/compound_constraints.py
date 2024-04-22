#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2024 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

"""
First, minimize the Rosenbrock function f(x) under the following constraints:
  - mean(x) = 1
  - x_i are integers
  - x != [1, 1, 1]

Then, minimize the Rosenbrock function f(x) under the following constraints:
  - mean(x) = 2
  - x_i are unique
  - x_i are in the discrete set {0, 2, 4, 6, 8}
"""

from mystic.models import rosen
from mystic.solvers import diffev
from mystic.constraints import unique, discrete, integers, with_mean, and_, not_
from mystic.math.measures import mean

bounds = [(0,10)]*3

# generate the constraints
c = and_(integers()(lambda x:x), not_(lambda x:[1.]*len(x)), with_mean(1)(lambda x:x))

from mystic.monitors import VerboseMonitor
stepmon = VerboseMonitor(1)

result = diffev(rosen, x0=bounds, bounds=bounds, constraints=c, npop=100, gtol=10, disp=False, full_output=True, itermon=stepmon)

print("solved: %s @ %s" % (result[0], result[1]))
assert list(result[0]) == [0, 1, 2]
assert result[1] == 201

# generate the second set of constraints
c = and_(unique, discrete(range(0,10,2))(lambda x:x), with_mean(2)(lambda x:x))

stepmon = VerboseMonitor(1)

result = diffev(rosen, x0=bounds, bounds=bounds, constraints=c, npop=100, gtol=10, disp=False, full_output=True, itermon=stepmon)

print("solved: %s @ %s" % (result[0], result[1]))
assert list(result[0]) == [0, 2, 4]
assert result[1] == 402
