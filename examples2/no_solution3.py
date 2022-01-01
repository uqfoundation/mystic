#!/usr/bin/env python
#
# Problem definition and original response:
# https://stackoverflow.com/q/12942153/2379433
# https://stackoverflow.com/a/43173143/2379433
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2018-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
'''
Attempt to solve equations with no solution, with a constrained search.

The equations are:
    (x-x1)^2 + (y-y1)^2 - r1^2 = 0
    (x-x2)^2 + (y-y2)^2 - r2^2 = 0
    (x-x3)^2 + (y-y3)^2 - r3^2 = 0

with the given points (x,y) and a radii (r):
    x1, y1, r1 = (0, 0, 0.88)
    x2, y2, r2 = (2, 0, 1)
    x3, y3, r3 = (0, 2, 0.75)
'''
def objective(x):
    return 0.0

equations = """
(x0 - 0)**2 + (x1 - 0)**2 - .88**2 == 0
(x0 - 2)**2 + (x1 - 0)**2 - 1**2 == 0
(x0 - 0)**2 + (x1 - 2)**2 - .75**2 == 0
"""

bounds = [(None,None),(None,None)] #unnecessary

from mystic.symbolic import generate_constraint, generate_solvers, simplify
from mystic.symbolic import generate_penalty, generate_conditions    
from mystic.solvers import diffev2

cf = generate_constraint(generate_solvers(simplify(equations)))

result = diffev2(objective, x0=bounds, bounds=bounds, \
                 constraints=cf, \
                 npop=40, gtol=50, disp=False, full_output=True)

print(result[0])
print(result[1])
