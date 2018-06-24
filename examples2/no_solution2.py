#!/usr/bin/env python
#
# Problem definition and original response:
# https://stackoverflow.com/q/12942153/2379433
# https://stackoverflow.com/a/43173143/2379433
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2018 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
'''
Attempt to solve equations with no solution, using penaltys.
'''
def objective(x):
    return 0.0

equations = """
(x0 - 0)**2 + (x1 - 0)**2 - .88**2 == 0
(x0 - 2)**2 + (x1 - 0)**2 - 1**2 == 0
(x0 - 0)**2 + (x1 - 2)**2 - .75**2 == 0
"""

bounds = [(None,None),(None,None)] #unnecessary

from mystic.symbolic import generate_penalty, generate_conditions
from mystic.solvers import diffev2

pf = generate_penalty(generate_conditions(equations), k=1e12)

result = diffev2(objective, x0=bounds, bounds=bounds, penalty=pf, \
                 npop=40, gtol=50, disp=False, full_output=True)

print(result[0])
print(result[1])
