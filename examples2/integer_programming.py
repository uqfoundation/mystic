#!/usr/bin/env python
#
# Problem definition:
# Example in google/or-tools
# https://github.com/google/or-tools/blob/master/examples/python/integer_programming.py
# with Copyright 2010-2013 Google
# and disclamer as stated at the above reference link.
# 
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
  Integer programming example

  Minimize:
            f(x) = x0 + 2*x1

  Where: 
            3*x0 + 2*x1 >= 17
            x0,x1 are integers
"""

def objective(x):
    return x[0] + 2*x[1]

bounds = [(0,None)]*2

# with penalty='penalty' applied, solution is:
xs = [6., 0.]
ys = 6.0

# constraints
equations = """
3.*x0 + 2.*x1 - 17. >= 0.0
"""

from mystic.symbolic import generate_penalty, generate_conditions
pf = generate_penalty(generate_conditions(equations))
from mystic.symbolic import generate_constraint, generate_solvers, simplify
cf = generate_constraint(generate_solvers(simplify(equations)))

from mystic.constraints import integers

@integers()
def round(x):
  return x


if __name__ == '__main__':

    from mystic.solvers import diffev2
    from mystic.math import almostEqual

    result = diffev2(objective, x0=bounds, bounds=bounds, penalty=pf, constraints=round, npop=20, gtol=50, disp=True, full_output=True)

    print(result[0])
    assert almostEqual(result[0], xs, tol=1e-8) #XXX: fails b/c rel & zero?
    assert almostEqual(result[1], ys, tol=1e-4)


# EOF
