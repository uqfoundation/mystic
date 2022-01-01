#!/usr/bin/env python
#
# Problem definition:
# Example in google/or-tools
# https://github.com/google/or-tools/blob/master/examples/python/ex10.py
# with Copyright 2010 Hakan Kjellerstrand hakank@bonetmail.com
# and disclamer as stated at the above reference link.
# 
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
  Eq 10 in Google CP Solver.

  Standard benchmark problem.
"""

def objective(x):
    return 0.0

bounds = [(0,10)]*7
# with penalty='penalty' applied, solution is:
xs = [6., 0., 8., 4., 9., 3., 9.]
ys = 0.0

# constraints
equations = """
98527*x0 + 34588*x1 + 5872*x2 + 59422*x4 + 65159*x6 - 1547604 - 30704*x3 - 29649*x5 == 0.0
98957*x1 + 83634*x2 + 69966*x3 + 62038*x4 + 37164*x5 + 85413*x6 - 1823553 - 93989*x0 == 0.0
900032 + 10949*x0 + 77761*x1 + 67052*x4 - 80197*x2 - 61944*x3 - 92964*x5 - 44550*x6 == 0.0
73947*x0 + 84391*x2 + 81310*x4 - 1164380 - 96253*x1 - 44247*x3 - 70582*x5 - 33054*x6 == 0.0
13057*x2 + 42253*x3 + 77527*x4 + 96552*x6 - 1185471 - 60152*x0 - 21103*x1 - 97932*x5 == 0.0
1394152 + 66920*x0 + 55679*x3 - 64234*x1 - 65337*x2 - 45581*x4 - 67707*x5 - 98038*x6 == 0.0
68550*x0 + 27886*x1 + 31716*x2 + 73597*x3 + 38835*x6 - 279091 - 88963*x4 - 76391*x5 == 0.0
76132*x1 + 71860*x2 + 22770*x3 + 68211*x4 + 78587*x5 - 480923 - 48224*x0 - 82817*x6 == 0.0
519878 + 94198*x1 + 87234*x2 + 37498*x3 - 71583*x0 - 25728*x4 - 25495*x5 - 70023*x6 == 0.0
361921 + 78693*x0 + 38592*x4 + 38478*x5 - 94129*x1 - 43188*x2 - 82528*x3 - 69025*x6 == 0.0
"""

from mystic.symbolic import generate_penalty, generate_conditions
pf = generate_penalty(generate_conditions(equations))
from mystic.symbolic import generate_constraint, generate_solvers, solve
cf = generate_constraint(generate_solvers(solve(equations)))

from mystic.constraints import and_ as combined
from numpy import round as npround

c = combined(npround, cf)


if __name__ == '__main__':

    from mystic.solvers import diffev2
    from mystic.math import almostEqual

   #result = diffev2(objective, x0=bounds, bounds=bounds, penalty=pf, npop=20, gtol=50, disp=True, full_output=True)
   #result = diffev2(objective, x0=bounds, bounds=bounds, penalty=pf, constraints=npround, npop=40, gtol=50, disp=True, full_output=True)
    result = diffev2(objective, x0=bounds, bounds=bounds, constraints=c, npop=4, gtol=1, disp=True, full_output=True)

    print(result[0])
    assert almostEqual(result[0], xs, tol=1e-8) #XXX: fails b/c rel & zero?
    assert almostEqual(result[1], ys, tol=1e-4)


# EOF
