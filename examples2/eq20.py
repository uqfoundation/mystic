#!/usr/bin/env python
#
# Problem definition:
# Example in google/or-tools
# https://github.com/google/or-tools/blob/master/examples/python/ex20.py
# with Copyright 2010 Hakan Kjellerstrand hakank@bonetmail.com
# and disclamer as stated at the above reference link.
# 
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
  Eq 20 in Google CP Solver.

  Standard benchmark problem.
"""

def objective(x):
    return 0.0

bounds = [(0,10)]*7
# with penalty='penalty' applied, solution is:
xs = [1., 4., 6., 6., 6., 3., 1.]
ys = 0.0

# constraints
equations = """
-76706*x0 + 98205*x1 + 23445*x2 + 67921*x3 + 24111*x4 - 48614*x5 - 41906*x6 - 821228 == 0.0
87059*x0 - 29101*x1 - 5513*x2 - 21219*x3 + 22128*x4 + 7276*x5 + 57308*x6 - 22167 == 0.0
-60113*x0 + 29475*x1 + 34421*x2 - 76870*x3 + 62646*x4 + 29278*x5 - 15212*x6 - 251591 == 0.0
49149*x0 + 52871*x1 - 7132*x2 + 56728*x3 - 33576*x4 - 49530*x5 - 62089*x6 - 146074 == 0.0
-10343*x0 + 87758*x1 - 11782*x2 + 19346*x3 + 70072*x4 - 36991*x5 + 44529*x6 - 740061 == 0.0
85176*x0 - 95332*x1 - 1268*x2 + 57898*x3 + 15883*x4 + 50547*x5 + 83287*x6 - 373854 == 0.0
-85698*x0 + 29958*x1 + 57308*x2 + 48789*x3 - 78219*x4 + 4657*x5 + 34539*x6 - 249912 == 0.0
-67456*x0 + 84750*x1 - 51553*x2 + 21239*x3 + 81675*x4 - 99395*x5 - 4254*x6 - 277271 == 0.0
94016*x0 - 82071*x1 + 35961*x2 + 66597*x3 - 30705*x4 - 44404*x5 - 38304*x6 - 25334 == 0.0
-60301*x0 + 31227*x1 + 93951*x2 + 73889*x3 + 81526*x4 - 72702*x5 + 68026*x6 - 1410723 == 0.0
-16835*x0 + 47385*x1 + 97715*x2 - 12640*x3 + 69028*x4 + 76212*x5 - 81102*x6 - 1244857 == 0.0
-43277*x0 + 43525*x1 + 92298*x2 + 58630*x3 + 92590*x4 - 9372*x5 - 60227*x6 - 1503588 == 0.0
-64919*x0 + 80460*x1 + 90840*x2 - 59624*x3 - 75542*x4 + 25145*x5 - 47935*x6 - 18465 == 0.0
-45086*x0 + 51830*x1 - 4578*x2 + 96120*x3 + 21231*x4 + 97919*x5 + 65651*x6 - 1198280 == 0.0
85268*x0 + 54180*x1 - 18810*x2 - 48219*x3 + 6013*x4 + 78169*x5 - 79785*x6 - 90614 == 0.0
8874*x0 - 58412*x1 + 73947*x2 + 17147*x3 + 62335*x4 + 16005*x5 + 8632*x6 - 752447 == 0.0
71202*x0 - 11119*x1 + 73017*x2 - 38875*x3 - 14413*x4 - 29234*x5 + 72370*x6 - 129768 == 0.0
1671*x0 - 34121*x1 + 10763*x2 + 80609*x3 + 42532*x4 + 93520*x5 - 33488*x6 - 915683 == 0.0
51637*x0 + 67761*x1 + 95951*x2 + 3834*x3 - 96722*x4 + 59190*x5 + 15280*x6 - 533909 == 0.0
-16105*x0 + 62397*x1 - 6704*x2 + 43340*x3 + 95100*x4 - 68610*x5 + 58301*x6 - 876370 == 0.0
"""

from mystic.symbolic import generate_penalty, generate_conditions
pf = generate_penalty(generate_conditions(equations))
from mystic.symbolic import generate_constraint, generate_solvers, solve
cf = generate_constraint(generate_solvers(solve(equations)))

from numpy import round as npround


if __name__ == '__main__':

    from mystic.solvers import diffev2
    from mystic.math import almostEqual

   #result = diffev2(objective, x0=bounds, bounds=bounds, penalty=pf, npop=20, gtol=50, disp=True, full_output=True)
   #result = diffev2(objective, x0=bounds, bounds=bounds, penalty=pf, constraints=npround, npop=40, gtol=50, disp=True, full_output=True)
    result = diffev2(objective, x0=bounds, bounds=bounds, constraints=cf, npop=4, gtol=1, disp=True, full_output=True)

    print(result[0])
    assert almostEqual(result[0], xs, tol=1e-8) #XXX: fails b/c rel & zero?
    assert almostEqual(result[1], ys, tol=1e-4)


# EOF
