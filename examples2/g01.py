#!/usr/bin/env python
#
# Problem definition:
# A-R Hedar and M Fukushima, "Derivative-Free Filter Simulated Annealing
# Method for Constrained Continuous Global Optimization", Journal of
# Global Optimization, 35(4), 521-549 (2006).
# 
# Original Matlab code written by A. Hedar (Nov. 23, 2005)
# http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/go.htm
# and ported to Python by Mike McKerns (December 2014)
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

def objective(x):
    return 5*sum(x[:4]) - 5*sum([xi**2 for xi in x[:4]]) - sum(x[4:])

bounds = [(0,1)]*9 + [(0,100)]*3 + [(0,1)]
# with penalty='penalty' applied, solution is:
xs = [1.0]*9 + [3.0]*3 + [1.0]
ys = -15.0

from mystic.symbolic import generate_constraint, generate_solvers, simplify
from mystic.symbolic import generate_penalty, generate_conditions

equations = """
2.0*x0 + 2.0*x1 + x9 + x10 - 10.0 <= 0.0
2.0*x0 + 2.0*x2 + x9 + x11 - 10.0 <= 0.0
2.0*x1 + 2.0*x2 + x10 + x11 - 10.0 <= 0.0
-8.0*x0 + x9 <= 0.0
-8.0*x1 + x10 <= 0.0
-8.0*x2 + x11 <= 0.0
-2.0*x3 - x4 + x9 <= 0.0
-2.0*x5 - x6 + x10 <= 0.0
-2.0*x7 - x8 + x11 <= 0.0
"""
cf = generate_constraint(generate_solvers(simplify(equations)))
pf = generate_penalty(generate_conditions(equations))



if __name__ == '__main__':
    x = [0]*len(xs)

    from mystic.solvers import fmin_powell
    from mystic.math import almostEqual

    result = fmin_powell(objective, x0=x, bounds=bounds, constraints=cf, disp=False, full_output=True)

    assert almostEqual(result[0], xs, tol=1e-2)
    assert almostEqual(result[1], ys, tol=1e-2)



# EOF
