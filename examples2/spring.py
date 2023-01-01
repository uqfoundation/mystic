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
# Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"a Tension-Compression String"

def objective(x):
    x0,x1,x2 = x
    return x0**2 * x1 * (x2 + 2)

bounds = [(0,100)]*3
# with penalty='penalty' applied, solution is:
xs = [0.05168906, 0.35671773, 11.28896619]
ys = 0.01266523

from mystic.symbolic import generate_constraint, generate_solvers, simplify
from mystic.symbolic import generate_penalty, generate_conditions

equations = """
1.0 - (x1**3 * x2)/(71785*x0**4) <= 0.0
(4*x1**2 - x0*x1)/(12566*x0**3 * (x1 - x0)) + 1./(5108*x0**2) - 1.0 <= 0.0
1.0 - 140.45*x0/(x2 * x1**2) <= 0.0
(x0 + x1)/1.5 - 1.0 <= 0.0
"""
# cf = generate_constraint(generate_solvers(simplify(equations))) #XXX: slow
pf = generate_penalty(generate_conditions(equations), k=1e12)



if __name__ == '__main__':

    from mystic.solvers import diffev2
    from mystic.math import almostEqual

    result = diffev2(objective, x0=bounds, bounds=bounds, penalty=pf, npop=40, gtol=500, disp=False, full_output=True)

    assert almostEqual(result[0], xs, rel=1e-2)
    assert almostEqual(result[1], ys, rel=1e-2)



# EOF
