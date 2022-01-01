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
    x0,x1 = x
    from math import sin, pi
    return -(sin(2*pi*x0)**3 * sin(2*pi*x1)) / (x0**3 * (x0 + x1))

bounds = [(0,10)]*2
# with penalty='penalty' applied, solution is:
xs = [1.22797136, 4.24537337]
ys = -0.09582504

from mystic.symbolic import generate_constraint, generate_solvers, simplify
from mystic.symbolic import generate_penalty, generate_conditions

equations = """
x0**2 - x1 + 1.0 <= 0.0
1.0 - x0 + (x1 - 4)**2 <= 0.0
"""
cf = generate_constraint(generate_solvers(simplify(equations)))
pf = generate_penalty(generate_conditions(equations), k=1e12)



if __name__ == '__main__':

    from mystic.solvers import buckshot
    from mystic.math import almostEqual

    result = buckshot(objective, 2, 40, bounds=bounds, penalty=pf, disp=False, full_output=True)

    assert almostEqual(result[0], xs, tol=1e-2)
    assert almostEqual(result[1], ys, rel=1e-2)



# EOF
