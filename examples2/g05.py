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
    x0,x1,x2,x3 = x
    return 3*x0 + 1.e-6*x0**3 + 2*x1 + 2.e-6/3*x1**3

bounds = [(0,1200)]*2 + [(-0.55,0.55)]*2
# with penalty='penalty' applied, solution is:
xs = [679.945794311, 1026.0666256385, 0.11887602615356, -0.3962337137961]
ys = 5126.49810960

from mystic.symbolic import generate_constraint, generate_solvers, simplify
from mystic.symbolic import generate_penalty, generate_conditions

equations = """
x2 - x3 - 0.55 <= 0.0
x3 - x2 - 0.55 <= 0.0
abs(1000*(sin(-x2-.25) + sin(-x3-0.25)) + 894.8 - x0) = 0.0
abs(1000*(sin(x2-.25) + sin(x2-x3-0.25)) + 894.8 - x1) = 0.0
abs(1000*(sin(x3-.25) + sin(x3-x2-0.25)) + 1294.8) = 0.0
"""
#cf = generate_constraint(generate_solvers(simplify(equations)))
pf = generate_penalty(generate_conditions(equations), k=1e12)



if __name__ == '__main__':

    from mystic.solvers import buckshot, sparsity
    from mystic.math import almostEqual

    result = buckshot(objective, len(xs), npts=100, bounds=bounds, penalty=pf, disp=False, full_output=True)
    #result = sparsity(objective, len(xs), npts=100, rtol=-10, bounds=bounds, penalty=pf, disp=False, full_output=True)

    assert almostEqual(result[0], xs, rel=1e-1)
    assert almostEqual(result[1], ys, rel=1e-1)



# EOF
