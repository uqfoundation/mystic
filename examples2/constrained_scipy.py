#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2019 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
  Example applying mystic to scipy.optimize

  Minimize:
            f(x) = 3*A + 1e-6*A**3 + 2*B + 2e-6/(3*B**3) + C**2
            x = A,B,C

  Where: 
            A*B + C >= 1
            B <= A
            10 >= A >= -10
            10 >= B >= -10
            10 >= C >= -10
"""
import mystic as my
import mystic.symbolic as ms
import scipy.optimize as so

equations = """
A*B + C >= 1
B <= A
A >= -10
B >= -10
C >= -10
10 >= A
10 >= B
10 >= C
"""
var = list('ABC')
eqns = ms.simplify(equations, variables=var, all=True)
constrain = ms.generate_constraint(ms.generate_solvers(eqns, var), join=my.constraints.and_)

def objective(x):
    return 3*x[0] + 1.e-6*x[0]**3 + 2*x[1] + 2.e-6/3*x[1]**3 + x[2]**2

mon = my.monitors.Monitor()

def cost(x):
    kx = constrain(x)
    y = objective(kx)
    mon(kx,y)
    return y

# with constraint='constrain' applied, solution is:
xs = [1.28899341e+00,  1.20724621e+00, -3.32434588e-05]
ys =  1.10512255e-09


if __name__ == '__main__':
    from mystic.math import almostEqual

    result = so.fmin(cost, [1,1,1], xtol=1e-4, ftol=1e-4, full_output=True, disp=False)
    assert almostEqual(result[0], xs, rel=1e-2)
    assert almostEqual(result[1], ys, rel=1e-2)

    # check results are consistent with monitor
    assert almostEqual(result[1], min(mon.y), rel=1e-2)

    # check results satisfy constraints
    A,B,C = result[0]
    assert A*B + C >= 1
    assert B <= A
    assert 10 >= A >= -10
    assert 10 >= B >= -10
    assert 10 >= C >= -10
