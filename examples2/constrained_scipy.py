#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2020-2022 The Uncertainty Quantification Foundation.
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
            5 >= A >= -5
            5 >= B >= -5
            5 >= C >= -5
"""
import mystic as my
import mystic.symbolic as ms
import scipy.optimize as so

equations = """
A*B + C >= 1
B <= A
A >= -5
B >= -5
C >= -5
5 >= A
5 >= B
5 >= C
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


if __name__ == '__main__':
    from mystic.math import almostEqual

    result = so.fmin(cost, [1,1,1], xtol=1e-6, ftol=1e-6, full_output=True, disp=False)
    # check results are consistent with monitor
    assert almostEqual(result[1], min(mon.y), rel=1e-2)

    # check results satisfy constraints
    A,B,C = result[0]
    print(dict(A=A, B=B, C=C))

    eps = 0.2 #XXX: give it some small wiggle room for small violations
    assert A*B + C >= 1-eps
    assert B <= A+eps
    assert (5+eps) >= A >= -(5+eps)
    assert (5+eps) >= B >= -(5+eps)
    assert (5+eps) >= C >= -(5+eps)
