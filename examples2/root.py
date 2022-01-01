#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
    Solve:
                3*x[0] - cos(x[1]*x[2]) + a = 0
  x[0]**2 - 81*(x[1]+.1)**2 + sin(x[2]) + b = 0
              exp(-x[0]*x[1]) + 20*x[2] + c = 0           

    where:
               a = -0.5
               b = 1.06
               c = (10 * pi - 3.0) / 3
"""

from math import pi, cos, sin, exp
from mystic import reduced

@reduced(lambda x,y: abs(x)+abs(y))
def objective(x, a, b, c):
  x0,x1,x2 = x
  eqns = \
         [3 * x0 - cos(x1*x2) + a,
          x0**2 - 81*(x1+0.1)**2 + sin(x2) + b,
          exp(-x0*x1) + 20*x2 + c]
  return eqns

bounds = [(-10, 10),(-10, 10),(-10, 10)] # bound, or exp causes OverflowError

a = -0.5
b = 1.06
c = (10 * pi - 3.0) / 3
args = (a,b,c)

# solution is:
xs = [0.5, 0.0, -0.523598776]
xs_ = [0.49814468, -0.1996059, -0.52882598]
ys = 0.0


if __name__ == '__main__':

  from mystic.solvers import diffev2, fmin_powell
  from mystic.math import almostEqual
# from mystic.monitors import VerboseMonitor
# mon = VerboseMonitor(10)

  result = diffev2(objective, args=args, x0=bounds, bounds=bounds, npop=40, ftol=1e-8, disp=False, full_output=True)#, itermon=mon)
# print(result[0])
  assert almostEqual(result[0], xs, tol=1e-8) \
      or almostEqual(result[0], xs_,tol=1e-8)
  assert almostEqual(result[1], ys, tol=1e-5)

  result = fmin_powell(objective, args=args, x0=[0.0,0.0,0.0], bounds=bounds, disp=False, full_output=True)
  assert almostEqual(result[0], xs, tol=1e-8) \
      or almostEqual(result[0], xs_,tol=1e-8)
  assert almostEqual(result[1], ys, tol=1e-5)


# EOF
