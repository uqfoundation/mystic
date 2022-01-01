#!/usr/bin/env python
#
# Problem definition:
# Example in reference documentation for cvxopt
# http://cvxopt.org/examples/tutorial/qp.html
# 
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

from cvxqp import objective, bounds, xs, ys
    
from mystic.penalty import quadratic_equality
from mystic.constraints import with_penalty

@with_penalty(quadratic_equality, k=1e4)
def penalty(x): # == 0.0
    return x[1] + x[0] - 1.0


if __name__ == '__main__':

  from mystic.solvers import diffev2, fmin_powell
  from mystic.math import almostEqual

  result = diffev2(objective, x0=bounds, bounds=bounds, penalty=penalty, npop=40, disp=False, full_output=True)
  assert almostEqual(result[0], xs, rel=2e-2)
  assert almostEqual(result[1], ys, rel=2e-2)

  result = fmin_powell(objective, x0=[0.0,0.0], bounds=bounds, penalty=penalty, disp=False, full_output=True)
  assert almostEqual(result[0], xs, rel=2e-2)
  assert almostEqual(result[1], ys, rel=2e-2)


# EOF
