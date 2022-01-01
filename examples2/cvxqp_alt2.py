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

from mystic.math.measures import normalize

def constraint(x): # impose exactly
    return normalize(x, 1.0)


if __name__ == '__main__':

  from mystic.solvers import diffev2, fmin_powell
  from mystic.math import almostEqual

  result = diffev2(objective, x0=bounds, bounds=bounds, constraints=constraint, npop=40, disp=False, full_output=True)
  assert almostEqual(result[0], xs, rel=1e-5)
  assert almostEqual(result[1], ys, rel=1e-5)


# EOF
