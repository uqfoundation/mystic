#!/usr/bin/env python
#
# Problem definition:
# Example in reference documentation for scipy.optimize
# http://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
# 
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

from optqp import _objective, bounds, xs, ys

from mystic.penalty import quadratic_equality
from mystic.constraints import with_penalty

@with_penalty(quadratic_equality, k=1e4)
def penalty(x): # == 0.0
    return x[0]**3 - x[1]


if __name__ == '__main__':

  from mystic.solvers import diffev2, fmin_powell
  from mystic.math import almostEqual

  result = diffev2(_objective, x0=bounds, bounds=bounds, penalty=penalty, npop=40, ftol=1e-8, gtol=100, disp=False, full_output=True)
  assert almostEqual(result[0], xs, rel=2e-2)
  assert almostEqual(result[1], ys, rel=2e-2)

  result = fmin_powell(_objective, x0=[-1.0,1.0], bounds=bounds, penalty=penalty, disp=False, full_output=True)
  assert almostEqual(result[0], xs, rel=2e-2)
  assert almostEqual(result[1], ys, rel=2e-2)


# EOD
