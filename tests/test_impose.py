#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE

from mystic.math.measures import *
from mystic.math import almostEqual

def test_impose_reweighted_mean():

  x0 = [1,2,3,4,5]
  w0 = [3,1,1,1,1]
  m = 3.5

  w = impose_reweighted_mean(m, x0, w0)
  assert almostEqual(mean(x0,w), m)


def test_impose_reweighted_variance():

  x0 = [1,2,3,4,5]
  w0 = [3,1,1,1,1]
  v = 1.0

  w = impose_reweighted_variance(v, x0, w0)
  assert almostEqual(variance(x0,w), v)
  assert almostEqual(mean(x0,w0), mean(x0,w))


if __name__ == '__main__':
  test_impose_reweighted_mean()
  test_impose_reweighted_variance()


# EOF
