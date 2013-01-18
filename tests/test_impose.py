from mystic.math.measures import *
from mystic.math import almostEqual

def test_impose_reweighted_mean():

  x0 = [1,2,3,4,5]
  w0 = [3,1,1,1,1]
  m = 3.5

  x, w = impose_reweighted_mean(m, x0, w0)
  assert x0 == x
  assert almostEqual(mean(x,w), m)


def test_impose_reweighted_variance():

  x0 = [1,2,3,4,5]
  w0 = [3,1,1,1,1]
  v = 1.0

  x, w = impose_reweighted_variance(v, x0, w0)
  assert x0 == x
  assert almostEqual(variance(x,w), v)
  assert almostEqual(mean(x0,w0), mean(x,w))


if __name__ == '__main__':
  test_impose_reweighted_mean()
  test_impose_reweighted_variance()


# EOF
