#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

from mystic.coupler import *
from mystic.math import almostEqual

def test_outer():

  def squared(x):
    return x**2

  @outer(outer=squared) 
  def plus_one_squared(x):
    return x+1

  from numpy import array
  x = array([1,2,3,4,5])
  assert all(plus_one_squared(x)) == all((x+1)**2)


def test_inner():

  def squared(x):
    return x**2

  @inner(inner=squared) 
  def squared_plus_one(x):
    return x+1

  from numpy import array
  x = array([1,2,3,4,5])
  assert all(squared_plus_one(x)) == all(x**2 + 1)


def test_outer_constraint():

  from mystic.math.measures import impose_mean, mean

  def impose_constraints(x, mean, weights=None):
    return impose_mean(mean, x, weights)

  @outer(outer=impose_constraints, kwds={'mean':5.0})
  def mean_of_squared(x):
    return [i**2 for i in x]

  from numpy import array
  x = array([1,2,3,4,5])
  y = impose_mean(5, [i**2 for i in x])
  assert mean(y) == 5.0
  assert mean_of_squared(x) == y


def test_inner_constraint():

  from mystic.math.measures import impose_mean

  def impose_constraints(x, mean, weights=None):
    return impose_mean(mean, x, weights)

  @inner(inner=impose_constraints, kwds={'mean':5.0})
  def mean_then_squared(x):
    return [i**2 for i in x]

  from numpy import array
  x = array([1,2,3,4,5])
  assert mean_then_squared(x) == [i**2 for i in impose_mean(5,x)]


def test_proxified_constraint():

  from mystic.math.measures import impose_mean

  @inner_proxy(inner=impose_mean)
  def mean_then_squared(x): #XXX: proxy doesn't preserve function signature
    return [i**2 for i in x]

  from numpy import array
  x = array([1,2,3,4,5])
  assert mean_then_squared(5,x) == [i**2 for i in impose_mean(5,x)]


def test_inner_constraints():

  from mystic.math.measures import impose_mean, impose_spread

  def impose_constraints(x, mean=0.0, spread=1.0):
    x = impose_mean(mean, x)
    x = impose_spread(spread, x)
    return x

  @inner(inner=impose_constraints, kwds={'mean':5.0, 'spread':50.0})
  def constrained_squared(x):
    return [i**2 for i in x]

  from numpy import array
  x = array([1,2,3,4,5])
  y = impose_spread(50.0, impose_mean(5.0,x))
  assert constrained_squared(x) == [i**2 for i in y]


def test_proxified_constraints():

  from mystic.math.measures import impose_mean, impose_spread

  def impose_constraints(x, mean=0.0, spread=1.0):
    x = impose_mean(mean, x)
    x = impose_spread(spread, x)
    return x

  @inner_proxy(inner=impose_constraints)
  def constrained_squared(x): #XXX: proxy doesn't preserve function signature
    return [i**2 for i in x]

  from numpy import array
  x = array([1,2,3,4,5])
  y = impose_spread(50.0, impose_mean(5.0,x))
  assert constrained_squared(x, 5.0, 50.0) == [i**2 for i in y]


def test_constrain():

  from mystic.math.measures import mean, spread
  from mystic.math.measures import impose_mean, impose_spread
  def mean_constraint(x, mean=0.0):
    return impose_mean(mean, x)

  def range_constraint(x, spread=1.0):
    return impose_spread(spread, x)

  @inner(inner=range_constraint, kwds={'spread':5.0})
  @inner(inner=mean_constraint, kwds={'mean':5.0})
  def constraints(x):
    return x

  def cost(x):
    return abs(sum(x) - 5.0)

  from mystic.solvers import fmin_powell
  from numpy import array
  x = array([1,2,3,4,5])
  y = fmin_powell(cost, x, constraints=constraints, disp=False)

  assert mean(y) == 5.0
  assert spread(y) == 5.0
  assert almostEqual(cost(y), 4*(5.0))


if __name__ == '__main__':
  test_outer()
  test_inner()
  test_outer_constraint()
  test_inner_constraint()
  test_proxified_constraint()
  test_inner_constraints()
  test_proxified_constraints()
  test_constrain()


# EOF
