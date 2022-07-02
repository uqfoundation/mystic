#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

from mystic.constraints import *
from mystic.penalty import quadratic_equality
from mystic.coupler import inner
from mystic.math import almostEqual
from mystic.tools import random_seed
random_seed(669)

def test_penalize():

  from mystic.math.measures import mean, spread
  def mean_constraint(x, target):
    return mean(x) - target

  def range_constraint(x, target):
    return spread(x) - target

  @quadratic_equality(condition=range_constraint, kwds={'target':5.0})
  @quadratic_equality(condition=mean_constraint, kwds={'target':5.0})
  def penalty(x):
    return 0.0

  def cost(x):
    return abs(sum(x) - 5.0)

  from mystic.solvers import fmin
  from numpy import array
  x = array([1,2,3,4,5])
  y = fmin(cost, x, penalty=penalty, disp=False)

  assert round(mean(y)) == 5.0
  assert round(spread(y)) == 5.0
  assert round(cost(y)) == 4*(5.0)


def test_solve():

  from mystic.math.measures import mean
  def mean_constraint(x, target):
    return mean(x) - target

  def parameter_constraint(x):
    return x[-1] - x[0]

  @quadratic_equality(condition=mean_constraint, kwds={'target':5.0})
  @quadratic_equality(condition=parameter_constraint)
  def penalty(x):
    return 0.0

  x = solve(penalty, guess=[2,3,1])
 #x = solve(penalty, solver='buckshot')

  assert round(mean_constraint(x, 5.0)) == 0.0
  assert round(parameter_constraint(x)) == 0.0
  assert issolution(penalty, x, tol=0.002)


def test_solve_constraint():

  from mystic.math.measures import mean
  @with_mean(1.0)
  def constraint(x):
    x[-1] = x[0]
    return x

  x = solve(constraint, guess=[2,3,1])
 #x = solve(constraint, solver='buckshot')

  assert almostEqual(mean(x), 1.0, tol=1e-15)
  assert x[-1] == x[0]
  assert issolution(constraint, x)


def test_as_constraint():

  from mystic.math.measures import mean, spread
  def mean_constraint(x, target):
    return mean(x) - target

  def range_constraint(x, target):
    return spread(x) - target

  @quadratic_equality(condition=range_constraint, kwds={'target':5.0})
  @quadratic_equality(condition=mean_constraint, kwds={'target':5.0})
  def penalty(x):
    return 0.0

  ndim = 3
  constraints = as_constraint(penalty)#, solver='fmin')
  #XXX: this is expensive to evaluate, as there are nested optimizations

  from numpy import arange
  x = arange(ndim)
  _x = constraints(x)
  
  assert round(mean(_x)) == 5.0
  assert round(spread(_x)) == 5.0
  assert round(penalty(_x)) == 0.0

  def cost(x):
    return abs(sum(x) - 5.0)

  npop = ndim*3
  from mystic.solvers import diffev
  y = diffev(cost, x, npop, constraints=constraints, disp=False, gtol=10)

  assert round(mean(y)) == 5.0
  assert round(spread(y)) == 5.0
  assert round(cost(y)) == 5.0*(ndim-1)


def test_as_penalty():

  from mystic.math.measures import mean, spread
  @with_spread(5.0)
  @with_mean(5.0)
  def constraint(x):
    return x

  penalty = as_penalty(constraint)

  from numpy import array
  x = array([1,2,3,4,5])
  
  def cost(x):
    return abs(sum(x) - 5.0)

  from mystic.solvers import fmin
  y = fmin(cost, x, penalty=penalty, disp=False)

  assert round(mean(y)) == 5.0
  assert round(spread(y)) == 5.0
  assert round(cost(y)) == 4*(5.0)


def test_with_penalty():

  from mystic.math.measures import mean, spread
  @with_penalty(quadratic_equality, kwds={'target':5.0})
  def penalty(x, target):
    return mean(x) - target

  def cost(x):
    return abs(sum(x) - 5.0)

  from mystic.solvers import fmin
  from numpy import array
  x = array([1,2,3,4,5])
  y = fmin(cost, x, penalty=penalty, disp=False)

  assert round(mean(y)) == 5.0
  assert round(cost(y)) == 4*(5.0)


def test_with_mean():

  from mystic.math.measures import mean, impose_mean

  @with_mean(5.0)
  def mean_of_squared(x):
    return [i**2 for i in x]

  from numpy import array
  x = array([1,2,3,4,5])
  y = impose_mean(5, [i**2 for i in x])
  assert mean(y) == 5.0
  assert mean_of_squared(x) == y


def test_with_mean_spread():

  from mystic.math.measures import mean, spread, impose_mean, impose_spread

  @with_spread(50.0)
  @with_mean(5.0)
  def constrained_squared(x):
    return [i**2 for i in x]

  from numpy import array
  x = array([1,2,3,4,5])
  y = impose_spread(50.0, impose_mean(5.0,[i**2 for i in x]))
  assert almostEqual(mean(y), 5.0, tol=1e-15)
  assert almostEqual(spread(y), 50.0, tol=1e-15)
  assert constrained_squared(x) == y


def test_constrained_solve():

  from mystic.math.measures import mean, spread
  @with_spread(5.0)
  @with_mean(5.0)
  def constraints(x):
    return x

  def cost(x):
    return abs(sum(x) - 5.0)

  from mystic.solvers import fmin_powell
  from numpy import array
  x = array([1,2,3,4,5])
  y = fmin_powell(cost, x, constraints=constraints, disp=False)

  assert almostEqual(mean(y), 5.0, tol=1e-15)
  assert almostEqual(spread(y), 5.0, tol=1e-15)
  assert almostEqual(cost(y), 4*(5.0), tol=1e-6)


def test_with_constraint():

  from mystic.math.measures import mean, impose_mean

  @with_constraint(inner, kwds={'target':5.0})
  def mean_of_squared(x, target):
    return impose_mean(target, [i**2 for i in x])

  from numpy import array
  x = array([1,2,3,4,5])
  y = impose_mean(5, [i**2 for i in x])
  assert mean(y) == 5.0
  assert mean_of_squared(x) == y


def test_discrete():

  @discrete([1.0, 3.5, 5.5, 7.0])
  def discrete_squared(x):
    return x**2

  from numpy import asarray
  assert discrete_squared(5.6) == 5.5**2
  assert all(discrete_squared(asarray([1, 3])) == asarray([1.0, 3.5])**2)
  discrete_squared.samples([1.0, 7.0])
  assert discrete_squared(5.6) == 7.0**2
  discrete_squared.index([0, -1])
  assert all(discrete_squared(asarray([0, 3, 6])) == asarray([1.0, 3.0, 7.0])**2)


if __name__ == '__main__':
  test_penalize()
  test_solve()
  test_solve_constraint()
  test_as_constraint()
  test_as_penalty()
  test_with_penalty()
  test_with_mean()
  test_with_mean_spread()
  test_constrained_solve()
  test_with_constraint()
  test_discrete()


# EOF
