#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

from mystic.constraints import *
from mystic.solvers import *
from mystic.math import almostEqual
from mystic.tools import random_seed
random_seed(42)

def test_one_liner(solver):

  from mystic.math.measures import mean, spread
  @with_spread(5.0)
  @with_mean(5.0)
  def constraints(x):
    return x

  def cost(x):
    return abs(sum(x) - 5.0)

  from numpy import array
  x = array([1,2,3,4,5])
  y = solver(cost, x, constraints=constraints, disp=False)

  assert almostEqual(mean(y), 5.0, tol=1e-15)
  assert almostEqual(spread(y), 5.0, tol=1e-15)
  assert almostEqual(cost(y), 4*(5.0), tol=1e-6)


def test_multi_liner(solver):

  from mystic.monitors import Monitor
  evalmon = Monitor()
  stepmon = Monitor()

  from mystic.math.measures import mean, spread
  @with_spread(5.0)
  @with_mean(5.0)
  def constraints(x):
    return x

  def cost(x):
    return abs(sum(x) - 5.0)

  from numpy import array
  x = array([1,2,3,4,5])
  solver = solver(len(x))
  solver.SetInitialPoints(x)
  solver.SetEvaluationMonitor(evalmon)
  solver.SetGenerationMonitor(stepmon)
  solver.SetConstraints(constraints)
  solver.Solve(cost, disp=False)
  y = solver.Solution()

  assert almostEqual(mean(y), 5.0, tol=1e-15)
  assert almostEqual(spread(y), 5.0, tol=1e-15)
  assert almostEqual(cost(y), 4*(5.0), tol=1e-6)


def test_nested_solver(nested, solver):

  from mystic.monitors import Monitor
  evalmon = Monitor()
  stepmon = Monitor()

  from mystic.math.measures import mean, spread
  @with_spread(5.0)
  @with_mean(5.0)
  def constraints(x):
    return x

  def cost(x):
    return abs(sum(x) - 5.0)

  from numpy import array
  nested = nested(5, 4)
  nested.SetEvaluationMonitor(evalmon)
  nested.SetGenerationMonitor(stepmon)
  nested.SetConstraints(constraints)
  nested.SetNestedSolver(solver)
  nested.Solve(cost, disp=False)
  y = nested.Solution()

  assert almostEqual(mean(y), 5.0, tol=1e-15)
  assert almostEqual(spread(y), 5.0, tol=1e-15)
  assert almostEqual(cost(y), 4*(5.0), tol=1e-6)


def test_inner_solver(nested, solver):

  from mystic.monitors import Monitor
  evalmon = Monitor()
  stepmon = Monitor()

  from mystic.math.measures import mean, spread
  @with_spread(5.0)
  @with_mean(5.0)
  def constraints(x):
    return x

  def cost(x):
    return abs(sum(x) - 5.0)

  from numpy import array
  solver = solver(5)
  lb,ub = [0,0,0,0,0],[100,100,100,100,100]
  solver.SetRandomInitialPoints(lb, ub)
  solver.SetConstraints(constraints)
  solver.SetStrictRanges(lb, ub, tight=True)
  nested = nested(5, 4)
  nested.SetEvaluationMonitor(evalmon)
  nested.SetGenerationMonitor(stepmon)
  nested.SetNestedSolver(solver)
  nested.Solve(cost, disp=False)
  y = nested.Solution()

  assert almostEqual(mean(y), 5.0, tol=1e-15)
  assert almostEqual(spread(y), 5.0, tol=1e-15)
  assert almostEqual(cost(y), 4*(5.0), tol=1e-6)


def test_mapped_solver(nested, solver, map):

  from mystic.monitors import Monitor
  evalmon = Monitor()
  stepmon = Monitor()

  from mystic.math.measures import mean, spread
  @with_spread(5.0)
  @with_mean(5.0)
  def constraints(x):
    return x

  def cost(x):
    return abs(sum(x) - 5.0)

  from numpy import array
  solver = solver(5)
  lb,ub = [0,0,0,0,0],[100,100,100,100,100]
  solver.SetRandomInitialPoints(lb, ub)
  solver.SetConstraints(constraints)
  solver.SetStrictRanges(lb, ub, tight=True)
  nested = nested(5, 4)
  nested.SetEvaluationMonitor(evalmon)
  nested.SetGenerationMonitor(stepmon)
  nested.SetNestedSolver(solver)
  nested.SetMapper(map)
  nested.Solve(cost, disp=False)
  y = nested.Solution()

  assert almostEqual(mean(y), 5.0, tol=1e-15)
  assert almostEqual(spread(y), 5.0, tol=1e-15)
  assert almostEqual(cost(y), 4*(5.0), tol=1e-6)


if __name__ == '__main__':
  solvers = [fmin_powell, fmin, diffev, diffev2]
  # test solver one-liners
  for solver in solvers:
    test_one_liner(solver)

  classes = [NelderMeadSimplexSolver, PowellDirectionalSolver, \
             DifferentialEvolutionSolver, DifferentialEvolutionSolver2]
  # nested solver one-liners (using inner solver class, not nested one-liner)
  for solver in classes:
    nested = BuckshotSolver
    test_nested_solver(nested, solver)

  # solver multi-line
  for solver in classes:
    test_multi_liner(solver)

  # nested solver multi-line
  for solver in classes:
    nested = BuckshotSolver
    test_inner_solver(nested, solver)

  # solver with mapper
 #from pathos.pools import ProcessPool as Pool
  from mystic.pools import SerialPool as Pool
  map = Pool(5).map
  for solver in classes:
    nested = BuckshotSolver
    test_mapped_solver(nested, solver, map)


# EOF
