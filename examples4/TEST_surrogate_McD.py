#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2009-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

#######################################################################
# scaling and mpi info; also optimizer configuration parameters
# hard-wired: use DE solver, don't use mpi, F-F' calculation
# (similar to concentration.in)
#######################################################################
scale = 1.0
#XXX: <mpi config goes here>

npop = 20
maxiter = 1000
maxfun = 1e+6
convergence_tol = 1e-4
crossover = 0.9
percent_change = 0.9


#######################################################################
# the model function
# (similar to Simulation.cpp)
#######################################################################
from surrogate import marc_surr as model


#######################################################################
# the subdiameter calculation
# (similar to driver.sh)
#######################################################################
def costFactory(i):
  """a cost factory for the cost function"""

  def cost(rv):
    """compute the diameter as a calculation of cost

  Input:
    - rv -- 1-d array of model parameters

  Output:
    - diameter -- scale * | F(x) - F(x')|**2
    """

    # prepare x and xprime
    params = rv[:-1]                         #XXX: assumes Xi' is at rv[-1]
    params_prime = rv[:i]+rv[-1:]+rv[i+1:-1] #XXX: assumes Xi' is at rv[-1]

    # get the F(x) response
    Fx = model(params)

    # get the F(x') response
    Fxp = model(params_prime)

    # compute diameter
    return -scale * (Fx - Fxp)**2

  return cost


#######################################################################
# the differential evolution optimizer
# (replaces the call to dakota)
#######################################################################
def optimize(cost,lb,ub):
  from mystic.solvers import DifferentialEvolutionSolver2
  from mystic.termination import CandidateRelativeTolerance as CRT
  from mystic.strategy import Best1Exp
  from mystic.monitors import VerboseMonitor, Monitor
  from mystic.tools import random_seed

  random_seed(123)

 #stepmon = VerboseMonitor(100)
  stepmon = Monitor()
  evalmon = Monitor()

  ndim = len(lb) # [(1 + RVend) - RVstart] + 1

  solver = DifferentialEvolutionSolver2(ndim,npop)
  solver.SetRandomInitialPoints(min=lb,max=ub)
  solver.SetStrictRanges(min=lb,max=ub)
  solver.SetEvaluationLimits(maxiter,maxfun)
  solver.SetEvaluationMonitor(evalmon)
  solver.SetGenerationMonitor(stepmon)

  tol = convergence_tol
  solver.Solve(cost,termination=CRT(tol,tol),strategy=Best1Exp, \
               CrossProbability=crossover,ScalingFactor=percent_change)

  print("solved: %s" % solver.bestSolution)
  diameter_squared = -solver.bestEnergy / scale  #XXX: scale != 0
  func_evals = solver.evaluations
  return diameter_squared, func_evals


#######################################################################
# loop over model parameters to calculate concentration of measure
# (similar to main.cc)
#######################################################################
def UQ(start,end,lower,upper):
  diameters = []
  function_evaluations = []
  total_func_evals = 0
  total_diameter = 0.0

  for i in range(start,end+1):
    lb = lower[start:end+1] + [lower[i]]
    ub = upper[start:end+1] + [upper[i]]
  
    #construct cost function and run optimizer
    cost = costFactory(i)
    subdiameter, func_evals = optimize(cost,lb,ub) #XXX: no initial conditions

    function_evaluations.append(func_evals)
    diameters.append(subdiameter)

    total_func_evals += function_evaluations[-1]
    total_diameter += diameters[-1]

  print("subdiameters (squared): %s" % diameters)
  print("diameter (squared): %s" % total_diameter)
  print("func_evals: %s => %s" % (function_evaluations, total_func_evals))

  return total_diameter


#######################################################################
# statistics
# probability mass, expectation, mean, diameter, McDiarmid
#######################################################################
from mystic.math.stats import volume, prob_mass, mean, mcdiarmid_bound
from mystic.math.integrate import integrate as expectation_value


#######################################################################
# rank, bounds, and restart information 
# (similar to concentration.variables)
#######################################################################
if __name__ == '__main__':
  from math import sqrt

  function_name = "marc_surr"
  lower_bounds = [60.0, 0.0, 2.1]
  upper_bounds = [105.0, 30.0, 2.8]
# h = thickness = [60,105]
# a = obliquity = [0,30]
# v = speed = [2.1,2.8]

  RVstart = 0; RVend = 2
  RVmax = len(lower_bounds)

  # when not a random variable, set the value to the lower bound
  for i in range(0,RVstart):
    upper_bounds[i] = lower_bounds[i]
  for i in range(RVend+1,RVmax):
    upper_bounds[i] = lower_bounds[i]

  lbounds = lower_bounds[RVstart:1+RVend]
  ubounds = upper_bounds[RVstart:1+RVend]

  print("...SETTINGS...")
  print("npop = %s" % npop)
  print("maxiter = %s" % maxiter)
  print("maxfun = %s" % maxfun)
  print("convergence_tol = %s" % convergence_tol)
  print("crossover = %s" % crossover)
  print("percent_change = %s" % percent_change)
  print("..............\n\n")

  print(" model: f(x) = %s(x)" % function_name)
  param_string = "["
  for i in range(RVmax): 
    param_string += "'x%s'" % str(i+1)
    if i == (RVmax - 1):
      param_string += "]"
    else:
      param_string += ", "

  print(" parameters: %s" % param_string)
  print(" lower bounds: %s" % lower_bounds)
  print(" upper bounds: %s" % upper_bounds)
# print(" ...")
 #cuboid_volume = volume(lower_bounds,upper_bounds)
  cuboid_volume = volume(lbounds,ubounds)
  probability_mass = prob_mass(cuboid_volume,cuboid_volume)
  print(" probability mass: %s" % probability_mass)
  expectation = expectation_value(model,lower_bounds,upper_bounds)
  print(" expectation: %s" % expectation)
  mean_value = mean(expectation,cuboid_volume)
  print(" mean value: %s" % mean_value)
  diameter = UQ(RVstart,RVend,lower_bounds,upper_bounds)
  mcdiarmid = mcdiarmid_bound(mean_value,sqrt(diameter))
  print("McDiarmid bound: %s" % mcdiarmid)

 #if RVstart != 0 or RVend != 2: # abort when not a 3-D problem
 #  break #FIXME: break? or exit? or pass/else? or ???

  weighted_bound = []
  sanity = []
  lb = []
  ub = []
# subdivisions at h=100, a=20, v=2.2
  lb.append([60.0, 0.0, 2.1])
  ub.append([100.0, 20.0, 2.2])

  lb.append([60.0, 0.0, 2.2])
  ub.append([100.0, 20.0, 2.8])

  lb.append([60.0, 20.0, 2.1])
  ub.append([100.0, 30.0, 2.2])

  lb.append([60.0, 20.0, 2.2])
  ub.append([100.0, 30.0, 2.8])

  lb.append([100.0, 0.0, 2.1])
  ub.append([105.0, 20.0, 2.2])

  lb.append([100.0, 0.0, 2.2])
  ub.append([105.0, 20.0, 2.8])

  lb.append([100.0, 20.0, 2.1])
  ub.append([105.0, 30.0, 2.2])

  lb.append([100.0, 20.0, 2.2])
  ub.append([105.0, 30.0, 2.8])

  for i in range(len(lb)):
    print("\n")
    print(" lower bounds: %s" % lb[i])
    print(" upper bounds: %s" % ub[i])
#   print(" ...")
    subcuboid_volume = volume(lb[i],ub[i])
    sub_prob_mass = prob_mass(subcuboid_volume,cuboid_volume)
    sanity.append(sub_prob_mass)
    print(" probability mass: %s" % sub_prob_mass)
    expect_value = expectation_value(model,lb[i],ub[i])
    print(" expectation: %s" % expect_value)
    sub_mean_value = mean(expect_value,subcuboid_volume)
    print(" mean value: %s" % sub_mean_value)
    sub_diameter = UQ(RVstart,RVend,lb[i],ub[i])
    sub_mcdiarmid = mcdiarmid_bound(sub_mean_value,sqrt(sub_diameter))
    print("McDiarmid bound: %s" % sub_mcdiarmid)
    weighted_bound.append(sub_prob_mass * sub_mcdiarmid)

  # compare weighted to McDiarmid
  print("\n\n..............")
  p_mcdiarmid = probability_mass * mcdiarmid
  print("McDiarmid: %s" % p_mcdiarmid)
  weighted = sum(weighted_bound)
  print("weighted McDiarmid: %s" % weighted)
  try:
    print("relative change: %s" % (weighted / p_mcdiarmid))
  except ZeroDivisionError:
    pass
  assert sum(sanity) == probability_mass
  
