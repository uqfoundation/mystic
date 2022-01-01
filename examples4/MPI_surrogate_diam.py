#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2009-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

#######################################################################
# scaling and mpi info; also optimizer configuration parameters
# hard-wired: use DE solver, use mpi, F-F' calculation
#######################################################################
#scale = 1.0

npop = 20
maxiter = 1000
maxfun = 1e+6
convergence_tol = 1e-4
crossover = 0.9
percent_change = 0.9


#######################################################################
# the model function
#######################################################################
#from surrogate import marc_surr as model
from surrogate import ballistic_limit as limit


#######################################################################
# the subdiameter calculation
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
    from surrogate import marc_surr as model

    # prepare x and xprime
    params = rv[:-1]                         #XXX: assumes Xi' is at rv[-1]
    params_prime = rv[:i]+rv[-1:]+rv[i+1:-1] #XXX: assumes Xi' is at rv[-1]

    # get the F(x) response
    Fx = model(params)

    # get the F(x') response
    Fxp = model(params_prime)

    # compute diameter
    scale = 1.0
    return -scale * (Fx - Fxp)**2

  return cost


#######################################################################
# the differential evolution optimizer
#######################################################################
def optimize(cost,lb,ub):
  from pyina.launchers import Mpi as Pool
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
  solver.SetMapper(Pool().map)

  tol = convergence_tol
  solver.Solve(cost,termination=CRT(tol,tol),strategy=Best1Exp, \
               CrossProbability=crossover,ScalingFactor=percent_change)

  print("solved: %s" % solver.bestSolution)
  scale = 1.0
  diameter_squared = -solver.bestEnergy / scale  #XXX: scale != 0
  func_evals = solver.evaluations
  return diameter_squared, func_evals


#######################################################################
# loop over model parameters to calculate concentration of measure
#######################################################################
def UQ(start,end,lower,upper):
  diameters = []
  function_evaluations = []
  total_func_evals = 0
  total_diameter = 0.0

  for i in range(start,end+1):
    lb = lower + [lower[i]]
    ub = upper + [upper[i]]
  
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
# rank, bounds, and restart information 
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
  RVmax = len(lower_bounds) - 1

  # when not a random variable, set the value to the lower bound
  for i in range(0,RVstart):
    upper_bounds[i] = lower_bounds[i]
  for i in range(RVend+1,RVmax+1):
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
  for i in range(RVmax+1): 
    param_string += "'x%s'" % str(i+1)
    if i == (RVmax):
      param_string += "]"
    else:
      param_string += ", "

  print(" parameters: %s" % param_string)
  print("  varying 'xi', with i = %s" % list(range(RVstart+1,RVend+2)))
  print(" lower bounds: %s" % lower_bounds)
  print(" upper bounds: %s" % upper_bounds)
# print(" ...")
  diameter = UQ(RVstart,RVend,lower_bounds,upper_bounds)

# EOF
