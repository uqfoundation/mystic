#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2009-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

DEBUG = False
#######################################################################
# scaling and mpi info; also optimizer configuration parameters
# hard-wired: use DE solver, don't use mpi, F-F' calculation
# (similar to concentration.in)
#######################################################################
from TEST_surrogate_diam import *  # model, limit
from mystic.math.stats import volume, prob_mass, mean, mcdiarmid_bound
from mystic.math.integrate import integrate as expectation_value
from mystic.math.samples import sample


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

  solved = solver.bestSolution
 #if DEBUG: print("solved: %s" % solved)
  diameter_squared = -solver.bestEnergy / scale  #XXX: scale != 0
  func_evals = solver.evaluations
  return solved, diameter_squared, func_evals


#######################################################################
# loop over model parameters to calculate concentration of measure
# (similar to main.cc)
#######################################################################
def UQ(start,end,lower,upper):
  params = []
  diameters = []
  function_evaluations = []
  total_func_evals = 0
  total_diameter = 0.0

  for i in range(start,end+1):
    lb = lower + [lower[i]]
    ub = upper + [upper[i]]
  
    # construct cost function and run optimizer
    cost = costFactory(i)
    # optimize, using no initial conditions
    solved, subdiameter, func_evals = optimize(cost,lb,ub)

    function_evaluations.append(func_evals)
    diameters.append(subdiameter)
    params.append(solved)

    total_func_evals += function_evaluations[-1]
    total_diameter += diameters[-1]

  if DEBUG:
    for solved in params:
      print("solved: %s" % solved)
    print("subdiameters (squared): %s" % diameters)
    print("diameter (squared): %s" % total_diameter)
    print("func_evals: %s => %s" % (function_evaluations, total_func_evals))

  return params, total_diameter, diameters


#######################################################################
# get solved_params, subdiameters, and prob_mass for a sliced cuboid
#######################################################################
PROBABILITY_MASS = []
SUB_DIAMETERS = []
TOTAL_DIAMETERS = []
SOLVED_PARAMETERS = []
NEW_SLICES = []

def test_cuboids(lb,ub,RVstart,RVend,cuboid_volume):

  probmass = []
  subdiams = []
  tot_diam = []
  solved_p = []

# subdivisions
  for i in range(len(lb)):
    if DEBUG:
      print("\n")
      print(" lower bounds: %s" % lb[i])
      print(" upper bounds: %s" % ub[i])
    if i in NEW_SLICES or not NEW_SLICES:
      subcuboid_volume = volume(lb[i],ub[i])
      sub_prob_mass = prob_mass(subcuboid_volume,cuboid_volume)
      probmass.append(sub_prob_mass)
      if DEBUG: print(" probability mass: %s" % sub_prob_mass)
      solved, diameter, subdiameters = UQ(RVstart,RVend,lb[i],ub[i])

      solved_p.append(solved)
      subdiams.append(subdiameters)
      tot_diam.append(diameter)
    else:
      probmass.append(PROBABILITY_MASS[i])
      if DEBUG: print(" probability mass: %s" % PROBABILITY_MASS[i])
      solved_p.append(SOLVED_PARAMETERS[i])
      subdiams.append(SUB_DIAMETERS[i])
      tot_diam.append(TOTAL_DIAMETERS[i])

  return solved_p, subdiams, tot_diam, probmass


#######################################################################
# slice the cuboid
#######################################################################
def make_cut(lb,ub,RVStart,RVend,vol):
  params, subdiams, diam, probmass = test_cuboids(lb,ub,RVstart,RVend,vol)
  SOLVED_PARAMETERS, SUB_DIAMETERS = params, subdiams
  TOTAL_DIAMETERS, PROBABILITY_MASS = diam, probmass

  #####XXX: probably needs to be some sort of loop ######
  # get region with largest probability mass
  region = probmass.index(max(probmass))
  NEW_SLICES = [region,region+1]

  # get direction with largest subdiameter
  direction = subdiams[region].index(max(subdiams[region]))

  #XXX: should check 'params' for 'no more improvement possilble'
  #XXX: ... if so, check all the directions in that region
  #XXX: ... if still so, then select region with next largest probmass
  #####XXX: probably needs to be some sort of loop ######

  # get the midpoint
  cutvalue = 0.5 * ( ub[region][direction] + lb[region][direction] )

  # modify bounds to include cut plane
  l = lb[:region+1]
  l += [lb[region][:direction] + [cutvalue] + lb[region][direction+1:]]
  l += lb[region+1:]
  u = ub[:region]
  u += [ub[region][:direction] + [cutvalue] + ub[region][direction+1:]]
  u += ub[region:]

  return l,u

  
#######################################################################
# rank, bounds, and restart information 
# (similar to concentration.variables)
#######################################################################
if __name__ == '__main__':
  from math import sqrt

  function_name = "marc_surr"

  lower_bounds = [60.0, 0.0, 2.1]
  upper_bounds = [105.0, 30.0, 2.8]
  RVstart = 0; RVend = 2
  max_number_of_cuts = 255  #NOTE: number of resulting subcuboids = cuts + 1

  print("...SETTINGS...")
  print("npop = %s" % npop)
  print("maxiter = %s" % maxiter)
  print("maxfun = %s" % maxfun)
  print("convergence_tol = %s" % convergence_tol)
  print("crossover = %s" % crossover)
  print("percent_change = %s" % percent_change)
  print("..............\n\n")

  print(" model: f(x) = %s(x)" % function_name)
  RVmax = len(lower_bounds)
  param_string = "["
  for i in range(RVmax): 
    param_string += "'x%s'" % str(i+1)
    if i == (RVmax - 1):
      param_string += "]"
    else:
      param_string += ", "

  print(" parameters: %s" % param_string)

  # get diameter for entire cuboid 
  lb,ub = [lower_bounds],[upper_bounds]
  cuboid_volume = volume(lb[0],ub[0])
  params0, subdiams0, diam0, probmass0 = test_cuboids(lb,ub,RVstart,RVend,\
                                                      cuboid_volume)
  SOLVED_PARAMETERS, SUB_DIAMETERS = params0, subdiams0
  TOTAL_DIAMETERS, PROBABILITY_MASS = diam0, probmass0

  if not DEBUG:
    failure,success = sample(model,lb[0],ub[0])
    pof = float(failure) / float(failure + success)
    print("Exact PoF: %s" % pof)

    for i in range(len(lb)):
      print("\n")
      print(" lower bounds: %s" % lb[i])
      print(" upper bounds: %s" % ub[i])
    for solved in params0[0]:
      print("solved: %s" % solved)
    print("subdiameters (squared): %s" % subdiams0[0])
    print("diameter (squared): %s" % diam0[0])
    print(" probability mass: %s" % probmass0[0])
    expectation = expectation_value(model,lower_bounds,upper_bounds)
    print(" expectation: %s" % expectation)
    mean_value = mean(expectation,cuboid_volume)
    print(" mean value: %s" % mean_value)
    mcdiarmid = mcdiarmid_bound(mean_value,sqrt(diam0[0]))
    print("McDiarmid bound: %s" % mcdiarmid)

  # determine 'best' cuts to cuboid
  for cut in range(max_number_of_cuts):
    print("\n..... cut #%s ....." % (cut+1))
    lb,ub = make_cut(lb,ub,RVstart,RVend,cuboid_volume)

  if DEBUG:
    print("\n..... cut #%s ....." % (max_number_of_cuts+1))
  # get diameter for each subcuboid 
  params, subdiams, diam, probmass = test_cuboids(lb,ub,RVstart,RVend,\
                                                  cuboid_volume)
  SOLVED_PARAMETERS, SUB_DIAMETERS = params, subdiams
  TOTAL_DIAMETERS, PROBABILITY_MASS = diam, probmass
  if not DEBUG:
    weighted_bound = []
    for i in range(len(lb)):
      print("\n")
      print(" lower bounds: %s" % lb[i])
      print(" upper bounds: %s" % ub[i])
      for solved in params[i]:
        print("solved: %s" % solved)
      print("subdiameters (squared): %s" % subdiams[i])
      print("diameter (squared): %s" % diam[i])
      print(" probability mass: %s" % probmass[i])
      #calculate remainder of the statistics, McDiarmid for cube & subcuboids
      subcuboid_volume = volume(lb[i],ub[i])
      expect_value = expectation_value(model,lb[i],ub[i])
      print(" expectation: %s" % expect_value)
      sub_mean_value = mean(expect_value,subcuboid_volume)
      print(" mean value: %s" % sub_mean_value)
      sub_mcdiarmid = mcdiarmid_bound(sub_mean_value,sqrt(diam[i]))
      print("McDiarmid bound: %s" % sub_mcdiarmid)
      weighted_bound.append(probmass[i] * sub_mcdiarmid)

    # compare weighted to McDiarmid
    print("\n\n..............")
    p_mcdiarmid = probmass0[0] * mcdiarmid
    print("McDiarmid: %s" % p_mcdiarmid)
    weighted = sum(weighted_bound)
    print("weighted McDiarmid: %s" % weighted)
    try:
      print("relative change: %s" % (weighted / p_mcdiarmid))
    except ZeroDivisionError:
      pass

  print("\n..............")
  print(" sum probability mass: %s" % sum(probmass))


# EOF
