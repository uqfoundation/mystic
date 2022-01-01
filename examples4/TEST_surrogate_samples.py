#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2009-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

DEBUG = False
PER_AI = True # if True, generate random_samples on each Ai
MCZERO = False # if True, McD[i] == 0 when STATUS[i] = SUCCESS
#######################################################################
# scaling and mpi info; also optimizer configuration parameters
# hard-wired: use DE solver, don't use mpi, F-F' calculation
# (similar to concentration.in)
#######################################################################
from TEST_surrogate_diam import *  # model, limit
from mystic.math.stats import volume, prob_mass, mean, mcdiarmid_bound
from mystic.math.integrate import integrate as expectation_value
from mystic.math.samples import random_samples, sampled_pts, sampled_prob
from mystic.math.samples import alpha, _pof_given_samples as sampled_pof
from mystic.tools import wrap_bounds

def sampled_mean(pts,lb,ub):
  from numpy import inf
  f = wrap_bounds(model,lb,ub)
  ave = 0; count = 0
  for i in range(len(pts[0])):
    Fx = f([pts[0][i],pts[1][i],pts[2][i]])
    if Fx != -inf: # outside of bounds evaluates to -inf
      ave += Fx
      count += 1
  if not count: return None  #XXX: define 0/0 = None
  ave = float(ave) / float(count)
  return ave

def minF(x):
  return scale * model(x)

def maxF(x):
  return -scale * model(x)


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
STATUS = [None]
TOTAL_CUTS = [0]
max_cuts = [9999]
SUCCESS = "S"
FAILURE = "F"
UNDERSAMPLED = "U"

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
  global STATUS #XXX: warning! global variable...
  params, subdiams, diam, probmass = test_cuboids(lb,ub,RVstart,RVend,vol)
  SOLVED_PARAMETERS, SUB_DIAMETERS = params, subdiams
  TOTAL_DIAMETERS, PROBABILITY_MASS = diam, probmass
  if DEBUG: print("\nSTATUS = %s" % STATUS)

  # get region with largest probability mass
  # region = probmass.index(max(probmass))
  # NEW_SLICES = [region,region+1]

  newstatus = []
  newslices = []
  # find interesting regions  (optimizer returns: solved, -Energy, f_evals)
  for i in range(len(lb)):
    if STATUS[i]: # status in [SUCCESS, FAILURE, UNDERSAMPLED]
      newstatus.append(STATUS[i]) # previously determined to skip this region
    elif not optimize(maxF,lb[i],ub[i])[1]:  # if max A = 0, then 'failure'
      newstatus.append(FAILURE) # mark as a failure region; skip
    elif -(optimize(minF,lb[i],ub[i])[1]): # if min A > 0, then 'success'
      newstatus.append(SUCCESS) # mark as a success region; skip
    else:
      newstatus.append(None) # each 'new' slice is a bisection
      newstatus.append(None) # ... thus appends TWO indicatiors
      newslices.append(i)

  ncut = 0
  NEW_SLICES = []
  for i in newslices:
    # get direction with largest subdiameter
    direction = subdiams[i].index(max(subdiams[i]))

    # adjust for ub,lb expanded by n slices
    region = i + ncut
    # get the midpoint
    cutvalue = 0.5 * ( ub[region][direction] + lb[region][direction] )

    # modify bounds to include cut plane
    l = lb[:region+1]
    l += [lb[region][:direction] + [cutvalue] + lb[region][direction+1:]]
    lb = l + lb[region+1:]
    u = ub[:region]
    u += [ub[region][:direction] + [cutvalue] + ub[region][direction+1:]]
    ub = u + ub[region:]

    # bean counting...
    NEW_SLICES.append(region)
    NEW_SLICES.append(region+1)
    ncut += 1
    TOTAL_CUTS[0] += 1
    if TOTAL_CUTS[0] >= max_cuts[0]:
      print("\nmaximum number of cuts performed.")
      max_cuts[0] = 0
      STATUS = newstatus[:i+1+ncut] + STATUS[i+1:] # patially use 'old' status
      return lb,ub

  STATUS = newstatus[:]
  return lb,ub

  
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
  max_cut_iterations = 4   #NOTE: number of resulting subcuboids = cuts + 1
  max_cuts[0] = 1          #      maximum number of cuts
  num_sample_points = 5    #NOTE: number of sample data points

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
  if DEBUG: print("\nSTATUS = %s" % STATUS)

  if not DEBUG:
    pts = random_samples(lb[0],ub[0])
    pof = sampled_pof(model,pts)
    print("Exact PoF: %s" % pof)
    # prepare new set of random samples (across entire domain) as 'data'
    if not PER_AI:
      pts = random_samples(lb[0],ub[0],num_sample_points)

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
   #print(" expectation: %s" % expectation)
    mean_value = mean(expectation,cuboid_volume)
    print(" mean value: %s" % mean_value)
    if STATUS[0] == SUCCESS and MCZERO: #XXX: should be false, or we are done
      mcdiarmid = 0.0
    else:
      mcdiarmid = mcdiarmid_bound(mean_value,sqrt(diam0[0]))
    print("McDiarmid bound: %s" % mcdiarmid)

  # determine 'best' cuts to cuboid
  for cut in range(max_cut_iterations):
    if max_cuts[0]:  #XXX: abort if max_cuts was set to zero
      print("\n..... cut iteration #%s ....." % (cut+1))
      lb,ub = make_cut(lb,ub,RVstart,RVend,cuboid_volume)

  if DEBUG:
    print("\n..... %s cuboids ....." % (cut+2)) #XXX: ?; was max_cut_iterations+1
  # get diameter for each subcuboid 
  params, subdiams, diam, probmass = test_cuboids(lb,ub,RVstart,RVend,\
                                                  cuboid_volume)
  SOLVED_PARAMETERS, SUB_DIAMETERS = params, subdiams
  TOTAL_DIAMETERS, PROBABILITY_MASS = diam, probmass
  print("\nSTATUS = %s" % STATUS)
  if not DEBUG:
    weighted_bound = []
    sampled_bound = []
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
     #print(" expectation: %s" % expect_value)
      sub_mean_value = mean(expect_value,subcuboid_volume)
      print(" mean value: %s" % sub_mean_value)
      if STATUS[i] == SUCCESS and MCZERO:
        sub_mcdiarmid = 0.0
      else:
        sub_mcdiarmid = mcdiarmid_bound(sub_mean_value,sqrt(diam[i]))
      print("McDiarmid bound: %s" % sub_mcdiarmid)
      weighted_bound.append(probmass[i] * sub_mcdiarmid)
      print("weighted McDiarmid: %s" % weighted_bound[-1])

      # prepare new set of random samples (on each subcuboid) as 'data'
      if PER_AI:
        pts = random_samples(lb[i],ub[i],num_sample_points)
      npts_i = sampled_pts(pts,lb[i],ub[i])
      print("Number of sample points: %s" % npts_i)
      if not npts_i:
        print("Warning, no sample points in bounded region")
        alpha_i = 0.0 #FIXME: defining undefined alpha to be 0.0
      else:
       #alpha_i = alpha(npts_i,sub_mcdiarmid)  #XXX: oops... was wrong
        alpha_i = alpha(npts_i,sqrt(diam[i]))
      print("alpha: %s" % alpha_i)

      s_prob = sampled_prob(pts,lb[i],ub[i])
      print("Sampled probability mass: %s" % s_prob)
      s_mean = sampled_mean(pts,lb[i],ub[i])
      if s_mean == None:
         s_mean = 0.0 #FIXME: defining undefined means to be 0.0
      print("Sampled mean value: %s" % s_mean)
      if STATUS[i] == SUCCESS and MCZERO:
        samp_mcdiarmid = 0.0
      else:
        samp_mcdiarmid = mcdiarmid_bound((s_mean-alpha_i),sqrt(diam[i]))
      print("Sampled McDiarmid bound: %s" % samp_mcdiarmid)
      if PER_AI: #XXX: 'cheat' by using probmass for uniform Ai
        sampled_bound.append(probmass[i] * samp_mcdiarmid)
      else:
        sampled_bound.append(s_prob * samp_mcdiarmid)
      print("weighted sampled McDiarmid: %s" % sampled_bound[-1])

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
   #if not PER_AI:
    sampled = sum(sampled_bound)
    print("weighted sampled McDiarmid: %s" % sampled)

  print("\n..............")
  print(" sum probability mass: %s" % sum(probmass))


# EOF
