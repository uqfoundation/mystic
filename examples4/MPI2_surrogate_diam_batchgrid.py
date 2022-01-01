#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2009-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

#######################################################################
# scaling and mpi info; also optimizer configuration parameters
# hard-wired: use fmin solver
#######################################################################
#scale = 1.0

#npop = 20
nbins = [2,2,2]
#maxiter = 1000
#maxfun = 1e+6
#convergence_tol = 1e-4


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
    rv = list(rv)
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
# make a pseudo-global optimizer from a steepest descent optimizer
#######################################################################
def optimize(cost,lower,upper,nbins):
  from mystic.tools import random_seed
  from pyina.launchers import Mpi as Pool
  random_seed(123)

  # generate arrays of points defining a grid in parameter space
  grid_dimensions = len(lower)
  bins = []
  for i in range(grid_dimensions):
    step = abs(upper[i] - lower[i])/nbins[i]
    bins.append( [lower[i] + (j+0.5)*step for j in range(nbins[i])] )

  # build a grid of starting points
  from pool_helper import local_optimize
  from mystic.math.grid import gridpts
  initial_values = gridpts(bins)

  # run optimizer for each grid point
  lb = [lower for i in range(len(initial_values))]
  ub = [upper for i in range(len(initial_values))]
  cf = [cost for i in range(len(initial_values))]
  nnodes = len(lb)

  # map:: params, energy, func_evals = local_optimize(cost,x0,lb,ub)
  results = Pool(nnodes).map(local_optimize, cf, initial_values, lb, ub)
  #print("results = %s" % results)

  # get the results with the lowest energy
  best = list(results[0][0]), results[0][1]
  func_evals = results[0][2]
  for result in results[1:]:
    func_evals += result[2] # add function evaluations
    if result[1] < best[1]: # compare energy
      best = list(result[0]), result[1]

  # return best
  print("solved: %s" % best[0])
  scale = 1.0
  diameter_squared = -best[1] / scale  #XXX: scale != 0
  return diameter_squared, func_evals


#######################################################################
# loop over model parameters to calculate concentration of measure
#######################################################################
def UQ(start,end,lower,upper):
 #from pyina.launchers import Mpi as Pool
  from pathos.pools import ProcessPool as Pool
 #from pool_helper import func_pickle  # if fails to pickle, try using a helper

  # run optimizer for each subdiameter 
  lb = [lower + [lower[i]] for i in range(start,end+1)]
  ub = [upper + [upper[i]] for i in range(start,end+1)]
  nb = [nbins[:]           for i in range(start,end+1)]
  for i in range(len(nb)): nb[i][-1] = nb[i][i]
  cf = [costFactory(i)     for i in range(start,end+1)]
 #cf = [func_pickle(i)     for i in cf]
 #cf = [cost.name          for cost in cf]
  nnodes = len(lb)

  #construct cost function and run optimizer
  results = Pool(nnodes).map(optimize, cf,lb,ub,nb)
  #print("results = %s" % results)

  results = list(zip(*results))
  diameters = list(results[0])
  function_evaluations = list(results[1])

  total_func_evals = sum(function_evaluations)
  total_diameter = sum(diameters)

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

  #FIXME: these are *copies*... actual values contained in 'local_optimize'
  maxiter = 1000
  maxfun = 1e+6
  convergence_tol = 1e-4

  print("...SETTINGS...")
  print("nbins = %s" % nbins)
  print("maxiter = %s" % maxiter)
  print("maxfun = %s" % maxfun)
  print("convergence_tol = %s" % convergence_tol)
  #print("crossover = %s" % crossover)
  #print("percent_change = %s" % percent_change)
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
  nbins.append(None) #XXX: kind of hackish
  diameter = UQ(RVstart,RVend,lower_bounds,upper_bounds)

  from pathos.helpers import shutdown
  shutdown()

# EOF
