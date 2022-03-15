#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2020-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
'''
calculate upper bound on mean value of 0th input

Test function is y = F(x), where:
  y0 = x0 + x1 * | x2 * x3**2 - (x4 / x1)**2 |**.5
  y1 = x0 - x1 * | x2 * x3**2 + (x4 / x1)**2 |**.5
  y2 = x0 - | x1 * x2 * x3 - x4 |

model = lambda x: F(x)[0]

Calculate upper bound on mean(x[0]), where:
  x in [(0,1), (1,10), (0,10), (0,10), (0,10)]
  wx in [(0,1), (1,1), (1,1), (1,1), (1,1)]
  npts = [2, 1, 1, 1, 1] (i.e. two Dirac masses on x[0], one elsewhere)
  sum(wx[i]_j) for j in [0,npts], for each i
  E|model(x)| = 11.0 +/- 1.0

Solves for two scenarios of x that produce upper bound on mean(x[0]),
given the bounds, normalization, and moment constraints.
'''
debug = False
MINMAX = -1  ## NOTE: sup = maximize = -1; inf = minimize = 1
#######################################################################
# scaling and mpi info; also optimizer configuration parameters
# hard-wired: use DE solver, don't use mpi, F-F' calculation
#######################################################################
npop = 80
maxiter = 1500
maxfun = 1e+6
convergence_tol = 1e-10; ngen = 100
crossover = 0.9
percent_change = 0.9


#######################################################################
# the model function
#######################################################################
from toys import function5 as model


#######################################################################
# the differential evolution optimizer
#######################################################################
def optimize(cost,_bounds,_constraints):
  from mystic.solvers import DifferentialEvolutionSolver2
  from mystic.termination import ChangeOverGeneration as COG
  from mystic.strategy import Best1Exp
  from mystic.monitors import VerboseMonitor, Monitor
  from mystic.tools import random_seed

 #random_seed(123)

  stepmon = VerboseMonitor(2)
 #stepmon = Monitor()
  evalmon = Monitor()

  lb,ub = _bounds
  ndim = len(lb)

  solver = DifferentialEvolutionSolver2(ndim,npop)
  solver.SetRandomInitialPoints(min=lb,max=ub)
  solver.SetStrictRanges(min=lb,max=ub)
  solver.SetEvaluationLimits(maxiter,maxfun)
  solver.SetEvaluationMonitor(evalmon)
  solver.SetGenerationMonitor(stepmon)
  solver.SetConstraints(_constraints)

  tol = convergence_tol
  solver.Solve(cost,termination=COG(tol,ngen),strategy=Best1Exp, \
               CrossProbability=crossover,ScalingFactor=percent_change)

  solved = solver.bestSolution
 #print("solved: %s" % solver.Solution())
  func_max = MINMAX * solver.bestEnergy       #NOTE: -solution assumes -Max
 #func_max = 1.0 + MINMAX*solver.bestEnergy   #NOTE: 1-sol => 1-success = fail
  func_evals = solver.evaluations
  from mystic.munge import write_support_file
  write_support_file(stepmon, npts=npts)
  return solved, func_max, func_evals


#######################################################################
# maximize the function
#######################################################################
def maximize(params,npts,bounds):

  from mystic.math.measures import split_param
  from mystic.math.discrete import product_measure
  from mystic.math import almostEqual
  from numpy import inf
  atol = 1e-18 # default is 1e-18
  rtol = 1e-7  # default is 1e-7
  target,error = params
  lb,ub = bounds

  # split lower & upper bounds into weight-only & sample-only
  w_lb, x_lb = split_param(lb, npts)
  w_ub, x_ub = split_param(ub, npts)

  # NOTE: rv, lb, ub are of the form:
  #    rv = [wxi]*nx + [xi]*nx + [wyi]*ny + [yi]*ny + [wzi]*nz + [zi]*nz

  # generate primary constraints function
  def constraints(rv):
    c = product_measure().load(rv, npts)
    # NOTE: bounds wi in [0,1] enforced by filtering
    # impose norm on each discrete measure
    for measure in c:
      if not almostEqual(float(measure.mass), 1.0, tol=atol, rel=rtol):
        measure.normalize()
    # impose expectation on product measure
    ##################### begin function-specific #####################
    E = float(c.expect(model))
    if not (E <= float(target[0] + error[0])) \
    or not (float(target[0] - error[0]) <= E):
      c.set_expect(target[0], model, (x_lb,x_ub), tol=error[0])
    ###################### end function-specific ######################
    # extract weights and positions
    return c.flatten()

  # generate maximizing function
  def cost(rv):
    c = product_measure().load(rv, npts)
    E = float(c.expect(model))
    if E > (target[0] + error[0]) or E < (target[0] - error[0]):
      if debug: print("skipping expect: %s" % E)
      return inf  #XXX: FORCE TO SATISFY E CONSTRAINTS
    return MINMAX * c[0].mean

  # maximize
  solved, func_max, func_evals = optimize(cost,(lb,ub),constraints)

  if MINMAX == 1:
    print("func_minimum: %s" % func_max)  # inf
  else:
    print("func_maximum: %s" % func_max)  # sup
  print("func_evals: %s" % func_evals)

  return solved, func_max


#######################################################################
# rank, bounds, and restart information 
#######################################################################
if __name__ == '__main__':
  function_name = model.__name__

  o_mean = 11.0   #NOTE: SET THE 'mean' HERE!
  o_range = 1.0   #NOTE: SET THE 'range' HERE!
  na = 2  #NOTE: SET THE NUMBER OF 'a' POINTS HERE!
  nb = 1  #NOTE: SET THE NUMBER OF 'b' POINTS HERE!
  nc = 1  #NOTE: SET THE NUMBER OF 'c' POINTS HERE!
  nd = 1  #NOTE: SET THE NUMBER OF 'd' POINTS HERE!
  ne = 1  #NOTE: SET THE NUMBER OF 'e' POINTS HERE!
  target = (o_mean,)
  error = (o_range,)

  w_lower = [0.0]
  w_upper = [1.0]
  a_lower = [0.0]; b_lower = [1.0];  c_lower = d_lower = e_lower = [0.0]
  a_upper = [1.0]; b_upper = [10.0]; c_upper = d_upper = e_upper = [10.0]

  lower_bounds = (na * w_lower) + (na * a_lower) \
               + (nb * w_lower) + (nb * b_lower) \
               + (nc * w_lower) + (nc * c_lower) \
               + (nd * w_lower) + (nd * d_lower) \
               + (ne * w_lower) + (ne * e_lower)
  upper_bounds = (na * w_upper) + (na * a_upper) \
               + (nb * w_upper) + (nb * b_upper) \
               + (nc * w_upper) + (nc * c_upper) \
               + (nd * w_upper) + (nd * d_upper) \
               + (ne * w_upper) + (ne * e_upper)

  print("...SETTINGS...")
  print("npop = %s" % npop)
  print("maxiter = %s" % maxiter)
  print("maxfun = %s" % maxfun)
  print("convergence_tol = %s" % convergence_tol)
  print("crossover = %s" % crossover)
  print("percent_change = %s" % percent_change)
  print("..............\n")

  print(" model: f(x) = %s(x)" % function_name)
  print(" target: %s" % str(target))
  print(" error: %s" % str(error))
  print(" npts: %s" % str((na,nb,nc,nd,ne)))
  print("..............\n")

  param_string = "["
  for i in range(na):
    param_string += "'wa%s', " % str(i+1)
  for i in range(na):
    param_string += "'a%s', " % str(i+1)
  for i in range(nb):
    param_string += "'wb%s', " % str(i+1)
  for i in range(nb):
    param_string += "'b%s', " % str(i+1)
  for i in range(nc):
    param_string += "'wc%s', " % str(i+1)
  for i in range(nc):
    param_string += "'c%s', " % str(i+1)
  for i in range(nd):
    param_string += "'wd%s', " % str(i+1)
  for i in range(nd):
    param_string += "'d%s', " % str(i+1)
  for i in range(ne):
    param_string += "'we%s', " % str(i+1)
  for i in range(ne):
    param_string += "'e%s', " % str(i+1)
  param_string = param_string[:-2] + "]"

  print(" parameters: %s" % param_string)
  print(" lower bounds: %s" % lower_bounds)
  print(" upper bounds: %s" % upper_bounds)
# print(" ...")
  pars = (target,error)
  npts = (na,nb,nc,nd,ne)
  bounds = (lower_bounds,upper_bounds)
  solved, diameter = maximize(pars,npts,bounds)

  from numpy import array
  from mystic.math.discrete import product_measure
  c = product_measure().load(solved,npts)
  print("solved: [wa,a]\n%s" % array(list(zip(c[0].weights,c[0].positions))))
  print("solved: [wb,b]\n%s" % array(list(zip(c[1].weights,c[1].positions))))
  print("solved: [wc,c]\n%s" % array(list(zip(c[2].weights,c[2].positions))))
  print("solved: [wd,d]\n%s" % array(list(zip(c[3].weights,c[3].positions))))
  print("solved: [we,e]\n%s" % array(list(zip(c[4].weights,c[4].positions))))

  print("expect: %s" % str( c.expect(model) ))

# EOF
