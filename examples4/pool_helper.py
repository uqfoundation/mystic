#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2009-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

# MPI config
#nnodes = 4   # = npop
nnodes = '4:core4:ppn=1'   # = npop

# QUEUE config
queue = 'weekdayQ'
timelimit = '00:10'


def func_pickle(func, suffix='.pik', dir='.'):
    """ standard pickle.dump of function to a NamedTemporaryFile """
    from dill.temp import dump
    return dump(func, suffix=suffix, dir=dir)


#######################################################################
# the steepest descent optimizer
#######################################################################
def local_optimize(cost,x0,lb,ub):
  from mystic.solvers import PowellDirectionalSolver
  from mystic.termination import NormalizedChangeOverGeneration as NCOG
  from mystic.monitors import VerboseMonitor, Monitor

  maxiter = 1000
  maxfun = 1e+6
  convergence_tol = 1e-4

 #def func_unpickle(filename):
 #  """ standard pickle.load of function from a File """
 #  import dill as pickle
 #  return pickle.load(open(filename,'r'))

 #stepmon = VerboseMonitor(100)
  stepmon = Monitor()
  evalmon = Monitor()

  ndim = len(lb)

  solver = PowellDirectionalSolver(ndim)
  solver.SetInitialPoints(x0)
  solver.SetStrictRanges(min=lb,max=ub)
  solver.SetEvaluationLimits(maxiter,maxfun)
  solver.SetEvaluationMonitor(evalmon)
  solver.SetGenerationMonitor(stepmon)

  tol = convergence_tol
 #cost = func_unpickle(cost)  #XXX: regenerate cost function from file
  solver.Solve(cost, termination=NCOG(tol))

  solved_params = solver.bestSolution
  solved_energy = solver.bestEnergy
  func_evals = solver.evaluations
  return solved_params, solved_energy, func_evals


# EOF
