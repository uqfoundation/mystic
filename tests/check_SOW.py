#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

from mystic.munge import write_support_file, write_converge_file, write_raw_file
## FIXME: 'converge' and 'raw' files are virtually unused and unsupported

def test0(monitor):
  from numpy import array
  x1 = array([1,2,3])
  x2 = array([3,4,5])
  x3 = array([5,6,7])
  x4 = array([7,8,9])

  monitor(x1,-1)
  monitor(x2,-3)
  monitor(x3,-5)
  monitor(x4,-7)

  print("...printing...")
  print("%s %s" % (monitor.x[0], monitor.y[0]))
  print("%s %s" % (monitor.x[1], monitor.y[1]))
  print("%s %s" % (monitor.x[2], monitor.y[2]))
  print("%s %s" % (monitor.x[3], monitor.y[3]))
  return


def test1(monitor):
  from numpy import array
  x1 = array([[1,2,3],[2,3,4]]); y1 = array([-1,-2])
  x2 = array([[3,4,5],[4,5,6]]); y2 = [-3,-4]
  x3 = [[5,6,7],[6,7,8]];        y3 = array([-5,-6])
  x4 = [[7,8,9],[8,9,0]];        y4 = [-7,-8]

  monitor(x1,y1)
  monitor(x2,y2)
  monitor(x3,y3)
  monitor(x4,y4)

  print("...printing...")
  print("%s %s" % (monitor.x[0], monitor.y[0]))
  print("%s %s" % (monitor.x[1], monitor.y[1]))
  print("%s %s" % (monitor.x[2], monitor.y[2]))
  print("%s %s" % (monitor.x[3], monitor.y[3]))
  return


def test2(monitor, diffenv=None):
  if diffenv == True:
   #from mystic.solvers import DifferentialEvolutionSolver as DE
    from mystic.solvers import DifferentialEvolutionSolver2 as DE
  elif diffenv == False:
    from mystic.solvers import NelderMeadSimplexSolver as noDE
  else:
    from mystic.solvers import PowellDirectionalSolver as noDE
  from mystic.termination import ChangeOverGeneration as COG
  from mystic.tools import getch, random_seed

  random_seed(123)

  lb = [-100,-100,-100]
  ub = [1000,1000,1000]
  ndim = len(lb)
  npop = 5
  maxiter = 10
  maxfun = 1e+6
  convergence_tol = 1e-10; ngen = 100
  crossover = 0.9
  percent_change = 0.9

  def cost(x):
    ax,bx,c = x
    return (ax)**2 - bx + c

  if diffenv == True:
    solver = DE(ndim,npop)
  else:
    solver = noDE(ndim)
  solver.SetRandomInitialPoints(min=lb,max=ub)
  solver.SetStrictRanges(min=lb,max=ub)
  solver.SetEvaluationLimits(maxiter,maxfun)
  solver.SetEvaluationMonitor(monitor)
 #solver.SetGenerationMonitor(monitor)

  tol = convergence_tol
  solver.Solve(cost, termination=COG(tol,ngen))

  solved = solver.Solution()
  monitor.info("solved: %s" % solved)
  func_max = -solver.bestEnergy 
  return solved, func_max


if __name__ == '__main__':

  from mystic.monitors import Monitor, VerboseMonitor, LoggingMonitor
  from mystic.monitors import VerboseLoggingMonitor
 #monitor = Monitor()
 #monitor = Monitor(all=True)
 #monitor = Monitor(all=False)
 #monitor = VerboseMonitor(1,1) 
 #monitor = VerboseMonitor(1,1, all=True) 
 #monitor = VerboseMonitor(1,1, all=False) 
 #monitor = VerboseMonitor(0,1)
  monitor = VerboseMonitor(1,0)
 #monitor = LoggingMonitor(1)
 #monitor = LoggingMonitor(1, all=True)
 #monitor = LoggingMonitor(1, all=False)
 #monitor = VerboseLoggingMonitor(1)
 #monitor = VerboseLoggingMonitor(0,1)

 #test0(monitor)
 #test1(monitor)
  test2(monitor)                 # GenerationMonitor works like test0
 #test2(monitor, diffenv=False)  # (to make like test1, add enclosing [])
 #test2(monitor, diffenv=True)

  # these are for "MonitorPlotter(s)"; need to adapt log.py plotters for test1
  write_support_file(monitor,'paramlog1.py')  # plot with 'support_*.py'
 #write_converge_file(monitor,'paramlog2.py') #XXX: no existing plotters?
 #write_raw_file(monitor,'paramlog3.py') #XXX: no existing plotters?

  import os
  for fname in ('paramlog1.py',):
    if os.path.exists(fname):
      os.remove(fname)


# EOF
