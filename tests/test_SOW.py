#!/usr/bin/env python
#
# Mike McKerns, Caltech

from mystic.munge import write_support_file, write_converge_file

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

  print "...printing..."
  print monitor.x[0], monitor.y[0]
  print monitor.x[1], monitor.y[1]
  print monitor.x[2], monitor.y[2]
  print monitor.x[3], monitor.y[3]
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

  print "...printing..."
  print monitor.x[0], monitor.y[0]
  print monitor.x[1], monitor.y[1]
  print monitor.x[2], monitor.y[2]
  print monitor.x[3], monitor.y[3]
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

  tol = convergence_tol
 #solver.Solve(cost, termination=COG(tol,ngen), StepMonitor=monitor)
  solver.Solve(cost, termination=COG(tol,ngen), EvaluationMonitor=monitor)

  solved = solver.Solution()
  print "solved: %s" % solver.Solution()
  func_max = -solver.bestEnergy 
  return solved, func_max


if __name__ == '__main__':

  from mystic.monitors import Monitor, VerboseMonitor, LoggingMonitor
 #monitor = Monitor()
 #monitor = Monitor(all=True)
 #monitor = Monitor(all=False)
 #monitor = VerboseMonitor(1,1) 
 #monitor = VerboseMonitor(1,1, all=True) 
 #monitor = VerboseMonitor(1,1, all=False) 
  monitor = LoggingMonitor(1)
 #monitor = LoggingMonitor(1, all=True)
 #monitor = LoggingMonitor(1, all=False)

 #test0(monitor)
 #test1(monitor)
  test2(monitor)                 # StepMonitor works like test0
 #test2(monitor, diffenv=False)  # (to make like test1, add enclosing [])
 #test2(monitor, diffenv=True)

  # these are for "MonitorPlotter(s)"; need to adapt log.py plotters for test1
  write_support_file(monitor,'paramlog1.py')
 #write_converge_file(monitor,'paramlog2.py')

# EOF
