#!/usr/bin/env python
#
# Mike McKerns, Caltech

def write_converge_file(mon,log_file='paramlog.py'):
  log = []
  steps = mon.x[:]
  energy = mon.y[:]
  f = open(log_file,'w')
 #f.write('# %s\n' % energy[-1])
  f.write('params = %s' % steps)
  f.write('\ncost = %s' % energy)
  f.close()
  return

def write_support_file(mon,log_file='paramlog.py'):
  log = []
  steps = mon.x[:]
  energy = mon.y[:]
  for p in range(len(steps[0])):
    q = []
    for s in range(len(steps)):
      q.append(steps[s][p])
    log.append(q)  
  f = open(log_file,'w')
 #f.write('# %s\n' % energy[-1])
  f.write('params = %s' % log)
  f.write('\ncost = %s' % energy)
  f.close()
  return

def test1(monitor):
  from numpy import array
  x1 = array([1,2,3])
  x2 = array([3,4,5])
  x3 = array([5,6,7])
  x4 = array([7,8,9])

  monitor(x1,1)
  monitor(x2,2)
  monitor(x3,3)
  monitor(x4,4)

  print "...printing..."
  print monitor.x[0], monitor.y[0]
  print monitor.x[1], monitor.y[1]
  print monitor.x[2], monitor.y[2]
  print monitor.x[3], monitor.y[3]
  return


def test2(monitor):
  from mystic.differential_evolution import DifferentialEvolutionSolver as DE
 #from mystic.differential_evolution import DifferentialEvolutionSolver2 as DE
  from mystic.scipy_optimize import NelderMeadSimplexSolver as noDE
  from mystic.termination import ChangeOverGeneration as COG
  from mystic import getch, random_seed

 #random_seed(123)

  lb = [-100,-100,-100]
  ub = [1000,1000,1000]
  ndim = len(lb)
  npop = 40
  maxiter = 30
  maxfun = 1e+6
  convergence_tol = 1e-10; ngen = 100
  crossover = 0.9
  percent_change = 0.9

  def cost(x):
    ax,bx,c = x
    return (ax)**2 - bx + c

  solver = DE(ndim,npop)
 #solver = noDE(ndim)
  solver.SetRandomInitialPoints(min=lb,max=ub)
  solver.SetStrictRanges(min=lb,max=ub)
  solver.SetEvaluationLimits(maxiter,maxfun)

  tol = convergence_tol
  solver.Solve(cost, termination=COG(tol,ngen), StepMonitor=monitor)

  solved = solver.Solution()
  print "solved: %s" % solver.Solution()
  func_max = -solver.bestEnergy 
  return solved, func_max


if __name__ == '__main__':

  from mystic.tools import Sow, VerboseSow
 #monitor = Sow()
  monitor = VerboseSow(1,1) 

 #test1(monitor)
  test2(monitor)

  write_support_file(monitor,'log1.py')
  write_converge_file(monitor,'log2.py')

