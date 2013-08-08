from mystic.monitors import *
a = Null()
b = Monitor()
c = VerboseMonitor(None)

a([1,],1)
a([2,],2)
b([0,],0)
b.extend(a)
assert b.x == [[0]]
assert len(b) == 1
a.extend(b)
assert a.x == Null()
#assert len(a) == None  #XXX: len(Null()) throws a TypeError

c([1,],1)
c([2,],2)
c.prepend(b)
assert len(b) == 1
assert len(c) == 3
c([3,],3)
assert c.x == [[0], [1], [2], [3]]
assert len(c) == 4

c.prepend(c)
assert c.x == [[0], [1], [2], [3], [0], [1], [2], [3]]
assert len(c) == 8

from mystic.solvers import NelderMeadSimplexSolver
from mystic.tools import random_seed
random_seed(123)

lb = [-100,-100,-100]
ub = [1000,1000,1000]
ndim = len(lb)
maxiter = 10
maxfun = 1e+6

def cost(x):
  ax,bx,c = x
  return (ax)**2 - bx + c

monitor = Monitor()
monitor2 = Monitor()
solver = NelderMeadSimplexSolver(ndim)
solver.SetRandomInitialPoints(min=lb,max=ub)
solver.SetStrictRanges(min=lb,max=ub)
solver.SetEvaluationLimits(maxiter,maxfun)
solver.SetGenerationMonitor(monitor)
solver.Solve(cost)

solved = solver.Solution()
monitor.info("solved: %s" % solved)

solver.SetEvaluationLimits(maxiter*2,maxfun)
solver.SetGenerationMonitor(monitor)
solver.Solve(cost)

solver.SetEvaluationLimits(maxiter*3,maxfun)
solver.SetGenerationMonitor(monitor, new=True)
solver.Solve(cost)



