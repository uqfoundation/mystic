try:
  from scipy.optimize import fmin, fmin_powell
  scipy_solvers = ['fmin_powell', 'fmin']
except ImportError:
  print "Warning: scipy not installed; comparison tests skipped"
  scipy_solvers = []

from mystic import solvers
from mystic.models import rosen
from mystic.math import almostEqual
from mystic.monitors import VerboseMonitor
from mystic.tools import random_seed
random_seed(321)

def test_solvers(solver1, solver2, x0, **kwds):
  exec "s1 = solvers.%s" % solver1
  exec "s2 = solvers.%s" % solver2
  maxiter = kwds.get('maxiter', None)
  maxfun = kwds.get('maxfun', None)
  s1_x = s1(rosen, x0, disp=0, full_output=True, **kwds)
  s2_x = s2(rosen, x0, disp=0, full_output=True, **kwds)
  # similar bestSolution and bestEnergy
# print 's1:', s1_x[0:2]
# print 's2:', s2_x[0:2]
  # print (iters, fcalls) and [maxiter, maxfun]
# print s1_x[2:4], s2_x[2:4], [maxiter, maxfun]
  if maxiter is not None:
    # test iters <= maxiter
    assert s1_x[2] <= maxiter
    # test same number of iters
    if s1_x[4] == s2_x[4]: assert s1_x[2] == s2_x[2]
  if maxiter and maxfun is not None:
    # test fcalls <= maxfun
#   assert s1_x[3] <= maxfun
    # test same number of fcalls
    if s1_x[4] == s2_x[4]: assert s1_x[3] == s2_x[3]
  return 

def test_compare(solvername, x0, **kwds):
  exec "my = solvers.%s" % solvername
  exec "sp = %s" % solvername
  maxiter = kwds.get('maxiter', None)
  maxfun = kwds.get('maxfun', None)
  my_x = my(rosen, x0, disp=0, full_output=True, **kwds)
# itermon = kwds.pop('itermon',None)
  sp_x = sp(rosen, x0, disp=0, full_output=True, **kwds)
  # similar bestSolution and bestEnergy
# print 'my:', my_x[0:2]
# print 'sp:', sp_x[0:2]
  assert almostEqual(my_x[0], sp_x[0])
  assert almostEqual(my_x[1], sp_x[1])
  # print (iters, fcalls) and [maxiter, maxfun]
# print my_x[2:4], (sp_x[-3],sp_x[-2]), [maxiter, maxfun]
  # test same number of iters and fcalls
  if maxiter and maxfun is not None:
    assert my_x[2] == sp_x[-3]
    assert my_x[3] == sp_x[-2]
#   # test fcalls <= maxfun
#   assert my_x[3] <= maxfun
  if maxiter is not None:
    # test iters <= maxiter
    if maxiter != 0: assert my_x[2] <= maxiter #FIXME: scipy* never stops at 0
  return 

if __name__ == '__main__':
  x0 = [0,0,0]

  # check solutions versus results based on the random_seed
  print "comparing against known results"
  sol = solvers.diffev(rosen, x0, npop=40, disp=0, full_output=True)
  assert almostEqual(sol[1], 0.0020640145337293249, tol=1e-3)
  sol = solvers.diffev2(rosen, x0, npop=40, disp=0, full_output=True)
  assert almostEqual(sol[1], 0.0017516784703663288)
  sol = solvers.fmin_powell(rosen, x0, disp=0, full_output=True)
  assert almostEqual(sol[1], 8.3173488898295291e-23)
  sol = solvers.fmin(rosen, x0, disp=0, full_output=True)
  assert almostEqual(sol[1], 1.1605792769954724e-09)

  solver2 = 'diffev2'
  for solver in ['diffev']:
    print "comparing %s and %s from mystic" % (solver, solver2)
    test_solvers(solver, solver2, x0, npop=40)
#   test_solvers(solver, solver2, x0, npop=40, maxiter=None, maxfun=0)
#   test_solvers(solver, solver2, x0, npop=40, maxiter=None, maxfun=1)
#   test_solvers(solver, solver2, x0, npop=40, maxiter=None, maxfun=2)
#   test_solvers(solver, solver2, x0, npop=40, maxiter=None, maxfun=9)
    test_solvers(solver, solver2, x0, npop=40, maxiter=0)
    test_solvers(solver, solver2, x0, npop=40, maxiter=1)
    test_solvers(solver, solver2, x0, npop=40, maxiter=2)
    test_solvers(solver, solver2, x0, npop=40, maxiter=9)

  for solver in scipy_solvers:
    print "comparing %s from mystic and scipy" % (solver)
    test_compare(solver, x0)
#   test_compare(solver, x0, maxiter=None, maxfun=0)
#   test_compare(solver, x0, maxiter=None, maxfun=1)
#   test_compare(solver, x0, maxiter=None, maxfun=2)
#   test_compare(solver, x0, maxiter=None, maxfun=9)
    test_compare(solver, x0, maxiter=0)
    test_compare(solver, x0, maxiter=1)
    test_compare(solver, x0, maxiter=2)#, itermon=VerboseMonitor(1,1))
    test_compare(solver, x0, maxiter=9)


# EOF
