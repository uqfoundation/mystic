#!/usr/bin/env python
#
# Author: Alta Fang (altafang @caltech and alta @princeton)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

try:
  from scipy.optimize import fmin, fmin_powell
  HAS_SCIPY = True
except ImportError:
  from mystic._scipyoptimize import fmin, fmin_powell
  HAS_SCIPY = False
# print("Warning: scipy not installed; defaulting to local solver copy")
scipy_solvers = ['fmin_powell', 'fmin']

import mystic.solvers as solvers
from mystic.models import rosen
from mystic.math import almostEqual
from mystic.monitors import VerboseMonitor
from mystic.tools import random_seed
random_seed(321)

def test_solvers(solver1, solver2, x0, **kwds):
  s1 = eval("solvers.%s" % solver1)
  s2 = eval("solvers.%s" % solver2)
  maxiter = kwds['maxiter'] if 'maxiter' in kwds else None
  maxfun = kwds['maxfun'] if 'maxfun' in kwds else None
  s1_x = s1(rosen, x0, disp=0, full_output=True, **kwds)
  s2_x = s2(rosen, x0, disp=0, full_output=True, **kwds)
  # similar bestSolution and bestEnergy
# print('s1: %s' % s1_x[0:2])
# print('s2: %s' % s2_x[0:2])
  # print (iters, fcalls) and [maxiter, maxfun]
# print('%s, %s, %s' % (s1_x[2:4], s2_x[2:4], [maxiter, maxfun]))
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
  my = eval("solvers.%s" % solvername)
  sp = eval("%s" % solvername)
  maxiter = kwds['maxiter'] if 'maxiter' in kwds else None
  maxfun = kwds['maxfun'] if 'maxfun' in kwds else None
  my_x = my(rosen, x0, disp=0, full_output=True, **kwds)
# itermon = kwds.pop('itermon',None)
  try:
    sp_x = sp(rosen, x0, disp=0, full_output=True, **kwds)
  except: # _MaxFuncCallError
    if HAS_SCIPY:
      return
    assert False
  # similar bestSolution and bestEnergy
# print('my: %s' % my_x[0:2])
# print('sp: %s' % sp_x[0:2])
  if my_x[3] == sp_x[-2]: # mystic can stop at iter=0, scipy can't
    assert almostEqual(my_x[0], sp_x[0])
    assert almostEqual(my_x[1], sp_x[1])
  # print (iters, fcalls) and [maxiter, maxfun]
# print('%s, %s, %s' % (my_x[2:4], (sp_x[-3],sp_x[-2]), [maxiter, maxfun]))
  # test same number of iters and fcalls
  if maxiter and maxfun is not None:
    assert my_x[2] == sp_x[-3]
    assert my_x[3] == sp_x[-2]
#   # test fcalls <= maxfun
#   assert my_x[3] <= maxfun
  if maxiter is not None:
    # test iters <= maxiter
    assert my_x[2] <= maxiter
  return 

if __name__ == '__main__':
  x0 = [0,0,0]

  # check solutions versus results based on the random_seed
# print("comparing against known results")
  sol = solvers.diffev(rosen, x0, npop=40, disp=0, full_output=True)
  assert almostEqual(sol[1], 0.0020640145337293249, tol=3e-3)
  sol = solvers.diffev2(rosen, x0, npop=40, disp=0, full_output=True)
  assert (almostEqual(sol[1], 0.0017516784703663288, tol=3e-3) or
         almostEqual(sol[1], 0.00496876027278, tol=3e-3)) # python3.x
  sol = solvers.fmin_powell(rosen, x0, disp=0, full_output=True)
  assert almostEqual(sol[1], 8.3173488898295291e-23)
  sol = solvers.fmin(rosen, x0, disp=0, full_output=True)
  assert almostEqual(sol[1], 1.1605792769954724e-09)

  solver2 = 'diffev2'
  for solver in ['diffev']:
#   print("comparing %s and %s from mystic" % (solver, solver2))
    test_solvers(solver, solver2, x0, npop=40)
    test_solvers(solver, solver2, x0, npop=40, maxiter=None, maxfun=0)
    test_solvers(solver, solver2, x0, npop=40, maxiter=None, maxfun=1)
    test_solvers(solver, solver2, x0, npop=40, maxiter=None, maxfun=2)
    test_solvers(solver, solver2, x0, npop=40, maxiter=None, maxfun=9)
    test_solvers(solver, solver2, x0, npop=40, maxiter=0)
    test_solvers(solver, solver2, x0, npop=40, maxiter=1)
    test_solvers(solver, solver2, x0, npop=40, maxiter=2)
    test_solvers(solver, solver2, x0, npop=40, maxiter=9)

  for solver in scipy_solvers:
#   print("comparing %s from mystic and scipy" % (solver))
    test_compare(solver, x0)
    test_compare(solver, x0, maxiter=None, maxfun=0)# _MaxFuncCallError
    test_compare(solver, x0, maxiter=None, maxfun=1)
    test_compare(solver, x0, maxiter=None, maxfun=2)
    test_compare(solver, x0, maxiter=None, maxfun=9)
    test_compare(solver, x0, maxiter=0)
    test_compare(solver, x0, maxiter=1)
    test_compare(solver, x0, maxiter=2)#, itermon=VerboseMonitor(1,1))
    test_compare(solver, x0, maxiter=9)


# EOF
