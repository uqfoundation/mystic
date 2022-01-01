#!/usr/bin/env python
#
# Author: Alta Fang (altafang @caltech and alta @princeton)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""test termination conditions. (defaults listed below)

VTR(tolerance = 0.005)
ChangeOverGeneration(tolerance = 1e-6, generations = 30)
NormalizedChangeOverGeneration(tolerance = 1e-4, generations = 10)
CandidateRelativeTolerance(xtol=1e-4, ftol=1e-4)
SolutionImprovement(tolerance = 1e-5)
NormalizedCostTarget(fval = None, tolerance = 1e-6, generations = 30)
VTRChangeOverGeneration(ftol = 0.005, gtol = 1e-6, generations = 30)
PopulationSpread(tolerance=1e-6)
GradientNormTolerance(tolerance=1e-5, norm=Inf)
"""
from mystic.termination import *
from numpy import inf

def test_terminators(test, func=lambda x:x[0], info=False, verbose=False, disp=False):
  res = test(lambda x,y:"", func, info, verbose)  #XXX: just check settings
  if disp: print(res) 
  else: assert bool(res) in (True, False)
  res = test(VTR(), func, info)
  if disp: print("VTR():%s" % res)
  else: assert bool(res) in (True, False)
  res = test(VTR(inf), func, info)
  if disp: print("VTR(inf):%s" % res)
  else: assert bool(res) in (True, False)
  res = test(ChangeOverGeneration(), func, info)
  if disp: print("COG():%s" %  res)
  else: assert bool(res) in (True, False)
  res = test(ChangeOverGeneration(generations=5), func, info)
  if disp: print("COG(gen=5):%s" % res)
  else: assert bool(res) in (True, False)
  res = test(NormalizedChangeOverGeneration(), func, info)
  if disp: print("NCOG():%s" % res)
  else: assert bool(res) in (True, False)
  res = test(NormalizedChangeOverGeneration(generations=5), func, info)
  if disp: print("NCOG(gen=5):%s" % res)
  else: assert bool(res) in (True, False)
  res = test(CandidateRelativeTolerance(), func, info)
  if disp: print("CTR():%s" % res)
  else: assert bool(res) in (True, False)
  res = test(CandidateRelativeTolerance(ftol=inf), func, info)
  if disp: print("CTR(ftol=inf):%s" % res)
  else: assert bool(res) in (True, False)
  res = test(CandidateRelativeTolerance(inf), func, info)
  if disp: print("CTR(inf):%s" % res)
  else: assert bool(res) in (True, False)
  res = test(SolutionImprovement(), func, info)
  if disp: print("SI():%s" % res)
  else: assert bool(res) in (True, False)
  res = test(SolutionImprovement(inf), func, info)
  if disp: print("SI(inf):%s" % res)
  else: assert bool(res) in (True, False)
  res = test(NormalizedCostTarget(), func, info)
  if disp: print("NCT():%s" % res)
  else: assert bool(res) in (True, False)
  res = test(NormalizedCostTarget(generations=5), func, info)
  if disp: print("NCT(gen=5):%s" % res)
  else: assert bool(res) in (True, False)
  res = test(NormalizedCostTarget(generations=None), func, info)
  if disp: print("NCT(gen=None):%s" % res)
  else: assert bool(res) in (True, False)
  res = test(NormalizedCostTarget(inf,inf), func, info)
  if disp: print("NCT(inf,inf):%s" % res)
  else: assert bool(res) in (True, False)
  res = test(VTRChangeOverGeneration(), func, info)
  if disp: print("VCOG():%s" % res)
  else: assert bool(res) in (True, False)
  res = test(VTRChangeOverGeneration(generations=5), func, info)
  if disp: print("VCOG(gen=5):%s" % res)
  else: assert bool(res) in (True, False)
  res = test(VTRChangeOverGeneration(inf), func, info)
  if disp: print("VCOG(inf):%s" % res)
  else: assert bool(res) in (True, False)
  res = test(PopulationSpread(), func, info)
  if disp: print("PS():%s" % res)
  else: assert bool(res) in (True, False)
  res = test(PopulationSpread(inf), func, info)
  if disp: print("PS(inf):%s" % res)
  else: assert bool(res) in (True, False)
 #print("GNT():%s" % test(GradientNormTolerance(), func, info))
  return

def verbosity(solver):
    print("energy_history:%s" % solver.energy_history)
    print("population:%s" % solver.population)
    print("popEnergy:%s" % solver.popEnergy)
    print("bestSolution:%s" % solver.bestSolution)
    print("trialSolution:%s" % solver.trialSolution)
    print("bestEnergy:%s" % solver.bestEnergy)
    return

def test01(terminate, func=lambda x:x[0], info=False, debug=False):
  from mystic.solvers import DifferentialEvolutionSolver2 as DE2
  solver = DE2(3,5)
  solver.SetRandomInitialPoints()
  solver.SetEvaluationLimits(8)
  solver.Solve(func, VTR())
  if debug: verbosity(solver)
  return terminate(solver, info)

def test02(terminate, func=lambda x:x[0], info=False, debug=False):
  from mystic.solvers import DifferentialEvolutionSolver2 as DE2
 #solver = DE2(3,1) #Solver throws ValueError "sample larger than population"
 #solver = DE2(1,1) #Solver throws ValueError "sample larger than population"
  solver = DE2(1,5)
  solver.SetRandomInitialPoints()
  solver.SetEvaluationLimits(8)
  solver.Solve(func, VTR())
  if debug: verbosity(solver)
  return terminate(solver, info)

def test03(terminate, func=lambda x:x[0], info=False, debug=False):
  from mystic.solvers import DifferentialEvolutionSolver as DE
  solver = DE(3,5)
  solver.SetRandomInitialPoints()
  solver.SetEvaluationLimits(8)
  solver.Solve(func, VTR())
  if debug: verbosity(solver)
  return terminate(solver, info)

def test04(terminate, func=lambda x:x[0], info=False, debug=False):
  from mystic.solvers import DifferentialEvolutionSolver as DE
  solver = DE(1,5)
  solver.SetRandomInitialPoints()
  solver.SetEvaluationLimits(8)
  solver.Solve(func, VTR())
  if debug: verbosity(solver)
  return terminate(solver, info)

def test05(terminate, func=lambda x:x[0], info=False, debug=False):
  from mystic.solvers import NelderMeadSimplexSolver as NM
  solver = NM(3)
  solver.SetRandomInitialPoints()
  solver.SetEvaluationLimits(8)
  solver.Solve(func, VTR())
  if debug: verbosity(solver)
  return terminate(solver, info)

def test06(terminate, func=lambda x:x[0], info=False, debug=False):
  from mystic.solvers import NelderMeadSimplexSolver as NM
  solver = NM(1)
  solver.SetRandomInitialPoints()
  solver.SetEvaluationLimits(8)
  solver.Solve(func, VTR())
  if debug: verbosity(solver)
  return terminate(solver, info)

def test07(terminate, func=lambda x:x[0], info=False, debug=False):
  from mystic.solvers import PowellDirectionalSolver as PDS
  solver = PDS(3)
  solver.SetRandomInitialPoints()
  solver.SetEvaluationLimits(8)
  solver.Solve(func, VTR())
  if debug: verbosity(solver)
  return terminate(solver, info)

def test08(terminate, func=lambda x:x[0], info=False, debug=False):
  from mystic.solvers import PowellDirectionalSolver as PDS
  solver = PDS(1)
  solver.SetRandomInitialPoints()
  solver.SetEvaluationLimits(8)
  solver.Solve(func, VTR())
  if debug: verbosity(solver)
  return terminate(solver, info)


if __name__ == "__main__":
  disp = False
  verbose = False
  info = False
  """NOTES: For x:x[0], test01-test04 returns either lists or floats;
while test05-test06 returns a ndarray for population, popEnergy, bestSolution;
test07-test08 throw a RuntimeError "Too many iterations" due to "bracket()".
  For x:inf, test01-test02 have 1e+20 in energy_history, popEnergy, bestEnergy;
while test03-test04 have inf; test05-test06 have Inf in popEnergy;
while test07-test08 have array(inf) for energy_history, popEnergy,
and inf for bestEnergy, and energy_history, popEnergy, population is list
of ndarrays, while bestSolution is an ndarray.
  For x:10.0, test01-test06 notes are as for x:x[0]; while test07-test08
notes are same as x:inf except with 'inf' replaced by '10.0'.
  For x:0.0, x:-10.0, x:-inf, test01-test08 notes are as for x:10.0 except
with '10.0' replaced by '0.0', '-10.0', '-inf' respectively.

ISSUES:
  Many of the 'internals' do not have a standard format. Is this a problem?
  Why does x:inf result in 1e+20 for DE2?
  """
 #function = lambda x:x[0]
 #function = lambda x:inf
 #function = lambda x:10.0
  function = lambda x:0.0
 #function = lambda x:-10.0
 #function = lambda x:-inf

 #test_terminators(test01,function,info,verbose,disp)
 #test_terminators(test02,function,info,verbose,disp)
 #test_terminators(test03,function,info,verbose,disp)
 #test_terminators(test04,function,info,verbose,disp)
 #test_terminators(test05,function,info,verbose,disp)
 #test_terminators(test06,function,info,verbose,disp)
  test_terminators(test07,function,info,verbose,disp)
 #test_terminators(test08,function,info,verbose,disp)

# EOF
