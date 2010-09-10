#!/usr/bin/env python
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

def test_terminators(test, func=lambda x:x[0], verbose=False):
  print test(lambda x:"", func, verbose) #XXX: just print settings
  print "VTR():", test(VTR(), func)
  print "VTR(inf):", test(VTR(inf), func)
  print "COG():", test(ChangeOverGeneration(), func)
  print "COG(gen=5):", test(ChangeOverGeneration(generations=5), func)
  print "NCOG():", test(NormalizedChangeOverGeneration(), func)
  print "NCOG(gen=5):", test(NormalizedChangeOverGeneration(generations=5),func)
  print "CTR():", test(CandidateRelativeTolerance(), func)
  print "CTR(ftol=inf):", test(CandidateRelativeTolerance(ftol=inf), func)
  print "CTR(inf):", test(CandidateRelativeTolerance(inf), func)
  print "SI():", test(SolutionImprovement(), func)
  print "SI(inf):", test(SolutionImprovement(inf), func)
  print "NCT():", test(NormalizedCostTarget(), func)
  print "NCT(gen=5):", test(NormalizedCostTarget(generations=5), func)
  print "NCT(gen=None):", test(NormalizedCostTarget(generations=None), func)
  print "NCT(inf,inf):", test(NormalizedCostTarget(inf,inf), func)
  print "VCOG():", test(VTRChangeOverGeneration(), func)
  print "VCOG(gen=5):", test(VTRChangeOverGeneration(generations=5), func)
  print "VCOG(inf):", test(VTRChangeOverGeneration(inf), func)
  print "PS():", test(PopulationSpread(), func)
  print "PS(inf):", test(PopulationSpread(inf), func)
 #print "GNT():", test(GradientNormTolerance(), func)
  return

def verbosity(solver):
    print "energy_history:", solver.energy_history
    print "population:", solver.population
    print "popEnergy:", solver.popEnergy
    print "bestSolution:", solver.bestSolution
    print "trialSolution:", solver.trialSolution
    print "bestEnergy:", solver.bestEnergy
    return

def test01(terminate, func=lambda x:x[0], debug=False):
  from mystic.differential_evolution import DifferentialEvolutionSolver2 as DE2
  solver = DE2(3,5)
  solver.SetRandomInitialPoints()
  solver.SetEvaluationLimits(8)
  solver.Solve(func, VTR())
  if debug: verbosity(solver)
  return terminate(solver)

def test02(terminate, func=lambda x:x[0], debug=False):
  from mystic.differential_evolution import DifferentialEvolutionSolver2 as DE2
 #solver = DE2(3,1) #Solver throws ValueError "sample larger than population"
 #solver = DE2(1,1) #Solver throws ValueError "sample larger than population"
  solver = DE2(1,5)
  solver.SetRandomInitialPoints()
  solver.SetEvaluationLimits(8)
  solver.Solve(func, VTR())
  if debug: verbosity(solver)
  return terminate(solver)

def test03(terminate, func=lambda x:x[0], debug=False):
  from mystic.differential_evolution import DifferentialEvolutionSolver as DE
  solver = DE(3,5)
  solver.SetRandomInitialPoints()
  solver.SetEvaluationLimits(8)
  solver.Solve(func, VTR())
  if debug: verbosity(solver)
  return terminate(solver)

def test04(terminate, func=lambda x:x[0], debug=False):
  from mystic.differential_evolution import DifferentialEvolutionSolver as DE
  solver = DE(1,5)
  solver.SetRandomInitialPoints()
  solver.SetEvaluationLimits(8)
  solver.Solve(func, VTR())
  if debug: verbosity(solver)
  return terminate(solver)

def test05(terminate, func=lambda x:x[0], debug=False):
  from mystic.scipy_optimize import NelderMeadSimplexSolver as NM
  solver = NM(3)
  solver.SetRandomInitialPoints()
  solver.SetEvaluationLimits(8)
  solver.Solve(func, VTR())
  if debug: verbosity(solver)
  return terminate(solver)

def test06(terminate, func=lambda x:x[0], debug=False):
  from mystic.scipy_optimize import NelderMeadSimplexSolver as NM
  solver = NM(1)
  solver.SetRandomInitialPoints()
  solver.SetEvaluationLimits(8)
  solver.Solve(func, VTR())
  if debug: verbosity(solver)
  return terminate(solver)

def test07(terminate, func=lambda x:x[0], debug=False):
  from mystic.scipy_optimize import PowellDirectionalSolver as PDS
  solver = PDS(3)
  solver.SetRandomInitialPoints()
  solver.SetEvaluationLimits(8)
  solver.Solve(func, VTR())
  if debug: verbosity(solver)
  return terminate(solver)

def test08(terminate, func=lambda x:x[0], debug=False):
  from mystic.scipy_optimize import PowellDirectionalSolver as PDS
  solver = PDS(1)
  solver.SetRandomInitialPoints()
  solver.SetEvaluationLimits(8)
  solver.Solve(func, VTR())
  if debug: verbosity(solver)
  return terminate(solver)


if __name__ == "__main__":
  verbose = True
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

 #test_terminators(test01,function,verbose)
 #test_terminators(test02,function,verbose)
 #test_terminators(test03,function,verbose)
 #test_terminators(test04,function,verbose)
 #test_terminators(test05,function,verbose)
 #test_terminators(test06,function,verbose)
  test_terminators(test07,function,verbose)
 #test_terminators(test08,function,verbose)

# EOF
