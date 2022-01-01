#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

from mystic.models import rosen
from mystic.solvers import *
from mystic.termination import VTRChangeOverGeneration
from mystic.monitors import VerboseMonitor, Monitor
from mystic.tools import random_seed
random_seed(123)
lb, ub = [-100.]*3, [100]*3
interval = None

if interval:
  _stepmon = VerboseMonitor(interval)
else:
  _stepmon = Monitor()
_term = VTRChangeOverGeneration(generations=200)
_solver = DifferentialEvolutionSolver(3, 20)#40)
_solver.SetRandomInitialPoints(lb,ub)
_solver.SetStrictRanges(lb,ub)
_solver.SetTermination(_term)
_solver.SetGenerationMonitor(_stepmon)
_solver.SetEvaluationLimits(100, 1000)
_solver.Solve(rosen)

_energy = _solver.bestEnergy
_solution =  _solver.bestSolution
_population = _solver.population

_solver.SetEvaluationLimits(10000, 100000)
_solver.Solve()

_energy = _solver.bestEnergy
_solution = _solver.bestSolution
_population = _solver.population

# again, with same method order
random_seed(123)
if interval:
  stepmon = VerboseMonitor(interval)
else:
  stepmon = Monitor()
term = VTRChangeOverGeneration(generations=200)
solver = DifferentialEvolutionSolver(3, 20)#40)
solver.SetRandomInitialPoints(lb,ub)
solver.SetStrictRanges(lb,ub)
solver.SetTermination(term)
solver.SetGenerationMonitor(stepmon)
solver.SetEvaluationLimits(100, 1000)
solver.Solve(rosen)

energy = solver.bestEnergy
solution =  solver.bestSolution
population = solver.population

solver.SetEvaluationLimits(10000, 100000)
solver.Solve()

energy = solver.bestEnergy
solution = solver.bestSolution
population = solver.population

# compare results
assert energy == _energy
assert all(solution == _solution)

# again, with different method order
random_seed(123)
if interval:
  stepmon = VerboseMonitor(interval)
else:
  stepmon = Monitor()
term = VTRChangeOverGeneration(generations=200)
solver = DifferentialEvolutionSolver(3, 20)#40)
solver.SetRandomInitialPoints(lb,ub)
solver.SetGenerationMonitor(stepmon)
solver.SetEvaluationLimits(100, 1000)
solver.SetTermination(term)
solver.SetStrictRanges(lb,ub)
solver.Solve(rosen)

energy = solver.bestEnergy
solution =  solver.bestSolution
population = solver.population

solver.SetEvaluationLimits(10000, 100000)
solver.Solve()

energy = solver.bestEnergy
solution = solver.bestSolution
population = solver.population

# compare results
assert energy == _energy
assert all(solution == _solution)

# again, but focused on method order for population
random_seed(123)
if interval:
  stepmon = VerboseMonitor(interval)
else:
  stepmon = Monitor()
term = VTRChangeOverGeneration(generations=200)
solver = DifferentialEvolutionSolver(3, 20)#40)
solver.SetGenerationMonitor(stepmon)
solver.SetEvaluationLimits(100, 1000)
solver.SetTermination(term)
solver.SetStrictRanges(lb,ub)
solver.SetRandomInitialPoints(lb,ub)
solver.Solve(rosen)

energy = solver.bestEnergy
solution =  solver.bestSolution
population = solver.population

solver.SetEvaluationLimits(10000, 100000)
solver.Solve()

energy = solver.bestEnergy
solution = solver.bestSolution
population = solver.population

# compare results
assert energy == _energy
assert all(solution == _solution)

# start over... this time focus on methods called on the restart
interval = None
random_seed(213)
if interval:
  _stepmon2 = VerboseMonitor(interval)
else:
  _stepmon2 = Monitor()
_term2 = VTRChangeOverGeneration(generations=2000)
_solver2 = DifferentialEvolutionSolver(3, 20)#40)
_solver2.SetEvaluationLimits(100, 1000)
_solver2.SetTermination(_term2)
_solver2.SetRandomInitialPoints(lb,ub)
_solver2.Solve(rosen)

_energy2 = _solver2.bestEnergy
_solution2 =  _solver2.bestSolution
_population2 = _solver2.population

_term2 = VTRChangeOverGeneration(generations=200)
_solver2.SetStrictRanges(lb,ub)
_solver2.SetEvaluationLimits(new=True)
_solver2.SetGenerationMonitor(_stepmon2)
_solver2.SetTermination(_term2)
_solver2.Solve()

_energy2 = _solver2.bestEnergy
_solution2 = _solver2.bestSolution
_population2 = _solver2.population

# again, but swap method order for restart
random_seed(213)
if interval:
  stepmon2 = VerboseMonitor(interval)
else:
  stepmon2 = Monitor()
term2 = VTRChangeOverGeneration(generations=2000)
solver2 = DifferentialEvolutionSolver(3, 20)#40)
solver2.SetEvaluationLimits(100, 1000)
solver2.SetTermination(term2)
solver2.SetRandomInitialPoints(lb,ub)
solver2.Solve(rosen)

energy2 = solver2.bestEnergy
solution2 =  solver2.bestSolution
population2 = solver2.population

term2 = VTRChangeOverGeneration(generations=200)
solver2.SetTermination(term2)
solver2.SetGenerationMonitor(stepmon2)
solver2.SetEvaluationLimits(new=True)
solver2.SetStrictRanges(lb,ub)
solver2.Solve()

energy2 = solver2.bestEnergy
solution2 = solver2.bestSolution
population2 = solver2.population

# compare results
assert energy2 == _energy2
assert all(solution2 == _solution2)

# start over... and change so initialize population in the restart
interval = None
random_seed(123)
if interval:
  _stepmon3 = VerboseMonitor(interval)
else:
  _stepmon3 = Monitor()
_term3 = VTRChangeOverGeneration(generations=2000)
_solver3 = DifferentialEvolutionSolver(3, 20)#40)
_solver3.SetEvaluationLimits(100, 1000)
_solver3.SetTermination(_term3)
_solver3.SetGenerationMonitor(_stepmon3)
_solver3.SetRandomInitialPoints(lb,ub)
_solver3.Solve(rosen)

_energy3 = _solver3.bestEnergy
_solution3 =  _solver3.bestSolution
_population3 = _solver3.population

_lb, _ub = [-10.]*3, [10]*3
_term3 = VTRChangeOverGeneration(generations=200)
_solver3.SetRandomInitialPoints(_lb,_ub) #FIXME: pretty much causes a flatline
_solver3.SetStrictRanges(lb,ub)          #       regardless of _lb,_ub value
_solver3.SetEvaluationLimits(200, 2000, new=True)# check if pop is uniform
_solver3.SetTermination(_term3)
_solver3.Solve()

_energy3 = _solver3.bestEnergy
_solution3 = _solver3.bestSolution
_population3 = _solver3.population
# FIXME: when population becomes uniform, the solver will get stuck
#        would be good to be able to inject some randomness into a restart

# again, but swap method order for restart
random_seed(123)
if interval:
  stepmon3 = VerboseMonitor(interval)
else:
  stepmon3 = Monitor()
term3 = VTRChangeOverGeneration(generations=2000)
solver3 = DifferentialEvolutionSolver(3, 20)#40)
solver3.SetEvaluationLimits(100, 1000)
solver3.SetTermination(term3)
solver3.SetGenerationMonitor(stepmon3)
solver3.SetRandomInitialPoints(lb,ub)
solver3.Solve(rosen)

energy3 = solver3.bestEnergy
solution3 =  solver3.bestSolution
population3 = solver3.population

term3 = VTRChangeOverGeneration(generations=200)
solver3.SetTermination(term3)
solver3.SetEvaluationLimits(200, 2000, new=True)
solver3.SetStrictRanges(lb,ub)
solver3.SetRandomInitialPoints(_lb,_ub)
solver3.Solve()

energy3 = solver3.bestEnergy
solution3 = solver3.bestSolution
population3 = solver3.population

# compare results
assert energy3 == _energy3
assert all(solution3 == _solution3)

# TODO: start over... but work with Step instead of Solve

# EOF
