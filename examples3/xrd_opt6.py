#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2022-2025 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
#
# adapted from XRD example at: https://spotlight.readthedocs.io/
"""
direct optimization of 6-input cost function using an ensemble solver
"""
from mystic.solvers import LatticeSolver
from mystic.solvers import NelderMeadSimplexSolver, PowellDirectionalSolver
from mystic.termination import VTR, ChangeOverGeneration as COG
from mystic.monitors import VerboseMonitor
from emulators import cost6 as cost, x6 as target, bounds6 as bounds

# set the ranges
lower_bounds, upper_bounds = zip(*bounds)

try: # parallel maps
    from pathos.maps import Map
    from pathos.pools import ProcessPool, ThreadPool, SerialPool
    pmap = Map(SerialPool) #ProcessPool
except ImportError:
    pmap = None

# specify optimization algorithm and limits on evaluations of the objective
subsolver = NelderMeadSimplexSolver(6)
subsolver.SetEvaluationLimits(5000, 50000)
subsolver.SetGenerationMonitor(VerboseMonitor(50))
subsolver.SetTermination(COG(1e-12, 200))

# create an ensemble solver, then set parallel map and optimization algorithm
solver = LatticeSolver(6, 8)
solver.SetMapper(pmap)
solver.SetNestedSolver(subsolver)

# set the range to search for all parameters
solver.SetStrictRanges(lower_bounds, upper_bounds)

# find the minimum
solver.Solve(cost, disp=True)

# shutdown mapper
if pmap is not None:
    pmap.close(); pmap.join(); pmap.clear()

# print the best parameters
print(f"Best solution is {solver.bestSolution} with Rwp {solver.bestEnergy}")
print(f"Reference solution: {target}")
ratios = [x / y for x, y in zip(target, solver.bestSolution)]
print(f"Ratios of best to reference solution: {ratios}")
