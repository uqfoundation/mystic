#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2018-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

from mystic.solvers import DifferentialEvolutionSolver
from mystic.models import rosen
from mystic.tools import solver_bounds
from mystic.termination import ChangeOverGeneration as COG, Or, CollapseCost
from mystic.monitors import VerboseLoggingMonitor as Monitor

solver = DifferentialEvolutionSolver(3,40)
solver.SetRandomInitialPoints([-100]*3, [100]*3)

mon = Monitor(1)
options = dict(limit=1.95, samples=50, clip=False)
mask = {} #solver_bounds(solver)
stop = Or(CollapseCost(mask=mask,**options), COG(generations=50))

solver.SetGenerationMonitor(mon)
solver.SetTermination(stop)
solver.SetObjective(rosen)
solver.Solve()

print(solver.Terminated(info=True))
print('%s @' % solver.bestEnergy)
print(solver.bestSolution)
