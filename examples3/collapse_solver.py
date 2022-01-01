#!/usr/bin/env python
#
# Author: Lan Huong Nguyen (lanhuong @stanford)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2012-2015 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

from mystic.termination import Or, CollapseAt, CollapseAs
from mystic.termination import ChangeOverGeneration as COG
#from mystic.termination import VTRChangeOverGeneration as COG

# update termination condition with new masks
## termination should be Or(*conditions), where
## conditions are: COG(), Collapse*(), and possibly others.
# takes termination condition (or solver?) and returns termination condition
# also requires updated masks as input
verbose = True#False

#from mystic.models import rosen as model; target = 1.0
from mystic.models import sphere as model; target = 0.0
n = 10

#term = Or((COG(generations=300), CollapseAt(None, generations=100), CollapseAs(generations=100)))
term = Or((COG(generations=500), CollapseAt(target, generations=100)))
#term = COG(generations=500)

from mystic import suppressed
@suppressed(1e-8)
def constrain(x):
    return x

from mystic.solvers import DifferentialEvolutionSolver as TheSolver
#from mystic.solvers import PowellDirectionalSolver as TheSolver
from mystic.solvers import BuckshotSolver
#solver = BuckshotSolver(n, 10)
solver = TheSolver(n)
solver.SetRandomInitialPoints()
solver.SetStrictRanges(min=[0]*n, max=[5]*n)
solver.SetConstraints(constrain)
solver.SetEvaluationLimits(evaluations=320000, generations=1000)
solver.SetTermination(term)

#from mystic.termination import state
#print(state(solver._termination).keys())
solver.Solve(model, disp=verbose)

# while collapse and solver.Collapse(verbose):
#   solver.Solve(model)

# we are done; get result
print(solver.Terminated(info=True))
print('%s @' % solver.bestEnergy)
print(solver.bestSolution)



# EOF
