#!/usr/bin/env python
#
# Author: Lan Huong Nguyen (lanhuong @stanford)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2012-2015 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE

from mystic.termination import Or, CollapseAt, CollapseAs
from mystic.termination import ChangeOverGeneration as COG

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


from mystic.solvers import DifferentialEvolutionSolver as TheSolver
#from mystic.solvers import PowellDirectionalSolver as TheSolver
from mystic.solvers import BuckshotSolver
#solver = BuckshotSolver(n, 10)
solver = TheSolver(n)
solver.SetRandomInitialPoints()
solver.SetStrictRanges(min=[0]*n, max=[5]*n)
solver.SetEvaluationLimits(evaluations=320000, generations=1000)
solver.SetTermination(term)

#from mystic.termination import state
#print state(solver._termination).keys()
solver.Solve(model, disp=verbose)

# while collapse and solver.Collapse(verbose):
#   solver.Solve(model)

# we are done; get result
print solver.Terminated(info=True)
print solver.bestEnergy, "@"
print solver.bestSolution



# EOF
