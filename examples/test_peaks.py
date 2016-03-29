#!/usr/bin/env python
#
# Author: Alta Fang (altafang @caltech and alta @princeton)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE
"""
Testing the 'Peaks" Function.

(tests VTR when minimum has negative value)
"""
from math import *
from mystic.models import peaks

nd = 2
npop = 20
tol = 0.05
lb = [-3.]*nd
ub = [3.]*nd

from mystic.tools import random_seed
#random_seed(123)
from mystic.differential_evolution import DifferentialEvolutionSolver
from mystic.termination import VTR
from mystic.termination import ChangeOverGeneration as COG

solver = DifferentialEvolutionSolver(nd, npop)
solver.SetRandomInitialPoints(lb, ub)
solver.SetStrictRanges(lb, ub)
term = VTR(tol) 
#term = COG()
solver.Solve(peaks, term, disp=True)
sol = solver.Solution()
print 'solution = ', sol
print 'expected = [0.23, -1.63]'

try:
    from scipy.stats import uniform
except ImportError:
    exit()

print '-'*60
print 'using a uniform distribution...'
solver = DifferentialEvolutionSolver(nd, npop)
solver.SetDistributionInitialPoints(uniform(lb[0],ub[0]))
solver.SetStrictRanges(lb, ub)
term = VTR(tol) 
#term = COG()
solver.Solve(peaks, term, disp=True)
sol = solver.Solution()
print 'solution = ', sol
print 'expected = [0.23, -1.63]'



