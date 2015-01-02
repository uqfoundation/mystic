#!/usr/bin/env python
#
# Author: Alta Fang (altafang @caltech and alta @princeton)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2015 California Institute of Technology.
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
term = VTR() 
#term = COG()
solver.Solve(peaks, term, disp=True)
sol = solver.Solution()
print 'solution = ', sol
print 'expected = [0.23, -1.63]'
