#!/usr/bin/env python
#
# Author: Alta Fang (altafang @caltech and alta @princeton)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2014 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE
"""
Testing the 'Peaks" Function.

(tests VTR when minimum has negative value)
"""
from math import *

def peaks(x_vector):
    """The peaks function. Optimize on the box [-3, 3]x[-3, 3]. Has 
several local minima and one global minimum at (0.23, -1.63) where 
the function value is about -6.55.

Source: http://www.nag.co.uk/numeric/FL/nagdoc_fl22/xhtml/E05/e05jbf.xml, 
example 9."""
    x = x_vector[0]
    y = x_vector[1]
    result = 3.*(1. - x)**2*exp(-x**2 - (y + 1.)**2) - \
            10.*(x*(1./5.) - x**3 - y**5)*exp(-x**2 - y**2) - \
            1./3.*exp(-(x + 1.)**2 - y**2)
    return result

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
