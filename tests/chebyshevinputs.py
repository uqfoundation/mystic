#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                       Patrick Hung & Mike McKerns, Caltech
#                        (C) 1998-2008  All Rights Reserved
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

"""
chebyshevinputs.py -- cost function container module for NelderMeadSimplexSolver 
and PowellDirectionalSolver for testsolvers_pyre.py
"""

from mystic.models.poly import chebyshev8cost as cost
from mystic.models.poly import chebyshev8coeffs
from mystic.termination import *

ND = 9
maxiter = 999

from numpy import inf
import random
random.seed(123)

x0 = [random.uniform(-5,5) + chebyshev8coeffs[i] for i in range(ND)]

# used with SetStrictRanges
min_bounds = [  0,-1,-300,-1,  0,-1,-100,-inf,-inf]
max_bounds = [200, 1,   0, 1,200, 1,   0, inf, inf]

termination = CandidateRelativeTolerance()
#termination = VTR()
#termination = ChangeOverGeneration()
#termination = NormalizedChangeOverGeneration()

# End of file
