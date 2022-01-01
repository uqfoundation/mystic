#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Sets up De Jong's Fourth function. This is problem 4 of testbed 1 in [1].

This is function fitting "with noise."
Reference:

[1] Storn, R. and Price, K. Differential Evolution - A Simple and Efficient
Heuristic for Global Optimization over Continuous Spaces. Journal of Global
Optimization 11: 341-359, 1997.

[2] Storn, R. and Proce, K. Same title as above, but as a technical report.
try: http://www.icsi.berkeley.edu/~storn/deshort1.ps
"""

from mystic.solvers import DifferentialEvolutionSolver
from mystic.termination import ChangeOverGeneration, VTR
from mystic.strategy import Best1Exp, Rand1Exp
from mystic.models.dejong import quartic as DeJong4

from mystic.tools import random_seed
random_seed(123)

ND = 30
NP = 10
MAX_GENERATIONS = 2500

def main():
    solver = DifferentialEvolutionSolver(ND, NP)

    solver.SetRandomInitialPoints(min = [-1.28]*ND, max = [1.28]*ND)
    solver.SetEvaluationLimits(generations=MAX_GENERATIONS)

    solver.Solve(DeJong4, termination=VTR(15), strategy=Rand1Exp, \
                 CrossProbability=0.3, ScalingFactor=1.0)

    solution = solver.Solution()
  
    print(solution)



if __name__ == '__main__':
    from timeit import Timer

    # optimize with DESolver
    t = Timer("main()", "from __main__ import main")
    timetaken =  t.timeit(number=1)
    print("CPU Time: %s\n" % timetaken)

    # optimize with fmin
    from mystic.solvers import fmin
    print(fmin(DeJong4, [0 for i in range(ND)]))

# end of file
