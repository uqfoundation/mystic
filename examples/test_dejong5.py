#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Sets up De Jong's Fifth function. This is problem 5 of testbed 1 in [1].

Exact answer: Min = 0 @ (-32, -32)

Reference:

[1] Storn, R. and Price, K. Differential Evolution - A Simple and Efficient
Heuristic for Global Optimization over Continuous Spaces. Journal of Global
Optimization 11: 341-359, 1997.
"""

from mystic.solvers import DifferentialEvolutionSolver
from mystic.termination import ChangeOverGeneration, VTR
from mystic.strategy import Best1Exp, Rand1Exp
from mystic.models.dejong import shekel as DeJong5

from mystic.tools import random_seed
random_seed(123)

ND = 2
NP = 15
MAX_GENERATIONS = 2500

def main():
    solver = DifferentialEvolutionSolver(ND, NP)

    solver.SetRandomInitialPoints(min = [-65.536]*ND, max = [65.536]*ND)
    solver.SetEvaluationLimits(generations=MAX_GENERATIONS)

    solver.Solve(DeJong5, termination=VTR(0.0000001), strategy=Rand1Exp, \
                 CrossProbability=0.5, ScalingFactor=0.9)

    solution = solver.Solution()
  
    print(solution)



if __name__ == '__main__':
    from timeit import Timer
    # optimize with DESolver
    t = Timer("main()", "from __main__ import main")
    timetaken =  t.timeit(number=1)
    print("CPU Time: %s\n" % timetaken)

# end of file
