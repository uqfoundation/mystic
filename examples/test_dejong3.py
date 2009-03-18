#!/usr/bin/env python

"""
Sets up De Jong's Third function. This is problem 3 of testbed 1 in [1].

Note: The function as defined by Eq.8 of [1] seems incomplete.

Reference:

[1] Storn, R. and Price, K. Differential Evolution - A Simple and Efficient
Heuristic for Global Optimization over Continuous Spaces. Journal of Global
Optimization 11: 341-359, 1997.

[2] Storn, R. and Proce, K. Same title as above, but as a technical report.
try: http://www.icsi.berkeley.edu/~storn/deshort1.ps
"""

from mystic.differential_evolution import DifferentialEvolutionSolver
from mystic.termination import ChangeOverGeneration, VTR
from mystic.models.dejong import step as DeJong3

import random
random.seed(123)

ND = 5
NP = 25
MAX_GENERATIONS = 2500

def main():
    solver = DifferentialEvolutionSolver(ND, NP)

    solver.SetRandomInitialPoints(min = [-5.12]*ND, max = [5.12]*ND)
    solver.SetEvaluationLimits(maxiter=MAX_GENERATIONS)

    solver.Solve(DeJong3, termination=VTR(0.00001), \
                 CrossProbability=0.3, ScalingFactor=1.0)

    solution = solver.Solution()
  
    print solution



if __name__ == '__main__':
    from timeit import Timer

    # optimize with DESolver
    t = Timer("main()", "from __main__ import main")
    timetaken =  t.timeit(number=1)
    print "CPU Time: %s\n" % timetaken

    # optimize with fmin
    from mystic.scipy_optimize import fmin
    print fmin(DeJong3, [0 for i in range(ND)])

# end of file
