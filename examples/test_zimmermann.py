#!/usr/bin/env python

"""
Sets up Zimmermann's problem. This is problem 8 of testbed 1 in [1] and [2].

Solution: Min of 0 @ Vector[0]

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

import random
random.seed(123)

# Eq. (24-26) of [2].
from mystic.models import zimmermann as CostFunction

ND = 2
NP = 20
MAX_GENERATIONS = 2500

def main():
    solver = DifferentialEvolutionSolver(ND, NP)

    solver.SetRandomInitialPoints(min = [0.]*ND, max = [5.]*ND)
    solver.SetEvaluationLimits(maxiter=MAX_GENERATIONS)

    solver.Solve(CostFunction, termination=VTR(0.0000001), strategy=Rand1Exp, \
                 CrossProbability=0.3, ScalingFactor=1.0)

    solution = solver.Solution()
  
    print solution



if __name__ == '__main__':
    from timeit import Timer
    t = Timer("main()", "from __main__ import main")
    timetaken =  t.timeit(number=1)
    print "CPU Time: %s" % timetaken

    from mystic.monitors import Monitor
    from mystic.solvers import NelderMeadSimplexSolver as fmin
    from mystic.termination import CandidateRelativeTolerance as CRT

    simplex = Monitor()
    esow = Monitor()
    xinit = [random.uniform(0,5) for j in range(ND)]

    solver = fmin(len(xinit))
    solver.SetInitialPoints(xinit)
    solver.Solve(CostFunction, CRT(), EvaluationMonitor = esow, StepMonitor = simplex)
    sol = solver.Solution()
    print "fmin solution: ", sol

# end of file
