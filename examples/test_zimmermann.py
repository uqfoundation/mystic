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

from mystic.differential_evolution import DifferentialEvolutionSolver
from mystic.detools import Best1Exp, Rand1Exp, ChangeOverGeneration, VTR

import random
random.seed(123)

def CostFunction(coeffs):
    """
Eq. (24-26) of [2].
    """
    x0, x1 = coeffs
    f8 = 9 - x0 - x1
    c0,c1,c2,c3 = 0,0,0,0
    if x0 < 0: c0 = -100 * x0
    if x1 < 0: c1 = -100 * x1
    xx =  (x0-3.)*(x0-3) + (x1-2.)*(x1-2)
    if xx > 16: c2 = 100 * (xx-16)
    if x0 * x1 > 14: c3 = 100 * (x0*x1-14.)
    return max(f8,c0,c1,c2,c3)



ND = 2
NP = 20
MAX_GENERATIONS = 2500

def main():
    solver = DifferentialEvolutionSolver(ND, NP)

    solver.SetRandomInitialPoints(min = [0.]*ND, max = [5.]*ND)

    solver.Solve(CostFunction, Rand1Exp, termination = VTR(0.0000001) , \
                 maxiter= MAX_GENERATIONS, CrossProbability=0.3, ScalingFactor=1.0)

    solution = solver.Solution()
  
    print solution



if __name__ == '__main__':
    from timeit import Timer
    t = Timer("main()", "from __main__ import main")
    timetaken =  t.timeit(number=1)
    print "CPU Time: %s" % timetaken

    from mystic import Sow
    from mystic.scipy_optimize_fmin import NelderMeadSimplexSolver as fmin
    from mystic.nmtools import IterationRelativeError as IRE

    simplex = Sow()
    esow = Sow()
    xinit = [random.uniform(0,5) for j in range(ND)]

    solver = fmin(len(xinit))
    solver.SetInitialPoints(xinit)
    solver.Solve(CostFunction, IRE(), EvaluationMonitor = esow, StepMonitor = simplex)
    sol = solver.Solution()
    print "fmin solution: ", sol

# end of file
