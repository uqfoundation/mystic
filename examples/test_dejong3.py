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
from mystic.detools import Best1Exp, Rand1Exp, ChangeOverGeneration, VTR

import random
random.seed(123)

def DeJong3(coeffs):
    """
The costfunction for the third De Jong function Eq. (8) of [1] / Eq. (19) of [2].
    """
    from math import floor
    f = 30.
    for c in coeffs:
        if abs(c) <= 5.12: 
            f += floor(c)
        elif c > 5.12:
            f += 30 * (c - 5.12)
        else:
            f += 30 * (5.12 - c)
    return f

ND = 5
NP = 25
MAX_GENERATIONS = 2500

def main():
    solver = DifferentialEvolutionSolver(ND, NP)

    solver.SetRandomInitialPoints(min = [-5.12]*ND, max = [5.12]*ND)

    solver.Solve(DeJong3, Best1Exp, termination = VTR(0.00001) , \
                 maxiter= MAX_GENERATIONS, CrossProbability=0.3, ScalingFactor=1.0)

    solution = solver.Solution()
  
    print solution



if __name__ == '__main__':
    from timeit import Timer
    import random

    # optimize with DESolver
    t = Timer("main()", "from __main__ import main")
    timetaken =  t.timeit(number=1)
    print "CPU Time: %s\n" % timetaken

    # optimize with fmin
    from mystic.scipy_optimize_fmin import fmin
    print fmin(DeJong3, [0 for i in range(ND)])

# end of file
