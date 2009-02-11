#!/usr/bin/env python

"""
Sets up De Jong's Fifth function. This is problem 5 of testbed 1 in [1].

Exact answer: Min = 0 @ (-32, -32)

Reference:

[1] Storn, R. and Price, K. Differential Evolution - A Simple and Efficient
Heuristic for Global Optimization over Continuous Spaces. Journal of Global
Optimization 11: 341-359, 1997.
"""

from mystic.differential_evolution import DifferentialEvolutionSolver
from mystic.detools import Best1Exp, Rand1Exp, ChangeOverGeneration, VTR

import random
random.seed(123)

A = [-32., -16., 0., 16., 32.] 
a1 = A * 5
a2 = reduce(lambda x1,x2: x1+x2, [[c] * 5 for c in A])

def DeJong5(coeffs):
    """
The costfunction for the Modified third De Jong function Eq. (8) of [1].
    """
    from math import pow
    x,y=coeffs
    r = 0.0
    for i in range(25):
        r += 1.0/ (1.0*i + pow(x-a1[i],6) + pow(y-a2[i],6) + 1e-15)
    return 1.0/(0.002 + r)


ND = 2
NP = 15
MAX_GENERATIONS = 2500

def main():
    solver = DifferentialEvolutionSolver(ND, NP)

    solver.SetRandomInitialPoints(min = [-65.536]*ND, max = [65.536]*ND)

    solver.Solve(DeJong5, Rand1Exp, termination = VTR(0.0000001) , \
                 maxiter= MAX_GENERATIONS, CrossProbability=0.5, ScalingFactor=0.9)

    solution = solver.Solution()
  
    print solution



if __name__ == '__main__':
    from timeit import Timer
    # optimize with DESolver
    t = Timer("main()", "from __main__ import main")
    timetaken =  t.timeit(number=1)
    print "CPU Time: %s\n" % timetaken

# end of file
