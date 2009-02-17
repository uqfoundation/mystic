#!/usr/bin/env python

"""
Sets up Griewangk's function. This is problem 7 of testbed 1 in [1].

Solution: Min of 0 @ Vector[0]

Reference:

[1] Storn, R. and Price, K. Differential Evolution - A Simple and Efficient
Heuristic for Global Optimization over Continuous Spaces. Journal of Global
Optimization 11: 341-359, 1997.
"""

from mystic.differential_evolution import DifferentialEvolutionSolver
from mystic.termination import ChangeOverGeneration, VTR
from mystic.strategy import Best1Exp, Rand1Exp

import random
random.seed(123)

# The costfunction for Griewangk's Function, Eq. (23) of [1].
from mystic.models import griewangk as Griewangk_cost


ND = 10
NP = ND*10
MAX_GENERATIONS = 2500

def main():
    solver = DifferentialEvolutionSolver(ND, NP)

    solver.SetRandomInitialPoints(min = [-400.0]*ND, max = [400.0]*ND)

    solver.Solve(Griewangk_cost, Rand1Exp, termination = VTR(0.00001) , \
                 maxiter= MAX_GENERATIONS, CrossProbability=0.3, ScalingFactor=1.0)

    solution = solver.Solution()
  
    print solution



if __name__ == '__main__':
    from mystic.scipy_optimize_fmin import fmin
    from timeit import Timer
    t = Timer("main()", "from __main__ import main")
    timetaken =  t.timeit(number=1)
    print "CPU Time: %s" % timetaken

    import random
    print "Scipy fmin"
    for i in [400,200,100,40,20,10,4,2,1]:
        print "\ninitializing with range (-%d, %d)" % (i,i)
        sol = fmin(Griewangk_cost, [random.uniform(-i,i) for j in range(10)])
        print "sol: ", sol
        print "cost: ", Griewangk_cost(sol)

# end of file
