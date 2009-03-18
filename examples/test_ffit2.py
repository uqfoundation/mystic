#!/usr/bin/env python

"""
Sets up Storn and Price's Polynomial 'Fitting' Problem for ChebyshevT16

Reference:

[1] Storn, R. and Price, K. Differential Evolution - A Simple and Efficient
Heuristic for Global Optimization over Continuous Spaces. Journal of Global
Optimization 11: 341-359, 1997.
"""

from mystic.differential_evolution import DifferentialEvolutionSolver
from mystic.termination import ChangeOverGeneration, VTR
from mystic.strategy import Best1Exp, Best1Bin, Rand1Exp, Best2Bin, Best2Exp
from mystic.models.poly import poly1d
from mystic import VerboseSow
from mystic.models.poly import chebyshev16cost

import random
random.seed(123)

# get the target coefficients
from mystic.models.poly import chebyshev16coeffs as Chebyshev16

def ChebyshevCost(trial):
    """
The costfunction for the fitting problem.
70 evaluation points between [-1, 1] with two end points
    """
    return chebyshev16cost(trial,M=70)*100


ND = 17
NP = 100
MAX_GENERATIONS = 5000

from test_ffit import plot_solution

def main():
    solver = DifferentialEvolutionSolver(ND, NP)
    solver.SetRandomInitialPoints()
    solver.SetEvaluationLimits(maxiter=MAX_GENERATIONS)
  
    #strategy = Best1Exp
    #strategy = Best1Bin
    #strategy = Best2Bin
    strategy = Best2Exp

    solver.Solve(ChebyshevCost, termination=VTR(0.0001), strategy=strategy, \
                 StepMonitor=VerboseSow(10), CrossProbability=1.0, ScalingFactor=0.6)

    solution = solver.Solution()
  
    print "\nsolved: "
    print poly1d(solution)
    print "\ntarget: "
    print poly1d(Chebyshev16)
   #print "actual coefficients vs computed:"
   #for actual,computed in zip(Chebyshev16, solution):
   #    print "%f %f" % (actual, computed)

    plot_solution(solution, Chebyshev16)


if __name__ == '__main__':
    from timeit import Timer
    t = Timer("main()", "from __main__ import main")
    timetaken =  t.timeit(number=1)
    print "\nCPU Time: %s" % timetaken

# end of file
