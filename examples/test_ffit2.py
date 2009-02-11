#!/usr/bin/env python

"""
Sets up Storn and Price's Polynomial 'Fitting' Problem for ChebyshevT16

Reference:

[1] Storn, R. and Price, K. Differential Evolution - A Simple and Efficient
Heuristic for Global Optimization over Continuous Spaces. Journal of Global
Optimization 11: 341-359, 1997.
"""

from mystic.differential_evolution import DifferentialEvolutionSolver
from mystic.detools import Best1Exp, Best1Bin, Rand1Exp, Best2Bin, Best2Exp, ChangeOverGeneration, VTR
from mystic.polytools import polyeval
from mystic.polytools import coefficients_to_polynomial as poly1d
from mystic import VerboseSow

import random
random.seed(123)

# These are the target coefficients
Chebyshev16 = [32768., 0., -131072., 0., 212992., 0., -180224., 0., 84480., 0., -21504., 0., 2688., 0., -128., 0., 1]

def ChebyshevCost(trial):
    """
The costfunction for the fitting problem.

100 evaluation points between [-1, 1] and two end points
    """
    M=70 # number of evaluation points between [-1, 1]

    result=0.0

    x=-1.0
    dx = 2.0 / (M-1)
    for i in range(M):
        px = polyeval(trial, x)
        if px<-1 or px>1:
            result += (1 - px) * (1 - px)
        x += dx

    px = polyeval(trial, 1.2) - 10558.1450229
    if px<0: result += px*px

    px = polyeval(trial, -1.2) - 10558.1450229
    if px<0: result += px*px

    return result*100



ND = 17
NP = 100
MAX_GENERATIONS = 5000

from test_ffit import plot_solution

def main():
    solver = DifferentialEvolutionSolver(ND, NP)

    solver.SetRandomInitialPoints(min = [-1000.0]*ND, max = [1000.0]*ND)
  
    #strategy = Best1Exp
    #strategy = Best1Bin
    #strategy = Best2Bin
    strategy = Best2Exp

    solver.Solve(ChebyshevCost, strategy, termination = VTR(0.0001) , \
                 StepMonitor=VerboseSow(1), maxiter= MAX_GENERATIONS, CrossProbability=1.0, ScalingFactor=0.6)

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
