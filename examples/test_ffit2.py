#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Sets up Storn and Price's Polynomial 'Fitting' Problem for ChebyshevT16

Reference:

[1] Storn, R. and Price, K. Differential Evolution - A Simple and Efficient
Heuristic for Global Optimization over Continuous Spaces. Journal of Global
Optimization 11: 341-359, 1997.
"""

from mystic.solvers import DifferentialEvolutionSolver
from mystic.termination import ChangeOverGeneration, VTR
from mystic.strategy import Best1Exp, Best1Bin, Rand1Exp, Best2Bin, Best2Exp
from mystic.math import poly1d
from mystic.monitors import VerboseMonitor
from mystic.models.poly import chebyshev16cost

from mystic.tools import random_seed
random_seed(123)

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
    solver.SetEvaluationLimits(generations=MAX_GENERATIONS)
    solver.SetGenerationMonitor(VerboseMonitor(10))
  
    #strategy = Best1Exp
    #strategy = Best1Bin
    #strategy = Best2Bin
    strategy = Best2Exp

    solver.Solve(ChebyshevCost, termination=VTR(0.0001), strategy=strategy, \
                 CrossProbability=1.0, ScalingFactor=0.6)

    solution = solver.Solution()
  
    print("\nsolved: ")
    print(poly1d(solution))
    print("\ntarget: ")
    print(poly1d(Chebyshev16))
   #print("actual coefficients vs computed:")
   #for actual,computed in zip(Chebyshev16, solution):
   #    print("%f %f" % (actual, computed))

    plot_solution(solution, Chebyshev16)


if __name__ == '__main__':
    from timeit import Timer
    t = Timer("main()", "from __main__ import main")
    timetaken =  t.timeit(number=1)
    print("\nCPU Time: %s" % timetaken)

# end of file
