#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Sets up Corana's parabola. This is problem 6 of testbed 1 in [1].

Exact answer: Min = 0 @ abs(x_j) < 0.05 for all j.

Reference:

[1] Storn, R. and Price, K. Differential Evolution - A Simple and Efficient
Heuristic for Global Optimization over Continuous Spaces. Journal of Global
Optimization 11: 341-359, 1997.
"""

from mystic.solvers import DifferentialEvolutionSolver
from mystic.termination import ChangeOverGeneration, VTR
from mystic.strategy import Best1Exp, Rand1Exp

from mystic.tools import random_seed
random_seed(123)

from mystic.models import corana
from mystic.models.storn import Corana as Corana1
corana1 = Corana1(1)

ND = 4
NP = 10
MAX_GENERATIONS = 2500

def main():
    solver = DifferentialEvolutionSolver(ND, NP)

    solver.SetRandomInitialPoints(min = [-1000]*ND, max = [1000]*ND)
    solver.SetEvaluationLimits(generations=MAX_GENERATIONS)

    solver.Solve(corana, termination=VTR(0.00000001), strategy=Rand1Exp,\
                 CrossProbability=0.5, ScalingFactor=0.9)

    solution = solver.Solution()
  
    print(solution)



if __name__ == '__main__':
    from timeit import Timer
    t = Timer("main()", "from __main__ import main")
    timetaken =  t.timeit(number=1)
    print("CPU Time: %s" % timetaken)

    try:
        from mystic.solvers import fmin
       #from mystic._scipyoptimize import fmin
        import random
        print( "\nScipy: ")
        sol = fmin(corana, [random.random() for j in range(4)], full_output=0, retall=1)
        print("solution: %s" % sol[-1][0])
        print("\nCorana 1 with Scipy")
        sol = fmin(corana1, [random.random()], full_output=1, retall=1)
        print("solution: %s" % sol[-1][0])
    except:
        pass

# end of file
