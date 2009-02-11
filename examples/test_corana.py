#!/usr/bin/env python

"""
Sets up Corana's parabola. This is problem 6 of testbed 1 in [1].

Exact answer: Min = 0 @ abs(x_j) < 0.05 for all j.

Reference:

[1] Storn, R. and Price, K. Differential Evolution - A Simple and Efficient
Heuristic for Global Optimization over Continuous Spaces. Journal of Global
Optimization 11: 341-359, 1997.
"""

from mystic.differential_evolution import DifferentialEvolutionSolver
from mystic.detools import Best1Exp, Rand1Exp, ChangeOverGeneration, VTR

import random
random.seed(123)

d = [1., 1000., 10., 100.]

def Corana(x):
    """
The costfunction for Corana's parabola. Eq. (11) of [1]
    """
    from math import pow
    from numpy import sign, floor
    r = 0
    for j in range(4):
        zj =  floor( abs(x[j]/0.2) + 0.49999 ) * sign(x[j]) * 0.2
        if abs(x[j]-zj) < 0.05:
            r += 0.15 * pow(zj - 0.05*sign(zj), 2) * d[j]
        else:
            r += d[j] * x[j] * x[j]
    return r

def Corana1(x):
    return Corana([x[0], 0, 0, 0])

ND = 4
NP = 10
MAX_GENERATIONS = 2500

def main():
    solver = DifferentialEvolutionSolver(ND, NP)

    solver.SetRandomInitialPoints(min = [-1000]*ND, max = [1000]*ND)

    solver.Solve(Corana, Rand1Exp, termination = VTR(0.00000001) , \
                 maxiter= MAX_GENERATIONS, CrossProbability=0.5, ScalingFactor=0.9)

    solution = solver.Solution()
  
    print solution



if __name__ == '__main__':
    from timeit import Timer
    t = Timer("main()", "from __main__ import main")
    timetaken =  t.timeit(number=1)
    print "CPU Time: %s" % timetaken

    try:
        import scipy.optimize, random
        print  "\nScipy: "
        sol = scipy.optimize.fmin(Corana, [random.random() for j in range(4)], full_output=0, retall=1)
        print "solution: ", sol[-1][0]
        print "\nCorana 1 with Scipy"
        sol = scipy.optimize.fmin(Corana1, [random.random()], full_output=1, retall=1)
        print "solution: ", sol[-1][0]
    except:
        pass

# end of file
