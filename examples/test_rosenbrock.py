#!/usr/bin/env python

"""
Testing Rosenbrock's Function.

This is a very popular function for testing minimization algorithm.

For direct searches, Nelder-Mead does very well on this problem. 

Run optimize.py in scipy.optimize to see a comparison of Nelder-Mead and a few other solvers
"""

from mystic.differential_evolution import DifferentialEvolutionSolver
from mystic.detools import Best1Exp, Rand1Exp, Best2Bin, ChangeOverGeneration, VTR
from mystic.scipy_optimize_fmin import __rosen as rosen
from mystic.scipy_optimize_fmin import NelderMeadSimplexSolver as fmin
from mystic.nmtools import IterationRelativeError as IRE
from mystic import Sow

import random
random.seed(123)

ND = 6
NP = 60
MAX_GENERATIONS = 99999

def main():
    solver = DifferentialEvolutionSolver(ND, NP)

    solver.SetRandomInitialPoints(min = [0]*ND, max = [2]*ND)

    solver.Solve(rosen, Best1Exp, termination = VTR(0.0001) , \
                 maxiter= MAX_GENERATIONS, CrossProbability=0.5, ScalingFactor=0.6)

    solution = solver.Solution()
  
    print solution



if __name__ == '__main__':
    from timeit import Timer
    t = Timer("main()", "from __main__ import main")
    timetaken =  t.timeit(number=1)
    print "CPU Time: %s\n" % timetaken

    esow= Sow()
    ssow= Sow()
    xinit = [random.random() for j in range(ND)]

    solver = fmin(len(xinit))
    solver.SetInitialPoints(xinit)
    solver.Solve(rosen, IRE(), EvaluationMonitor = esow, StepMonitor = ssow)
    sol = solver.Solution()
    print sol
 
   #print len(esow.x)
   #print len(ssow.x)
    print "\nstep x:\n", ssow.x[1:10][0]
    print "\nstep y:\n", ssow.y[1:10][0]



# end of file
