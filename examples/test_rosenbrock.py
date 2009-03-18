#!/usr/bin/env python

"""
Testing Rosenbrock's Function.

This is a very popular function for testing minimization algorithm.

For direct searches, Nelder-Mead does very well on this problem. 

Run optimize.py in scipy.optimize to see a comparison of Nelder-Mead and a few other solvers
"""

from mystic.differential_evolution import DifferentialEvolutionSolver
from mystic.termination import ChangeOverGeneration, VTR
from mystic.models import rosen
from mystic.scipy_optimize import NelderMeadSimplexSolver as fmin
from mystic.termination import CandidateRelativeTolerance as CRT
from mystic import Sow

import random
random.seed(123)

ND = 6
NP = 60
MAX_GENERATIONS = 99999

def main():
    solver = DifferentialEvolutionSolver(ND, NP)

    solver.SetRandomInitialPoints(min = [0]*ND, max = [2]*ND)
    solver.SetEvaluationLimits(maxiter=MAX_GENERATIONS)

    solver.Solve(rosen, termination = VTR(0.0001), \
                 CrossProbability=0.5, ScalingFactor=0.6)

    solution = solver.Solution()
  
    print solution



if __name__ == '__main__':
    from numpy import inf
    print "without bounds..."
    from timeit import Timer
    t = Timer("main()", "from __main__ import main")
    timetaken =  t.timeit(number=1)
    print "CPU Time: %s\n" % timetaken

    print "with bounds..."
    import time
    times = []
    algor = []

    print "Differential Evolution"
    print "======================"
    start = time.time()
    esow= Sow()
    ssow= Sow()

#   xinit = [random.random() for j in range(ND)]
    xinit = [0.8,1.2,0.7]*2
   #xinit = [0.8,1.2,1.7]*2             #... better when using "bad" range
    min = [-0.999, -0.999, 0.999]*2     #XXX: behaves badly when large range
    max = [200.001, 100.001, inf]*2     #... for >=1 x0 out of bounds; (up xtol)
  # min = [-0.999, -0.999, -0.999]*2
  # max = [200.001, 100.001, inf]*2
 #  min = [-0.999, -0.999, 0.999]*2
 #  max = [2.001, 1.001, 1.001]*2

    solver = fmin(len(xinit))
    solver.SetInitialPoints(xinit)
    solver.SetStrictRanges(min,max)
    solver.Solve(rosen, CRT(), EvaluationMonitor = esow, StepMonitor = ssow)
    sol = solver.Solution()
    print sol
 
    times.append(time.time() - start)
    algor.append('Differential Evolution\t')

    for k in range(len(algor)):
        print algor[k], "\t -- took", times[k]

   #print len(esow.x)
   #print len(ssow.x)
    print "\nstep x:\n", ssow.x[1:10][0]
    print "\nstep y:\n", ssow.y[1:10][0]



# end of file
