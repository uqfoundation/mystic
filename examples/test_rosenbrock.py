#!/usr/bin/env python

"""
Testing Rosenbrock's Function.

This is a very popular function for testing minimization algorithm.
The following provides tests for both bounded and unbounded minimization
of the Rosenbrock function with Differential Evolution.

For direct searches, Nelder-Mead does very well on this problem. 
For a direct comparison between DE and steepest-descent solvers, run
test_rosenbrock*.py (or optimize.py in scipy.optimize).
"""

from mystic.differential_evolution import DifferentialEvolutionSolver, diffev
from mystic.termination import ChangeOverGeneration, VTR
from mystic.models import rosen
from mystic import Sow, VerboseSow

import random
random.seed(123)

ND = 3
NP = 30
#MAX_GENERATIONS = 29
MAX_GENERATIONS = 99999

def main():
    solver = DifferentialEvolutionSolver(ND, NP)

    solver.SetRandomInitialPoints(min = [0]*ND, max = [2]*ND)
    solver.SetEvaluationLimits(maxiter=MAX_GENERATIONS)

    solver.Solve(rosen, termination = VTR(0.0001), \
                 CrossProbability=0.5, ScalingFactor=0.6, disp=1)

    solution = solver.Solution()
   #print "Current function value: %s" % solver.bestEnergy
   #print "Iterations: %s" % solver.generations
   #print "Function evaluations: %s" % solver.evaluations
  
    print solution



if __name__ == '__main__':
    from numpy import inf
    print "without bounds..."
    from timeit import Timer
    print "Differential Evolution"
    print "======================"
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
   #ssow= VerboseSow(1)

 #  xinit = [random.random() for j in range(ND)]
    xinit = [0.8,1.2,0.7]
  # xinit = [0.8,1.2,1.7]             #... better when using "bad" range
    min = [-0.999, -0.999, 0.999]     #XXX: behaves badly when large range
    max = [200.001, 100.001, inf]     #... for >=1 x0 out of bounds; (up xtol)
  # min = [-0.999, -0.999, -0.999]
  # max = [200.001, 100.001, inf]
 #  min = [-0.999, -0.999, 0.999]     #XXX: tight range and non-randomness
 #  max = [2.001, 1.001, 1.001]       #...: is _bad_ for DE solvers

   #print diffev(rosen,xinit,NP,retall=0,full_output=0)
    solver = DifferentialEvolutionSolver(len(xinit), NP)
    solver.SetInitialPoints(xinit)
    solver.SetStrictRanges(min,max)
    solver.SetEvaluationLimits(maxiter=MAX_GENERATIONS)
    solver.Solve(rosen, VTR(0.0001), EvaluationMonitor=esow, StepMonitor=ssow,\
                 CrossProbability=0.5, ScalingFactor=0.6, disp=1)
    sol = solver.Solution()
    print sol
   #print "Current function value: %s" % solver.bestEnergy
   #print "Iterations: %s" % solver.generations
   #print "Function evaluations: %s" % solver.evaluations
 
    times.append(time.time() - start)
    algor.append('Differential Evolution\t')

    for k in range(len(algor)):
        print algor[k], "\t -- took", times[k]

   #print len(esow.x)
   #print len(ssow.x)
   #print "\nstep x:\n", ssow.x[1:10][0]
   #print "\nstep y:\n", ssow.y[1:10][0]



# end of file
