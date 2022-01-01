#!/usr/bin/env python
#
# Author: Alta Fang (altafang @caltech and alta @princeton)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
example of using DifferentialEvolutionSolver on the rosenbrock function
"""

from mystic.models import rosen
import numpy

def constrain(x):
  x[1] = x[0]
  return x

if __name__=='__main__':
    import time

    times = []
    algor = []
    x0 = [0.8,1.2,0.7]
   #x0 = [0.8,1.2,1.7]                  #... better when using "bad" range
    min = [-0.999, -0.999, 0.999]       #XXX: behaves badly when large range
    max = [200.001, 100.001, numpy.inf] #... for >=1 x0 out of bounds; (up xtol)
  # min = [-0.999, -0.999, -0.999]
  # max = [200.001, 100.001, numpy.inf]
 #  min = [-0.999, -0.999, 0.999]
 #  max = [2.001, 1.001, 1.001]
    npop = 5*len(x0)
    print("Differential Evolution")
    print("======================")
    start = time.time()
    from mystic.monitors import Monitor, VerboseMonitor
    stepmon = VerboseMonitor(1,1)
   #stepmon = Monitor() #VerboseMonitor(10)
    from mystic.termination import NormalizedChangeOverGeneration as NCOG

   #from mystic.solvers import diffev, DifferentialEvolutionSolver
    from mystic.solvers import diffev2, DifferentialEvolutionSolver2
   #print(diffev2(rosen,x0,npop,retall=0,full_output=0)#,maxiter=14))
    solver = DifferentialEvolutionSolver2(len(x0), npop)
    solver.SetInitialPoints(x0)
    solver.SetStrictRanges(min,max)
   #solver.SetEvaluationLimits(generations=13)
    solver.SetGenerationMonitor(stepmon)
    solver.SetConstraints(constrain)
    solver.enable_signal_handler()
    solver.Solve(rosen, NCOG(tolerance=1e-4), disp=1)
    print(solver.bestSolution)
   #print("Current function value: %s" % solver.bestEnergy)
   #print("Iterations: %s" % solver.generations)
   #print("Function evaluations: %s" % solver.evaluations)

    times.append(time.time() - start)
    algor.append("Differential Evolution\t")

    for k,t in zip(algor,times):
        print("%s\t -- took %s" % (k, t))

# end of file
