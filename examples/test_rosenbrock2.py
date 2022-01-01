#!/usr/bin/env python
#
# Author: Alta Fang (altafang @caltech and alta @princeton)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
example of using NelderMeadSimplexSolver on the rosenbrock function
"""

from mystic.models import rosen
import numpy

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
    print("Nelder-Mead Simplex")
    print("===================")
    start = time.time()
    from mystic.monitors import Monitor, VerboseMonitor
   #stepmon = VerboseMonitor(1)
    stepmon = Monitor() #VerboseMonitor(10)
    from mystic.termination import CandidateRelativeTolerance as CRT

   #from mystic._scipyoptimize import fmin
    from mystic.solvers import fmin, NelderMeadSimplexSolver
   #print(fmin(rosen,x0,retall=0,full_output=0,maxiter=121))
    solver = NelderMeadSimplexSolver(len(x0))
    solver.SetInitialPoints(x0)
    solver.SetStrictRanges(min,max)
    solver.SetEvaluationLimits(generations=146)
    solver.SetGenerationMonitor(stepmon)
    solver.enable_signal_handler()
    solver.Solve(rosen, CRT(xtol=4e-5), disp=1)
    print(solver.bestSolution)
   #print("Current function value: %s" % solver.bestEnergy)
   #print("Iterations: %s" % solver.generations)
   #print("Function evaluations: %s" % solver.evaluations)

    times.append(time.time() - start)
    algor.append('Nelder-Mead Simplex\t')

    for k,t in zip(algor,times):
        print("%s\t -- took %s" % (k, t))

# end of file
