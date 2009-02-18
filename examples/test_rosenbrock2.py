#!/usr/bin/env python

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
    print "Nelder-Mead Simplex"
    print "==================="
    start = time.time()
    from mystic.tools import VerboseSow
    stepmon = VerboseSow(10)
    from mystic.termination import IterationRelativeTolerance as IRT

    from mystic.scipy_optimize import fmin, NelderMeadSimplexSolver
   #print fmin(rosen,x0,retall=0,full_output=0)
    solver = NelderMeadSimplexSolver(len(x0))
    solver.SetInitialPoints(x0)
    solver.SetStrictRanges(min,max)
    solver.enable_signal_handler()
    solver.Solve(rosen,termination=IRT(xtol=1e-5),StepMonitor=stepmon)
    print solver.Solution()

    times.append(time.time() - start)
    algor.append('Nelder-Mead Simplex\t')

    for k in range(len(algor)):
        print algor[k], "\t -- took", times[k]

# end of file
