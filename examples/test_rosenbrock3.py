#!/usr/bin/env python

"""
example of using PowellDirectionalSolver on the rosenbrock function
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
    print "Powell Direction Set Method"
    print "==========================="
    start = time.time()
    from mystic.tools import Sow, VerboseSow
    stepmon = VerboseSow(1,1)
   #stepmon = Sow() #VerboseSow(10)
    from mystic.termination import NormalizedChangeOverGeneration as NCOG

   #from scipy.optimize import fmin_powell
    from mystic.solvers import fmin_powell, PowellDirectionalSolver
   #print fmin_powell(rosen,x0,retall=0,full_output=0)#,maxiter=14)
    solver = PowellDirectionalSolver(len(x0))
    solver.SetInitialPoints(x0)
    solver.SetStrictRanges(min,max)
   #solver.SetEvaluationLimits(maxiter=13)
    solver.enable_signal_handler()
    solver.Solve(rosen,termination=NCOG(tolerance=1e-4),StepMonitor=stepmon,disp=1, constraints=constrain)
    print solver.Solution()
   #print "Current function value: %s" % solver.bestEnergy
   #print "Iterations: %s" % solver.generations
   #print "Function evaluations: %s" % solver.evaluations

    times.append(time.time() - start)
    algor.append("Powell's Method\t")

    for k in range(len(algor)):
        print algor[k], "\t -- took", times[k]

# end of file
