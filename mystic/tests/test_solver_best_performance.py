#!/usr/bin/env python
#
# Author: Alta Fang (altafang @caltech and alta @princeton)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""Test Mystic's performance on some benchmark problems, with Mystic's settings
adjusted to achieve the best results.
"""
from mystic.monitors import Monitor
from mystic.math import almostEqual
from mystic.tools import random_seed
random_seed(123)
import time

verbose = False


def test_rosenbrock(verbose=False):
    """Test the 2-dimensional Rosenbrock function.

Testing 2-D Rosenbrock:
Expected: x=[1., 1.] and f=0

Using DifferentialEvolutionSolver:
Solution:  [ 1.00000037  1.0000007 ]
f value:  2.29478683682e-13
Iterations:  99
Function evaluations:  3996
Time elapsed:  0.582273006439  seconds

Using DifferentialEvolutionSolver2:
Solution:  [ 0.99999999  0.99999999]
f value:  3.84824937598e-15
Iterations:  100
Function evaluations:  4040
Time elapsed:  0.577210903168  seconds

Using NelderMeadSimplexSolver:
Solution:  [ 0.99999921  1.00000171]
f value:  1.08732211477e-09
Iterations:  70
Function evaluations:  130
Time elapsed:  0.0190329551697  seconds

Using PowellDirectionalSolver:
Solution:  [ 1.  1.]
f value:  0.0
Iterations:  28
Function evaluations:  859
Time elapsed:  0.113857030869  seconds
"""

    if verbose:
        print("Testing 2-D Rosenbrock:")
        print("Expected: x=[1., 1.] and f=0")
    from mystic.models import rosen as costfunc
    ndim = 2
    lb = [-5.]*ndim
    ub = [5.]*ndim
    x0 = [2., 3.]
    maxiter = 10000
    
    # DifferentialEvolutionSolver
    if verbose: print("\nUsing DifferentialEvolutionSolver:")
    npop = 40
    from mystic.solvers import DifferentialEvolutionSolver
    from mystic.termination import ChangeOverGeneration as COG
    from mystic.strategy import Rand1Bin
    esow = Monitor()
    ssow = Monitor() 
    solver = DifferentialEvolutionSolver(ndim, npop)
    solver.SetInitialPoints(x0)
    solver.SetStrictRanges(lb, ub)
    solver.SetEvaluationLimits(generations=maxiter)
    solver.SetEvaluationMonitor(esow)
    solver.SetGenerationMonitor(ssow)
    term = COG(1e-10)
    time1 = time.time() # Is this an ok way of timing?
    solver.Solve(costfunc, term, strategy=Rand1Bin)
    sol = solver.Solution()
    time_elapsed = time.time() - time1
    fx = solver.bestEnergy
    if verbose:
        print("Solution: %s" % sol)
        print("f value: %s" % fx)
        print("Iterations: %s" % solver.generations)
        print("Function evaluations: %s" % len(esow.x))
        print("Time elapsed: %s seconds" % time_elapsed)
    assert almostEqual(fx, 2.29478683682e-13, tol=3e-3)

    # DifferentialEvolutionSolver2
    if verbose: print("\nUsing DifferentialEvolutionSolver2:")
    npop = 40
    from mystic.solvers import DifferentialEvolutionSolver2
    from mystic.termination import ChangeOverGeneration as COG
    from mystic.strategy import Rand1Bin
    esow = Monitor()
    ssow = Monitor() 
    solver = DifferentialEvolutionSolver2(ndim, npop)
    solver.SetInitialPoints(x0)
    solver.SetStrictRanges(lb, ub)
    solver.SetEvaluationLimits(generations=maxiter)
    solver.SetEvaluationMonitor(esow)
    solver.SetGenerationMonitor(ssow)
    term = COG(1e-10)
    time1 = time.time() # Is this an ok way of timing?
    solver.Solve(costfunc, term, strategy=Rand1Bin)
    sol = solver.Solution()
    time_elapsed = time.time() - time1
    fx = solver.bestEnergy
    if verbose:
        print("Solution: %s" % sol)
        print("f value: %s" % fx)
        print("Iterations: %s" % solver.generations)
        print("Function evaluations: %s" % len(esow.x))
        print("Time elapsed: %s seconds" % time_elapsed)
    assert almostEqual(fx, 3.84824937598e-15, tol=3e-3)

    # NelderMeadSimplexSolver
    if verbose: print("\nUsing NelderMeadSimplexSolver:")
    from mystic.solvers import NelderMeadSimplexSolver
    from mystic.termination import CandidateRelativeTolerance as CRT
    esow = Monitor()
    ssow = Monitor() 
    solver = NelderMeadSimplexSolver(ndim)
    solver.SetInitialPoints(x0)
    solver.SetStrictRanges(lb, ub)
    solver.SetEvaluationLimits(generations=maxiter)
    solver.SetEvaluationMonitor(esow)
    solver.SetGenerationMonitor(ssow)
    term = CRT()
    time1 = time.time() # Is this an ok way of timing?
    solver.Solve(costfunc, term)
    sol = solver.Solution()
    time_elapsed = time.time() - time1
    fx = solver.bestEnergy
    if verbose:
        print("Solution: %s" % sol)
        print("f value: %s" % fx)
        print("Iterations: %s" % solver.generations)
        print("Function evaluations: %s" % len(esow.x))
        print("Time elapsed: %s seconds" % time_elapsed)
    assert almostEqual(fx, 1.08732211477e-09, tol=3e-3)

    # PowellDirectionalSolver
    if verbose: print("\nUsing PowellDirectionalSolver:")
    from mystic.solvers import PowellDirectionalSolver
    from mystic.termination import NormalizedChangeOverGeneration as NCOG
    esow = Monitor()
    ssow = Monitor() 
    solver = PowellDirectionalSolver(ndim)
    solver.SetInitialPoints(x0)
    solver.SetStrictRanges(lb, ub)
    solver.SetEvaluationLimits(generations=maxiter)
    solver.SetEvaluationMonitor(esow)
    solver.SetGenerationMonitor(ssow)
    term = NCOG(1e-10)
    time1 = time.time() # Is this an ok way of timing?
    solver.Solve(costfunc, term)
    sol = solver.Solution()
    time_elapsed = time.time() - time1
    fx = solver.bestEnergy
    if verbose:
        print("Solution: %s" % sol)
        print("f value: %s" % fx)
        print("Iterations: %s" % solver.generations)
        print("Function evaluations: %s" % len(esow.x))
        print("Time elapsed: %s seconds" % time_elapsed)
    assert almostEqual(fx, 0.0, tol=3e-3)

def test_griewangk(verbose=False):
    """Test Griewangk's function, which has many local minima.

Testing Griewangk:
Expected: x=[0.]*10 and f=0

Using DifferentialEvolutionSolver:
Solution:  [  8.87516194e-09   7.26058147e-09   1.02076001e-08   1.54219038e-08
  -1.54328461e-08   2.34589663e-08   2.02809360e-08  -1.36385836e-08
   1.38670373e-08   1.59668900e-08]
f value:  0.0
Iterations:  4120
Function evaluations:  205669
Time elapsed:  34.4936850071  seconds

Using DifferentialEvolutionSolver2:
Solution:  [ -2.02709316e-09   3.22017968e-09   1.55275472e-08   5.26739541e-09
  -2.18490470e-08   3.73725584e-09  -1.02315312e-09   1.24680355e-08
  -9.47898116e-09   2.22243557e-08]
f value:  0.0
Iterations:  4011
Function evaluations:  200215
Time elapsed:  32.8412370682  seconds
"""

    if verbose:
        print("Testing Griewangk:")
        print("Expected: x=[0.]*10 and f=0")
    from mystic.models import griewangk as costfunc
    ndim = 10
    lb = [-400.]*ndim
    ub = [400.]*ndim
    maxiter = 10000
    seed = 123 # Re-seed for each solver to have them all start at same x0
    
    # DifferentialEvolutionSolver
    if verbose: print("\nUsing DifferentialEvolutionSolver:")
    npop = 50
    random_seed(seed)
    from mystic.solvers import DifferentialEvolutionSolver
    from mystic.termination import ChangeOverGeneration as COG
    from mystic.termination import CandidateRelativeTolerance as CRT
    from mystic.termination import VTR
    from mystic.strategy import Rand1Bin, Best1Bin, Rand1Exp
    esow = Monitor()
    ssow = Monitor() 
    solver = DifferentialEvolutionSolver(ndim, npop)
    solver.SetRandomInitialPoints(lb, ub)
    solver.SetStrictRanges(lb, ub)
    solver.SetEvaluationLimits(generations=maxiter)
    solver.SetEvaluationMonitor(esow)
    solver.SetGenerationMonitor(ssow)
    solver.enable_signal_handler()
    #term = COG(1e-10)
    #term = CRT()
    term = VTR(0.)
    time1 = time.time() # Is this an ok way of timing?
    solver.Solve(costfunc, term, strategy=Rand1Exp, \
                 CrossProbability=0.3, ScalingFactor=1.0)
    sol = solver.Solution()
    time_elapsed = time.time() - time1
    fx = solver.bestEnergy
    if verbose:
        print("Solution: %s" % sol)
        print("f value: %s" % fx)
        print("Iterations: %s" % solver.generations)
        print("Function evaluations: %s" % len(esow.x))
        print("Time elapsed: %s seconds" % time_elapsed)
    assert almostEqual(fx, 0.0, tol=3e-3)

    # DifferentialEvolutionSolver2
    if verbose: print("\nUsing DifferentialEvolutionSolver2:")
    npop = 50
    random_seed(seed)
    from mystic.solvers import DifferentialEvolutionSolver2
    from mystic.termination import ChangeOverGeneration as COG
    from mystic.termination import CandidateRelativeTolerance as CRT
    from mystic.termination import VTR
    from mystic.strategy import Rand1Bin, Best1Bin, Rand1Exp
    esow = Monitor()
    ssow = Monitor() 
    solver = DifferentialEvolutionSolver2(ndim, npop)
    solver.SetRandomInitialPoints(lb, ub)
    solver.SetStrictRanges(lb, ub)
    solver.SetEvaluationLimits(generations=maxiter)
    solver.SetEvaluationMonitor(esow)
    solver.SetGenerationMonitor(ssow)
    #term = COG(1e-10)
    #term = CRT()
    term = VTR(0.)
    time1 = time.time() # Is this an ok way of timing?
    solver.Solve(costfunc, term, strategy=Rand1Exp, \
                 CrossProbability=0.3, ScalingFactor=1.0)
    sol = solver.Solution()
    time_elapsed = time.time() - time1
    fx = solver.bestEnergy
    if verbose:
        print("Solution: %s" % sol)
        print("f value: %s" % fx)
        print("Iterations: %s" % solver.generations)
        print("Function evaluations: %s" % len(esow.x))
        print("Time elapsed: %s seconds" % time_elapsed)
    assert almostEqual(fx, 0.0, tol=3e-3)

if __name__ == '__main__':
    test_rosenbrock(verbose)
   #test_griewangk(verbose)
