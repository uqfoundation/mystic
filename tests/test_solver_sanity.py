#!/usr/bin/env python
#
# Author: Alta Fang (altafang @caltech and alta @princeton)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""A sanity test suite for Mystic solvers."""
# should report clock-time, # of iterations, and # of function evaluations

import sys
PY3 = (sys.hexversion >= 0x30000f0)
if PY3:
    from io import StringIO
else:
    from StringIO import StringIO
import unittest
from math import *
from mystic.math import almostEqual

verbosity = 0 # Verbosity setting for unittests (default is 1).

def trap_stdout(): #XXX: better with contextmanager?
    "temporarily trap stdout; return original sys.stdout"
    orig, sys.stdout = sys.stdout, StringIO()
    return orig

def release_stdout(orig):
    "release stdout; return any trapped output as a string"
    out = sys.stdout.getvalue()
    sys.stdout.close()
    sys.stdout = orig
    return out


class TestRosenbrock(unittest.TestCase):
    """Test the 2-dimensional rosenbrock optimization problem."""

    def setUp(self):
        from mystic.models import rosen
        self.costfunction = rosen
        self.exact=[1., 1.]
        self.NP = 40
        self.ND = len(self.exact)
        self.min = [-5.]*self.ND
        self.max = [5.]*self.ND
        self.maxiter = my_maxiter
        self.maxfun = 100000
        self.precision = 0 # precision compared to exact
        self.usebounds = False
        self.uselimits = True#False
        self.useevalmon = True#False
        self.usestepmon = True#False

    def _run_solver(self, early_terminate=False, **kwds):
        from mystic.monitors import Monitor
        import numpy
        from mystic.tools import random_seed
        seed = 111 if self.maxiter is None else 321 #XXX: good numbers...
        random_seed(seed)
        esow = Monitor()
        ssow = Monitor() 

        solver = self.solver
        solver.SetRandomInitialPoints(min = self.min, max = self.max)
        if self.usebounds: solver.SetStrictRanges(self.min, self.max)
        if self.uselimits: solver.SetEvaluationLimits(self.maxiter, self.maxfun)
        if self.useevalmon: solver.SetEvaluationMonitor(esow)
        if self.usestepmon: solver.SetGenerationMonitor(ssow)
        #### run solver, but trap output
        _stdout = trap_stdout()
        solver.Solve(self.costfunction, self.term, **kwds)
        out = release_stdout(_stdout)
        ################################
        sol = solver.Solution()

        iter=1
       #if self.uselimits and self.maxiter == 0: iter=0
        # sanity check solver internals
        self.assertTrue(solver.generations == len(solver._stepmon._y)-iter)
        self.assertTrue(list(solver.bestSolution) == solver._stepmon.x[-1]) #XXX
        self.assertTrue(solver.bestEnergy == solver._stepmon.y[-1])
        self.assertTrue(solver.solution_history == solver._stepmon.x)
        self.assertTrue(solver.energy_history == solver._stepmon.y)
        if self.usestepmon:
            self.assertTrue(ssow.x == solver._stepmon.x)
            self.assertTrue(ssow.y == solver._stepmon.y)
            self.assertTrue(ssow._y == solver._stepmon._y)
        if self.useevalmon:
            self.assertTrue(solver.evaluations == len(solver._evalmon._y))
            self.assertTrue(esow.x == solver._evalmon.x)
            self.assertTrue(esow.y == solver._evalmon.y)
            self.assertTrue(esow._y == solver._evalmon._y)

        # Fail appropriately for solver/termination mismatch
        if early_terminate:
            self.assertTrue(solver.generations < 2)
            warn = "Warning: Invalid termination condition (nPop < 2)"
            self.assertTrue(warn in out)
            return

        g = solver.generations
        calls = [(g+1)*self.NP, (2*g)+1]
        iters = [g]
        # Test early terminations
        if self.uselimits and self.maxfun == 0:
            calls += [1, 20] #XXX: scipy*
            iters += [1]     #XXX: scipy*
            self.assertTrue(solver.evaluations in calls) 
            self.assertTrue(solver.generations in iters)
            return
        if self.uselimits and self.maxfun == 1:
            calls += [1, 20] #XXX: scipy*
            iters += [1]     #XXX: scipy*
            self.assertTrue(solver.evaluations in calls) 
            self.assertTrue(solver.generations in iters)
            return
        if self.uselimits and self.maxiter == 0:
            calls += [1, 20] #XXX: scipy*
            iters += [1]     #XXX: scipy*
            self.assertTrue(solver.evaluations in calls) 
            self.assertTrue(solver.generations in iters)
            return
        if self.uselimits and self.maxiter == 1:
            calls += [20] #Powell's
            self.assertTrue(solver.evaluations in calls) 
            self.assertTrue(solver.generations in iters)
            return
        if self.uselimits and self.maxiter and 2 <= self.maxiter <= 5:
            calls += [52, 79, 107, 141] #Powell's
            self.assertTrue(solver.evaluations in calls) 
            self.assertTrue(solver.generations in iters)
            return

        # Verify solution is close to exact
       #print(sol)
        for i in range(len(sol)):
            self.assertAlmostEqual(sol[i], self.exact[i], self.precision)
        return

    def test_DifferentialEvolutionSolver_VTR(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import VTR
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = VTR()
        self._run_solver()

    def test_DifferentialEvolutionSolver_COG(self): # Default for this solver
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import ChangeOverGeneration as COG
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = COG()
        self._run_solver()

    def test_DifferentialEvolutionSolver_NCOG(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = NCOG()
        self._run_solver()

    def test_DifferentialEvolutionSolver_CRT(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = CRT()
        self._run_solver()

#--------------------------------------------------------------------------

    def test_DifferentialEvolutionSolver2_VTR(self): 
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import VTR
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = VTR()
        self._run_solver()

    def test_DifferentialEvolutionSolver2_COG(self): # Default for this solver
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import ChangeOverGeneration as COG
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = COG()
        self._run_solver()

    def test_DifferentialEvolutionSolver2_NCOG(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = NCOG()
        self._run_solver()

    def test_DifferentialEvolutionSolver2_CRT(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import CandidateRelativeTolerance as CRT
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = CRT()
        self._run_solver()

#--------------------------------------------------------------

    def test_NelderMeadSimplexSolver_CRT(self): # Default for this solver
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = CRT()
        self._run_solver()

    def test_NelderMeadSimplexSolver_VTR(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import VTR
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = VTR()
        self._run_solver()

    def test_NelderMeadSimplexSolver_COG(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import ChangeOverGeneration as COG
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = COG()
        self._run_solver()

    def test_NelderMeadSimplexSolver_NCOG(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = NCOG()
        self._run_solver()

#--------------------------------------------------------------------------

    def test_PowellDirectionalSolver_NCOG(self): # Default for this solver
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = NCOG()
        self._run_solver()

    def test_PowellDirectionalSolver_COG(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import ChangeOverGeneration as COG
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = COG()
        self._run_solver()

    def test_PowellDirectionalSolver_VTR(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import VTR
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = VTR()
        self._run_solver()

    def test_PowellDirectionalSolver_CRT(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = CRT()
        self._run_solver(early_terminate=True)



if __name__ == '__main__':
    def prepare():
        suite1 = unittest.TestLoader().loadTestsFromTestCase(TestRosenbrock)
        allsuites = unittest.TestSuite([suite1])
        runner = unittest.TextTestRunner(verbosity=verbosity)
        return allsuites, runner

    allsuites, runner = prepare()
    my_maxiter = 0
    runner.run(allsuites)
    allsuites, runner = prepare()
    my_maxiter = 1
    runner.run(allsuites)
    allsuites, runner = prepare()
    my_maxiter = 2
    runner.run(allsuites)
    allsuites, runner = prepare()
    my_maxiter = None
    runner.run(allsuites)


# EOF
