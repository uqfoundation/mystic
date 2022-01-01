#!/usr/bin/env python
#
# Author: Alta Fang (altafang @caltech and alta @princeton)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""A test suite for Mystic solvers.
Note: VTR termination with default tolerance shouldn't work for functions 
whose value at the minimum is negative!
Also, the two differential evolution solvers are global, while the other solvers
are local optimizers."""
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

disp = False  # Flag for whether to display number of iterations 
              #  and function evaluations.
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


class TestZimmermann(unittest.TestCase):
    """Test the zimmermann optimization problem."""

    def setUp(self):
        from mystic.models import zimmermann
        self.costfunction = zimmermann
        self.expected=[7., 2.]
        self.ND = len(self.expected)
        self.min = [0.]*self.ND
        self.max = [5.]*self.ND
        self.maxiter = 2500
        self.nplaces = 0 # Precision of answer
        self.local = [ 2.35393787,  5.94748068] # local minimum

    def _run_solver(self, iter_limit=False, local=False, **kwds):
        from mystic.monitors import Monitor
        from mystic.tools import random_seed
        random_seed(321) # Number of failures is quite dependent on random seed!
        esow = Monitor()
        ssow = Monitor() 

        solver = self.solver
        solver.SetRandomInitialPoints(min = self.min, max = self.max)
        solver.SetEvaluationLimits(generations=self.maxiter)
        solver.SetEvaluationMonitor(esow)
        solver.SetGenerationMonitor(ssow)
        _stdout = trap_stdout()
        solver.Solve(self.costfunction, self.term, **kwds)
        out = release_stdout(_stdout)
        sol = solver.Solution()
        #print('\nsol: %s' % sol)

        if disp:
            # Print the number of iterations and function evaluations
            iters = len(ssow.x)
            func_evals = len(esow.x)
            print("\nNumber of iterations = %s" % iters)
            print("Number of function evaluations = %s" % func_evals)

        # If solver should terminate immediately, check for that only.
        if iter_limit:
            self.assertTrue(solver.generations < 2)
            warn = "Warning: Invalid termination condition (nPop < 2)"
            self.assertTrue(warn in out)
            return

        # Allow success if local solvers find the local or global minimum
        if local:
            tol = 1. # Tolerance for almostEqual.
            for i in range(len(sol)):
                self.assertTrue(almostEqual(sol[i], self.local[i], tol=tol) or \
                                almostEqual(sol[i], self.expected[i], tol=tol))
            return

        # Verify solution is close to expected
        for i in range(len(sol)):
            self.assertAlmostEqual(sol[i], self.expected[i], self.nplaces)

    def test_DifferentialEvolutionSolver_VTR(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = VTR()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_COG(self): # Default for this solver
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = COG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_NCOG(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = NCOG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_CRT(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = CRT()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

#--------------------------------------------------------------------------

    def test_DifferentialEvolutionSolver2_VTR(self): 
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import VTR
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = VTR()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_COG(self): # Default for this solver
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = COG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_NCOG(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = NCOG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_CRT(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = CRT()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

#--------------------------------------------------------------------------

    def test_NelderMeadSimplexSolver_CRT(self): # Default for this solver
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = CRT()#tol)
        self._run_solver(local=True)

    def test_NelderMeadSimplexSolver_VTR(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = VTR()#tol)
        self._run_solver(local=True)

    def test_NelderMeadSimplexSolver_COG(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = COG()#tol)
        self._run_solver(local=True)

    def test_NelderMeadSimplexSolver_NCOG(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = NCOG()#tol)
        self._run_solver(local=True)

#--------------------------------------------------------------------------

    def test_PowellDirectionalSolver_NCOG(self): # Default for this solver
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = NCOG()#tol)
        self._run_solver(local=True)

    def test_PowellDirectionalSolver_COG(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = COG()#tol)
        self._run_solver(local=True)

    def test_PowellDirectionalSolver_VTR(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = VTR()#tol)
        self._run_solver(local=True)

    def test_PowellDirectionalSolver_CRT(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = CRT()#tol)
        self._run_solver(iter_limit=True, local=True)

#####################################################################


class TestRosenbrock(unittest.TestCase):
    """Test the 2-dimensional rosenbrock optimization problem."""

    def setUp(self):
        from mystic.models import rosen
        self.costfunction = rosen
        self.expected=[1., 1.]
        self.ND = len(self.expected)
        self.usebounds = False
        self.min = [-5.]*self.ND
        self.max = [5.]*self.ND
        self.maxiter = 10000
        self.nplaces = 0 # Precision of answer

    def _run_solver(self, iter_limit=False, **kwds):
        from mystic.monitors import Monitor
        from mystic.tools import random_seed
        #random_seed(123)
        esow = Monitor()
        ssow = Monitor() 

        solver = self.solver
        solver.SetRandomInitialPoints(min = self.min, max = self.max)
        if self.usebounds:
            solver.SetStrictRanges(self.min, self.max)
        solver.SetEvaluationLimits(generations=self.maxiter)
        solver.SetEvaluationMonitor(esow)
        solver.SetGenerationMonitor(ssow)
        _stdout = trap_stdout()
        solver.Solve(self.costfunction, self.term, **kwds)
        out = release_stdout(_stdout)
        sol = solver.Solution()

        if disp:
            # Print the number of iterations and function evaluations
            iters = len(ssow.x) #XXX Should use solver.generations instead?
            func_evals = len(esow.x)
            print("\nNumber of iterations = %s" % iters)
            print("Number of function evaluations = %s" % func_evals)

        # If solver should terminate immediately, check for that only.
        if iter_limit:
            self.assertTrue(solver.generations < 2)
            warn = "Warning: Invalid termination condition (nPop < 2)"
            self.assertTrue(warn in out)
            return

        # Verify solution is close to expected
        for i in range(len(sol)):
            self.assertAlmostEqual(sol[i], self.expected[i], self.nplaces)

    def test_DifferentialEvolutionSolver_VTR(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = VTR()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_COG(self): # Default for this solver
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = COG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_NCOG(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = NCOG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_CRT(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = CRT()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

#--------------------------------------------------------------------------

    def test_DifferentialEvolutionSolver2_VTR(self): 
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import VTR
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = VTR()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_COG(self): # Default for this solver
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = COG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_NCOG(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = NCOG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_CRT(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = CRT()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

#--------------------------------------------------------------

    def test_NelderMeadSimplexSolver_CRT(self): # Default for this solver
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = CRT()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_VTR(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = VTR()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_COG(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = COG()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_NCOG(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = NCOG()#tol)
        self._run_solver()

#--------------------------------------------------------------------------

    def test_PowellDirectionalSolver_NCOG(self): # Default for this solver
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = NCOG()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_COG(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = COG()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_VTR(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = VTR()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_CRT(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = CRT()#tol)
        self._run_solver(iter_limit=True)

#####################################################################


class TestCorana(unittest.TestCase):
    """Test the 4-dimensional Corana optimization problem. Many local
minima."""

    def setUp(self):
        from mystic.models import corana
        self.costfunction = corana
        self.ND = 4
        self.maxexpected=[0.05]*self.ND
        self.min = [-1000]*self.ND
        self.max = [1000]*self.ND
        self.maxiter = 10000
        #self.nplaces = 0 # Precision of answer

    def _run_solver(self, iter_limit=False, **kwds):
        from mystic.monitors import Monitor
        from mystic.tools import random_seed
        random_seed(123)
        esow = Monitor()
        ssow = Monitor() 

        solver = self.solver
        solver.SetRandomInitialPoints(self.min, self.max)
        solver.SetStrictRanges(self.min, self.max)
        solver.SetEvaluationLimits(generations=self.maxiter)
        solver.SetEvaluationMonitor(esow)
        solver.SetGenerationMonitor(ssow)
        _stdout = trap_stdout()
        solver.Solve(self.costfunction, self.term, **kwds)
        out = release_stdout(_stdout)
        sol = solver.Solution()

        if disp:
            # Print the number of iterations and function evaluations
            iters = len(ssow.x)
            func_evals = len(esow.x)
            print("\nNumber of iterations = %s" % iters)
            print("Number of function evaluations = %s" % func_evals)

        # If solver should terminate immediately, check for that only.
        if iter_limit:
            self.assertTrue(solver.generations < 2)
            warn = "Warning: Invalid termination condition (nPop < 2)"
            self.assertTrue(warn in out)
            return

        # Verify solution is close to expected (here, absolute value is within the
        # inequality) 
        error = 1. # Allowed error in either direction
        for i in range(len(sol)):
            self.assertTrue(abs(sol[i]) < self.maxexpected[i] + error)

    def test_DifferentialEvolutionSolver_VTR(self): 
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = VTR()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_COG(self): # Default for this solver
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = COG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_NCOG(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = NCOG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_CRT(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = CRT()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

#--------------------------------------------------------------------------

    def test_DifferentialEvolutionSolver2_VTR(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import VTR
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = VTR()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_COG(self): # Default for this solver
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = COG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_NCOG(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = NCOG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_CRT(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = CRT()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

#--------------------------------------------------------------

    def test_NelderMeadSimplexSolver_CRT(self): # Default for this solver
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = CRT()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_VTR(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = VTR()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_COG(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = COG()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_NCOG(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = NCOG()#tol)
        self._run_solver()

#--------------------------------------------------------------------------

    def test_PowellDirectionalSolver_NCOG(self): # Default for this solver
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = NCOG()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_COG(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = COG()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_VTR(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = VTR()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_CRT(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = CRT()#tol)
        self._run_solver(iter_limit=True)

#####################################################################


class TestQuartic(unittest.TestCase):
    """Test the quartic (noisy) optimization problem."""

    def setUp(self):
        from mystic.models import quartic
        self.costfunction = quartic
        self.ND = 30
        self.expected=[0.]*self.ND
        self.min = [-1.28]*self.ND
        self.max = [1.28]*self.ND
        self.maxiter = 2500
        self.nplaces = 0 # Precision of answer 

    def _run_solver(self, iter_limit=False, **kwds):
        from mystic.monitors import Monitor
        from mystic.tools import random_seed
        #random_seed(123)
        esow = Monitor()
        ssow = Monitor() 

        solver = self.solver
        solver.SetRandomInitialPoints(self.min, self.max)
        solver.SetStrictRanges(self.min, self.max)
        solver.SetEvaluationLimits(generations=self.maxiter)
        solver.SetEvaluationMonitor(esow)
        solver.SetGenerationMonitor(ssow)
        _stdout = trap_stdout()
        solver.Solve(self.costfunction, self.term, **kwds)
        out = release_stdout(_stdout)
        sol = solver.Solution()

        if disp:
            # Print the number of iterations and function evaluations
            iters = len(ssow.x)
            func_evals = len(esow.x)
            print("\nNumber of iterations = %s" % iters)
            print("Number of function evaluations = %s" % func_evals)

        # If solver should terminate immediately, check for that only.
        if iter_limit:
            self.assertTrue(solver.generations < 2)
            warn = "Warning: Invalid termination condition (nPop < 2)"
            self.assertTrue(warn in out)
            return

        # Verify solution is close to expected
        for i in range(len(sol)):
            self.assertAlmostEqual(sol[i], self.expected[i], self.nplaces)

    def test_DifferentialEvolutionSolver_VTR(self): 
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = VTR()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_COG(self): # Default for this solver
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = COG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_NCOG(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = NCOG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_CRT(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = CRT()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

#--------------------------------------------------------------------------

    def test_DifferentialEvolutionSolver2_VTR(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import VTR
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = VTR()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_COG(self): # Default for this solver
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = COG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_NCOG(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = NCOG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_CRT(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = CRT()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

#--------------------------------------------------------------

    def test_NelderMeadSimplexSolver_CRT(self): # Default for this solver
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = CRT()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_VTR(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = VTR()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_COG(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = COG()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_NCOG(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = NCOG()#tol)
        self._run_solver()

#--------------------------------------------------------------------------

    def test_PowellDirectionalSolver_NCOG(self): # Default for this solver
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = NCOG()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_COG(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = COG()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_VTR(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = VTR()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_CRT(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = CRT()#tol)
        self._run_solver(iter_limit=True)

#####################################################################


class TestShekel(unittest.TestCase):
    """Test the 5th DeJong function (Shekel) optimization problem."""

    def setUp(self):
        from mystic.models import shekel
        self.costfunction = shekel
        self.ND = 2
        self.expected=[-32., -32.]
        self.min = [-65.536]*self.ND
        self.max = [65.536]*self.ND
        self.maxiter = 10000
        self.nplaces = 0 # Precision of answer

    def _run_solver(self, iter_limit=False, **kwds):
        from mystic.monitors import Monitor
        from mystic.tools import random_seed
        random_seed(123)
        esow = Monitor()
        ssow = Monitor() 

        solver = self.solver
        solver.SetRandomInitialPoints(self.min, self.max)
        solver.SetStrictRanges(self.min, self.max)
        solver.SetEvaluationLimits(generations=self.maxiter)
        solver.SetEvaluationMonitor(esow)
        solver.SetGenerationMonitor(ssow)
        _stdout = trap_stdout()
        solver.Solve(self.costfunction, self.term, **kwds)
        out = release_stdout(_stdout)
        sol = solver.Solution()

        if disp:
            # Print the number of iterations and function evaluations
            iters = len(ssow.x)
            func_evals = len(esow.x)
            print("\nNumber of iterations = %s" % iters)
            print("Number of function evaluations = %s" % func_evals)

        # If solver should terminate immediately, check for that only.
        if iter_limit:
            self.assertTrue(solver.generations < 2)
            warn = "Warning: Invalid termination condition (nPop < 2)"
            self.assertTrue(warn in out)
            return

        # Verify solution is close to expected
        for i in range(len(sol)):
            self.assertAlmostEqual(sol[i], self.expected[i], self.nplaces)

    def test_DifferentialEvolutionSolver_VTR(self): 
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = VTR()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_COG(self): # Default for this solver
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = COG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_NCOG(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = NCOG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_CRT(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = CRT()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

#--------------------------------------------------------------------------

    def test_DifferentialEvolutionSolver2_VTR(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import VTR
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = VTR()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_COG(self): # Default for this solver
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = COG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_NCOG(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = NCOG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_CRT(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = CRT()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

#--------------------------------------------------------------

    def test_NelderMeadSimplexSolver_CRT(self): # Default for this solver
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = CRT()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_VTR(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = VTR()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_COG(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = COG()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_NCOG(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = NCOG()#tol)
        self._run_solver()

#--------------------------------------------------------------------------

    def test_PowellDirectionalSolver_NCOG(self): # Default for this solver
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = NCOG()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_COG(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = COG()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_VTR(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = VTR()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_CRT(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = CRT()#tol)
        self._run_solver(iter_limit=True)


#####################################################################


class TestStep(unittest.TestCase):
    """Test the 3rd DeJong function (step) optimization problem."""

    def setUp(self):
        from mystic.models import step
        self.costfunction = step
        self.ND = 5
        self.expected = [-5.]*self.ND # xi=-5-n where n=[0.0,0.12]
        self.min = [-5.12]*self.ND
        self.max = [5.12]*self.ND
        self.maxiter = 10000
        #self.nplaces = 0 # Precision of answer

    def _run_solver(self, iter_limit = False, **kwds):
        from mystic.monitors import Monitor
        from mystic.tools import random_seed
       #random_seed(123)
        esow = Monitor()
        ssow = Monitor() 

        solver = self.solver
        solver.SetRandomInitialPoints(self.min, self.max)
        solver.SetStrictRanges(self.min, self.max)
        solver.SetEvaluationLimits(generations=self.maxiter)
        solver.SetEvaluationMonitor(esow)
        solver.SetGenerationMonitor(ssow)
        _stdout = trap_stdout()
        solver.Solve(self.costfunction, self.term, **kwds)
        out = release_stdout(_stdout)
        sol = solver.Solution()

        if disp:
            # Print the number of iterations and function evaluations
            iters = len(ssow.x)
            func_evals = len(esow.x)
            print("\nNumber of iterations = %s" % iters)
            print("Number of function evaluations = %s" % func_evals)

        # If solver should terminate immediately, check for that only.
        if iter_limit:
            self.assertTrue(solver.generations < 2)
            warn = "Warning: Invalid termination condition (nPop < 2)"
            self.assertTrue(warn in out)
            return

        # Expected: xi=-5-n where n=[0.0,0.12]
        #XXX Again, no cushion.
        for i in range(len(sol)):
            self.assertTrue(sol[i] > self.expected[i] - 0.12 or sol[i] < self.expected[i])

    def test_DifferentialEvolutionSolver_VTR(self): 
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = VTR()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_COG(self): # Default for this solver
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = COG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_NCOG(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = NCOG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_CRT(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = CRT()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

#--------------------------------------------------------------------------

    def test_DifferentialEvolutionSolver2_VTR(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import VTR
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = VTR()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_COG(self): # Default for this solver
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = COG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_NCOG(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = NCOG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_CRT(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = CRT()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

#--------------------------------------------------------------

    def test_NelderMeadSimplexSolver_CRT(self): # Default for this solver
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = CRT()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_VTR(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = VTR()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_COG(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = COG()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_NCOG(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = NCOG()#tol)
        self._run_solver()

#--------------------------------------------------------------------------

    def test_PowellDirectionalSolver_NCOG(self): # Default for this solver
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = NCOG()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_COG(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = COG()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_VTR(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = VTR()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_CRT(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = CRT()#tol)
        self._run_solver(iter_limit=True)


#####################################################################


class TestGriewangk(unittest.TestCase):
    """Test the Griewangk optimization problem."""

    def setUp(self):
        from mystic.models import griewangk
        self.costfunction = griewangk
        self.ND = 10
        self.expected = [0.]*self.ND 
        self.min = [-400.0]*self.ND
        self.max = [400.0]*self.ND
        self.maxiter = 2500
        self.nplaces = 0 # Precision of answer

    def _run_solver(self, iter_limit=False, **kwds):
        from mystic.monitors import Monitor
        from mystic.tools import random_seed
        #random_seed(123)
        esow = Monitor()
        ssow = Monitor() 

        solver = self.solver
        solver.SetRandomInitialPoints(self.min, self.max)
        solver.SetEvaluationLimits(generations=self.maxiter)
        solver.SetStrictRanges(self.min, self.max)
        solver.SetEvaluationMonitor(esow)
        solver.SetGenerationMonitor(ssow)
        _stdout = trap_stdout()
        solver.Solve(self.costfunction, self.term, **kwds)
        out = release_stdout(_stdout)
        sol = solver.Solution()

        if disp:
            # Print the number of iterations and function evaluations
            iters = len(ssow.x)
            func_evals = len(esow.x)
            print("\nNumber of iterations = %s" % iters)
            print("Number of function evaluations = %s" % func_evals)

        # If solver should terminate immediately, check for that only.
        if iter_limit:
            self.assertTrue(solver.generations < 2)
            warn = "Warning: Invalid termination condition (nPop < 2)"
            self.assertTrue(warn in out)
            return

        # Expected: xi=-5-n where n=[0.0,0.12]
        #XXX Again, no cushion.
        for i in range(len(sol)):
            self.assertTrue(sol[i] > self.expected[i] - 0.12 or sol[i] < self.expected[i])

    def test_DifferentialEvolutionSolver_VTR(self): 
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = VTR()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_COG(self): # Default for this solver
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = COG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_NCOG(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = NCOG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_CRT(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = CRT()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

#--------------------------------------------------------------------------

    def test_DifferentialEvolutionSolver2_VTR(self): 
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import VTR
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = VTR()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_COG(self): # Default for this solver
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = COG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_NCOG(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = NCOG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_CRT(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = CRT()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

#-------------------------------------------------------------------------

    def test_NelderMeadSimplexSolver_CRT(self): # Default for this solver
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = CRT()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_VTR(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = VTR()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_COG(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = COG()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_NCOG(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = NCOG()#tol)
        self._run_solver()

#--------------------------------------------------------------------------

    def test_PowellDirectionalSolver_NCOG(self): # Default for this solver
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = NCOG()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_COG(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = COG()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_VTR(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = VTR()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_CRT(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = CRT()#tol)
        self._run_solver(iter_limit=True)


#####################################################################

class TestPeaks(unittest.TestCase):
    """Test the peaks optimization problem."""

    def setUp(self):
        from mystic.models import peaks
        self.costfunction = peaks
        self.ND = 2
        self.expected = [0.23, -1.63]
        self.min = [-3.0]*self.ND
        self.max = [3.0]*self.ND
        self.maxiter = 2500
        self.nplaces = 0 # Precision of answer

    def _run_solver(self, iter_limit=False, **kwds):
        from mystic.monitors import Monitor
        from mystic.tools import random_seed
        #random_seed(123)
        esow = Monitor()
        ssow = Monitor() 

        solver = self.solver
        solver.SetRandomInitialPoints(self.min, self.max)
        solver.SetEvaluationLimits(generations=self.maxiter)
        solver.SetStrictRanges(self.min, self.max)
        solver.SetEvaluationMonitor(esow)
        solver.SetGenerationMonitor(ssow)
        _stdout = trap_stdout()
        solver.Solve(self.costfunction, self.term, **kwds)
        out = release_stdout(_stdout)
        sol = solver.Solution()

        if disp:
            # Print the number of iterations and function evaluations
            iters = len(ssow.x)
            func_evals = len(esow.x)
            print("\nNumber of iterations = %s" % iters)
            print("Number of function evaluations = %s" % func_evals)

        # If solver should terminate immediately, check for that only.
        if iter_limit:
            self.assertTrue(solver.generations < 2)
            warn = "Warning: Invalid termination condition (nPop < 2)"
            self.assertTrue(warn in out)
            return

        # Verify solution is close to expected
        for i in range(len(sol)):
            self.assertAlmostEqual(sol[i], self.expected[i], self.nplaces)

    def test_DifferentialEvolutionSolver_VTR(self): 
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = VTR()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_COG(self): # Default for this solver
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = COG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_NCOG(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = NCOG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_CRT(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = CRT()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

#--------------------------------------------------------------------------

    def test_DifferentialEvolutionSolver2_VTR(self): 
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import VTR
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = VTR()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_COG(self): # Default for this solver
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = COG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_NCOG(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = NCOG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_CRT(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = CRT()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

#-------------------------------------------------------------------------

    def test_NelderMeadSimplexSolver_CRT(self): # Default for this solver
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = CRT()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_VTR(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = VTR()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_COG(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = COG()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_NCOG(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = NCOG()#tol)
        self._run_solver()

#--------------------------------------------------------------------------

    def test_PowellDirectionalSolver_NCOG(self): # Default for this solver
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = NCOG()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_COG(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = COG()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_VTR(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = VTR()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_CRT(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = CRT()#tol)
        self._run_solver(iter_limit=True)


#---------------------------------------------------------------------------------

##############################################################################

class TestVenkataraman91(unittest.TestCase):
    """Test Venkataraman's sinc optimization problem."""

    def setUp(self):
        from mystic.models import venkat91
        self.costfunction = venkat91
        self.ND = 2
        self.expected = [4., 4.]
        self.min = [-10.0]*self.ND
        self.max = [10.0]*self.ND
        self.maxiter = 2500
        self.nplaces = 0 # Precision of answer

    def _run_solver(self, iter_limit=False, **kwds):
        from mystic.monitors import Monitor
        from mystic.tools import random_seed
        #random_seed(123)
        esow = Monitor()
        ssow = Monitor() 

        solver = self.solver
        solver.SetRandomInitialPoints(self.min, self.max)
        solver.SetEvaluationLimits(generations=self.maxiter)
        #solver.SetStrictRanges(self.min, self.max)
        solver.SetEvaluationMonitor(esow)
        solver.SetGenerationMonitor(ssow)
        _stdout = trap_stdout()
        solver.Solve(self.costfunction, self.term, **kwds)
        out = release_stdout(_stdout)
        sol = solver.Solution()

        if disp:
            # Print the number of iterations and function evaluations
            iters = len(ssow.x)
            func_evals = len(esow.x)
            print("\nNumber of iterations = %s" % iters)
            print("Number of function evaluations = %s" % func_evals)

        # Verify solution is close to expected
        for i in range(len(sol)):
            self.assertAlmostEqual(sol[i], self.expected[i], self.nplaces)

        # If solver should terminate immediately, check for that only.
        if iter_limit:
            self.assertTrue(solver.generations < 2)
            warn = "Warning: Invalid termination condition (nPop < 2)"
            self.assertTrue(warn in out)
            return

    def test_DifferentialEvolutionSolver_VTR(self): 
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = VTR()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_COG(self): # Default for this solver
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = COG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_NCOG(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = NCOG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_CRT(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = CRT()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

#--------------------------------------------------------------------------

    def test_DifferentialEvolutionSolver2_VTR(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import VTR
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = VTR()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_COG(self): # Default for this solver
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = COG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_NCOG(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = NCOG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_CRT(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = CRT()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

#-------------------------------------------------------------------------

    def test_NelderMeadSimplexSolver_CRT(self): # Default for this solver
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = CRT()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_VTR(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = VTR()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_COG(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = COG()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_NCOG(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = NCOG()#tol)
        self._run_solver()

#--------------------------------------------------------------------------

    def test_PowellDirectionalSolver_NCOG(self): # Default for this solver
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = NCOG()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_COG(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = COG()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_VTR(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = VTR()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_CRT(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = CRT()#tol)
        self._run_solver(iter_limit=True)


##############################################################################

class TestSchwefel(unittest.TestCase):
    """Test Schwefel's optimization problem."""

    def setUp(self):
        from mystic.models import schwefel
        self.costfunction = schwefel
        self.ND = 2
        self.expected = [420.9687]*self.ND
        self.min = [-500.0]*self.ND
        self.max = [500.0]*self.ND
        self.maxiter = 10000
        self.nplaces = -1 # Precision of answer

    def _run_solver(self, iter_limit=False, **kwds):
        from mystic.monitors import Monitor
        from mystic.tools import random_seed
        random_seed(123)
        esow = Monitor()
        ssow = Monitor() 

        solver = self.solver
        solver.SetRandomInitialPoints(self.min, self.max)
        solver.SetEvaluationLimits(generations=self.maxiter)
        solver.SetStrictRanges(self.min, self.max)
        solver.SetEvaluationMonitor(esow)
        solver.SetGenerationMonitor(ssow)
        _stdout = trap_stdout()
        solver.Solve(self.costfunction, self.term, **kwds)
        out = release_stdout(_stdout)
        sol = solver.Solution()

        if disp:
            # Print the number of iterations and function evaluations
            iters = len(ssow.x)
            func_evals = len(esow.x)
            print("\nNumber of iterations = %s" % iters)
            print("Number of function evaluations = %s" % func_evals)

        # If solver should terminate immediately, check for that only.
        if iter_limit:
            self.assertTrue(solver.generations < 2)
            warn = "Warning: Invalid termination condition (nPop < 2)"
            self.assertTrue(warn in out)
            return

        # Verify solution is close to expected
        for i in range(len(sol)):
            self.assertAlmostEqual(sol[i], self.expected[i], self.nplaces)

    def test_DifferentialEvolutionSolver_VTR(self): 
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = VTR()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_COG(self): # Default for this solver
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = COG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_NCOG(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = NCOG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_CRT(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = CRT()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

#--------------------------------------------------------------------------

    def test_DifferentialEvolutionSolver2_VTR(self): 
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import VTR
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = VTR()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_COG(self): # Default for this solver
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = COG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_NCOG(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = NCOG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_CRT(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = CRT()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

#-------------------------------------------------------------------------

    def test_NelderMeadSimplexSolver_CRT(self): # Default for this solver
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = CRT()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_VTR(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = VTR()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_COG(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = COG()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_NCOG(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = NCOG()#tol)
        self._run_solver()

#--------------------------------------------------------------------------

    def test_PowellDirectionalSolver_NCOG(self): # Default for this solver
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = NCOG()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_COG(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = COG()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_VTR(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = VTR()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_CRT(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = CRT()#tol)
        self._run_solver(iter_limit=True)


##############################################################################

class TestEasom(unittest.TestCase):
    """Test Easom's function."""

    def setUp(self):
        from mystic.models import easom
        self.costfunction = easom
        self.ND = 2
        self.expected = [pi]*self.ND
        self.min = [-100.]*self.ND
        self.max = [100.]*self.ND
        self.maxiter = 10000
        self.nplaces = -1 # Precision of answer

    def _run_solver(self, iter_limit=False, **kwds):
        from mystic.monitors import Monitor
        from mystic.tools import random_seed
        random_seed(123)
        esow = Monitor()
        ssow = Monitor() 

        solver = self.solver
        solver.SetRandomInitialPoints(self.min, self.max)
        solver.SetEvaluationLimits(generations=self.maxiter)
        solver.SetStrictRanges(self.min, self.max)
        solver.SetEvaluationMonitor(esow)
        solver.SetGenerationMonitor(ssow)
        _stdout = trap_stdout()
        solver.Solve(self.costfunction, self.term, **kwds)
        out = release_stdout(_stdout)
        sol = solver.Solution()

        if disp:
            # Print the number of iterations and function evaluations
            iters = len(ssow.x)
            func_evals = len(esow.x)
            print("\nNumber of iterations = %s" % iters)
            print("Number of function evaluations = %s" % func_evals)

        # If solver should terminate immediately, check for that only.
        if iter_limit:
            self.assertTrue(solver.generations < 2)
            warn = "Warning: Invalid termination condition (nPop < 2)"
            self.assertTrue(warn in out)
            return

        # Verify solution is close to expected
        for i in range(len(sol)):
            self.assertAlmostEqual(sol[i], self.expected[i], self.nplaces)

    def test_DifferentialEvolutionSolver_VTR(self): 
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = VTR()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_COG(self): # Default for this solver
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = COG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_NCOG(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = NCOG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_CRT(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = CRT()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

#--------------------------------------------------------------------------

    def test_DifferentialEvolutionSolver2_VTR(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import VTR
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = VTR()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_COG(self): # Default for this solver
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = COG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_NCOG(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = NCOG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_CRT(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = CRT()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

#-------------------------------------------------------------------------

    def test_NelderMeadSimplexSolver_CRT(self): # Default for this solver
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = CRT()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_VTR(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = VTR()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_COG(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = COG()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_NCOG(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = NCOG()#tol)
        self._run_solver()

#--------------------------------------------------------------------------

    def test_PowellDirectionalSolver_NCOG(self): # Default for this solver
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = NCOG()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_COG(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = COG()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_VTR(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = VTR()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_CRT(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = CRT()#tol)
        self._run_solver(iter_limit=True)


##############################################################################

class TestRotatedEllipsoid(unittest.TestCase):
    """Test the rotated ellipsoid function in 2 dimensions."""

    def setUp(self):
        from mystic.models import ellipsoid
        self.costfunction = ellipsoid
        self.ND = 2
        self.expected = [0.]*self.ND
        self.min = [-65.536]*self.ND
        self.max = [65.536]*self.ND
        self.maxiter = 10000
        self.nplaces = -1 # Precision of answer

    def _run_solver(self, iter_limit=False, **kwds):
        from mystic.monitors import Monitor
        from mystic.tools import random_seed
        random_seed(123)
        esow = Monitor()
        ssow = Monitor() 

        solver = self.solver
        solver.SetRandomInitialPoints(self.min, self.max)
        solver.SetEvaluationLimits(generations=self.maxiter)
        solver.SetStrictRanges(self.min, self.max)
        solver.SetEvaluationMonitor(esow)
        solver.SetGenerationMonitor(ssow)
        _stdout = trap_stdout()
        solver.Solve(self.costfunction, self.term, **kwds)
        out = release_stdout(_stdout)
        sol = solver.Solution()

        if disp:
            # Print the number of iterations and function evaluations
            iters = len(ssow.x)
            func_evals = len(esow.x)
            print("\nNumber of iterations = %s" % iters)
            print("Number of function evaluations = %s" % func_evals)

        # If solver should terminate immediately, check for that only.
        if iter_limit:
            self.assertTrue(solver.generations < 2)
            warn = "Warning: Invalid termination condition (nPop < 2)"
            self.assertTrue(warn in out)
            return

        # Verify solution is close to expected
        for i in range(len(sol)):
            self.assertAlmostEqual(sol[i], self.expected[i], self.nplaces)

    def test_DifferentialEvolutionSolver_VTR(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = VTR()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_COG(self): # Default for this solver
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = COG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_NCOG(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = NCOG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_CRT(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = CRT()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

#--------------------------------------------------------------------------

    def test_DifferentialEvolutionSolver2_VTR(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import VTR
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = VTR()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_COG(self): # Default for this solver
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = COG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_NCOG(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = NCOG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_CRT(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = CRT()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

#-------------------------------------------------------------------------

    def test_NelderMeadSimplexSolver_CRT(self): # Default for this solver
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = CRT()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_VTR(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = VTR()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_COG(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = COG()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_NCOG(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = NCOG()#tol)
        self._run_solver()

#--------------------------------------------------------------------------

    def test_PowellDirectionalSolver_NCOG(self): # Default for this solver
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = NCOG()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_COG(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = COG()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_VTR(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = VTR()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_CRT(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = CRT()#tol)
        self._run_solver(iter_limit=True)


##############################################################################

class TestAckley(unittest.TestCase):
    """Test Ackley's path function in 2 dimensions."""

    def setUp(self):
        from mystic.models import ackley
        self.costfunction = ackley
        self.ND = 2
        self.expected = [0.]*self.ND
        self.min = [-32.768]*self.ND
        self.max = [32.768]*self.ND
        self.maxiter = 10000
        self.nplaces = 0 # Precision of answer

    def _run_solver(self, iter_limit=False, **kwds):
        from mystic.monitors import Monitor
        from mystic.tools import random_seed
        random_seed(123)
        esow = Monitor()
        ssow = Monitor() 

        solver = self.solver
        solver.SetRandomInitialPoints(self.min, self.max)
        solver.SetEvaluationLimits(generations=self.maxiter)
        solver.SetStrictRanges(self.min, self.max)
        solver.SetEvaluationMonitor(esow)
        solver.SetGenerationMonitor(ssow)
        _stdout = trap_stdout()
        solver.Solve(self.costfunction, self.term, **kwds)
        out = release_stdout(_stdout)
        sol = solver.Solution()

        if disp:
            # Print the number of iterations and function evaluations
            iters = len(ssow.x)
            func_evals = len(esow.x)
            print("\nNumber of iterations = %s" % iters)
            print("Number of function evaluations = %s" % func_evals)

        # If solver should terminate immediately, check for that only.
        if iter_limit:
            self.assertTrue(solver.generations < 2)
            warn = "Warning: Invalid termination condition (nPop < 2)"
            self.assertTrue(warn in out)
            return

        # Verify solution is close to expected
        for i in range(len(sol)):
            self.assertAlmostEqual(sol[i], self.expected[i], self.nplaces)

    def test_DifferentialEvolutionSolver_VTR(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = VTR()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_COG(self): # Default for this solver
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = COG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_NCOG(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = NCOG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_CRT(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = CRT()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

#--------------------------------------------------------------------------

    def test_DifferentialEvolutionSolver2_VTR(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import VTR
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = VTR()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_COG(self): # Default for this solver
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = COG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_NCOG(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = NCOG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_CRT(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = CRT()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

#-------------------------------------------------------------------------

    def test_NelderMeadSimplexSolver_CRT(self): # Default for this solver
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = CRT()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_VTR(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = VTR()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_COG(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = COG()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_NCOG(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = NCOG()#tol)
        self._run_solver()

#--------------------------------------------------------------------------

    def test_PowellDirectionalSolver_NCOG(self): # Default for this solver
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = NCOG()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_COG(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = COG()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_VTR(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = VTR()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_CRT(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = CRT()#tol)
        self._run_solver(iter_limit=True)


##############################################################################

class TestRastrigin(unittest.TestCase):
    """Test Rastrigin's function in 2 dimensions."""

    def setUp(self):
        from mystic.models import rastrigin
        self.costfunction = rastrigin
        self.ND = 2
        self.expected = [0.]*self.ND
        self.min = [-5.12]*self.ND
        self.max = [5.12]*self.ND
        self.maxiter = 10000
        self.nplaces = 0 # Precision of answer

    def _run_solver(self, iter_limit=False, **kwds):
        from mystic.monitors import Monitor
        from mystic.tools import random_seed
        random_seed(123)
        esow = Monitor()
        ssow = Monitor() 

        solver = self.solver
        solver.SetRandomInitialPoints(self.min, self.max)
        solver.SetEvaluationLimits(generations=self.maxiter)
        solver.SetStrictRanges(self.min, self.max)
        solver.SetEvaluationMonitor(esow)
        solver.SetGenerationMonitor(ssow)
        _stdout = trap_stdout()
        solver.Solve(self.costfunction, self.term, **kwds)
        out = release_stdout(_stdout)
        sol = solver.Solution()

        if disp:
            # Print the number of iterations and function evaluations
            iters = len(ssow.x)
            func_evals = len(esow.x)
            print("\nNumber of iterations = %s" % iters)
            print("Number of function evaluations = %s" % func_evals)

        # If solver should terminate immediately, check for that only.
        if iter_limit:
            self.assertTrue(solver.generations < 2)
            warn = "Warning: Invalid termination condition (nPop < 2)"
            self.assertTrue(warn in out)
            return

        # Verify solution is close to expected
        for i in range(len(sol)):
            self.assertAlmostEqual(sol[i], self.expected[i], self.nplaces)

    def test_DifferentialEvolutionSolver_VTR(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = VTR()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_COG(self): # Default for this solver
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = COG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_NCOG(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = NCOG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_CRT(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = CRT()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

#--------------------------------------------------------------------------

    def test_DifferentialEvolutionSolver2_VTR(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import VTR
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = VTR()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_COG(self): # Default for this solver
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = COG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_NCOG(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = NCOG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_CRT(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = CRT()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

#-------------------------------------------------------------------------

    def test_NelderMeadSimplexSolver_CRT(self): # Default for this solver
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = CRT()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_VTR(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = VTR()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_COG(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = COG()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_NCOG(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = NCOG()#tol)
        self._run_solver()

#--------------------------------------------------------------------------

    def test_PowellDirectionalSolver_NCOG(self): # Default for this solver
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = NCOG()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_COG(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = COG()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_VTR(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = VTR()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_CRT(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = CRT()#tol)
        self._run_solver(iter_limit=True)

##############################################################################

class TestGoldsteinPrice(unittest.TestCase):
    """Test the Goldstein-Price function."""

    def setUp(self):
        from mystic.models import goldstein
        self.costfunction = goldstein
        self.ND = 2
        self.expected = [0., -1.]
        self.min = [-2.]*self.ND
        self.max = [2.]*self.ND
        self.maxiter = 10000
        self.nplaces = 0 # Precision of answer

    def _run_solver(self, iter_limit=False, **kwds):
        from mystic.monitors import Monitor
        from mystic.tools import random_seed
        random_seed(123)
        esow = Monitor()
        ssow = Monitor() 

        solver = self.solver
        solver.SetRandomInitialPoints(self.min, self.max)
        solver.SetEvaluationLimits(generations=self.maxiter)
        solver.SetStrictRanges(self.min, self.max)
        solver.SetEvaluationMonitor(esow)
        solver.SetGenerationMonitor(ssow)
        _stdout = trap_stdout()
        solver.Solve(self.costfunction, self.term, **kwds)
        out = release_stdout(_stdout)
        sol = solver.Solution()

        if disp:
            # Print the number of iterations and function evaluations
            iters = len(ssow.x)
            func_evals = len(esow.x)
            print("\nNumber of iterations = %s" % iters)
            print("Number of function evaluations = %s" % func_evals)

        # If solver should terminate immediately, check for that only.
        if iter_limit:
            self.assertTrue(solver.generations < 2)
            warn = "Warning: Invalid termination condition (nPop < 2)"
            self.assertTrue(warn in out)
            return

        # Verify solution is close to expected
        for i in range(len(sol)):
            self.assertAlmostEqual(sol[i], self.expected[i], self.nplaces)


    def test_DifferentialEvolutionSolver_VTR(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = VTR()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_COG(self): # Default for this solver
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = COG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_NCOG(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = NCOG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_CRT(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = CRT()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

#--------------------------------------------------------------------------

    def test_DifferentialEvolutionSolver2_VTR(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import VTR
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = VTR()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_COG(self): # Default for this solver
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = COG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_NCOG(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = NCOG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_CRT(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = CRT()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

#-------------------------------------------------------------------------

    def test_NelderMeadSimplexSolver_CRT(self): # Default for this solver
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = CRT()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_VTR(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = VTR()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_COG(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = COG()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_NCOG(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = NCOG()#tol)
        self._run_solver()

#--------------------------------------------------------------------------

    def test_PowellDirectionalSolver_NCOG(self): # Default for this solver
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = NCOG()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_COG(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = COG()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_VTR(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = VTR()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_CRT(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = CRT()#tol)
        self._run_solver(iter_limit=True)


##############################################################################

class TestChampion(unittest.TestCase):
    """Test Champion's NMinimize test function 51."""

    def setUp(self):
        from mystic.models import nmin51
        self.costfunction = nmin51
        self.ND = 2
        self.expected = [-0.0244031,0.210612]
        self.min = [-1.]*self.ND
        self.max = [1.]*self.ND
        self.maxiter = 10000
        self.nplaces = 0 # Precision of answer

    def _run_solver(self, iter_limit=False, **kwds):
        from mystic.monitors import Monitor
        from mystic.tools import random_seed
        random_seed(123)
        esow = Monitor()
        ssow = Monitor() 

        solver = self.solver
        solver.SetRandomInitialPoints(self.min, self.max)
        solver.SetEvaluationLimits(generations=self.maxiter)
        #solver.SetStrictRanges(self.min, self.max)
        solver.SetEvaluationMonitor(esow)
        solver.SetGenerationMonitor(ssow)
        _stdout = trap_stdout()
        solver.Solve(self.costfunction, self.term, **kwds)
        out = release_stdout(_stdout)
        sol = solver.Solution()
        #print('\n%s' % sol)

        if disp:
            # Print the number of iterations and function evaluations
            iters = len(ssow.x)
            func_evals = len(esow.x)
            print("\nNumber of iterations = %s" % iters)
            print("Number of function evaluations = %s" % func_evals)

        # If solver should terminate immediately, check for that only.
        if iter_limit:
            self.assertTrue(solver.generations < 2)
            warn = "Warning: Invalid termination condition (nPop < 2)"
            self.assertTrue(warn in out)
            return

        # Verify solution is close to expected
        for i in range(len(sol)):
            self.assertAlmostEqual(sol[i], self.expected[i], self.nplaces)

    def test_DifferentialEvolutionSolver_VTR(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = VTR()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_COG(self): # Default for this solver
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = COG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_NCOG(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = NCOG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_CRT(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = CRT()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

#--------------------------------------------------------------------------

    def test_DifferentialEvolutionSolver2_VTR(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import VTR
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = VTR()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_COG(self): # Default for this solver
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = COG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_NCOG(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = NCOG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_CRT(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = CRT()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

#-------------------------------------------------------------------------

    def test_NelderMeadSimplexSolver_CRT(self): # Default for this solver
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = CRT()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_VTR(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = VTR()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_COG(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = COG()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_NCOG(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = NCOG()#tol)
        self._run_solver()

#--------------------------------------------------------------------------

    def test_PowellDirectionalSolver_NCOG(self): # Default for this solver
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = NCOG()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_COG(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = COG()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_VTR(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = VTR()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_CRT(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = CRT()#tol)
        self._run_solver(iter_limit=True)


##############################################################################

class TestPaviani(unittest.TestCase):
    """Test Paviani's function, or TP110 of Schittkowski's test problems."""

    def setUp(self):
        from mystic.models import paviani
        self.costfunction = paviani
        self.ND = 10
        self.x0 = [9.]*self.ND
        self.expected = [9.35025655]*self.ND
        self.min = [2.001]*self.ND
        self.max = [9.999]*self.ND
        self.maxiter = 10000
        self.nplaces = 0 # Precision of answer

    def _run_solver(self, iter_limit=False, **kwds):
        from mystic.monitors import Monitor
        from mystic.tools import random_seed
        random_seed(123)
        esow = Monitor()
        ssow = Monitor() 

        solver = self.solver
        solver.SetInitialPoints(self.x0)
        solver.SetEvaluationLimits(generations=self.maxiter)
        solver.SetStrictRanges(self.min, self.max)
        solver.SetEvaluationMonitor(esow)
        solver.SetGenerationMonitor(ssow)
        _stdout = trap_stdout()
        solver.Solve(self.costfunction, self.term, **kwds)
        out = release_stdout(_stdout)
        sol = solver.Solution()
        #print('\n%s' % sol)

        if disp:
            # Print the number of iterations and function evaluations
            iters = len(ssow.x)
            func_evals = len(esow.x)
            print("\nNumber of iterations = %s" % iters)
            print("Number of function evaluations = %s" % func_evals)

        # If solver should terminate immediately, check for that only.
        if iter_limit:
            self.assertTrue(solver.generations < 2)
            warn = "Warning: Invalid termination condition (nPop < 2)"
            self.assertTrue(warn in out)
            return

        # Verify solution is close to expected
        for i in range(len(sol)):
            self.assertAlmostEqual(sol[i], self.expected[i], self.nplaces)


    def test_DifferentialEvolutionSolver_VTR(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = VTR()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_COG(self): # Default for this solver
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = COG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_NCOG(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = NCOG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver_CRT(self):
        from mystic.solvers import DifferentialEvolutionSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver(self.ND, self.NP)
        self.term = CRT()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

#--------------------------------------------------------------------------

    def test_DifferentialEvolutionSolver2_VTR(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import VTR
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = VTR()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_COG(self): # Default for this solver
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = COG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_NCOG(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.NP = 40
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = NCOG()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

    def test_DifferentialEvolutionSolver2_CRT(self):
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.NP = 40 
        self.solver = DifferentialEvolutionSolver2(self.ND, self.NP)
        self.term = CRT()#tol)
        from mystic.strategy import Rand1Bin
        self._run_solver( strategy=Rand1Bin )

#-------------------------------------------------------------------------

    def test_NelderMeadSimplexSolver_CRT(self): # Default for this solver
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = CRT()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_VTR(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = VTR()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_COG(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = COG()#tol)
        self._run_solver()

    def test_NelderMeadSimplexSolver_NCOG(self): 
        from mystic.solvers import NelderMeadSimplexSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.solver = NelderMeadSimplexSolver(self.ND)
        self.term = NCOG()#tol)
        self._run_solver()

#--------------------------------------------------------------------------

    def test_PowellDirectionalSolver_NCOG(self): # Default for this solver
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = NCOG()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_COG(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import ChangeOverGeneration as COG
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = COG()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_VTR(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import VTR
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = VTR()#tol)
        self._run_solver()

    def test_PowellDirectionalSolver_CRT(self): 
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import CandidateRelativeTolerance as CRT
        tol = 1e-7
        self.solver = PowellDirectionalSolver(self.ND)
        self.term = CRT()#tol)
        self._run_solver(iter_limit=True)


#---------------------------------------------------------------------------------

if __name__ == '__main__':
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestZimmermann)
    suite2 = unittest.TestLoader().loadTestsFromTestCase(TestRosenbrock)
    suite3 = unittest.TestLoader().loadTestsFromTestCase(TestCorana)
    suite4 = unittest.TestLoader().loadTestsFromTestCase(TestQuartic)
    suite5 = unittest.TestLoader().loadTestsFromTestCase(TestShekel)
    suite6 = unittest.TestLoader().loadTestsFromTestCase(TestStep)
    suite7 = unittest.TestLoader().loadTestsFromTestCase(TestGriewangk)
    suite8 = unittest.TestLoader().loadTestsFromTestCase(TestPeaks) 
    suite9 = unittest.TestLoader().loadTestsFromTestCase(TestVenkataraman91)
    suite10 = unittest.TestLoader().loadTestsFromTestCase(TestSchwefel)  
    suite11 = unittest.TestLoader().loadTestsFromTestCase(TestEasom)  
    suite12 = unittest.TestLoader().loadTestsFromTestCase(TestRotatedEllipsoid)
    suite13 = unittest.TestLoader().loadTestsFromTestCase(TestAckley) 
    suite14 = unittest.TestLoader().loadTestsFromTestCase(TestRastrigin) 
    suite15 = unittest.TestLoader().loadTestsFromTestCase(TestGoldsteinPrice) 
    suite16 = unittest.TestLoader().loadTestsFromTestCase(TestChampion) 
    suite17 = unittest.TestLoader().loadTestsFromTestCase(TestPaviani)
    # Comment out suites in the list below to test specific test cost functions only
    # (Testing all the problems will take some time)
    allsuites = unittest.TestSuite([suite1,   # Zimmermann
#                                   suite2,   # Rosenbrock
#                                   suite3,   # Corana
#                                   suite4,   # Quartic
#                                   suite5,   # Shekel
 #                                  suite6,   # Step
 #                                  suite7,   # Griewangk
#                                   suite8,   # Peaks
#                                   suite9,   # Venkataraman91
#                                   suite10,  # Schwefel
#                                   suite11,  # Easom
                                    suite12,  # RotatedEllipsoid
#                                   suite13,  # Ackley
#                                   suite14,  # Rastrigin
#                                   suite15,  # GoldsteinPrice
#                                   suite16,  # Champion
 #                                  suite17,  # Paviani
                                    ])
    unittest.TextTestRunner(verbosity=verbosity).run(allsuites)

# EOF
