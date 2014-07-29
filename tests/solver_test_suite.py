#!/usr/bin/env python
#
# Author: Alta Fang (altafang @caltech and alta @princeton)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2014 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE
"""A test suite for Mystic solvers.
Note: VTR termination with default tolerance shouldn't work for functions 
whose value at the minimum is negative!
Also, the two differential evolution solvers are global, while the other solvers
are local optimizers."""
# should report clock-time, # of iterations, and # of function evaluations

import sys
from StringIO import StringIO
import unittest
from math import *
from mystic.math import almostEqual

disp = False  # Flag for whether to display number of iterations 
              #  and function evaluations.
verbosity = 2 # Verbosity setting for unittests (default is 1).

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
        #print '\nsol:', sol

        if disp:
            # Print the number of iterations and function evaluations
            iters = len(ssow.x)
            func_evals = len(esow.x)
            print "\nNumber of iterations = ", iters
            print "Number of function evaluations = ", func_evals

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
            print "\nNumber of iterations = ", iters
            print "Number of function evaluations = ", func_evals

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
            print "\nNumber of iterations = ", iters
            print "Number of function evaluations = ", func_evals

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
            print "\nNumber of iterations = ", iters
            print "Number of function evaluations = ", func_evals

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
            print "\nNumber of iterations = ", iters
            print "Number of function evaluations = ", func_evals

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
            print "\nNumber of iterations = ", iters
            print "Number of function evaluations = ", func_evals

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
            print "\nNumber of iterations = ", iters
            print "Number of function evaluations = ", func_evals

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
    """Test the peaks optimization problem.
Source: http://www.nag.co.uk/numeric/FL/nagdoc_fl22/xhtml/E05/e05jbf.xml"""

    def setUp(self):
        def peaks(x_vector):
            """The peaks function. Optimize on the box [-3, 3]x[-3, 3]. Has 
        several local minima and one global minimum at (0.23, -1.63) where 
        the function value is about -6.55.

        Source: http://www.nag.co.uk/numeric/FL/nagdoc_fl22/xhtml/E05/e05jbf.xml, 
        example 9."""
            x = x_vector[0]
            y = x_vector[1]
            result = 3.*(1. - x)**2*exp(-x**2 - (y + 1.)**2) - \
                    10.*(x*(1./5.) - x**3 - y**5)*exp(-x**2 - y**2) - \
                    1./3.*exp(-(x + 1.)**2 - y**2)
            return result
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
            print "\nNumber of iterations = ", iters
            print "Number of function evaluations = ", func_evals

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
    """Test the optimization found in Example 9.1, page 442, in
Applied Optimization with MATLAB Programming, by Venkataraman, 
Wiley, 2nd edition, 2009."""

    def setUp(self):
        def venkataraman_91(x):
            """This function has several minima and maxima and 
        strong global minimum at x1 = 4, x2 = 4.

        Source: Example 9.1, page 442, in Applied Optimization with MATLAB Programming, 
        by Venkataraman, Wiley, 2nd edition, 2009."""
            x1 = x[0]
            x2 = x[1]
            return -20.*sin(0.1 + ((x1 - 4.)**2 + (x2 - 4.)**2)**(0.5))/   \
                   (0.1 + ((x1 - 4.)**2 + (x2 - 4.)**2)**0.5)
        self.costfunction = venkataraman_91
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
            print "\nNumber of iterations = ", iters
            print "Number of function evaluations = ", func_evals

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
    """Test Schwefel's function in 2 dimensions."""

    def setUp(self):
        import numpy
        def schwefel(x):
            """'Schwefel's function [Sch81] is deceptive in that the global minimum is 
        geometrically distant, over the parameter space, from the next best local 
        minima. Therefore, the search algorithms are potentially prone to convergence 
        in the wrong direction.' - http://www.geatbx.com/docu/fcnindex-01.html
        xi in [-500, 500]
        global minimum: f(x) = -n*418.9829, xi=420.9687
        Can be n-dimensional.
        """
            x = numpy.asarray(x)
            return numpy.sum(-x*numpy.sin(abs(x)**0.5))
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
            print "\nNumber of iterations = ", iters
            print "Number of function evaluations = ", func_evals

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
        def easom(x):
            """'The Easom function [Eas90] is a unimodal test function, where the global 
        minimum has a small area relative to the search space. The function was 
        inverted for minimization.' - http://www.geatbx.com/docu/fcnindex-01.html
        Global minimum f(x1, x2) = -1 at (x1, x2) = (pi, pi)
        xi in [-100, 100]. 2-dimensional."""
            x1 = x[0]
            x2 = x[1]
            return -cos(x1)*cos(x2)*exp(-(x1-pi)**2 - (x2 - pi)**2)
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
            print "\nNumber of iterations = ", iters
            print "Number of function evaluations = ", func_evals

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
        def rotated_ellipsoid(x):
            """'An extension of the axis parallel hyper-ellipsoid is Schwefel's 
        function1.2. With respect to the coordinate axes, this function produces 
        rotated hyper-ellipsoids. It is continuous, convex and unimodal.' 
        - http://www.geatbx.com/docu/fcnindex-01.html
        xi in [-65.536, 65.536]. n-dimensional.
        global minimum f(x)=0 at xi=0."""
            result = 0.
            for i in range(len(x)):
                s = 0.
                for j in range(i+1):
                    s += x[j]
                result += s**2
            return result
        self.costfunction = rotated_ellipsoid
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
            print "\nNumber of iterations = ", iters
            print "Number of function evaluations = ", func_evals

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
        import numpy
        def ackley(x):
            """Ackley's Path function.
        xi in [-32.768., 32.768]
        global minimum f(x)=0 at xi=0
        can be n-dimensional. http://www.geatbx.com/docu/fcnindex-01.html"""
            x = numpy.asarray(x)
            n = len(x)
            return -20.*exp(-0.2*sqrt(1./n*numpy.sum(x**2))) - exp(1./n*numpy.sum(
                    numpy.cos(2.*pi*x))) + 20. + exp(1)
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
            print "\nNumber of iterations = ", iters
            print "Number of function evaluations = ", func_evals

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
    """Test Rastrigin's function in 2 dimensions. Has many local minima and
one global minimum."""

    def setUp(self):
        import numpy
        def rastrigin(x):
            """Rastrigin's function. Global minimum at xi=0, f(x)=0. Contains
        many local minima regularly distributed. Can be n-dimensional.
        xi in [-5.12, 5.12]
        http://www.geatbx.com/docu/fcnindex-01.html"""
            x = numpy.asarray(x)
            return 10.*len(x) + numpy.sum(x**2 - 10*numpy.cos(2.*pi*x))
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
            print "\nNumber of iterations = ", iters
            print "Number of function evaluations = ", func_evals

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
        def goldstein_price(x):
            """Goldstein-Price function. Global minimum f(x1, x2) = 3 at 
        (x1, x2) = (0, -1)
        x1, x2 in [-2., 2.]
        2-dimensional.  http://www.geatbx.com/docu/fcnindex-01.html"""
            x1 = x[0]
            x2 = x[1]
            return (1.+(x1+x2+1.)**2*(19.-14.*x1+3.*x1**2-14.*x2+6.*x1*x2+3.*x2**2))*\
                   (30.+(2.*x1-3.*x2)**2*(18.-32.*x1+12.*x1**2+48.*x2-36.*x1*x2+\
                    27.*x2**2))
        self.costfunction = goldstein_price
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
            print "\nNumber of iterations = ", iters
            print "Number of function evaluations = ", func_evals

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

class TestMathematicaDoc(unittest.TestCase):
    """Test the function with many local minima found at 
http://reference.wolfram.com/mathematica/tutorial/ConstrainedOptimizationGlobalNumerical.html."""

    def setUp(self):
        def mathematica_doc1(x):
            """This function has many local minima. Even mathematica itself
        has a hard time finding the global minimum.

        Source: http://reference.wolfram.com/mathematica/tutorial/ConstrainedOptimizationGlobalNumerical.html

        global minimum at [-0.0244031,0.210612]"""
            x1 = x[0]
            x2 = x[1]
            return exp(sin(50.*x1)) + sin(60.*exp(x2)) + sin(70.*sin(x1)) + \
                   sin(sin(80.*x2)) - sin(10.*(x1 + x2)) + 1./4.*(x1**2 + x2**2)
        self.costfunction = mathematica_doc1
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
        #print '\n', sol

        if disp:
            # Print the number of iterations and function evaluations
            iters = len(ssow.x)
            func_evals = len(esow.x)
            print "\nNumber of iterations = ", iters
            print "Number of function evaluations = ", func_evals

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
    """Paviani's function, or TP110 of Schittkowski's test problems.
F(min) is negative, so VTR default can fail."""

    def setUp(self):
        def tp110(x):
            """TP110 of Schittkowski's test problems. Paviani's function.
        Minimize sum(ln(xi-2)**2 + ln(10-xi)**2, i=1:10) - product(xi, i=1:10)**0.2
        xi in [2.001, 9.999]
        ndim = 10.
        x0 = [9.]*ndim
        expected solution: [9.35025655]*ndim where f=-45.77846971"""
            p = 1.
            for i in range(10):
                p *= x[i]
            s = 0.
            for i in range(10):
                s += log(x[i] - 2.)**2 + log(10.-x[i])**2
            return s - p**0.2
        self.costfunction = tp110
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
        #print '\n', sol

        if disp:
            # Print the number of iterations and function evaluations
            iters = len(ssow.x)
            func_evals = len(esow.x)
            print "\nNumber of iterations = ", iters
            print "Number of function evaluations = ", func_evals

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
    suite16 = unittest.TestLoader().loadTestsFromTestCase(TestMathematicaDoc) 
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
#                                   suite16,  # MathematicaDoc
 #                                  suite17,  # Paviani
                                    ])
    unittest.TextTestRunner(verbosity=verbosity).run(allsuites)

# EOF
