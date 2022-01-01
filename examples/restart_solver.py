#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

from mystic.solvers import DifferentialEvolutionSolver
from mystic.solvers import NelderMeadSimplexSolver
from mystic.termination import VTR, ChangeOverGeneration, When, Or
from mystic.monitors import VerboseMonitor
from mystic.models import rosen
from mystic.solvers import LoadSolver
import dill
import os

def runme():
    # instantiate the solver
    _solver = NelderMeadSimplexSolver(3)
    lb,ub = [0.,0.,0.],[10.,10.,10.]
    _solver.SetRandomInitialPoints(lb, ub)
    _solver.SetEvaluationLimits(1000)
    # add a monitor stream
    stepmon = VerboseMonitor(1)
    _solver.SetGenerationMonitor(stepmon)
    # configure the bounds
    _solver.SetStrictRanges(lb, ub)
    # configure stop conditions
    term = Or( VTR(), ChangeOverGeneration() )
    _solver.SetTermination(term)
    # add a periodic dump to an archive
    tmpfile = 'mysolver.pkl'
    _solver.SetSaveFrequency(10, tmpfile)
    # run the optimizer
    _solver.Solve(rosen)
    # get results
    x = _solver.bestSolution
    y = _solver.bestEnergy  
    # load the saved solver
    solver = LoadSolver(tmpfile)
    #os.remove(tmpfile)
    # obligatory check that state is the same
    assert all(x == solver.bestSolution)
    assert y == solver.bestEnergy  
    # modify the termination condition
    term = VTR(0.0001)
    solver.SetTermination(term)
    # run the optimizer
    solver.Solve(rosen)
    os.remove(tmpfile)
    # check the solver serializes
    _s = dill.dumps(solver)
    return dill.loads(_s)


solver = runme()

# EOF
