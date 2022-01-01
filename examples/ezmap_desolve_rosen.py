#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
#
# Adapted from parallel_desolve.py

__doc__ = """
# Tests MPI version of Storn and Price's Polynomial 'Fitting' Problem,
# using a single solver with the population launched in parallel.
# This can be much slower, due to overhead of repeatedly setting up MPI.
# 
# Exact answer: [1,1,1]
  
# Reference:
#
# [1] Storn, R. and Price, K. Differential Evolution - A Simple and Efficient
# Heuristic for Global Optimization over Continuous Spaces. Journal of Global
# Optimization 11: 341-359, 1997.

# To run in parallel:  (must install 'pyina')
python ezmap_desolve_rosen.py
"""

try:
  from pyina.launchers import Mpi as Pool
# from pyina.launchers import TorqueMpi as Pool
except:
  print(__doc__)


from mystic.solvers import DifferentialEvolutionSolver2
from mystic.termination import ChangeOverGeneration, VTR
from mystic.strategy import Best1Exp
from mystic.monitors import VerboseMonitor
from mystic.tools import random_seed

#from raw_rosen import rosen as myCost     # with a helper function
from mystic.models import rosen as myCost  # without a helper function

ND = 3
NP = 20
MAX_GENERATIONS = NP*NP
NNODES = "5:ppn=4"
QUEUE = "weekdayQ"
TIMELIMIT = "00:30:00"

TOL = 0.01
CROSS = 0.9
SCALE = 0.8
seed = 100


if __name__=='__main__':
    def print_solution(func):
        print(func)
        return

    psow = VerboseMonitor(10)
    ssow = VerboseMonitor(10)

    random_seed(seed)
    print("first sequential...")
    solver = DifferentialEvolutionSolver2(ND,NP)  #XXX: sequential
    solver.SetRandomInitialPoints(min=[-100.0]*ND, max=[100.0]*ND)
    solver.SetEvaluationLimits(generations=MAX_GENERATIONS)
    solver.SetGenerationMonitor(ssow)
    solver.Solve(myCost, VTR(TOL), strategy=Best1Exp, \
                 CrossProbability=CROSS, ScalingFactor=SCALE, disp=1)
    print("")
    print_solution( solver.bestSolution )

    random_seed(seed)
    print("\n and now parallel...")
    solver2 = DifferentialEvolutionSolver2(ND,NP)  #XXX: parallel
    solver2.SetMapper(Pool(NNODES).map)
#   solver2.SetMapper(Pool(NNODES, queue=QUEUE, timelimit=TIMELIMIT).map)
    solver2.SetRandomInitialPoints(min=[-100.0]*ND, max=[100.0]*ND)
    solver2.SetEvaluationLimits(generations=MAX_GENERATIONS)
    solver2.SetGenerationMonitor(psow)
    solver2.Solve(myCost, VTR(TOL), strategy=Best1Exp, \
                  CrossProbability=CROSS, ScalingFactor=SCALE, disp=1)
    print("")
    print_solution( solver2.bestSolution )

# end of file
