#!/usr/bin/env python
#Adapted from parallel_desolve.py by mmckerns@caltech.edu

__doc__ = """
# Tests MPI version of Storn and Price's Polynomial 'Fitting' Problem.
# 
# Exact answer: [1,1,1]
  
# Reference:
#
# [1] Storn, R. and Price, K. Differential Evolution - A Simple and Efficient
# Heuristic for Global Optimization over Continuous Spaces. Journal of Global
# Optimization 11: 341-359, 1997.

# To run in parallel:  (must install 'pyina')
mpipython.exe ezmap_desolve_rosen.py
"""

try:
  import pyina.launchers as launchers
  from pyina.launchers import mpirun_launcher
  from pyina.mappers import equalportion_mapper
  from pyina.ez_map import ez_map2 as ez_map
except:
  print __doc__


from mystic.differential_evolution import DifferentialEvolutionSolver2
from mystic.termination import ChangeOverGeneration, VTR
from mystic.strategy import Best1Exp
from mystic.tools import VerboseSow, random_seed

#from raw_rosen import rosen as myCost     # ez_map needs a helper function
from mystic.models import rosen as myCost  # ez_map2 doesn't require help

ND = 3
NP = ND*10
MAX_GENERATIONS = ND*NP
NNODES = ND

TOL = 0.01
CROSS = 0.9
SCALE = 0.8
seed = 100


if __name__=='__main__':
    def print_solution(func):
        print func
        return

    psow = VerboseSow(10)
    ssow = VerboseSow(10)

    random_seed(seed)
    print "first sequential..."
    solver = DifferentialEvolutionSolver2(ND,NP)  #XXX: sequential
    solver.SetRandomInitialPoints(min=[-100.0]*ND, max=[100.0]*ND)
    solver.SetEvaluationLimits(maxiter=MAX_GENERATIONS)
    solver.Solve(myCost, VTR(TOL), strategy=Best1Exp, \
                StepMonitor=ssow, CrossProbability=CROSS, ScalingFactor=SCALE, \
                disp=1)
    print ""
    print_solution( solver.Solution() )

    random_seed(seed)
    print "\n and now parallel..."
    solver2 = DifferentialEvolutionSolver2(ND,NP)  #XXX: parallel
    solver2.SetMapper(ez_map, equalportion_mapper)
    solver2.SetLauncher(mpirun_launcher, NNODES)
    solver2.SetRandomInitialPoints(min=[-100.0]*ND, max=[100.0]*ND)
    solver2.SetEvaluationLimits(maxiter=MAX_GENERATIONS)
    solver2.Solve(myCost, VTR(TOL), strategy=Best1Exp, \
                StepMonitor=psow, CrossProbability=CROSS, ScalingFactor=SCALE, \
                disp=1)
    print ""
    print_solution( solver2.Solution() )

# end of file
