#!/usr/bin/env python

"""
Same as test_ffit.py
but uses DifferentialEvolutionSolver2 instead.

Note:
 1. MPIDifferentialEvolutionSolver is functionally identical to DifferentialEvolultionSolver2.
 2. In DifferentialEvolutionSolver, as each trial vector is compared to its target, once the trial beats
    the target it enters the generation, replacing the old vector and immediately becoming available
    as candidates for creating difference vectors, and for mutations, etc.
"""

from test_ffit import *

def main():
    from mystic.differential_evolution import DifferentialEvolutionSolver2
    from pathos.pp_map import pp_map

    solver = DifferentialEvolutionSolver2(ND, NP)
    solver.SetMapper(pp_map)
    solver.SetRandomInitialPoints(min = [-100.0]*ND, max = [100.0]*ND)
    solver.SetEvaluationLimits(maxiter=MAX_GENERATIONS)
    solver.enable_signal_handler()
  
    strategy = Best1Exp
    #strategy = Best1Bin

    solver.Solve(ChebyshevCost, termination=VTR(0.01), strategy=strategy, \
                 CrossProbability=1.0, ScalingFactor=0.9 , \
                 StepMonitor=VerboseSow(30), sigint_callback=plot_solution)

    solution = solver.Solution()

    return solution
  

if __name__ == '__main__':
    solution = main()
    print_solution(solution)
    plot_solution(solution)

# end of file
