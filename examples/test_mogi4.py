#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Similar to test_mogi2 (two sources) (See that one first)
"""

from test_mogi2 import params0, params1, stations, data, data_z, ND, NP, plot_sol, plot_noisy_data, MAX_GENERATIONS, ForwardMogiFactory
from mystic.solvers import DifferentialEvolutionSolver
from mystic.termination import ChangeOverGeneration, VTR
from mystic.monitors import Monitor
from mystic.tools import getch, random_seed

from mystic.forward_model import CostFactory
from mystic.filters import component

def de_solve(CF):
    solver = DifferentialEvolutionSolver(ND, NP)

    solver.enable_signal_handler()

    stepmon = Monitor()
    minrange = [-1000., -1000., -100., -1.]*2;
    maxrange = [1000., 1000., 100., 1.]*2;
    solver.SetRandomInitialPoints(min = minrange, max = maxrange)
    solver.SetEvaluationLimits(generations=MAX_GENERATIONS)
    solver.SetGenerationMonitor(stepmon)

    solver.Solve(CF, termination = ChangeOverGeneration(generations=300), \
                 CrossProbability=0.5, ScalingFactor=0.5, \
                 sigint_callback = plot_sol)

    solution = solver.Solution()
  
    return solution, stepmon

if __name__ == '__main__':

    F = CostFactory()
    F.addModel(ForwardMogiFactory, 4, 'mogi1', outputFilter=component(2, -1))
    F.addModel(ForwardMogiFactory, 4, 'mogi2', outputFilter=component(2, -1))
    myCostFunction = F.getCostFunction(evalpts=stations, observations=data_z)
    print(F)

    def C2(x):
        "This is the new version"
        return 100000 * myCostFunction(x)

    def C3(x):
        "Cost function constructed by hand"
        from test_mogi2 import cost_function
        return cost_function(x)

    def test():
        "call me to see if the functions return the same thing"
        rp = F.getRandomParams()
        print("C2: %s" % C2(rp))
        print("C3: %s" % C3(rp))

    test()
    import matplotlib.pyplot as plt
    plot_noisy_data()
    desol, dstepmon = de_solve(C2)
    print("desol: %s" % desol)

   #plot_sol(dstepmon.x[-100],'k-')
    plot_sol(desol,'r-')

    getch()

# end of file
