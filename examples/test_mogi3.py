#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
One mogi source, similar to test_mogi, but uses
CostFactory objects

"""

from test_mogi import *

from mystic.forward_model import CostFactory
from mystic.filters import component

def de_solve(CF):
    solver = DifferentialEvolutionSolver(ND, NP)

    stepmon = Monitor()
    minrange = [-1000., -1000., -100., -10.];
    maxrange = [1000., 1000., 100., 10.];
    solver.SetRandomInitialPoints(min = minrange, max = maxrange)
    solver.SetEvaluationLimits(generations=MAX_GENERATIONS)
    solver.SetGenerationMonitor(stepmon)

    solver.Solve(CF, termination = ChangeOverGeneration(generations=100), \
                 CrossProbability=0.5, ScalingFactor=0.5)

    solution = solver.Solution()
  
    return solution, stepmon

if __name__ == '__main__':
    F = CostFactory()
    F.addModel(ForwardMogiFactory, 4, 'mogi1', outputFilter=component(2))
    myCostFunction = F.getCostFunction(evalpts=stations, observations=data_z)
    print(F)
    rp =  F.getRandomParams()
    print("new Cost Function : %s " % myCostFunction(rp))
    print("orig Cost Function: %s " % cost_function(rp))

    f1 = ForwardMogiFactory(rp)
    f2 = ForwardMogiFactory(rp)

    print('start cf')
    for i in range(3000):
        xx = cost_function(rp)
    print('end cf')

    print('start cf2')
    for i in range(3000):
        xx = myCostFunction(rp)
    print('end cf2')

    #de_solve(cost_function)
    de_solve(myCostFunction)

# end of file
