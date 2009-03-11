#!/usr/bin/env python

"""
One mogi source, similar to test_mogi, but uses
CostFactory objects

"""

from test_mogi import *

from mystic.forward_model import CostFactory
from mystic.filters import PickComponent

def de_solve(CF):
    solver = DifferentialEvolutionSolver(ND, NP)

    stepmon = Sow()
    minrange = [-1000., -1000., -100., -10.];
    maxrange = [1000., 1000., 100., 10.];
    solver.SetRandomInitialPoints(min = minrange, max = maxrange)
    solver.SetEvaluationLimits(maxiter=MAX_GENERATIONS)

    solver.Solve(CF, Best1Exp,\
                 termination = ChangeOverGeneration(generations=100) , \
                 CrossProbability=0.5, ScalingFactor=0.5, \
                 StepMonitor = stepmon)

    solution = solver.Solution()
  
    return solution, stepmon

if __name__ == '__main__':
    F = CostFactory()
    F.addModel(ForwardMogiFactory, 'mogi1', 4, outputFilter = PickComponent(2))
    myCostFunction = F.getCostFunction(evalpts = stations, observations = data_z)
    print F
    rp =  F.getRandomParams()
    print "new Cost Function : %s " % myCostFunction(rp)
    print "orig Cost Function: %s " % cost_function(rp)

    f1 = ForwardMogiFactory(rp)
    f2 = ForwardMogiFactory(rp)

    print 'start cf'
    for i in range(3000):
        xx = cost_function(rp)
    print 'end cf'

    print 'start cf2'
    for i in range(3000):
        xx = myCostFunction(rp)
    print 'end cf2'

    #de_solve(cost_function)
    de_solve(myCostFunction)

# end of file
