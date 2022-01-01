#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
test_getCost.py
Example to demonstrate use of CostFactory
"""

from mystic.termination import *
from mystic.strategy import *
from forward_model import *

from mystic.math import poly1d as ForwardPolyFactory
from mystic.models import poly; PolyCostFactory = poly.CostFactory
from mystic.solvers import DifferentialEvolutionSolver
from mystic.monitors import VerboseMonitor
from mystic.tools import getch

ND = 3
NP = 80
MAX_GENERATIONS = ND*NP

from numpy import array

def data(params):
    fwd = ForwardPolyFactory(params)
    x = (array([list(range(101))])-50.)[0]
    return x,fwd(x)

def de_solve(CF):
    solver = DifferentialEvolutionSolver(ND, NP)
    solver.enable_signal_handler()

    stepmon = VerboseMonitor(10,50)
    minrange = [-100., -100., -100.];
    maxrange = [100., 100., 100.];
    solver.SetRandomInitialPoints(min = minrange, max = maxrange)
    solver.SetStrictRanges(min = minrange, max = maxrange)
    solver.SetEvaluationLimits(generations=MAX_GENERATIONS)
    solver.SetGenerationMonitor(stepmon)

    solver.Solve(CF, termination=ChangeOverGeneration(generations=300),\
                CrossProbability=0.5, ScalingFactor=0.5,\
                sigint_callback=plot_sol)

    solution = solver.Solution()
    return solution, stepmon

def plot_sol(params,linestyle='b-'):
    d = data(params)
    plt.plot(d[0],d[1],'%s'%linestyle,linewidth=2.0)
    plt.axis(plotview)
    plt.draw()
    plt.pause(0.001)
    return

from numpy import sum as numpysum
def cost_function(params):
    x = data(params)[1] - datapts
    return numpysum(real((conjugate(x)*x)))


if __name__ == '__main__':
    plotview = [-60,60, 0,2500]
    target = [1., 2., 1.]
    x,datapts = data(target)

   #myCost = cost_function
  ##myCost = PolyCostFactory(target,x)
    F = CostFactory()
    F.addModel(ForwardPolyFactory,len(target),'poly')
    myCost = F.getCostFunction(evalpts=x, observations=datapts)    

    import matplotlib.pyplot as plt
    plt.ion()

    print("target: %s" % target)
    plot_sol(target,'r-')
    solution, stepmon = de_solve(myCost)
    print("solution: %s" % solution)
    plot_sol(solution,'g-')
    print("")
#   print("at step 10: %s" % stepmon.x[10])

    getch()

# End of file
