#!/usr/bin/env python
"""
example_getCost.py
Example to demonstrate use of CostFactory
"""

from mystic.termination import *
from mystic.strategy import *
from forward_model import *

from mystic.models.poly import poly1d as ForwardPolyFactory
from mystic.models import poly; PolyCostFactory = poly.CostFactory
from mystic.differential_evolution import DifferentialEvolutionSolver
from mystic.tools import VerboseSow, getch

ND = 3
NP = 80
MAX_GENERATIONS = ND*NP

from numpy import array

def data(params):
    fwd = ForwardPolyFactory(params)
    x = (array([range(101)])-50.)[0]
    return x,fwd(x)

def de_solve(CF):
    solver = DifferentialEvolutionSolver(ND, NP)
    solver.enable_signal_handler()

    stepmon = VerboseSow(10,50)
    minrange = [-100., -100., -100.];
    maxrange = [100., 100., 100.];
    solver.SetRandomInitialPoints(min = minrange, max = maxrange)
    solver.SetStrictRanges(min = minrange, max = maxrange)
    solver.SetEvaluationLimits(maxiter=MAX_GENERATIONS)

    solver.Solve(CF,Best1Exp,termination=ChangeOverGeneration(generations=300),\
                CrossProbability=0.5,ScalingFactor=0.5,\
                StepMonitor=stepmon,sigint_callback=plot_sol)

    solution = solver.Solution()
    return solution, stepmon

def plot_sol(params,linestyle='b-'):
    d = data(params)
    pylab.plot(d[0],d[1],'%s'%linestyle,linewidth=2.0)
    pylab.axis(plotview)
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
    F.addModel(ForwardPolyFactory,'poly',len(target))
    myCost = F.getCostFunction(evalpts=x, observations=datapts)    

    import pylab
    pylab.ion()

    print "target: ",target
    plot_sol(target,'r-')
    solution, stepmon = de_solve(myCost)
    print "solution: ",solution
    plot_sol(solution,'g-')
    print ""
#   print "at step 10: ",stepmon.x[10]

    getch()

# End of file
