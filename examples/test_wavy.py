#!/usr/bin/env python

"""
test some simple multi-minima functions, such as |x + 3 sin[x]|
"""

from mystic.solvers import DifferentialEvolutionSolver2 as DifferentialEvolutionSolver
from mystic.termination import ChangeOverGeneration, VTR
from mystic.strategy import Best1Exp, Best1Bin, Rand1Exp
from mystic.tools import getch, VerboseSow
from numpy import arange
from mystic.solvers import fmin
#from scipy.optimize import fmin

import random
random.seed(123)

from mystic.models import wavy1, wavy2
wavy = wavy1

def show():
    import pylab, Image
    pylab.savefig('test_wavy_out',dpi=100)
    im = Image.open('test_wavy_out.png')
    im.show()
    return

def plot_solution(sol=None):
    try:
        import pylab
        x = arange(-40,40,0.01)
        y = wavy(x)
        pylab.plot(x,y)
        if sol is not None:
            pylab.plot(sol, wavy(sol), 'r+')
        show()
    except ImportError:
        print "Install matplotlib for plotting"
        pass


ND = 1
NP = 20
MAX_GENERATIONS = 100

def main():
    solver = DifferentialEvolutionSolver(ND, NP)
    solver.SetRandomInitialPoints(min = [-100.0]*ND, max = [100.0]*ND)
    solver.SetEvaluationLimits(maxiter=MAX_GENERATIONS)

    solver.enable_signal_handler()

    strategy = Best1Bin
    stepmon = VerboseSow(1)
    solver.Solve(wavy,
                 termination = ChangeOverGeneration(generations=50), \
                 strategy=strategy, CrossProbability=1.0, ScalingFactor=0.9, \
                 StepMonitor = stepmon,  sigint_callback = plot_solution)

    solution = solver.Solution()

    return solution, solver
  


if __name__ == '__main__':
    #solution = main()
    scipysol = fmin(wavy, [0.1])
    desol, solver = main()
    #plot_solution(scipysol)
    #plot_solution(desol)
    print "scipy: ", scipysol, wavy(scipysol)
    print "desol: ", desol, wavy(desol)
    try:
        import pylab
        x = arange(-40,40,0.01)
        pylab.plot(x,wavy(x))
        pylab.plot(scipysol, wavy(scipysol), 'r+',markersize=8)
        pylab.plot(desol, wavy(desol), 'bo',markersize=8)
        pylab.legend(('|x + 3 sin(x+pi)|','scipy','DE'))
        pylab.plot(solver.genealogy[10], wavy(solver.genealogy[10]), 'g-')
        print "genealogy:\n"
        for xx in solver.genealogy[4]:
            print xx
            pylab.plot(xx, wavy(xx), 'go',markersize=3)
        show()
    except ImportError:
        print "Install matplotlib for plotting"

# end of file
