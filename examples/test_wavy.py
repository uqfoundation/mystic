#!/usr/bin/env python

"""
|x + 3 sin[x]|
"""

from mystic.differential_evolution import DifferentialEvolutionSolver2 as DifferentialEvolutionSolver
from mystic.detools import Best1Exp, Best1Bin, Rand1Exp, ChangeOverGeneration, VTR
from mystic.polytools import polyeval
from mystic import getch, VerboseSow
from numpy import *
import scipy.optimize

import random
random.seed(123)

def wavy1(x):
    x = array(x)
    return abs(x+3.*sin(x+pi)+pi)

def wavy2(x):
    x = array(x)
    return 4 *sin(x)+sin(4*x) + sin(8*x)+sin(16*x)+sin(32*x)+sin(64*x)

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

    solver.enable_signal_handler()
  
    strategy = Best1Bin

    stepmon = VerboseSow(1)
    solver.Solve(wavy, strategy, termination = ChangeOverGeneration(generations=50) , \
                 maxiter= MAX_GENERATIONS, CrossProbability=1.0, ScalingFactor=0.9 , \
                 StepMonitor = stepmon,  sigint_callback = plot_solution \
                 )

    solution = solver.Solution()

    return solution, solver
  


if __name__ == '__main__':
    #solution = main()
    scipysol = scipy.optimize.fmin(wavy, [0.1])
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
