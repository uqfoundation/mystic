#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
test some simple multi-minima functions, such as |x + 3 sin[x]|
"""

from mystic.solvers import DifferentialEvolutionSolver2 as DifferentialEvolutionSolver
from mystic.termination import ChangeOverGeneration, VTR
from mystic.strategy import Best1Exp, Best1Bin, Rand1Exp
from mystic.monitors import VerboseMonitor
from mystic.tools import getch
from numpy import arange
from mystic.solvers import fmin
#from mystic._scipyoptimize import fmin

from mystic.tools import random_seed
random_seed(123)

from mystic.models import wavy1, wavy2
wavy = wavy1

def show():
    import matplotlib.pyplot as plt, Image
    plt.savefig('test_wavy_out',dpi=100)
    im = Image.open('test_wavy_out.png')
    im.show()
    return

def plot_solution(sol=None):
    try:
        import matplotlib.pyplot as plt
        x = arange(-40,40,0.01)
        y = wavy(x)
        plt.plot(x,y)
        if sol is not None:
            plt.plot(sol, wavy(sol), 'r+')
        try: show()
        except ImportError: plt.show()
    except ImportError:
        print("Install matplotlib for visualization")
        pass


ND = 1
NP = 20
MAX_GENERATIONS = 100

def main():
    solver = DifferentialEvolutionSolver(ND, NP)
    solver.SetRandomInitialPoints(min = [-100.0]*ND, max = [100.0]*ND)
    solver.SetEvaluationLimits(generations=MAX_GENERATIONS)

    solver.enable_signal_handler()

    strategy = Best1Bin
    stepmon = VerboseMonitor(1)
    solver.SetGenerationMonitor(stepmon)
   #solver.SetReducer(sum, arraylike=True) # reduce wavy's multi-valued return
    solver.Solve(wavy, ChangeOverGeneration(generations=50), \
                 strategy=strategy, CrossProbability=1.0, ScalingFactor=0.9, \
                 sigint_callback = plot_solution)

    solution = solver.Solution()

    return solution, solver
  


if __name__ == '__main__':
    #solution = main()
    scipysol = fmin(wavy, [0.1])
    desol, solver = main()
    #plot_solution(scipysol)
    #plot_solution(desol)
    print("fmin: %s %s" % (scipysol, wavy(scipysol)))
    print("dife: %s %s" % (desol, wavy(desol)))
    try:
        import matplotlib.pyplot as plt
        x = arange(-40,40,0.01)
        plt.plot(x,wavy(x))
        plt.plot(scipysol, wavy(scipysol), 'r+',markersize=8)
        plt.plot(desol, wavy(desol), 'bo',markersize=8)
        plt.legend(('|x + 3 sin(x+pi)|','fmin','dife'))
        if hasattr(solver, 'genealogy'):
            xx = solver.genealogy
            plt.plot(xx[4], wavy(xx[4]), 'g-',markersize=3)
            plt.plot(xx[10], wavy(xx[10]), 'y-',markersize=3)
        try: show()
        except ImportError: plt.show()
    except ImportError:
        print("Install matplotlib for visualization")

# end of file
