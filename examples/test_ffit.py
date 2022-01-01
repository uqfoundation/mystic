#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Adapted from DETest.py by Patrick Hung

Sets up Storn and Price's Polynomial 'Fitting' Problem.

Exact answer: Chebyshev Polynomial of the first kind. T8(x)

Reference:

[1] Storn, R. and Price, K. Differential Evolution - A Simple and Efficient
Heuristic for Global Optimization over Continuous Spaces. Journal of Global
Optimization 11: 341-359, 1997.
"""

from mystic.solvers import DifferentialEvolutionSolver
from mystic.termination import ChangeOverGeneration, VTR
from mystic.strategy import Best1Exp, Best1Bin, Rand1Exp
from mystic.math import poly1d
from mystic.monitors import VerboseMonitor
from mystic.tools import getch

from mystic.tools import random_seed
random_seed(123)

# get the target coefficients and cost function
from mystic.models.poly import chebyshev8coeffs as Chebyshev8
from mystic.models.poly import chebyshev8cost as ChebyshevCost

def print_solution(func):
    print(poly1d(func))
    return

def plot_solution(func, benchmark=Chebyshev8):
    try:
        import matplotlib.pyplot as plt, numpy
        x = numpy.arange(-1.2, 1.2001, 0.01)
        x2 = numpy.array([-1.0, 1.0])
        p = poly1d(func)
        chebyshev = poly1d(benchmark)
        y = p(x)
        exact = chebyshev(x)
        plt.plot(x,y,'b-',linewidth=2)
        plt.plot(x,exact,'r-',linewidth=2)
        plt.plot(x,0*x-1,'k-')
        plt.plot(x2, 0*x2+1,'k-')
        plt.plot([-1.2, -1.2],[-1, 10],'k-')
        plt.plot([1.2, 1.2],[-1, 10],'k-')
        plt.plot([-1.0, -1.0],[1, 10],'k-')
        plt.plot([1.0, 1.0],[1, 10],'k-')
        plt.axis([-1.4, 1.4, -2, 8])#,'k-')
        plt.legend(('Fitted','Chebyshev'))
        plt.show()
    except ImportError:
        print("Install matplotlib for visualization")
        pass


ND = 9
NP = ND*10
MAX_GENERATIONS = ND*NP

def main():
    solver = DifferentialEvolutionSolver(ND, NP)
    solver.SetRandomInitialPoints(min = [-100.0]*ND, max = [100.0]*ND)
    solver.SetEvaluationLimits(generations=MAX_GENERATIONS)
    solver.SetGenerationMonitor(VerboseMonitor(30))
    solver.enable_signal_handler()
  
    strategy = Best1Exp
    #strategy = Best1Bin

    solver.Solve(ChebyshevCost, termination=VTR(0.01), strategy=strategy, \
                 CrossProbability=1.0, ScalingFactor=0.9, \
                 sigint_callback=plot_solution)

    solution = solver.Solution()
    return solution
  


if __name__ == '__main__':
   #plot_solution(Chebyshev8)
    solution = main()
    print_solution(solution)
    plot_solution(solution)
    getch()

# end of file
