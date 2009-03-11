#!/usr/bin/env python

"""
Adapted from DETest.py by Patrick Hung

Sets up Storn and Price's Polynomial 'Fitting' Problem.

Exact answer: Chebyshev Polynomial of the first kind. T8(x)

Reference:

[1] Storn, R. and Price, K. Differential Evolution - A Simple and Efficient
Heuristic for Global Optimization over Continuous Spaces. Journal of Global
Optimization 11: 341-359, 1997.
"""

from mystic.differential_evolution import DifferentialEvolutionSolver
from mystic.termination import ChangeOverGeneration, VTR
from mystic.strategy import Best1Exp, Best1Bin, Rand1Exp
from mystic.models.poly import poly1d
from mystic import getch, VerboseSow

import random
random.seed(123)

# get the target coefficients and cost function
from mystic.models.poly import chebyshev8coeffs as Chebyshev8
from mystic.models.poly import chebyshev8cost as ChebyshevCost

def print_solution(func):
    print poly1d(func)
    return

def plot_solution(func, benchmark=Chebyshev8):
    try:
        import pylab, numpy
        x = numpy.arange(-1.2, 1.2001, 0.01)
        x2 = numpy.array([-1.0, 1.0])
        p = poly1d(func)
        chebyshev = poly1d(benchmark)
        y = p(x)
        exact = chebyshev(x)
        pylab.plot(x,y,'b-',linewidth=2)
        pylab.plot(x,exact,'r-',linewidth=2)
        pylab.plot(x,0*x-1,'k-')
        pylab.plot(x2, 0*x2+1,'k-')
        pylab.plot([-1.2, -1.2],[-1, 10],'k-')
        pylab.plot([1.2, 1.2],[-1, 10],'k-')
        pylab.plot([-1.0, -1.0],[1, 10],'k-')
        pylab.plot([1.0, 1.0],[1, 10],'k-')
        pylab.axis([-1.4, 1.4, -2, 8],'k-')
        pylab.legend(('Fitted','Chebyshev'))
        pylab.show()
    except ImportError:
        print "Install matplotlib/numpy for visualization"
        pass


ND = 9
NP = ND*10
MAX_GENERATIONS = ND*NP

def main():
    solver = DifferentialEvolutionSolver(ND, NP)
    solver.SetRandomInitialPoints(min = [-100.0]*ND, max = [100.0]*ND)
    solver.SetEvaluationLimits(maxiter=MAX_GENERATIONS)
    solver.enable_signal_handler()
  
    strategy = Best1Exp
    #strategy = Best1Bin

    solver.Solve(ChebyshevCost, strategy, termination = VTR(0.01) , \
                 CrossProbability=1.0, ScalingFactor=0.9 , \
                 StepMonitor=VerboseSow(30), sigint_callback=plot_solution)

    solution = solver.Solution()
    return solution
  


if __name__ == '__main__':
   #plot_solution(Chebyshev8)
    solution = main()
    print_solution(solution)
    plot_solution(solution)
    getch()

# end of file
