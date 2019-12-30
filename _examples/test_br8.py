#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2020 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Data from Chapter 8 of Bevington and Robinson

Wants to fit
y = a1 + a2 Exp[-t / a4] + a3 Exp[-t/a5] to data
"""

from numpy import *
from scipy.integrate import romberg
from mystic.solvers import DifferentialEvolutionSolver2 as DifferentialEvolutionSolver
from mystic.termination import ChangeOverGeneration, VTR
from mystic.tools import getch, random_seed
from mystic.monitors import VerboseMonitor as MyMonitor

from mystic.models.br8 import decay; F = decay.ForwardFactory
from mystic.models.br8 import cost as myCF
from mystic.models.br8 import data
# evalpts = data[:,0], observations = data[:,1]

def myshow():
    import matplotlib.pyplot as plt
    try:
        import Image
        plt.savefig('test_br8_out',dpi=72)
        im = Image.open('test_br8_out.png')
        im.show()
    except ImportError:
        pass
    plt.show()
    
def plot_sol(solver=None, linestyle='k-'):
    import matplotlib.pyplot as plt
    def _(params):
        import mystic._signal as signal
        print("plotting params: %s" % params)
        # because of the log ordinate axis, will draw errorbars the dumb way
        plt.semilogy(data[:,0],data[:,1],'k.')
        for i, j in data:
            plt.semilogy([i, i], [j-sqrt(j), j+sqrt(j)],'k-')
        plt.grid()
        plt.xlabel('Time (s)')
        plt.ylabel('Number of counts')
        x = arange(15, 900, 5)
        f = F(params)
        plt.plot(x, f(x), linestyle)
        if solver is not None:
            signal.signal(signal.SIGINT, signal.Hander(solver))
    return _

ND = 5
NP = 50
MAX_GENERATIONS = 2000

def de_solve(CF, a4=None, a5=None):
    """solve with DE for given cost funciton; fix a4 and/or a5 if provided"""
    minrange = [0, 100, 100, 1, 1]
    maxrange = [100, 2000, 200, 200, 200]
    interval = 10
    if a5 != None:
        minrange[4] = maxrange[4] = a5
        interval = 20
    if a4 != None:
        minrange[3] = maxrange[3] = a4
        if interval == 20: interval = 1000

    solver = DifferentialEvolutionSolver(ND, NP)
    solver.enable_signal_handler()
    stepmon = MyMonitor(interval)
    solver.SetRandomInitialPoints(min=minrange,max=maxrange)
    solver.SetStrictRanges(min=minrange, max=maxrange)
    solver.SetEvaluationLimits(generations=MAX_GENERATIONS)
    solver.SetGenerationMonitor(stepmon)
    solver.Solve(CF,termination=ChangeOverGeneration(generations=50))
    solution = solver.Solution()
    return solution, stepmon


def error2d(myCF, sol, cx, cy):
    from scipy.optimize import fmin
    # This is the center-point
    x0 = [sol[0], sol[1], sol[2]]
    def curry(x):
        return myCF([x[0], x[1], x[2], cx, cy])
    sol = fmin(curry, x0)
    print(sol)

if __name__ == '__main__':
    sol, steps = de_solve(myCF)
    a1, a2, a3, a4, a5 = sol
    minChiSq = steps.y[-1]
    #
    print(sol)
    plot_sol()(sol)
    # plot the "precomputed" confidence interval too
    s4, ss4 = de_solve(myCF, a4=29.5, a5=163)
    s5, ss5 = de_solve(myCF, a4=38.9, a5=290)
    plot_sol(linestyle='r-')(s4)
    plot_sol(linestyle='r-')(s5)
    myshow()
    #

    # compute the 'uncertainty' of the last parameter, and fit a parabola
    # disable for now
    ## print("redo a5 at several points")
    ## a5x = [a5-30, a5, a5+30]
    ## a5y = [] 
    ## for a in a5x:
    ##    sol2, steps2 = de_solve(myCF, a5=a)
    ##    a5y.append(steps2.y[-1])
    ## import matplotlib.pyplot as plt
    ## plt.clf()
    ## plt.plot(a5x, a5y, 'r+')
    ## # fitting a parabola
    ## x1,x2,x3=a5x
    ## y1,y2,y3=a5y
    ## a1 = -(-x2*y1 + x3*y1 + x1*y2-x3*y2-x1*y3+x2*y3)/(x2-x3)/(x1*x1-x1*x2-x1*x3+x2*x3)
    ## a2 = -(x2*x2*y1-x3*x3*y1-x1*x1*y2+x3*x3*y2+x1*x1*y3-x2*x2*y3)/(x1-x2)/(x1-x3)/(x2-x3)
    ## a3 = -(-x2*x2*x3*y1+x2*x3*x3*y1+x1*x1*x3*y2-x1*x3*x3*y2-x1*x1*x2*y3+x1*x2*x2*y3)/((x2-x3)*(x1*x1-x1*x2-x1*x3+x2*x3))
    ## print("%s %s %s" % (a1, a2, a3))
    ## x = arange(150,270)
    ## from mystic.math import polyeval
    ## plt.plot(x, polyeval([a1,a2,a3],x),'k-')
    ## myshow()

    # 2D (a fine mesh solution can be computed by test_br8_mpi.py)
    try:
        import matplotlib.pyplot as plt
        X = loadtxt('test_br8_mpi.out.X')
        Y = loadtxt('test_br8_mpi.out.Y')
        V = loadtxt('test_br8_mpi.out.V')
        plt.clf()
        plt.plot([[a4]],[[a5]],'k+')
        plt.xlabel('a4')
        plt.ylabel('a5')
        plt.grid()
        plt.contour(X,Y,V, minChiSq + array([1,2,3]),colors='black')
        myshow()
    except IOError:
        print("Run test_br8_mpi to create dataset.")

# end of file
