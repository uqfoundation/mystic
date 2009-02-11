#!/usr/bin/env python

"""
Data from Chapter 8 of Bevington and Robinson

Wants to fit
y = a1 + a2 Exp[-t / a4] + a3 Exp[-t/a5] to data
"""

from numpy import *
from scipy.integrate import romberg
from mystic.differential_evolution import DifferentialEvolutionSolver2 as DifferentialEvolutionSolver
from mystic.detools import Best1Exp, Rand1Exp, Best1Bin, ChangeOverGeneration, VTR, Best2Exp, Best2Exp
from mystic import getch, Sow, random_seed
from mystic.polytools import polyeval
from mystic.forward_model import CostFactory

data = array([[15, 775], [30, 479], [45, 380], [60, 302],
[75, 185], [90, 157], [105,137], [120, 119], [135, 110],
[150, 89], [165, 74], [180, 61], [195, 66], [210, 68],
[225, 48], [240, 54], [255, 51], [270, 46], [285, 55],
[300, 29], [315, 28], [330, 37], [345, 49], [360, 26],
[375, 35], [390, 29], [405, 31], [420, 24], [435, 25],
[450, 35], [465, 24], [480, 30], [495, 26], [510, 28],
[525, 21], [540, 18], [555, 20], [570, 27], [585, 17],
[600, 17], [615, 14], [630, 17], [645, 24], [660, 11],
[675, 22], [690, 17], [705, 12], [720, 10], [735, 13],
[750, 16], [765, 9], [780, 9], [795, 14], [810, 21],
[825, 17], [840, 13], [855, 12], [870, 18], [885, 10]])

class MySow(Sow):
    def __init__(self, interval = 1000):
        Sow.__init__(self)
        self._step = 0
        self._interval = interval
    def __call__(self, x, y):
        Sow.__call__(self, x, y)
        self._step += 1
        if self._step % self._interval == 0:
            print "Generation %d has best Chi-Squared at : %f" % (self._step, y)
        return

def myshow():
    import pylab, Image
    pylab.savefig('test_br8_out',dpi=72)
    im = Image.open('test_br8_out.png')
    im.show()
    
def show_figure81():
    import pylab
    # because of the log ordinate axis, will draw errorbars the dumb way
    pylab.semilogy(data[:,0],data[:,1],'k.')
    for i, j in data:
        pylab.semilogy([i, i], [j-sqrt(j), j+sqrt(j)],'k-')
    pylab.grid()
    pylab.xlabel('Time (s)')
    pylab.ylabel('Number of counts')
    myshow()

def F(alpha):
    a1,a2,a3,a4,a5 = alpha
    def _(t):
        return a1 + a2*exp(-t/a4) + a3*exp(-t/a5)
    return _

def plot_sol(solver=None, linestyle='k-'):
    import pylab
    def _(params):
        import signal
        print "plotting params: ", params
        pylab.semilogy(data[:,0],data[:,1],'k.')
        for i, j in data:
            pylab.semilogy([i, i], [j-sqrt(j), j+sqrt(j)],'k-')
        pylab.grid()
        pylab.xlabel('Time (s)')
        pylab.ylabel('Number of counts')
        x = arange(15, 900, 5)
        f = F(params)
        pylab.plot(x, f(x), linestyle)
        if solver is not None:
            signal.signal(signal.SIGINT, solver.signal_handler)
    return _

def de_solve(CF, ND, NP):
    MAX_GENERATIONS = 2000
    solver = DifferentialEvolutionSolver(ND, NP)
    solver.enable_signal_handler()
    stepmon = MySow(10)
    minrange = [0, 100, 100, 1, 1]
    maxrange = [100, 2000, 200, 200, 200]
    solver.SetRandomInitialPoints(min=minrange,max=maxrange)
    solver.Solve(CF, Best1Bin, termination=ChangeOverGeneration(generations=50), \
                 maxiter = MAX_GENERATIONS, StepMonitor=stepmon, sigint_callback = plot_sol(solver))
    solution = solver.Solution()
    return solution, stepmon

def de_solve5(a5, CF, ND, NP):
    "fix a5 at a certain value and solve)"
    MAX_GENERATIONS = 2000
    solver = DifferentialEvolutionSolver(ND, NP)
    solver.enable_signal_handler()
    stepmon = MySow(20)
    minrange = [0, 100, 100, 1, a5 ]
    maxrange = [100, 2000, 200, 200,  a5 ]
    solver.SetRandomInitialPoints(min=minrange,max=maxrange)
    solver.SetStrictRanges(min=minrange, max=maxrange)
    solver.Solve(CF, Best1Exp, termination=ChangeOverGeneration(generations=50), \
                 maxiter = MAX_GENERATIONS, StepMonitor=stepmon, sigint_callback = plot_sol(solver))
    solution = solver.Solution()
    return solution, stepmon

def de_solve45(a4, a5, CF, ND, NP):
    "fix a5 at a certain value and solve)"
    MAX_GENERATIONS = 2000
    solver = DifferentialEvolutionSolver(ND, NP)
    solver.enable_signal_handler()
    stepmon = MySow(1000)
    minrange = [0, 100, 100, a4, a5 ]
    maxrange = [100, 2000, 200, a4,  a5 ]
    solver.SetRandomInitialPoints(min=minrange,max=maxrange)
    solver.SetStrictRanges(min=minrange, max=maxrange)
    solver.Solve(CF, Best1Exp, termination=ChangeOverGeneration(generations=50), \
                 maxiter = MAX_GENERATIONS, StepMonitor=stepmon)
    solution = solver.Solution()
    return solution, stepmon

def error2d(myCF, sol, cx, cy):
    from scipy.optimize import fmin
    # This is the center-point
    x0 = [sol[0], sol[1], sol[2]]
    def curry(x):
        return myCF([x[0], x[1], x[2], cx, cy])
    sol = fmin(curry, x0)
    print sol

if __name__ == '__main__':
    CF = CostFactory()
    CF.addModel(F, 'decay', 5)
    myCF = CF.getCostFunction(data[:,0], data[:,1], sqrt(data[:,1]))
    sol, steps = de_solve(myCF, 5, 50)
    a1, a2, a3, a4, a5 = sol
    minChiSq = steps.y[-1]
    #
    print sol
    plot_sol()(sol)
    # plot the "precomputed" confidence interval too
    s4, ss4 = de_solve45(29.5, 163,  myCF, 5, 50)
    s5, ss5 = de_solve45(38.9, 290, myCF, 5, 50)
    plot_sol(linestyle='r-')(s4)
    plot_sol(linestyle='r-')(s5)
    myshow()
    #

    # compute the 'uncertainty' of the last parameter, and fit a parabola
    # disable for now
    ## print "redo a5 at several points"
    ## a5x = [a5-30, a5, a5+30]
    ## a5y = [] 
    ## for a in a5x:
    ##    sol2, steps2 = de_solve5(a, myCF, 5, 50)
    ##    a5y.append(steps2.y[-1])
    ## import pylab
    ## pylab.clf()
    ## pylab.plot(a5x, a5y, 'r+')
    ## # fitting a parabola
    ## x1,x2,x3=a5x
    ## y1,y2,y3=a5y
    ## a1 = -(-x2*y1 + x3*y1 + x1*y2-x3*y2-x1*y3+x2*y3)/(x2-x3)/(x1*x1-x1*x2-x1*x3+x2*x3)
    ## a2 = -(x2*x2*y1-x3*x3*y1-x1*x1*y2+x3*x3*y2+x1*x1*y3-x2*x2*y3)/(x1-x2)/(x1-x3)/(x2-x3)
    ## a3 = -(-x2*x2*x3*y1+x2*x3*x3*y1+x1*x1*x3*y2-x1*x3*x3*y2-x1*x1*x2*y3+x1*x2*x2*y3)/((x2-x3)*(x1*x1-x1*x2-x1*x3+x2*x3))
    ## print a1, a2, a3
    ## x = arange(150,270)
    ## pylab.plot(x, polyeval([a1,a2,a3],x),'k-')
    ## myshow()

    # 2D (a fine mesh solution can be computed by test_br8_mpi.py)
    try:
        import scipy.io, pylab
        X = scipy.io.read_array(open('test_br8_mpi.out.X'))
        Y = scipy.io.read_array(open('test_br8_mpi.out.Y'))
        V = scipy.io.read_array(open('test_br8_mpi.out.V'))
        pylab.clf()
        pylab.plot([[a4]],[[a5]],'k+')
        pylab.xlabel('a4')
        pylab.ylabel('a5')
        pylab.grid()
        pylab.contour(X,Y,V, minChiSq + array([1,2,3]),colors='black')
        myshow()
    except IOError:
        print "Run test_br8_mpi to create dataset."

# end of file
