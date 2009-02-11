#!/usr/bin/env python

"""
Testing the fitting of lorentzian peak.

"""

import pylab, matplotlib, Image
from numpy import *
from scipy.integrate import romberg
from mystic.differential_evolution import DifferentialEvolutionSolver
from mystic.detools import Best1Exp, Rand1Exp, ChangeOverGeneration, VTR, Best2Exp, Best2Exp
from mystic import getch, Sow, random_seed
from mystic.forward_model import CostFactory
from mystic.forward_model import *

random.seed(123)

def F(alpha):
    a1,a2,a3,A0,E0,G0 = alpha
    def _(x):
        return a1 + a2*x + a3*x*x + A0 * ( G0/(2*pi) )/( (x-E0)*(x-E0)+(G0/2)*(G0/2) )
    return _

def getPDF(alpha):
    f = F(alpha)
    n = romberg(f, 0, 3) # this is the reason for the slowness.
    def _(x):
        return f(x)/n
    return _

def getsample(F, xmin, xmax):
    import numpy
    a = numpy.arange(xmin, xmax, (xmax-xmin)/200.)
    ymin = 0
    ymax = F(a).max()
    while 1:
        t1 = random.random() * (xmax-xmin) + xmin
        t2 = random.random() * (ymax-ymin) + ymin
        t3 = F(t1)
        if t2 < t3:
            return t1


def plot_sol(solver=None):
    def _(params):
        import signal
        print "plotting params: ", params
        pylab.errorbar(binsc, histo,sqrt(histo), fmt='b+')
        x = arange(0,3,0.01)
        pylab.plot(x, pdf(x)*400,'b:')
        pylab.plot(x, getPDF(params)(x)*400,'r-')
        pylab.xlabel('E (GeV)')
        pylab.ylabel('Counts')
        pylab.savefig('test_lorentzian_out',dpi=72)
        im = Image.open('test_lorentzian_out.png')
        im.show()
        if solver is not None:
            signal.signal(signal.SIGINT, solver.signal_handler)
    return _

def de_solve(CF, ND, NP):
    MAX_GENERATIONS = 200
    solver = DifferentialEvolutionSolver(ND, NP)
    solver.enable_signal_handler()
    stepmon = Sow()
    minrange = [0.5,30, -15, 10, 0, 0]
    maxrange = [2, 60., -5, 50, 2, 1]
    solver.SetRandomInitialPoints(min=minrange,max=maxrange)
    solver.Solve(CF, Rand1Exp, termination=ChangeOverGeneration(generations=80), \
                 maxiter = MAX_GENERATIONS, StepMonitor=stepmon, sigint_callback = plot_sol(solver))
    solution = solver.Solution()
    return solution, stepmon


if __name__ == '__main__':
    params = [1., 45., -10., 20., 1., 0.1]
    f = F(params)
    pdf = getPDF(params)
    print "pdf(1): ", pdf(1)
    data = array([getsample(f, 0,3) for i in xrange(4000)])
    bins = arange(0,3, 0.1)
    binsc = bins + 0.05
    histo = histogram(data, bins)[0]
    print "binsc:  ", binsc
    print "count:  ", histo
    print "ncount: ", histo/400. # divides by N, and then divide by bin width
    print "exact : ", pdf(binsc)
    #
    print "now with DE..."
    CF = CostFactory()
    CF.addModel(getPDF, 'lorentzian', 6)
    myCF = CF.getCostFunction(binsc, histo/400.)
    sol, steps = de_solve(myCF, 6, 50)
    plot_sol()(sol)
    #pylab.errorbar(binsc, histo,sqrt(histo), fmt='b+')
    #x = arange(0,3,0.01)
    #pylab.plot(x, pdf(x)*400,'b:')
    #pylab.plot(x, getPDF(sol)(x)*400,'r-')
    #pylab.xlabel('E (GeV)')
    #pylab.ylabel('Counts')
    #pylab.show()


# end of file
