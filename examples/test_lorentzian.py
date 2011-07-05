#!/usr/bin/env python

"""
Alternate fitting of a lorentzian peak (see test_lorentzian2.py)
"""

import pylab, matplotlib, Image
from numpy import *

from mystic.models import lorentzian
from test_lorentzian2 import *

def F(alpha):
    "lorentzian, with norm calcualted"
    alpha[-1] = 1. # always set norm = 1
    f = lorentzian.ForwardFactory(alpha)
    from scipy.integrate import romberg
    n = romberg(f, 0, 3) #NOTE: this step is _SLOW_
    def _(x):
        return f(x)/n
    return _

def show():
    pylab.savefig('test_lorentzian_out',dpi=72)
    im = Image.open('test_lorentzian_out.png')
    im.show()
    return

def plot_sol(solver=None):
    def _(params):
        import signal
        print "plotting params: ", params
        pylab.errorbar(binsc, histo, sqrt(histo), fmt='b+')
        x = arange(xmin, xmax, (0.1* binwidth))
        pylab.plot(x, pdf(x)*N,'b:')
        pylab.plot(x, F(params)(x)*N,'r-')
        pylab.xlabel('E (GeV)')
        pylab.ylabel('Counts')
        show()
        if solver is not None:
            signal.signal(signal.SIGINT, solver.signal_handler)
    return _

ND = 7
NP = 50
MAX_GENERATIONS = 200
generations = 80
minrange = [0.5,30, -15, 10, 0, 0, 1]
maxrange = [2, 60., -5, 50, 2, 1, 1]

def de_solve(CF):
    solver = DifferentialEvolutionSolver(ND, NP)
    solver.enable_signal_handler()
    stepmon = Monitor()
    solver.SetRandomInitialPoints(min=minrange,max=maxrange)
    solver.SetStrictRanges(min=minrange,max=maxrange)
    solver.SetEvaluationLimits(maxiter=MAX_GENERATIONS)
    solver.SetGenerationMonitor(stepmon)
    termination=ChangeOverGeneration(generations=generations)
    solver.Solve(CF, termination=termination, strategy=Rand1Exp, \
                 sigint_callback = plot_sol(solver))
    solution = solver.Solution()
    return solution, stepmon


if __name__ == '__main__':
    #NOTE: we will calculate the norm, not solve for it as a parameter
    target = [1., 45., -10., 20., 1., 0.1, 1.0] # norm set to 1.0

    from mystic.models.lorentzian import gendata, histogram
    npts = 4000; binwidth = 0.1
    N = npts * binwidth
    xmin, xmax = 0.0, 3.0
    pdf = F(target)          # normalized
    print "pdf(1): ", pdf(1)

    data = gendata(target, xmin, xmax, npts)  # data is 'unnormalized'
    #pylab.plot(data[1:N],0*data[1:N],'k.')
    #pylab.title('Samples drawn from density to be estimated.')
    #show()
    #pylab.clf()

    binsc, histo = histogram(data, binwidth, xmin,xmax)
    print "binsc:  ", binsc
    print "count:  ", histo
    print "ncount: ", histo/N
    print "exact : ", pdf(binsc)

    print "now with DE..."
    from mystic.forward_model import CostFactory
    CF = CostFactory()
    CF.addModel(F, 'lorentz', ND)
    myCF = CF.getCostFunction(binsc, histo/N)
    sol, steps = de_solve(myCF)
    plot_sol()(sol)
    #print "steps: ", steps.x, steps.y

# end of file
