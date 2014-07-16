#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2014 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE
"""
Same as test_lorentzian, but with n being a fitted variable

This is MUCH faster than test_lorentzian because the cost function no
longer has to do an "integral" as an intermediate step
"""

import pylab, matplotlib
from numpy import *
from mystic.solvers import DifferentialEvolutionSolver
from mystic.termination import ChangeOverGeneration, VTR
from mystic.strategy import Best1Exp, Rand1Exp, Best2Exp, Best2Exp
from mystic.monitors import Monitor
from mystic.tools import getch, random_seed

random.seed(123)

#matplotlib.interactive(True)

from mystic.models import lorentzian
F = lorentzian.ForwardFactory

def show():
    import Image
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
        try: show()
        except ImportError: pylab.show()
        if solver is not None:
            signal.signal(signal.SIGINT, solver.signal_handler)
    return _

ND = 7
NP = 50
MAX_GENERATIONS = 200
generations = 200
minrange = [0.5,30, -15, 10, 0, 0, 100]
maxrange = [2, 60., -5, 50, 2, 1, 200]

def de_solve(CF):
    solver = DifferentialEvolutionSolver(ND, NP)
    solver.enable_signal_handler()
    stepmon = Monitor()
    solver.SetRandomInitialPoints(min=minrange,max=maxrange)
    solver.SetStrictRanges(min=minrange,max=maxrange)
    solver.SetEvaluationLimits(generations=MAX_GENERATIONS)
    solver.SetGenerationMonitor(stepmon)
    termination=ChangeOverGeneration(generations=generations)
    solver.Solve(CF, termination=termination, strategy=Rand1Exp, \
                 sigint_callback = plot_sol(solver))
    solution = solver.Solution()
    return solution, stepmon


if __name__ == '__main__':
    target = [1., 45., -10., 20., 1., 0.1, 120.]

    from mystic.models.lorentzian import gendata, histogram
    npts = 4000; binwidth = 0.1
    N = npts * binwidth
    xmin, xmax = 0.0, 3.0
    pdf = F(target)
    print "pdf(1): ", pdf(1)

    data = gendata(target, xmin, xmax, npts)
    pylab.plot(data[1:N],0*data[1:N],'k.')
    pylab.title('Samples drawn from density to be estimated.')
    try: show()
    except ImportError: pylab.show()
    pylab.clf()

    binsc, histo = histogram(data, binwidth, xmin,xmax)
    print "binsc:  ", binsc
    print "count:  ", histo
    print "ncount: ", histo/N
    print "exact : ", pdf(binsc)

    print "now with DE..."
    myCF = lorentzian.CostFactory2(binsc, histo/N, ND)
    sol, steps = de_solve(myCF)
    plot_sol()(sol)
    #print "steps: ", steps.x, steps.y

# end of file
