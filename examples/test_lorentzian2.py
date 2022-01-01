#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Same as test_lorentzian, but with n being a fitted variable

This is MUCH faster than test_lorentzian because the cost function no
longer has to do an "integral" as an intermediate step
"""

import matplotlib.pyplot as plt, matplotlib
from numpy import *
from mystic.solvers import DifferentialEvolutionSolver
from mystic.termination import ChangeOverGeneration, VTR
from mystic.strategy import Best1Exp, Rand1Exp, Best2Exp, Best2Exp
from mystic.monitors import Monitor
from mystic.tools import getch

from mystic.tools import random_seed
random_seed(123)

#matplotlib.interactive(True)

from mystic.models import lorentzian
F = lorentzian.ForwardFactory

def show():
    import Image
    plt.savefig('test_lorentzian_out',dpi=72)
    im = Image.open('test_lorentzian_out.png')
    im.show()
    return

def plot_sol(solver=None):
    def _(params):
        import mystic._signal as signal
        print("plotting params: %s" % params)
        plt.errorbar(binsc, histo, sqrt(histo), fmt='b+')
        x = arange(xmin, xmax, (0.1* binwidth))
        plt.plot(x, pdf(x)*N,'b:')
        plt.plot(x, F(params)(x)*N,'r-')
        plt.xlabel('E (GeV)')
        plt.ylabel('Counts')
        try: show()
        except ImportError: plt.show()
        if solver is not None:
            signal.signal(signal.SIGINT, signal.Handler(solver))
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
    print("pdf(1): %s" % pdf(1))

    data = gendata(target, xmin, xmax, npts)
    plt.plot(data[1:int(N)],0*data[1:int(N)],'k.')
    plt.title('Samples drawn from density to be estimated.')
    try: show()
    except ImportError: plt.show()
    plt.clf()

    binsc, histo = histogram(data, binwidth, xmin,xmax)
    print("binsc:  %s" % binsc)
    print("count:  %s" % histo)
    print("ncount: %s" % (histo//N))
    print("exact : %s" % pdf(binsc))

    print("now with DE...")
    myCF = lorentzian.CostFactory2(binsc, histo//N, ND)
    sol, steps = de_solve(myCF)
    plot_sol()(sol)
    #print("steps: %s %s" % (steps.x, steps.y))

# end of file
