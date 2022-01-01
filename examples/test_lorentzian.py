#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Alternate fitting of a lorentzian peak (see test_lorentzian2.py)
"""

import matplotlib.pyplot as plt, matplotlib
from numpy import *

from mystic.models import lorentzian
from test_lorentzian2 import *
import warnings

def F(alpha):
    "lorentzian, with norm calcualted"
    alpha[-1] = 1. # always set norm = 1
    f = lorentzian.ForwardFactory(alpha)
    from scipy.integrate import romberg
    with warnings.catch_warnings():
        # suppress: "AccuracyWarning: divmax (10) exceeded"
        warnings.simplefilter('ignore')
        n = romberg(f,0,3,divmax=5) #NOTE: this step is _SLOW_ (tweak divmax)
    def _(x):
        return f(x)/n
    return _

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
generations = 80
minrange = [0.5,30, -15, 10, 0, 0, 1]
maxrange = [2, 60., -5, 50, 2, 1, 1]

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
    #NOTE: we will calculate the norm, not solve for it as a parameter
    target = [1., 45., -10., 20., 1., 0.1, 1.0] # norm set to 1.0

    from mystic.models.lorentzian import gendata, histogram
    npts = 4000; binwidth = 0.1
    N = npts * binwidth
    xmin, xmax = 0.0, 3.0
    pdf = F(target)          # normalized
    print("pdf(1): %s" % pdf(1))

    data = gendata(target, xmin, xmax, npts)  # data is 'unnormalized'
    #plt.plot(data[1:N],0*data[1:N],'k.')
    #plt.title('Samples drawn from density to be estimated.')
    #show()
    #plt.clf()

    binsc, histo = histogram(data, binwidth, xmin,xmax)
    print("binsc:  %s" % binsc)
    print("count:  %s" % histo)
    print("ncount: %s" % (histo//N))
    print("exact : %s" % pdf(binsc))

    print("now with DE...")
    from mystic.forward_model import CostFactory
    CF = CostFactory()
    CF.addModel(F, ND, 'lorentz')
    myCF = CF.getCostFunction(binsc, histo//N)
    sol, steps = de_solve(myCF)
    plot_sol()(sol)
    #print("steps: %s %s" % (steps.x, steps.y))

# end of file
