#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Tests a single mogi fitting. 

Script is pretty crude right now.

Requirement: 
   numpy  --- for data-structures and "vectorized" ops on them, like sqrt, pow
   matplotlib for plotting

Computes surface displacements Ux, Uy, Uz in meters from a point spherical
pressure source in an elastic half space [1].

Reference:

[1] Mogi, K. Relations between the eruptions of various
volcanoes and the deformations of the ground surfaces around them, 
Bull. Earthquake. Res. Inst., 36, 99-134, 1958.
"""

import matplotlib.pyplot as plt
from mystic.solvers import DifferentialEvolutionSolver
from mystic.termination import ChangeOverGeneration, VTR
from mystic.monitors import Monitor, VerboseMonitor
from mystic.tools import getch, random_seed

from numpy import pi, sqrt, array, mgrid, random, real, conjugate, arange
from numpy.random import rand
import numpy

random_seed(123)

from mystic.models import mogi; ForwardMogiFactory = mogi.ForwardFactory

# Let the "actual parameters" be :
actual_params = [1234.,-500., 10., .1]
actual_forward = ForwardMogiFactory(actual_params)

# The data to be "fitted" 
xstations = array([random.uniform(-30,30) for i in range(300)])+actual_params[0]
ystations =  0*xstations + actual_params[1]
stations  = array((xstations, ystations))

data = actual_forward(stations)
# noisy data, gaussian distribution with mean 0, sig 0.1e-3
noise =  array([[random.normal(0,0.1e-3) for i in range(data.shape[1])] for j in range(data.shape[0])])

# Here is the "observed data"
data_z = -data[2,:] + noise[2,:]
# the stations are still at : stations

def plot_noisy_data():
    import matplotlib.pyplot as plt
    plt.plot(stations[0,:],data_z,'k.')

# we need a filter for the forward model
def filter_for_zdisp(input):
    return -input[2,:]

# Here is the cost function
def vec_cost_function(params):
    model = ForwardMogiFactory(params)
    zdisp = filter_for_zdisp(model(stations))
    return 100. * (zdisp - data_z)

# Here is the normed version
def cost_function(params):
    x = vec_cost_function(params)
    return numpy.sum(real((conjugate(x)*x)))

# a cost function with parameters "normalized"
def vec_cost_function2(params):
    sca = numpy.array([1000, 100., 10., 0.1])
    return vec_cost_function(sca * params)


ND = 4
NP = 50
MAX_GENERATIONS = 2500

def de_solve():
    solver = DifferentialEvolutionSolver(ND, NP)

    stepmon = Monitor()
    minrange = [-1000., -1000., -100., -10.];
    maxrange = [1000., 1000., 100., 10.];
    solver.SetRandomInitialPoints(min = minrange, max = maxrange)
    solver.SetEvaluationLimits(generations=MAX_GENERATIONS)
    solver.SetGenerationMonitor(stepmon)

   #termination = VTR(0.0000029)
    termination = ChangeOverGeneration(generations=100)

    solver.Solve(cost_function, termination=termination, \
                 CrossProbability=0.5, ScalingFactor=0.5)

    solution = solver.Solution()
  
    return solution, stepmon


def plot_sol(params, linestyle = 'b-'):
    forward_solution = ForwardMogiFactory(params)
    xx = arange(-30,30,0.1)+actual_params[0]
    yy = 0*xx + actual_params[1]
    ss  = array((xx, yy))
    dd = forward_solution(ss)
    plt.plot(ss[0,:],-dd[2,:],'%s'%linestyle,linewidth=2.0)

if __name__ == '__main__':

    from mystic.solvers import NelderMeadSimplexSolver as fmin
    from mystic.termination import CandidateRelativeTolerance as CRT
    try:
        from scipy.optimize import leastsq, fmin_cg
    except ImportError:
        from mystic._scipyoptimize import fmin_cg
        leastsq = None
    #
    desol, dstepmon = de_solve()
    print("desol: %s" % desol)
    print("dstepmon 50: %s" % dstepmon.x[50])
    print("dstepmon 100: %s" % dstepmon.x[100])
    #
    # this will try to use nelder_mean from a relatively "near by" point (very sensitive)
    point = [1234., -500., 10., 0.001] # both cg and nm does fine
    point = [1000,-100,0,1] # cg will do badly on this one
    # this will try nelder-mead from an unconverged DE solution 
    #point = dstepmon.x[-150]
    #
    simplex, esow = Monitor(), Monitor()
    solver = fmin(len(point))
    solver.SetInitialPoints(point)
    solver.SetEvaluationMonitor(esow)
    solver.SetGenerationMonitor(simplex)
    solver.Solve(cost_function, CRT())
    sol = solver.Solution()

    print("\nsimplex solution: %s" % sol)
    #
    solcg = fmin_cg(cost_function, point)
    print("\nConjugate-Gradient (Polak Rubiere) : %s" % solcg)
    #
    if leastsq:
        sollsq = leastsq(vec_cost_function, point)
        sollsq = sollsq[0]
        print("\nLeast Squares (Levenberg Marquardt) : %s" % sollsq)
    #
    legend = ['Noisy data', 'Differential Evolution', 'Nelder Mead', 'Polak Ribiere']
    plot_noisy_data()
    plot_sol(desol,'r-')
    plot_sol(sol,'k--')
    plot_sol(solcg,'b-.')
    if leastsq:
        plot_sol(sollsq,'g-.')
        legend += ['Levenberg Marquardt']
    plt.legend(legend) 
    plt.show()

# end of file
