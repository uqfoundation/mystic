#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Two mogi sources. Similar to test_mogi.py. (See that one first)

Reference:

[1] Mogi, K. Relations between the eruptions of various
volcanoes and the deformations of the ground surfaces around them, 
Bull. Earthquake. Res. Inst., 36, 99-134, 1958.
"""

from test_mogi import *

# Let the "actual parameters" be :
params0 = [1000.,-100., 10., .2]
params1 = [1500.,-400., 40., 1.5]

forward0 = ForwardMogiFactory(params0)
forward1 = ForwardMogiFactory(params1)

# The data to be "fitted" 
xstations = array([random.uniform(-500,500) for i in range(300)])+1250.
ystations =  0*xstations - 200.
stations  = array((xstations, ystations))

data = forward0(stations) + forward1(stations)
# noisy data, gaussian distribution
noise =  array([[random.normal(0,0.05e-5) for i in range(data.shape[1])] for j in range(data.shape[0])])

# observed data
data_z = -data[2,:] + noise[2,:]

def cost_function(params):
    m0 = ForwardMogiFactory(params[0:4])
    m1 = ForwardMogiFactory(params[4:])
    zdisp = filter_for_zdisp(m0(stations) + m1(stations))
    x = zdisp - data_z
    return 100000. * numpy.sum(real((conjugate(x)*x)))

def plot_noisy_data():
    import matplotlib.pyplot as plt
    plt.plot(stations[0,:],-data[2,:]+noise[2,:],'k.')
    plt.draw()
    plt.pause(0.001)

def plot_sol(params,linestyle='b-'):
    import matplotlib.pyplot as plt
    s0 = ForwardMogiFactory(params[0:4])
    s1 = ForwardMogiFactory(params[4:])
    xx = arange(-500,500,1)+1250.
    yy = 0*xx - 200.
    ss  = array((xx, yy))
    dd = s0(ss) + s1(ss)
    plt.plot(ss[0,:],-dd[2,:],'%s'%linestyle,linewidth=2.0)
    plt.draw()
    plt.pause(0.001)

ND = 8
NP = 80
MAX_GENERATIONS = 5000

def de_solve():
    solver = DifferentialEvolutionSolver(ND, NP)

    solver.enable_signal_handler()

    stepmon = VerboseMonitor()
    minrange = [-1000., -1000., -100., -1.]*2;
    maxrange = [1000., 1000., 100., 1.]*2;
    solver.SetRandomInitialPoints(min = minrange, max = maxrange)
    solver.SetEvaluationLimits(generations=MAX_GENERATIONS)
    solver.SetGenerationMonitor(stepmon)

    solver.Solve(cost_function, \
                 termination=ChangeOverGeneration(generations=300), \
                 CrossProbability=0.5, ScalingFactor=0.5, \
                 sigint_callback = plot_sol)

    solution = solver.Solution()
  
    return solution, stepmon

if __name__ == '__main__':

    plt.ion()
    plot_noisy_data()
    desol, dstepmon = de_solve()
    print("desol: %s" % desol)
   #plot_sol(dstepmon.x[-100],'k-')

    plot_sol(desol,'r-')

    getch()

# end of file
