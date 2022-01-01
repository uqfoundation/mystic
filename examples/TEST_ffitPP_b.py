#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Same as test_ffit.py
but uses DifferentialEvolutionSolver2 instead.

Note:
 1. MPIDifferentialEvolutionSolver is functionally identical to DifferentialEvolultionSolver2.
 2. In DifferentialEvolutionSolver, as each trial vector is compared to its target, once the trial beats
    the target it enters the generation, replacing the old vector and immediately becoming available
    as candidates for creating difference vectors, and for mutations, etc.
"""

from test_ffit import *

# get the target coefficients and cost function
from mystic.math import poly1d

def ChebyshevCost(trial,M=61):
    """The costfunction for order-n Chebyshev fitting.
M evaluation points between [-1, 1], and two end points"""

    from mystic.models.poly import chebyshev8coeffs as target
    from mystic.math import polyeval
    result=0.0
    x=-1.0
    dx = 2.0 / (M-1)
    for i in range(M):
        px = polyeval(trial, x)
        if px<-1 or px>1:
            result += (1 - px) * (1 - px)
        x += dx

    px = polyeval(trial, 1.2) - polyeval(target, 1.2)
    if px<0: result += px*px

    px = polyeval(trial, -1.2) - polyeval(target, -1.2)
    if px<0: result += px*px

    return result


def main(servers,ncpus):
    from mystic.solvers import DifferentialEvolutionSolver2
   #from pathos.pools import ProcessPool as Pool
    from pathos.pools import ParallelPool as Pool

    solver = DifferentialEvolutionSolver2(ND, NP)
    solver.SetMapper(Pool().map)
    solver.SetRandomInitialPoints(min = [-100.0]*ND, max = [100.0]*ND)
    solver.SetEvaluationLimits(generations=MAX_GENERATIONS)
    solver.SetGenerationMonitor(VerboseMonitor(30))
    solver.enable_signal_handler()
  
    strategy = Best1Exp
    #strategy = Best1Bin

    solver.SelectServers(servers,ncpus)
    solver.Solve(ChebyshevCost, termination=VTR(0.01), strategy=strategy, \
                 CrossProbability=1.0, ScalingFactor=0.9 , \
                 sigint_callback=plot_solution)

    solution = solver.Solution()

    return solution
  

if __name__ == '__main__':
    from pathos.helpers import freeze_support, shutdown
    freeze_support() # help Windows use multiprocessing

    # number of local processors
    ncpus = 'autodetect' #XXX: None == autodetect; otherwise select n=0,1,2,...

    import sys
    servers = []
    # get tunneled ports from sys.argv
    for i in range(1,len(sys.argv)):
        tunnelport = int(sys.argv[i])
        servers.append("localhost:%s" % tunnelport)
    servers = tuple(servers)

    import time
    t1 = time.time()
    solution = main(servers,ncpus) # solve
    t2 = time.time()

    shutdown() # help multiprocessing shutdown all workers
    print_solution(solution)
    print("Finished in %0.3f s" % ((t2-t1)))
    plot_solution(solution)

# end of file
