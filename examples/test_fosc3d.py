#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Adapted from The Mathematica Guidebook, Numerics.

"""

from mystic.solvers import DifferentialEvolutionSolver

from mystic.termination import ChangeOverGeneration, VTR
from mystic.strategy import Best1Exp, Best1Bin, Rand1Exp

from mystic.tools import random_seed
random_seed(123)

from mystic.models import fosc3d as fOsc3D

def draw_contour():
    import matplotlib.pyplot as plt, numpy
    x, y = numpy.mgrid[-1:2:0.02,-0.5:2:0.02]
    c = 0*x
    s,t = x.shape
    for i in range(s):
       for j in range(t):
          xx,yy = x[i,j], y[i,j]
          c[i,j] = fOsc3D([xx,yy])
    plt.contourf(x,y,c,100)

ND = 2
NP = ND*10
MAX_GENERATIONS = 2000

def main():
    solver = DifferentialEvolutionSolver(ND, NP)
    solver.SetRandomInitialPoints(min = [-2.0]*ND, max = [2.0]*ND)
    solver.SetEvaluationLimits(generations=MAX_GENERATIONS)
  
    strategy = Best1Exp
    #strategy = Best1Bin

    solver.Solve(fOsc3D,termination=ChangeOverGeneration(1e-5, 30), \
                 strategy=strategy,CrossProbability=1.0,ScalingFactor=0.9)

    return solver.Solution()
  


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mystic.solvers import fmin
   #from mystic._scipyoptimize import fmin
    draw_contour()
    solution = main()
    print("solution: %s" % solution)
    plt.plot([solution[0]],[solution[1]],'wo',markersize=10)
    print("Differential Evolution: Min: %s, sol = %s" % (fOsc3D(solution), solution))

    print("\nTrying scipy.optimize.fmin (Nelder-Mead Simplex)...")

    m = fmin(fOsc3D, [0.1, 0.1])
    plt.plot([m[0]],[m[1]],'ro',markersize=5)
    print("solution w/ initial conditions (0.1,0.1): %s\n" % m)

    m = fmin(fOsc3D, [1, 1])
    plt.plot([m[0]],[m[1]],'ro',markersize=5)
    print("solution w/ initial conditions (1,1): %s\n" % m)

    m = fmin(fOsc3D, [-1, 1])
    print("solution w/ initial conditions (-1,1): %s\n" % m)
    plt.plot([m[0]],[m[1]],'ro',markersize=5)

#   m = fmin(fOsc3D, [0, 2])
#   print("solution w/ initial conditions (0,2): %s\n" % m)
#   plt.plot([m[0]],[m[1]],'ro',markersize=5)

    plt.title('White dot: DE, Red dots: Nelder-Mead')

    try:
        import Image
        plt.savefig('test_fosc3d_out',dpi=72)
        im = Image.open('test_fosc3d_out.png')
        im.show()
    except ImportError:
        plt.show()


# end of file
