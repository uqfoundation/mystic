#!/usr/bin/env python

"""
Similar to test_mogi.py

but trying to use scipy's levenberg marquardt.

"""

from test_mogi import *

# Here is the new cost function definition
def cost_function_levenbergmarquardt(params):
    model = ForwardMogiFactory(params)
    zdisp = filter_for_zdisp(model(stations))
    return zdisp - data_z

import pylab
from scipy.optimize import fmin_cg

if __name__ == '__main__':

    sol = fmin_cg(cost_function, [1000., -500., -10., 0.1], full_output=1)
    print "scipy solution: ", sol[0]
    plot_sol(sol[0],'r-')
    pylab.show()

# end of file
