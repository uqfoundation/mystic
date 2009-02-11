#!/usr/bin/env python

"""
Similar to test_mogi.py

but trying to use scipy's levenberg marquardt.

"""

from test_mogi import *
from scipy.optimize import anneal
import pylab

if __name__ == '__main__':

    lower = array([1000,-1000,0,0])
    upper = array([5000,-0,20,0.5])
    sol = anneal(cost_function, [1000., -500., -10., 0.1], lower=lower, upper=upper, feps=1e-10, dwell=100,T0=10)
    print "scipy solution: ", sol[0]
    plot_noisy_data()
    plot_sol(sol[0],'r-')
    pylab.show()

# end of file
