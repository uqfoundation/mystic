#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Similar to test_mogi.py
but using scipy's basinhopping algorithm
"""

from test_mogi import *
from scipy.optimize import basinhopping
import matplotlib.pyplot as plt

if __name__ == '__main__':

    sol = basinhopping(cost_function, [1000., -500., -10., 0.1], niter=100,T=10)
    print("scipy solution: %s" % sol.x)
    plot_noisy_data()
    plot_sol(sol.x,'r-')
    plt.show()

# end of file
