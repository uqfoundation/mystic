#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2006-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
'''a signal handler for a mystic solver instance'''

try:
    input = raw_input
except NameError:
    pass

# pull from the signal module
from signal import *

class Handler(object):
    def __init__(self, solver, sigint_callback=None):
        """factory to generate signal handler

Inputs::
    - solver: a mystic.solver instance
    - sigint_callback: a callback function

Available switches::
    - sol  --> Print current best solution.
    - cont --> Continue calculation.
    - call --> Executes sigint_callback, if provided.
    - exit --> Exits with current best solution.
"""
        self.solver = solver
        self.sigint_callback = sigint_callback or solver.sigint_callback
    def __call__(self, signum, frame):
        import inspect
        print(inspect.getframeinfo(frame))
        print(inspect.trace())
        while 1:
            s = input(\
"""
 
 Enter sense switch.

    sol:  Print current best solution.
    cont: Continue calculation.
    call: Executes sigint_callback [%s].
    exit: Exits with current best solution.

 >>> """ % self.sigint_callback)
            if s.lower() == 'sol':
                print(self.solver.bestSolution)
            elif s.lower() == 'cont':
                return
            elif s.lower() == 'call':
                # sigint call_back
                if self.sigint_callback is not None:
                    self.sigint_callback(self.solver.bestSolution)
            elif s.lower() == 'exit':
                self.solver._EARLYEXIT = True
                return
            else:
                print("unknown option : %s" % s)
        return

# EOF
