#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2020 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Runs differential evolution as a pyre application

Based on DifferentialEvolutionSolver
"""

from pyre.applications.Script import Script
from mystic.helputil import paginate
from mystic.solvers import DifferentialEvolutionSolver
import logging

class DerunApp(Script):
    """Differential Evolution Solver wrapped into a Pyre Application."""

    class Inventory(Script.Inventory):

        import pyre.inventory

        # the defaults
        scale = pyre.inventory.float('scale', default = 0.7)
        scale.meta['tip'] = 'Differential Evolution scale factor.'

        probability = pyre.inventory.float('probability', default = 0.5)
        probability.meta['tip'] = 'Differential Evolution crossover probability.'

        costfunc = pyre.inventory.str('costfunc', default = 'dummy')
        costfunc.meta['tip'] = 'The python module containing the cost-function and other data.'

        strategy = pyre.inventory.str('strategy', default = 'Best1Exp')
        strategy.meta['tip'] = 'The differential evolution strategy.'
        strategy.meta['known_alternatives'] = ['Best1Bin', 'Rand1Exp']

        verbose = pyre.inventory.bool('verbose', default = False)
        verbose.meta['tip'] = 'Turns on logging.'


    def main(self, *args, **kwds):
        solver = DifferentialEvolutionSolver(self.mod.ND, self.mod.NP)
        costfunction  = self.mod.cost
        termination = self.mod.termination
        MAX_GENERATIONS  = self.mod.MAX_GENERATIONS
        solver.SetRandomInitialPoints(min = self.mod.min, max = self.mod.max)
        solver.SetEvaluationLimits(generations=MAX_GENERATIONS)
        solver.Solve(costfunction, termination, strategy=self.strategy,\
                     CrossProbability=self.probability, \
                     ScalingFactor=self.scale)
        self.solution = solver.Solution()
        return


    def __init__(self):
        Script.__init__(self, 'derunapp')
        self.scale = ''
        self.probability = ''
        self.mod = ''
        self.strategy = ''
        self.solution = None
        return


    def _defaults(self):
        Script._defaults(self)
        return


    def _configure(self):
        from mystic import strategy as detools
        Script._configure(self)
        mod = __import__(self.inventory.costfunc)
        self.mod = mod

        self.scale = self.inventory.scale
        self.probability = self.inventory.probability
        try:
            self.probability = self.mod.probability
        except:
            pass
        try:
            self.scale = self.mod.scale
        except:
            pass
        self.strategy = getattr(detools,self.inventory.strategy) 
       
        if self.inventory.verbose:
            logging.basicConfig(level=logging.DEBUG,
                                format='%(asctime)s %(levelname)s %(message)s',
                                datefmt='%a, %d %b %Y %H:%M:%S')

    def _init(self):
        Script._init(self)
        return

    def help(self):
        doc =  """
# this will solve the default "example" problem
%(name)s

        """ % {'name' : __file__}
        paginate(doc)
        return



# main
if __name__ == '__main__':
    app = DerunApp()
    app.run()

    # Chebyshev8 polynomial (used with "dummy.py")
    from mystic.models.poly import chebyshev8coeffs as target_coeffs
    from mystic.math import poly1d
    print("target:\n%s" % poly1d(target_coeffs))
    print("\nDE Solution:\n%s" % poly1d(app.solution))


# End of file
