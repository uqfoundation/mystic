#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                       Patrick Hung & Mike McKerns, Caltech
#                        (C) 1998-2008  All Rights Reserved
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

"""
Runs differential_evolution as a pyre application

Based on differential_evolution.py
"""

from pyre.applications.Script import Script
from mystic.helputil import paginate
from mystic.differential_evolution import *
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

        costfunc = pyre.inventory.str('costfunc', default = 'example')
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
        strategy  = self.strategy
        solver.SetRandomInitialPoints(min = self.mod.min, max = self.mod.max)
        solver.Solve(costfunction, strategy, termination, CrossProbability=self.probability, \
                     maxiter = MAX_GENERATIONS,ScalingFactor=self.scale)
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
        from mystic import detools
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

    # Chebyshev8 polynomial (used with "example.py")
    target_coef = [128., 0., -256., 0., 160., 0., -32., 0., 1.]

    from mystic.polytools import coefficients_to_polynomial as poly1d
    print "target:\n", poly1d(target_coef)
    print "DE Solution:\n", poly1d(app.solution)


# End of file
