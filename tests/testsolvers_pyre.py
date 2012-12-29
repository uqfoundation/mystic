#!/usr/bin/env python
#
# Alta Fang & Mike McKerns, Caltech

"""
Runs mystic solvers as a pyre application

Test solver/termination combinations by editing input files
  - chebyshevinputs      -- for NelderMeadSimplex, PowellDirectional with
                            chebyshev8
  - chebyshevinputs_de   -- for DifferentialEvolution, DifferentialEvolution2 with
                            chebyshev8
  - roseninputs          -- for NelderMeadSimplex, PowellDirectional, 
                            DifferentialEvolution(2) with rosenbrock

and running, for example:
$python testsolvers_pyre.py --solver=NelderMeadSimplex --inputs=chebyshevinputs

"""

from pyre.applications.Script import Script
from mystic.helputil import paginate
from mystic.solvers import *
from mystic.termination import *
import logging

class TestSolverApp(Script):
    """Solvers wrapped into a Pyre Application."""

    class Inventory(Script.Inventory):

        import pyre.inventory

        # the defaults

        inputs = pyre.inventory.str('inputs', default = 'chebyshevinputs_de')
        inputs.meta['tip'] = 'The python module containing the cost-function and other data.'

        verbose = pyre.inventory.bool('verbose', default = False)
        verbose.meta['tip'] = 'Turns on logging.'

        solver = pyre.inventory.str('solver', default = 'DifferentialEvolution')
	solver.meta['tip'] = 'The solver to be used.'


    def main(self, *args, **kwds):
	# general solver

	# exception for DifferentialEvolutionSolver2
	if self.inventory.solver == 'DifferentialEvolution2':
            solvername = DifferentialEvolutionSolver2
        else:
	    solvername = eval(self.inventory.solver + 'Solver')

        # create the solver
	try:
            NP = self.mod.NP
	    solver = solvername(self.mod.ND, NP)
	except:
	    solver = solvername(self.mod.ND)

	costfunction  = self.mod.cost
        termination = self.mod.termination

	import random
	random.seed(123)

        # set initial points
	try:
            solver.SetInitialPoints(self.mod.x0)
	except:
	    solver.SetRandomInitialPoints(self.mod.min, self.mod.max)

        # set maximum number of iterations
        try:
            maxiter = self.mod.maxiter
            solver.SetEvaluationLimits(generations=maxiter)
        except:
            pass

        # set bounds, if applicable
        try:
            min_bounds = self.mod.min_bounds
            max_bounds = self.mod.max_bounds
            solver.SetStrictRanges(min_bounds, max_bounds)
        except:
            pass

        # additional arguments/kwds to the Solve() call
        try:
            solverkwds = self.mod.solverkwds
        except:
            solverkwds = {}
        
        solver.Solve(costfunction, termination, **solverkwds)
        self.solution = solver.Solution()
	return


    def __init__(self):
        Script.__init__(self, 'testsolverapp')
        self.mod = ''
        self.solution = None
        return


    def _defaults(self):
        Script._defaults(self)
        return


    def _configure(self):
        from mystic import strategy as detools
        Script._configure(self)
        mod = __import__(self.inventory.inputs)
        self.mod = mod
       
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

#-----------------------------------

def output_chebyshev():
    # Chebyshev8 polynomial
    from mystic.models.poly import chebyshev8coeffs as target_coeffs
    from mystic.models.poly import poly1d
    print "target:\n", poly1d(target_coeffs)
    print "\nSolver Solution:\n", poly1d(app.solution)

def output_rosen():
    # rosenbrock
    print "target: [1. 1. 1.]"
    print "solver solution: ", app.solution

# main
if __name__ == '__main__':
    app = TestSolverApp()
    app.run()

    # select the correct output format
    # redirects to output_chebyshev or output_rosen
    inputs = app.inventory.inputs
    i = inputs.find('input')
    funcname = inputs[:i]
    eval('output_' + funcname + '()')


# End of file
