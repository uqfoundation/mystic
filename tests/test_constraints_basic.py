#!/usr/bin/env python
#
# coded by Alta Fang, 2010
"""
A few basic constraints tests, but definitely not a comprehensive suite.
"""
from mystic.constraints import *
from mystic.tools import random_seed
random_seed(123)

def test_sumt1():
    def costfunc(x):
        x1 = x[0]
        x2 = x[1]
        return  x1**4 - 2.*x1**2*x2 + x1**2 + x1*x2**2 - 2.*x1 + 4.

    constraints_string = """
    x1**2 + x2**2 - 2. = 0.
    0.25*x1**2 + 0.75*x2**2 - 1. <= 0.
        """

    ndim = 2
    x0 = [3., 2.]
    npop = 25

    from mystic.differential_evolution import DifferentialEvolutionSolver
    from mystic.scipy_optimize import NelderMeadSimplexSolver
    from mystic.termination import VTR
    #solver = DifferentialEvolutionSolver(ndim, npop)
    solver = NelderMeadSimplexSolver(ndim)
    solver.SetInitialPoints(x0)
    term = VTR()
    end_solver = sumt(constraints_string, ndim, costfunc, solver,\
                                term, disp=True)
    soln = end_solver.Solution()
    print "final answer:", soln
    print "constraints satisfied:", issolution(constraints_string, soln)
    print "expected: [1., 1.]"

def test_sumt2():
    def costfunc(x):
        return (x[0] - 1.)**2 + (x[1]-2.)**2 + (x[2]-3.)**4
    x0 = [0., 0., 0.]
    ndim = 3
    constraints_string = """
    x2 > 5.
    4.*x1-5.*x3 < -1.
    (x1-10.)**2 + (x2+1.)**2 < 50. 
    """
    from mystic.differential_evolution import DifferentialEvolutionSolver
    from mystic.scipy_optimize import NelderMeadSimplexSolver
    from mystic.termination import VTR
    #solver = DifferentialEvolutionSolver(ndim, npop)
    solver = NelderMeadSimplexSolver(ndim)
    solver.SetInitialPoints(x0)
    term = VTR()
    end_solver = sumt(constraints_string, ndim, costfunc, solver,\
                                term, disp=True)
    soln = end_solver.Solution()
    print "final answer:", soln
    print "constraints satisfied:", issolution(constraints_string, soln)
    print "expected: [ 6.25827968  4.999961    5.20662288]"

def test_form_constraints_function():
    # Test a nonlinear constraints example.
    string = """
x2*x3 = 1.
x3 = x1 - 3.
"""
    x0 = [0.8,1.2,-0.7]
    cf = parse(string)
    c = cf(x0)
    print 'constraints satisfied:', issolution(string, c)
    print 'strictly... constraints(x0) =', asarray(c)

def test_matrix_interface():
    # Demonstrates linear_symbolic()
    A = asarray([[3., 4., 5.],
         [1., 6., -9.]])
    b = asarray([0., 0.])
    G = [1., 0., 0.]
    h = [5.]
    costfunc = lambda x: sum(x)
    constraints_string = linear_symbolic(A=A, b=b, G=G, h=h)
    print "symbolic string:"
    print constraints_string
    cf = wrap_constraints(constraints_string, costfunc)
    print cf([1., 1., 1.])

def test_varnamelist():
    # Demonstrates usage of varnamelist
    varnamelist = ['length', 'width', 'height']
    string = "length = height**2 - 3.5*width"
    costfunc = lambda x: x[0]   
    wrappedfunc = wrap_constraints(string, costfunc, variables=varnamelist)
    print wrappedfunc([2., 2., 3.]) # Expected: 2.0

def test_feasible_pt():
    constraints = """x1 + x2 + x7 - 2*x4 > 3"""
    bad_constraints = """
x1 + x2 < 3.
x1 + x2 > 4."""
    soln = solve(constraints, guess=[1.]*8)
    print 'actual solution:', soln

def test_varnamelist2():
    # Test tricky cases of varnamelist
    varnamelist = ['x', 'y', 'x3']
    string = "x + y + x3 = 0"
    newstring = substitute_symbolic(string, varnamelist, variables='x')
    print newstring


if __name__ == '__main__':
    test_sumt1()
    test_sumt2()
    test_form_constraints_function()
    test_matrix_interface()
    test_varnamelist()
    test_feasible_pt()
    test_varnamelist2()
    pass

#EOF
