#!/usr/bin/env python
#
# Author: Alta Fang (altafang @caltech and alta @princeton)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2010-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#
# coded by Alta Fang, 2010
"""
A few basic symbolic constraints tests, but in no way a comprehensive suite.
"""
from numpy import asarray
#from mystic.restarts import sumt
from mystic.constraints import issolution, as_constraint, solve as _solve
from mystic.coupler import inner
from mystic.symbolic import *
from mystic.tools import random_seed
from mystic.math import almostEqual
random_seed(24)


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
    # print("constraints equations:%s" % (constraints_string.rstrip(),))

    from mystic.solvers import DifferentialEvolutionSolver
    from mystic.solvers import NelderMeadSimplexSolver
    from mystic.termination import VTR
    #solver = DifferentialEvolutionSolver(ndim, npop)
    solver = NelderMeadSimplexSolver(ndim)
    solver.SetInitialPoints(x0)
    term = VTR()
    #FIXME: sumt, issolution no longer take constraints strings
    end_solver = sumt(constraints_string, ndim, costfunc, solver,\
                                term, disp=True)
    soln = end_solver.Solution()
    assert issolution(constraints_string, soln)
    # print("final answer: %s" % soln)
    # print("constraints satisfied: %s" % issolution(constraints_string, soln))
    # print("expected: [1., 1.]\n")

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
    # print("constraints equations:%s" % (constraints_string.rstrip(),))
    from mystic.solvers import DifferentialEvolutionSolver
    from mystic.solvers import NelderMeadSimplexSolver
    from mystic.termination import VTR
    #solver = DifferentialEvolutionSolver(ndim, npop)
    solver = NelderMeadSimplexSolver(ndim)
    solver.SetInitialPoints(x0)
    term = VTR()
    #FIXME: sumt, issolution no longer take constraints strings
    end_solver = sumt(constraints_string, ndim, costfunc, solver,\
                                term, disp=True)
    soln = end_solver.Solution()
    assert issolution(constraints_string, soln)
    # print("final answer: %s" % soln)
    # print("constraints satisfied: %s" % issolution(constraints_string, soln))
    # print("expected: [ 6.25827968  4.999961    5.20662288]\n")

def test_form_constraints_function():
    # Test a nonlinear constraints example.
    string = """
    x1*x2 = 1.
    x2 = x0 - 3.
    """
    # print("building constraints function for:%s" % string.rstrip())
    x0 = [0.8,1.2,-0.7]
    # print('initial parameters: %s' % asarray(x0))
    cf = generate_constraint(generate_solvers(solve(string)))
    # print('constraints satisfied? %s' % issolution(cf, x0))
    assert not issolution(cf, x0)
    x = cf(x0)
    # print('after imposing constraints: %s' % asarray(x))
    # print('constraints satisfied? %s' % issolution(cf, x))
    assert issolution(cf, x)

    x0 = [1.,1.,1.]
    # print('initial parameters: %s' % asarray(x0))
    # print('constraints satisfied? %s' % issolution(cf, x0))
    assert not issolution(cf, x0)
    x = cf(x0)
    # print('after imposing constraints: %s' % asarray(x))
    # print('constraints satisfied? %s' % issolution(cf, x), "\n")
    assert issolution(cf, x)

def test_matrix_interface():
    # Demonstrates linear_symbolic()
    A = asarray([[3., 4., 5.],
         [1., 6., -9.]])
    b = asarray([0., 0.])
    G = [1., 0., 0.]
    h = [5.]
    # print("equality constraints")
    # print("G: %s" % G)
    # print("h: %s" % h)
    # print("inequality constraints")
    # print("A:\n%s" % A)
    # print("b: %s" % b)
    constraints_string = linear_symbolic(A=A, b=b, G=G, h=h)
    cs = constraints_string.split('\n')
    assert cs[0] == "1.0*x0 + 0.0*x1 + 0.0*x2 <= 5.0"
    assert cs[1] == "3.0*x0 + 4.0*x1 + 5.0*x2 = 0.0"
    assert cs[2] == "1.0*x0 + 6.0*x1 + -9.0*x2 = 0.0"
    # print("symbolic string:\n%s" % constraints_string.rstrip())
    pf = generate_penalty(generate_conditions(constraints_string))
    cn = as_constraint(pf)

    x0 = [1., 1., 1.]
    assert almostEqual(pf(cn(x0)), 0.0, tol=2e-2)
    #XXX: implement: wrap_constraint( as_constraint(pf), sum, ctype='inner') ?

def test_varnamelist():
    # Demonstrates usage of varnamelist
    varnamelist = ['length', 'width', 'height']
    string = "length = height**2 - 3.5*width"
    # print("symbolic string:\n%s" % string.rstrip())
    string = replace_variables(string, varnamelist, 'x')
    cf = generate_constraint(generate_solvers(string))

    @inner(cf)
    def wrappedfunc(x):
        return x[0]

    #XXX: implement: wrap_constraint( cf, lambda x: x[0], ctype='inner') ?
    # print("c = constraints wrapped around x[0]")
    x0 = [2., 2., 3.]
    # print("c(%s): %s\n" % (x0, wrappedfunc(x0))) # Expected: 2.0
    assert almostEqual(wrappedfunc(x0), 2.0, tol=1e-15)

def test_feasible_pt():
    # constraints = """x0 + x1 + x6 - 2*x3 > 3""" #FIXME: allow inequalities
    constraints = """x0 + x1 + x6 - 2*x3 = 3"""
    constraints = solve(constraints)
    assert constraints == 'x0 = -x1 + 2*x3 - x6 + 3'
    solv = generate_solvers(constraints)
    conf = generate_constraint(solv)
    soln = _solve(conf, guess=[1.]*8)
    assert soln == [3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # print('actual solution: %s\n' % soln)

def test_varnamelist2():
    # Test tricky cases of varnamelist
    varnamelist = ['x', 'y', 'x3']
    string = "x + y + x3 = 0"
    # print("symbolic string:\n%s" % string.rstrip())
    newstring = replace_variables(string, varnamelist, 'x')
    # print("new symbolic string:\n%s" % newstring.rstrip()
    assert newstring == 'x0 + x1 + x2 = 0'


if __name__ == '__main__':
####test_sumt1()
####test_sumt2()
    test_form_constraints_function()
    test_matrix_interface()
    test_varnamelist()
    test_varnamelist2()
    test_feasible_pt()


#EOF
