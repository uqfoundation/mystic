#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Example use of Forward Poly Model
in mystic and PARK optimization frameworks.
(built for mystic "trunk" and with park-1.2)

for help, type "python rosetta_parabola_example.py --help"
"""

from math import pi
from numpy import array, real, conjugate
import numpy

try: # check if park is installed
    import park
   #import park.parksnob
    import park.parkde
    Model = park.Model
    __park = True
except ImportError:
    Model = object
    __park = False


def ForwardPolyFactory(params):
    a,b,c = params
    def forward_poly(x):
        """ x should be a 1D (1 by N) numpy array """
        return array((a*x*x + b*x + c))
    return forward_poly

def data(params):
    fwd = ForwardPolyFactory(params)
    x = (array([list(range(101))])-50.)[0]
    return x,fwd(x)


# --- Cost Function stuff ---
# Here is the cost function
def vec_cost_function(params):
    return data(params)[1] - datapts

# Here is the normed version
def cost_function(params):
    x = vec_cost_function(params)
    return numpy.sum(real((conjugate(x)*x)))
# --- Cost Function end ---


# --- Plotting stuff ---
import matplotlib.pyplot as plt
def plot_sol(params,linestyle='b-'):
    d = data(params)
    plt.plot(d[0],d[1],'%s'%linestyle,linewidth=2.0)
    plt.axis(plotview)
    return
# --- Plotting end ---


# --- Call to Mystic ---
def mystic_optimize(point):
    from mystic.monitors import Monitor, VerboseMonitor
    from mystic.solvers import NelderMeadSimplexSolver as fmin
    from mystic.termination import CandidateRelativeTolerance as CRT
    simplex, esow = VerboseMonitor(50), Monitor()
    solver = fmin(len(point))
    solver.SetInitialPoints(point)
    min = [-100,-100,-100]; max = [100,100,100]
    solver.SetStrictRanges(min,max)
    solver.SetEvaluationMonitor(esow)
    solver.SetGenerationMonitor(simplex)
    solver.Solve(cost_function, CRT(1e-7,1e-7))
    solution = solver.Solution()
    return solution
# --- Mystic end ---


# --- Call to Park ---
class PolyModel(Model):
    """a park model:
 - parameters are passed as named strings to set them as class attributes
 - function that does the evaluation must be named "eval"
 - __call__ generated that takes namestring and parameter-named keywords
"""
    parameters = ["a","b","c"]
    def eval(self, x):
        a = self.a
        b = self.b
        c = self.c
        f = ForwardPolyFactory((a,b,c))
        return f(x)
    pass

class Data1D(object):
    """1d model data with the required park functions"""
    def __init__(self,z):
        self.z = z
        return

    def residuals(self,model):
        x = (array([list(range(101))])-50.)[0]
        return (model(x) - self.z).flatten()
    pass


def park_optimize(point):
    # build the data instance
    data1d = Data1D(datapts)

    # build the model instance
    a,b,c = point 
    model = PolyModel("mymodel",a=a,b=b,c=c)
    # required to set bounds on the parameters
    model.a = [-100,100]
    model.b = [-100,100]
    model.c = [-100,100]

    # add a monitor, and set to print results to the console
    handler=park.fitresult.ConsoleUpdate()

    # select the fitter, and do the fit
   #fitter=park.parksnob.Snobfit()
    fitter=park.parkde.DiffEv()
    # 'fit' requires a list of tuples of (model,data)
    result=park.fit.fit([(model,data1d)],fitter=fitter,handler=handler)

    # print results
   #print(result.call)     # print number of function calls
   #result.print_summary() # print solution

    # get the results back into a python object
    solution = {}
    for fitparam in result.parameters:
        solution[fitparam.name] = fitparam.value
    solution = [ solution['mymodel.a'],
                 solution['mymodel.b'],
                 solution['mymodel.c'] ]
    return solution
# --- Park end ---


if __name__ == '__main__':
    # parse user selection to solve with "mystic" [default] or "park"
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-p","--park",action="store_true",dest="park",\
                      default=False,help="solve with park (instead of mystic)")
    parsed_opts, parsed_args = parser.parse_args()

    # set plot window
    from mystic.tools import getch
    plotview = [-10,10, 0,100]

    # Let the "actual parameters" be :
    target = [1., 2., 1.]
    print("Target: %s" % target)

    # Here is the "observed data"
    x,datapts = data(target)
    plt.ion()
    plot_sol(target,'r-')
    plt.draw()
    plt.pause(0.001)

    # initial values
    point = [100,-100,0]

    # DO OPTIMIZATION STUFF HERE TO GET SOLUTION
    if parsed_opts.park:
        if __park:
            print("Solving with park's DE optimizer...")
            solution = park_optimize(point)
        else:
            print('This option requires park to be installed')
            exit()
    else:
        print("Solving with mystic's fmin optimizer...")
        solution = mystic_optimize(point)
    print("Solved: %s" % solution)

    # plot the solution
    plot_sol(solution,'g-')
    plt.draw()
    plt.pause(0.001)

    getch()

# End of file
