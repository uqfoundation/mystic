#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Example use of Forward Mogi Model
in mystic and PARK optimization frameworks.
(built for mystic "trunk" and with park-1.2)

for help, type "python rosetta_mogi_example.py --help"
"""

from math import pi
from numpy import array

try: # check if park is installed
    import park
   #import park.parksnob
    import park.parkde
    Model = park.Model
    __park = True
except ImportError:
    Model = object
    __park = False


def ForwardMogiFactory(params):
    x0,y0,z0,dV = params
    def forward_mogi(evalpts):
        """ evalpts should be a 2D (2 by N) numpy array """
        dx = evalpts[0,:] - x0
        dy = evalpts[1,:] - y0
        dz = 0 - z0
        c = dV * 3. / 4. * pi
        # or equivalently c= (3/4) a^3 dP / rigidity
        # where a = sphere radius, dP = delta Pressure
        r2 = dx*dx + dy*dy + dz*dz
        C = c / pow(r2, 1.5)
        return array((C*dx,C*dy,C*dz))
    return forward_mogi


# --- Cost Function stuff ---
def filter_for_zdisp(input):
    return -input[2,:]

# Here is the cost function
def vec_cost_function(params):
    model = ForwardMogiFactory(params)
    zdisp = filter_for_zdisp(model(stations))
    return 100. * (zdisp - data_z)

# Here is the normed version  [NOTE: fit this one!]
def cost_function(params):
    x = vec_cost_function(params)
    return numpy.sum(real((conjugate(x)*x)))

# a cost function with parameters "normalized"
def vec_cost_function2(params):
    sca = numpy.array([1000, 100., 10., 0.1])
    return vec_cost_function(sca * params)
# --- Cost Function end ---


# --- Plotting stuff ---
import matplotlib.pyplot as plt
def plot_sol(params,linestyle='b-'):
    forward_solution = ForwardMogiFactory(params)
    xx = arange(-30,30,0.1)+actual_params[0]
    yy = 0*xx + actual_params[1]
    ss  = array((xx, yy))
    dd = forward_solution(ss)
    plt.plot(ss[0,:],-dd[2,:],'%s'%linestyle,linewidth=2.0)

def plot_noisy_data():
    import matplotlib.pyplot as plt
    plt.plot(stations[0,:],-data[2,:]+noise[2,:],'k.')
# --- Plotting end ---


# --- Call to Mystic's Fmin optimizer ---
def mystic_optimize(point):
    from mystic.monitors import Monitor, VerboseMonitor
    from mystic.tools import getch, random_seed
    random_seed(123)
    from mystic.solvers import NelderMeadSimplexSolver as fmin
    from mystic.termination import CandidateRelativeTolerance as CRT
    simplex, esow = VerboseMonitor(50), Monitor()
    solver = fmin(len(point))
    solver.SetInitialPoints(point)
    solver.SetEvaluationMonitor(esow)
    solver.SetGenerationMonitor(simplex)
    solver.Solve(cost_function, CRT())
    solution = solver.Solution()
    return solution
# --- Mystic end ---

# --- Call to Mystic's DE optimizer ---
def mystic_optimize2(point):
    from mystic.monitors import Monitor, VerboseMonitor
    from mystic.tools import getch, random_seed
    random_seed(123)
    from mystic.solvers import DifferentialEvolutionSolver as de
    from mystic.termination import ChangeOverGeneration as COG
    NPOP = 50
    simplex, esow = VerboseMonitor(50), Monitor()
    solver = de(len(point),NPOP)
    solver.SetInitialPoints(point)
    solver.SetEvaluationMonitor(esow)
    solver.SetGenerationMonitor(simplex)
    solver.Solve(cost_function, COG(generations=100), \
                 CrossProbability=0.5, ScalingFactor=0.5)
    solution = solver.Solution()
    return solution
# --- Mystic end ---

# --- Call to Park ---
class MogiModel(Model):
    """a park model:
 - parameters are passed as named strings to set them as class attributes
 - function that does the evaluation must be named "eval"
 - __call__ generated that takes namestring and parameter-named keywords
"""
    parameters = ["x0","y0","z0","dV"]
    def eval(self, x):
        x0 = self.x0
        y0 = self.y0
        z0 = self.z0
        dV = self.dV
        f = ForwardMogiFactory((x0,y0,z0,dV))
        return f(x)
    pass

class Data2D(object):
    """2d model data with the required park functions"""
    def __init__(self,z):
        self.z = z
        return

    def residuals(self,model):
        zdisp = filter_for_zdisp(model(stations))
        return (100. * (zdisp - self.z)).flatten()
    pass


def park_optimize(point):
    # build the data instance
    data2d = Data2D(data_z)

    # build the model instance
    x0,y0,z0,dV = point 
    model = MogiModel("mymodel",x0=x0,y0=y0,z0=z0,dV=dV)
    # required to set bounds on the parameters
   #model.x0 = [-numpy.inf,numpy.inf]
   #model.y0 = [-numpy.inf,numpy.inf]
   #model.z0 = [-numpy.inf,numpy.inf]
   #model.dV = [-numpy.inf,numpy.inf]
    model.x0 = [-5000,5000]
    model.y0 = [-5000,5000]
    model.z0 = [-5000,5000]
    model.dV = [-5000,5000]

    # add a monitor, and set to print results to the console
    handler=park.fitresult.ConsoleUpdate()

    # select the fitter, and do the fit
   #fitter=park.parksnob.Snobfit()
    fitter=park.parkde.DiffEv()
    # 'fit' requires a list of tuples of (model,data)
    result=park.fit.fit([(model,data2d)],fitter=fitter,handler=handler)

    # print results
   #print(result.calls)    # print number of function calls
   #result.print_summary() # print solution

    # get the results back into a python object
    solution = {}
    for fitparam in result.parameters:
        solution[fitparam.name] = fitparam.value
    solution = [ solution['mymodel.x0'],
                 solution['mymodel.y0'],
                 solution['mymodel.z0'],
                 solution['mymodel.dV'] ]
    return solution
# --- Park end ---


if __name__ == '__main__':
    # parse user selection to solve with "mystic" [default] or "park"
    # also can select mystic's optimizer: "diffev" or "fmin" [default]
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-p","--park",action="store_true",dest="park",\
                      default=False,help="solve with park (instead of mystic)")
    parser.add_option("-m","--mystic",action="store_true",dest="mystic",\
                      default=False,help="solve with mystic's DE optimizer)")
    parsed_opts, parsed_args = parser.parse_args()

    from numpy import pi, sqrt, array, mgrid, random, real, conjugate, arange
    from numpy.random import rand
    import numpy
    # Let the "actual parameters" be :
    actual_params = [1234.,-500., 10., .1]
    actual_forward = ForwardMogiFactory(actual_params)
    print("Target: %s" % actual_params)

    # The data to be "fitted" 
    xstations = array([random.uniform(-30,30) for i in range(300)])+actual_params[0]
    ystations =  0*xstations + actual_params[1]
    stations  = array((xstations, ystations))
    data = actual_forward(stations)

    # generate noise... gaussian distribution with mean 0, sig 0.1e-3
    noise =  array([[random.normal(0,0.1e-3) for i in range(data.shape[1])] for j in range(data.shape[0])])

    # Here is the "observed data"
    data_z = -data[2,:] + noise[2,:]

    # plot the noisy data
    plot_noisy_data()

    point = [1000,-100,0,1] # cg will do badly on this one

    # DO OPTIMIZATION STUFF HERE TO GET SOLUTION
    if parsed_opts.park:     #solve with park's DE
        if __park:
            print("Solving with park's DE optimizer...")
            solution = park_optimize(point)
        else:
            print('This option requires park to be installed')
            exit()
    elif parsed_opts.mystic: #solve with mystic's DE
        print("Solving with mystic's DE optimizer...")
        solution = mystic_optimize2(point)
    else:                    #solve with mystic's fmin
        print("Solving with mystic's fmin optimizer...")
        solution = mystic_optimize(point)
    print("Solved: %s" % solution)

    # plot the solution
    plot_sol(solution,'r-')
    plt.show()


# End of file
