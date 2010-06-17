"""The function get_McDiarmid() is a generalization of 
mystic/branches/UQ/math/examples/TEST_surrogate_McD.py
"""

#XXX Consider adding everything in this file to the file statistics.py?

from statistics import mcdiarmid_bound, integrated_mean
from samples import random_samples, sampled_mean, wrap_bounds

def get_McDiarmid(model, lower_bounds, upper_bounds, a=0.0, sampled=True):
    """
This function finds the mcdiarmid bound probability and diameter of a 
model given lower and upper bounds and an optional threshhold.

Input parameters:
model	-- a function
lower_bounds -- a list
upper_bounds -- a list
a -- threshhold for mcdiarmid bound. default is 0.0. (does this belong?)
sampled -- True if you want the mean to be calculated by uniform sampling,
           and False if you want the mean to be calculated by integrating.

Output parameters:
[mcdiarmid bound probability, diameter squared]
"""

    scale = 1.0
    npop = 20
    maxiter = 1000
    maxfun = 1e+6
    convergence_tol = 1e-4
    crossover = 0.9
    percent_change = 0.9
    #######################################################################
    # the subdiameter calculation
    # (similar to driver.sh)
    #######################################################################
    def costFactory(i):
      """a cost factory for the cost function"""

      def cost(rv):
        """compute the diameter as a calculation of cost

      Input:
        - rv -- 1-d array of model parameters

      Output:
        - diameter -- scale * | F(x) - F(x')|**2
        """

        # prepare x and xprime
        params = rv[:-1]                         #XXX: assumes Xi' is at rv[-1]
        params_prime = rv[:i]+rv[-1:]+rv[i+1:-1] #XXX: assumes Xi' is at rv[-1]

        # get the F(x) response
        Fx = model(params)

        # get the F(x') response
        Fxp = model(params_prime)

        # compute diameter
        return -scale * (Fx - Fxp)**2

      return cost


    #######################################################################
    # the differential evolution optimizer
    # (replaces the call to dakota)
    #######################################################################
    def optimize(cost,lb,ub):
      from mystic.differential_evolution import DifferentialEvolutionSolver2
      from mystic.termination import CandidateRelativeTolerance as CRT
      from mystic.strategy import Best1Exp
      from mystic import getch, random_seed, VerboseSow, Sow

      random_seed(123)

     #stepmon = VerboseSow(100)
      stepmon = Sow()
      evalmon = Sow()

      ndim = len(lb) # [(1 + RVend) - RVstart] + 1

      solver = DifferentialEvolutionSolver2(ndim,npop)
      solver.SetRandomInitialPoints(min=lb,max=ub)
      solver.SetStrictRanges(min=lb,max=ub)
      solver.SetEvaluationLimits(maxiter,maxfun)

      tol = convergence_tol
      solver.Solve(cost,termination=CRT(tol,tol),strategy=Best1Exp, \
                   CrossProbability=crossover,ScalingFactor=percent_change, \
                   StepMonitor=stepmon, EvaluationMonitor=evalmon)

      print "solved: %s" % solver.Solution()
      diameter_squared = -solver.bestEnergy / scale  #XXX: scale != 0
      func_evals = len(evalmon.y)
      return diameter_squared, func_evals


    #######################################################################
    # loop over model parameters to calculate concentration of measure
    # (similar to main.cc)
    #######################################################################
    def UQ(start,end,lower,upper):
      diameters = []
      function_evaluations = []
      total_func_evals = 0
      total_diameter = 0.0

      for i in range(start,end+1):
        lb = lower[start:end+1] + [lower[i]]
        ub = upper[start:end+1] + [upper[i]]
      
        #construct cost function and run optimizer
        cost = costFactory(i)
        subdiameter, func_evals = optimize(cost,lb,ub) #XXX: no initial conditions

        function_evaluations.append(func_evals)
        diameters.append(subdiameter)

        total_func_evals += function_evaluations[-1]
        total_diameter += diameters[-1]

      print "subdiameters (squared): %s" % diameters
      print "diameter (squared): %s" % total_diameter
      print "func_evals: %s => %s" % (function_evaluations, total_func_evals)

      return total_diameter


    #######################################################################
    # statistics
    # expectation, diameter, McDiarmid
    #######################################################################
 
    RVstart = 0; RVend = len(lower_bounds) - 1
    RVmax = len(lower_bounds)
    # when not a random variable, set the value to the lower bound
    for i in range(0,RVstart):
      upper_bounds[i] = lower_bounds[i]
    for i in range(RVend+1,RVmax):
      upper_bounds[i] = lower_bounds[i]

    #lbounds = lower_bounds[RVstart:1+RVend]
    #ubounds = upper_bounds[RVstart:1+RVend] 

    if sampled == False:
        mean_value = integrated_mean(model, lower_bounds, upper_bounds)
    else:
        pts = random_samples(lower_bounds, upper_bounds, npts=5000)
        mean_value = sampled_mean(model, pts, lower_bounds, upper_bounds)

    print "mean value: %s" % mean_value
    diameter = UQ(RVstart,RVend,lower_bounds,upper_bounds)
    from math import sqrt
    mcdiarmid = mcdiarmid_bound(mean_value - a, sqrt(diameter))
    print "McDiarmid bound: %s" % mcdiarmid
    return mcdiarmid, diameter

#-----------------------------------------------------------------
# Test function
def test_get_McDiarmid():
    def model(x):
       # x is of type numpy array, although it could also be list
       total = 1.
       n = len(x)
       for xi in x:
           total *= xi
       return total**(1./n) 

    n = 3
    lower_bounds = [1.]*n
    upper_bounds = [2.]*n
    
    print get_McDiarmid(model, lower_bounds, upper_bounds, a=1.4, sampled=True)

if __name__ == '__main__':
    test_get_McDiarmid()
