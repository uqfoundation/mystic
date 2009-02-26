#!/usr/bin/env python

## Differential Evolution Solver Class
## Based on algorithms developed by Dr. Rainer Storn & Kenneth Price
## Written By: Lester E. Godwin
##             PushCorp, Inc.
##             Dallas, Texas
##             972-840-0208 x102
##             godwin@pushcorp.com
## Created: 6/8/98
## Last Modified: 6/8/98
## Revision: 1.0
##
## Ported To Python From C++ July 2002
## Ported To Python By: James R. Phillips
##                      Birmingham, Alabama USA
##                      zunzun@zunzun.com
##
## bounds added by mmckerns@caltech.edu

"""
Differential Evolution Solver of Storn and Price
Ported to Python from Python by Parick Hung, May 2006. 

Two classes are exported.

  -- DifferentialEvolutionSolver, whose logic is identical to that found in DETest.py

  -- DifferentialEvoltuionSolver2, whose logic follows [2] in that a current generation,
     and a trial generation is kept. All the vectors for creating difference vectors and for 
     mutations draw from the current_generation, which remains invariant the end of the 
     round. (see tests/test_ffit.py and test/test_ffitB.py)
  
Reference:

[1] Storn, R. and Price, K. Differential Evolution - A Simple and Efficient
Heuristic for Global Optimization over Continuous Spaces. Journal of Global
Optimization 11: 341-359, 1997.

[2] Price, K., Storn, R., and Lampinen, J. - Differential Evolution, A Practical
Approach to Global Optimization. Springer, 1st Edition, 2005

"""
__all__ = ['DifferentialEvolutionSolver','DifferentialEvolutionSolver2']

from mystic.tools import Null, wrap_function

from abstract_solver import AbstractSolver

class DifferentialEvolutionSolver(AbstractSolver):
    """
    Differential Evolution optimization of Storn and Price.
    """
    
    def __init__(self, dim, NP):
        """
 Takes two inputs: 
   dim      -- dimensionality of the problem
   NP       -- size of the population (> 4)
        """
        #XXX: raise Error if npop <= 4?
        AbstractSolver.__init__(self,dim,npop=NP)
        self.genealogy     = [ [] for j in range(NP)]
        self.scale         = 0.7
        self.probability   = 0.5
        
    def _keepSolutionWithinRangeBoundary(self, base): #XXX: could be smarter?
        """scale trialSolution to be between base value and range boundary"""
        if not self._useStrictRange:
            return
        min = self._strictMin
        max = self._strictMax
        import random
        for i in range(self.nDim):
            if base[i] < min[i] or base[i] > max[i]:
                self.trialSolution[i] = random.uniform(min[i],max[i])
            elif self.trialSolution[i] < min[i]:
                self.trialSolution[i] = random.uniform(min[i],base[i])
            elif self.trialSolution[i] > max[i]:
                self.trialSolution[i] = random.uniform(base[i],max[i])
        return

    def UpdateGenealogyRecords(self, id, newchild):
        """
        Override me for more refined behavior. Currently all changes are logged.
        """
        self.genealogy[id].append(newchild)
        return

    def Solve(self, costfunction, strategy, termination,
              maxiter, CrossProbability = 0.5, ScalingFactor = 0.7,
              sigint_callback = None,
              EvaluationMonitor=Null, StepMonitor=Null, ExtraArgs=()):
        """Minimize a function using differential evolution.

    Description:

      <doc here>

    Inputs:

      <doc here>

    Additional Inputs:

      <doc here>

    Further Inputs:

      <doc here>

        """
        #FIXME: Solve() interface does not conform to AbstractSolver interface
        import signal
        import mystic.termination as detools
        detools.EARLYEXIT = False

        fcalls, costfunction = wrap_function(costfunction, ExtraArgs, EvaluationMonitor)

        def handler(signum, frame):
            import inspect
            print inspect.getframeinfo(frame)
            print inspect.trace()
            while 1:
                s = raw_input(\
"""
 
 Enter sense switch.

   sol: Write current best solution.
   cont: Continue calculation.
   call: Executes sigint_callback [%s].
   exit: Exits with current best solution.

 >>> """ % sigint_callback)
                if s.lower() == 'sol': 
                    print "sw1."
                    print self.bestSolution
                elif s.lower() == 'cont': 
                    return
                elif s.lower() == 'call': 
                    # sigint call_back
                    if sigint_callback is not None:
                        sigint_callback(self.bestSolution)
                elif s.lower() == 'exit': 
                    detools.EARLYEXIT = True
                    return
                else:
                    print "unknown option : %s ", s

        self.signal_handler = handler

        if self._handle_sigint: signal.signal(signal.SIGINT, self.signal_handler)

        self.probability = CrossProbability
        self.scale = ScalingFactor

        self.bestEnergy = 1.0E20
         
        generation = 0
        for generation in range(maxiter):
            StepMonitor(self.bestSolution[:], self.bestEnergy)
            for candidate in range(self.nPop):
                # generate trialSolution (within valid range)
                strategy(self, candidate)
                trialEnergy = costfunction(self.trialSolution)

                if trialEnergy < self.popEnergy[candidate]:
                    # New low for this candidate
                    self.popEnergy[candidate] = trialEnergy
                    self.population[candidate][:] = self.trialSolution[:]
                    self.UpdateGenealogyRecords(candidate, self.trialSolution[:])

                    # Check if all-time low
                    if trialEnergy < self.bestEnergy:
                        self.bestEnergy = trialEnergy
                        self.bestSolution[:] = self.trialSolution[:]
                            
            #To print this line, use StepMonitor instead
            #print "Generation %d has best cost function : %f" % (generation, self.bestEnergy)
            self.energy_history.append(self.bestEnergy)
            
            if detools.EARLYEXIT or termination(self):
                break

        self.generations = generation

        signal.signal(signal.SIGINT,signal.default_int_handler)

        return 



##########################################################################################
# DifferentialEvolutionSolver2 is functionally identical to MPIDifferentialEvolutionSolver
##########################################################################################

class DifferentialEvolutionSolver2(DifferentialEvolutionSolver):
    """
    Differential Evolution optimization of Storn and Price.

    Alternate implementaiton: 
      - functionally equivalent to MPIDifferentialEvolutionSolver.
      - both a current and a next generation are kept, while the current
        generation is invariant during the main DE logic.
    """
    def Solve(self, costfunction, strategy, termination,
              maxiter, CrossProbability = 0.5, ScalingFactor = 0.7,
              sigint_callback = None,
              EvaluationMonitor=Null, StepMonitor=Null, ExtraArgs=()):
        """Minimize a function using differential evolution.

    Description:

      <doc here>

    Inputs:

      <doc here>

    Additional Inputs:

      <doc here>

    Further Inputs:

      <doc here>

        """
        #FIXME: Solve() interface does not conform to AbstractSolver interface
        import signal
        import mystic.termination as detools
        detools.EARLYEXIT = False

        fcalls, costfunction = wrap_function(costfunction, ExtraArgs, EvaluationMonitor)

        def handler(signum, frame):
            import inspect
            print inspect.getframeinfo(frame)
            print inspect.trace()
            while 1:
                s = raw_input(\
"""
 
 Enter sense switch.

   sol: Write current best solution.
   cont: Continue calculation.
   call: Executes sigint_callback [%s].
   exit: Exits with current best solution.

 >>> """ % sigint_callback)
                if s.lower() == 'sol': 
                    print "sw1."
                    print self.bestSolution
                elif s.lower() == 'cont': 
                    return
                elif s.lower() == 'call': 
                    # sigint call_back
                    if sigint_callback is not None:
                        sigint_callback(self.bestSolution)
                elif s.lower() == 'exit': 
                    detools.EARLYEXIT = True
                    return
                else:
                    print "unknown option : %s ", s

        if self._handle_sigint: signal.signal(signal.SIGINT, handler)

        self.probability = CrossProbability
        self.scale = ScalingFactor

        self.bestEnergy = 1.0E20
         
        trialPop = [[0.0 for i in range(self.nDim)] for j in range(self.nPop)]

        generation = 0
        for generation in range(maxiter):
            StepMonitor(self.bestSolution[:], self.bestEnergy)
            for candidate in range(self.nPop):
                # generate trialSolution (within valid range)
                strategy(self, candidate)
                trialPop[candidate][:] = self.trialSolution[:]

            trialEnergy = map(costfunction, trialPop)

            for candidate in range(self.nPop):
                if trialEnergy[candidate] < self.popEnergy[candidate]:
                    # New low for this candidate
                    self.popEnergy[candidate] = trialEnergy[candidate]
                    self.population[candidate][:] = trialPop[candidate][:]
                    self.UpdateGenealogyRecords(candidate, self.trialSolution[:])

                    # Check if all-time low
                    if trialEnergy[candidate] < self.bestEnergy:
                        self.bestEnergy = trialEnergy[candidate]
                        self.bestSolution[:] = trialPop[candidate][:]
                            
            #To print this line, use StepMonitor instead
            #print "Generation %d has best cost function : %f" % (generation, self.bestEnergy)
            self.energy_history.append(self.bestEnergy)
            
            if detools.EARLYEXIT or termination(self):
                break

        self.generations = generation

        signal.signal(signal.SIGINT,signal.default_int_handler)

        return 


if __name__=='__main__':
    help(__name__)

# end of file
