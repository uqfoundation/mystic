#!/usr/bin/env python

"""
Differential Evolution Strategies adapted from DESolver.py by Patrick Hung

These strategies are to be passed into DifferentialEvolutionSolver's Solve method.
"""

import random

def get_random_candidates(NP, exclude, N):
    return random.sample(range(exclude)+range(exclude+1,NP), N)


#################### #################### #################### ####################
#  Code below are the different crossovers/mutation strategies
#################### #################### #################### ####################

def Best1Exp(inst, candidate):
    r1,r2 = get_random_candidates(inst.nPop, candidate, 2) 
    n = random.randrange(inst.nDim)
   
    inst.trialSolution[:] = inst.population[candidate][:]

    i = 0
    while 1:
        if random.random() >= inst.probability or i == inst.nDim:
            break
        inst.trialSolution[n] = inst.bestSolution[n] + \
                                inst.scale * (inst.population[r1][n] - \
                                              inst.population[r2][n])
        n = (n + 1) % inst.nDim
        i += 1

    inst.scaleSolutionWithRangeBoundary(inst.bestSolution)
    return

def Best1Bin(inst, candidate):
    """
In DESolve, Best1Bin was identical to Best1Exp.
But the logic of Best1Bin is different from [1]. Reimplementing here.
    """
    r1,r2 = get_random_candidates(inst.nPop, candidate, 2) 

    inst.trialSolution[:] = inst.population[candidate][:]

    # Randomly chosen index between [0, ND-1] (See Eq.4 of [1] )
    n = random.randrange(inst.nDim)

    for i in range(inst.nDim):
        cross = random.random()
        if i==n or cross < inst.probability:
            # this component of trial vector will come from vector v
            inst.trialSolution[i] = inst.bestSolution[i] + \
                                    inst.scale * (inst.population[r1][i] - \
                                                  inst.population[r2][i])

    inst.scaleSolutionWithRangeBoundary(inst.bestSolution)
    return

def Rand1Exp(inst, candidate):
    r1,r2,r3 = get_random_candidates(inst.nPop, candidate, 3) 
    n = random.randrange(inst.nDim)

    inst.trialSolution[:] = inst.population[candidate][:]

    i = 0
    while 1:
        if random.random() >= inst.probability or i == inst.nDim:
            break
        inst.trialSolution[n] = inst.population[r1][n] + \
                                inst.scale * (inst.population[r2][n] - \
                                              inst.population[r3][n])
        n = (n + 1) % inst.nDim
        i += 1

    inst.scaleSolutionWithRangeBoundary(inst.population[r1])
    return

# WARNING, stuff below are not debugged

def RandToBest1Exp(inst, candidate):
    r1,r2 = get_random_candidates(inst.nPop, candidate, 2) 
    n = random.randrange(inst.nDim)

    inst.trialSolution[:] = inst.population[candidate][:]
    i = 0
    while 1:
        if random.random() >= inst.probability or i == inst.nDim:
            break
        inst.trialSolution[n] += inst.scale * (inst.bestSolution[n] - \
                                               inst.trialSolution[n]) + \
                                 inst.scale * (inst.population[r1][n] - \
                                               inst.population[r2][n])
        n = (n + 1) % inst.nDim
        i += 1

    inst.scaleSolutionWithRangeBoundary(inst.trialSolution)
    return

def Best2Exp(inst, candidate):
    r1,r2,r3,r4 = get_random_candidates(inst.nPop, candidate, 4) 
    n = random.randrange(inst.nDim)

    inst.trialSolution[:] = inst.population[candidate][:]
    i = 0
    while 1:
        if random.random() >= inst.probability or i == inst.nDim:
            break
        inst.trialSolution[n] = inst.bestSolution[n] + \
                                inst.scale * (inst.population[r1][n] + \
                                              inst.population[r2][n] - \
                                              inst.population[r3][n] - \
                                              inst.population[r4][n])
        n = (n + 1) % inst.nDim
        i += 1

    inst.scaleSolutionWithRangeBoundary(inst.bestSolution)
    return

def Rand2Exp(inst, candidate):
    r1,r2,r3,r4,r5 = get_random_candidates(inst.nPop, candidate, 5) 
    n = random.randrange(inst.nDim)

    inst.trialSolution[:] = inst.population[candidate][:]
    i = 0
    while 1:
        if random.random() >= inst.probability or i == inst.nDim:
            break
        inst.trialSolution[n] = inst.population[r1][n] + \
                                inst.scale * (inst.population[r2][n] + \
                                              inst.population[r3][n] - \
                                              inst.population[r4][n] - \
                                              inst.population[r5][n])
        n = (n + 1) % inst.nDim
        i += 1

    inst.scaleSolutionWithRangeBoundary(inst.population[r1])
    return

def Rand1Bin(inst, candidate):
    r1,r2,r3 = get_random_candidates(inst.nPop, candidate, 3) 
    n = random.randrange(inst.nDim)

    inst.trialSolution[:] = inst.population[candidate][:]
    i = 0
    while 1:
        if random.random() >= inst.probability or i == inst.nDim:
            break
        inst.trialSolution[n] = inst.population[r1][n] + \
                                inst.scale * (inst.population[r2][n] -\
                                              inst.population[r3][n])
        n = (n + 1) % inst.nDim
        i += 1

    inst.scaleSolutionWithRangeBoundary(inst.population[r1])
    return

def RandToBest1Bin(inst, candidate):
    r1,r2 = get_random_candidates(inst.nPop, candidate, 2) 
    n = random.randrange(inst.nDim)

    inst.trialSolution[:] = inst.population[candidate][:]
    i = 0
    while 1:
        if random.random() >= inst.probability or i == inst.nDim:
            break
        inst.trialSolution[n] += inst.scale * (inst.bestSolution[n] - \
                                               inst.trialSolution[n]) + \
                                 inst.scale * (inst.population[r1][n] - \
                                               inst.population[r2][n])
        n = (n + 1) % inst.nDim
        i += 1

    inst.scaleSolutionWithRangeBoundary(inst.trialSolution)
    return

def Best2Bin(inst, candidate):
    r1,r2,r3,r4 = get_random_candidates(inst.nPop, candidate, 4) 
    n = random.randrange(inst.nDim)

    inst.trialSolution[:] = inst.population[candidate][:]
    i = 0
    while 1:
        if random.random() >= inst.probability or i == inst.nDim:
            break
        inst.trialSolution[n] = inst.bestSolution[n] + \
                                inst.scale * (inst.population[r1][n] + \
                                              inst.population[r2][n] - \
                                              inst.population[r3][n] - \
                                              inst.population[r4][n])
        n = (n + 1) % inst.nDim
        i += 1

    inst.scaleSolutionWithRangeBoundary(inst.bestSolution)
    return

def Rand2Bin(inst, candidate):
    r1,r2,r3,r4,r5 = get_random_candidates(inst.nPop, candidate, 5) 
    n = random.randrange(inst.nDim)

    inst.trialSolution[:] = inst.population[candidate][:]
    i = 0
    while 1:
        if random.random() >= inst.probability or i == inst.nDim:
            break
        inst.trialSolution[n] = inst.population[r1][n] + \
                                inst.scale * (inst.population[r2][n] + \
                                              inst.population[r3][n] - \
                                              inst.population[r4][n] - \
                                              inst.population[r5][n])
        n = (n + 1) % inst.nDim
        i += 1

    inst.scaleSolutionWithRangeBoundary(inst.population[r1])
    return
    
# end of file
