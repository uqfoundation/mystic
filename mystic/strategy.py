#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
#
# Differential Evolution Strategies adapted from DESolver.py by Patrick Hung
"""
Differential Evolution Strategies

These strategies are to be passed into DifferentialEvolutionSolver's
Solve method, and determine how the candidate parameter values mutate
across a population.
"""

import random

def get_random_candidates(NP, exclude, N):
    """select N random candidates from population of size NP,
where exclude is the candidate to exclude from selection.

Thus, get_random_candidates(x,1,2) randomly selects two nPop[i],
where i != 1"""
    return random.sample(list(range(exclude))+list(range(exclude+1,NP)), N)


#################### #################### #################### ####################
#  Code below are the different crossovers/mutation strategies
#################### #################### #################### ####################

def Best1Exp(inst, candidate):
    """trial solution is current best solution plus scaled difference
of two randomly chosen candidates; mutates until random stop

trial = best + scale*(candidate1 - candidate2)"""
    r1,r2 = get_random_candidates(inst.nPop, candidate, 2) 
    n = random.randrange(inst.nDim)

    if inst._map_solver:
        trialSolution = inst.trialSolution[candidate]
    else:
        trialSolution = inst.trialSolution
    trialSolution[:] = inst.population[candidate]

    i = 0
    while 1:
        if random.random() >= inst.probability or i == inst.nDim:
            break
        trialSolution[n] = inst.bestSolution[n] + \
                           inst.scale * (inst.population[r1][n] - \
                                         inst.population[r2][n])
        n = (n + 1) % inst.nDim
        i += 1

#   inst._keepSolutionWithinRangeBoundary(inst.bestSolution)
    return

def Best1Bin(inst, candidate):
    """trial solution is current best solution plus scaled difference
of two randomly chosen candidates; mutates at random

trial = best + scale*(candidate1 - candidate2)"""
    # In DESolve, Best1Bin was identical to Best1Exp.
    # But the logic of Best1Bin is different from [1]. Reimplementing here.
    r1,r2 = get_random_candidates(inst.nPop, candidate, 2) 

    if inst._map_solver:
        trialSolution = inst.trialSolution[candidate]
    else:
        trialSolution = inst.trialSolution
    trialSolution[:] = inst.population[candidate]

    # Randomly chosen index between [0, ND-1] (See Eq.4 of [1] )
    n = random.randrange(inst.nDim)

    for i in range(inst.nDim):
        cross = random.random()
        if i==n or cross < inst.probability:
            # this component of trial vector will come from vector v
            trialSolution[i] = inst.bestSolution[i] + \
                               inst.scale * (inst.population[r1][i] - \
                                             inst.population[r2][i])

#   inst._keepSolutionWithinRangeBoundary(inst.bestSolution)
    return

def Rand1Exp(inst, candidate):
    """trial solution is randomly chosen candidate plus scaled difference
of two other randomly chosen candidates; mutates until random stop

trial = candidate1 + scale*(candidate2 - candidate3)"""
    r1,r2,r3 = get_random_candidates(inst.nPop, candidate, 3) 
    n = random.randrange(inst.nDim)

    if inst._map_solver:
        trialSolution = inst.trialSolution[candidate]
    else:
        trialSolution = inst.trialSolution
    trialSolution[:] = inst.population[candidate]

    i = 0
    while 1:
        if random.random() >= inst.probability or i == inst.nDim:
            break
        trialSolution[n] = inst.population[r1][n] + \
                           inst.scale * (inst.population[r2][n] - \
                                         inst.population[r3][n])
        n = (n + 1) % inst.nDim
        i += 1

#   inst._keepSolutionWithinRangeBoundary(inst.population[r1])
    return

# WARNING, stuff below are not debugged

def RandToBest1Exp(inst, candidate):
    """trial solution is itself plus scaled difference of best solution
and trial solution, plus the difference of two randomly chosen candidates;
mutates at random

trial += scale*(best - trial) + scale*(candidate1 - candidate2)"""
    r1,r2 = get_random_candidates(inst.nPop, candidate, 2) 
    n = random.randrange(inst.nDim)

    if inst._map_solver:
        trialSolution = inst.trialSolution[candidate]
    else:
        trialSolution = inst.trialSolution
    trialSolution[:] = inst.population[candidate][:]
    i = 0
    while 1:
        if random.random() >= inst.probability or i == inst.nDim:
            break
        trialSolution[n] += inst.scale * (inst.bestSolution[n] - \
                                          trialSolution[n]) + \
                            inst.scale * (inst.population[r1][n] - \
                                          inst.population[r2][n])
        n = (n + 1) % inst.nDim
        i += 1

#   inst._keepSolutionWithinRangeBoundary(inst.trialSolution)
    return

def Best2Exp(inst, candidate):
    """trial solution is current best solution plus scaled contributions
from four randomly chosen candidates; mutates until random stop

trial = best + scale*(candidate1 + candidate2 - candidate3 - candidate4)"""
    r1,r2,r3,r4 = get_random_candidates(inst.nPop, candidate, 4) 
    n = random.randrange(inst.nDim)

    if inst._map_solver:
        trialSolution = inst.trialSolution[candidate]
    else:
        trialSolution = inst.trialSolution
    trialSolution[:] = inst.population[candidate]
    i = 0
    while 1:
        if random.random() >= inst.probability or i == inst.nDim:
            break
        trialSolution[n] = inst.bestSolution[n] + \
                           inst.scale * (inst.population[r1][n] + \
                                         inst.population[r2][n] - \
                                         inst.population[r3][n] - \
                                         inst.population[r4][n])
        n = (n + 1) % inst.nDim
        i += 1

#   inst._keepSolutionWithinRangeBoundary(inst.bestSolution)
    return

def Rand2Exp(inst, candidate):
    """trial solution is randomly chosen candidate plus scaled contributions
from four other randomly chosen candidates; mutates until random stop

trial = candidate1 + scale*(candidate2 + candidate3 - candidate4 - candidate5)"""
    r1,r2,r3,r4,r5 = get_random_candidates(inst.nPop, candidate, 5) 
    n = random.randrange(inst.nDim)

    if inst._map_solver:
        trialSolution = inst.trialSolution[candidate]
    else:
        trialSolution = inst.trialSolution
    trialSolution[:] = inst.population[candidate]
    i = 0
    while 1:
        if random.random() >= inst.probability or i == inst.nDim:
            break
        trialSolution[n] = inst.population[r1][n] + \
                           inst.scale * (inst.population[r2][n] + \
                                         inst.population[r3][n] - \
                                         inst.population[r4][n] - \
                                         inst.population[r5][n])
        n = (n + 1) % inst.nDim
        i += 1

#   inst._keepSolutionWithinRangeBoundary(inst.population[r1])
    return

def Rand1Bin(inst, candidate):
    """trial solution is randomly chosen candidate plus scaled difference
of two other randomly chosen candidates; mutates at random

trial = candidate1 + scale*(candidate2 - candidate3)"""
    r1,r2,r3 = get_random_candidates(inst.nPop, candidate, 3) 
    n = random.randrange(inst.nDim)

    if inst._map_solver:
        trialSolution = inst.trialSolution[candidate]
    else:
        trialSolution = inst.trialSolution
    trialSolution[:] = inst.population[candidate]
    i = 0
    while 1:
        if random.random() >= inst.probability or i == inst.nDim:
            break
        trialSolution[n] = inst.population[r1][n] + \
                           inst.scale * (inst.population[r2][n] -\
                                         inst.population[r3][n])
        n = (n + 1) % inst.nDim
        i += 1

#   inst._keepSolutionWithinRangeBoundary(inst.population[r1])
    return

def RandToBest1Bin(inst, candidate):
    """trial solution is itself plus scaled difference of best solution
and trial solution, plus the difference of two randomly chosen candidates;
mutates until random stop

trial += scale*(best - trial) + scale*(candidate1 - candidate2)"""
    r1,r2 = get_random_candidates(inst.nPop, candidate, 2) 
    n = random.randrange(inst.nDim)

    if inst._map_solver:
        trialSolution = inst.trialSolution[candidate]
    else:
        trialSolution = inst.trialSolution
    trialSolution[:] = inst.population[candidate]
    i = 0
    while 1:
        if random.random() >= inst.probability or i == inst.nDim:
            break
        trialSolution[n] += inst.scale * (inst.bestSolution[n] - \
                                          trialSolution[n]) + \
                            inst.scale * (inst.population[r1][n] - \
                                          inst.population[r2][n])
        n = (n + 1) % inst.nDim
        i += 1

#   inst._keepSolutionWithinRangeBoundary(inst.trialSolution)
    return

def Best2Bin(inst, candidate):
    """trial solution is current best solution plus scaled contributions
of four randomly chosen candidates; mutates at random

trial = best + scale*(candidate1 - candidate2 - candidate3 - candidate4)"""
    r1,r2,r3,r4 = get_random_candidates(inst.nPop, candidate, 4) 
    n = random.randrange(inst.nDim)

    if inst._map_solver:
        trialSolution = inst.trialSolution[candidate]
    else:
        trialSolution = inst.trialSolution
    trialSolution[:] = inst.population[candidate]
    i = 0
    while 1:
        if random.random() >= inst.probability or i == inst.nDim:
            break
        trialSolution[n] = inst.bestSolution[n] + \
                           inst.scale * (inst.population[r1][n] + \
                                         inst.population[r2][n] - \
                                         inst.population[r3][n] - \
                                         inst.population[r4][n])
        n = (n + 1) % inst.nDim
        i += 1

#   inst._keepSolutionWithinRangeBoundary(inst.bestSolution)
    return

def Rand2Bin(inst, candidate):
    """trial solution is randomly chosen candidate plus scaled contributions
of four other randomly chosen candidates; mutates at random

trial = candidate1 + scale*(candidate2 - candidate3 - candidate4 - candidate5)"""
    r1,r2,r3,r4,r5 = get_random_candidates(inst.nPop, candidate, 5) 
    n = random.randrange(inst.nDim)

    if inst._map_solver:
        trialSolution = inst.trialSolution[candidate]
    else:
        trialSolution = inst.trialSolution
    trialSolution[:] = inst.population[candidate]
    i = 0
    while 1:
        if random.random() >= inst.probability or i == inst.nDim:
            break
        trialSolution[n] = inst.population[r1][n] + \
                           inst.scale * (inst.population[r2][n] + \
                                         inst.population[r3][n] - \
                                         inst.population[r4][n] - \
                                         inst.population[r5][n])
        n = (n + 1) % inst.nDim
        i += 1

#   inst._keepSolutionWithinRangeBoundary(inst.population[r1])
    return
    
# end of file
