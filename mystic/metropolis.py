#!/usr/bin/env python

"""
Implements a simple version of the Metropolis-Hastings algorithm
"""


def metropolis_hastings(proposal, target, x):
    """ 
Proposal(x) -> next point. Must be symmetric.
This is because otherwise the PDF of the proposal density is needed
(not just a way to draw from the proposal)
    """
    from numpy import random
    Yt = proposal(x)
    r = min(target(Yt)/(target(x) + 1e-100), 1)
    if random.random() <= r:
        return Yt
    else:
        return x

    
if __name__=='__main__':
    pass

# end of file
