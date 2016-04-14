#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2016 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE
"""
Implements a simple version of the Metropolis-Hastings algorithm
"""
from __future__ import division
from past.utils import old_div


def metropolis_hastings(proposal, target, x):
    """ 
Proposal(x) -> next point. Must be symmetric.
This is because otherwise the PDF of the proposal density is needed
(not just a way to draw from the proposal)
    """
    import random
    Yt = proposal(x)
    r = min(old_div(target(Yt),(target(x) + 1e-100)), 1)
    if random.random() <= r:
        return Yt
    else:
        return x

    
if __name__=='__main__':
    pass

# end of file
