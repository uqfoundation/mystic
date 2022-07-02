#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

import numpy as np
from mystic.math.measures import _k


if __name__ == '__main__':
    # even-length
    w = [3,1,1,1,3,3]
    assert _k(w) == w
    # even-length clipping
    assert (np.array(_k(w,(10,10),clip=True)) > 0).tolist() == [1,1,1,1,1,1]
    assert (np.array(_k(w,(25,25),clip=True)) > 0).tolist() == [0,1,1,1,1,0]
    assert (np.array(_k(w,(50,50),clip=True)) > 0).tolist() == [0,0,0,1,1,0]
    assert (np.array(_k(w,(49,50),clip=True)) > 0).tolist() == [0,0,0,1,0,0]
    assert (np.array(_k(w,(50,49),clip=True)) > 0).tolist() == [0,0,0,0,1,0]
    assert (np.array(_k(w,(49,49),clip=True)) > 0).tolist() == [0,0,0,1,1,0]
    assert (np.array(_k(w,(25,75),clip=True)) > 0).tolist() == [1,1,0,0,0,0]
    assert (np.array(_k(w,(24,75),clip=True)) > 0).tolist() == [1,0,0,0,0,0]
    assert (np.array(_k(w,(25,74),clip=True)) > 0).tolist() == [0,1,0,0,0,0]
    assert (np.array(_k(w,(24,74),clip=True)) > 0).tolist() == [1,1,0,0,0,0]
    assert (np.array(_k(w,(75,25),clip=True)) > 0).tolist() == [0,0,0,0,1,1]
    assert (np.array(_k(w,(74,25),clip=True)) > 0).tolist() == [0,0,0,0,1,0]
    assert (np.array(_k(w,(75,24),clip=True)) > 0).tolist() == [0,0,0,0,0,1]
    assert (np.array(_k(w,(74,24),clip=True)) > 0).tolist() == [0,0,0,0,1,1]
    # even-length trimming
    assert (np.array(_k(w,(10,10))) > 0).tolist() == [1,1,1,1,1,1]
    assert (np.array(_k(w,(25,25))) > 0).tolist() == [0,1,1,1,1,0]
    assert (np.array(_k(w,(50,50))) > 0).tolist() == [0,0,0,0,0,0]
    assert (np.array(_k(w,(49,50))) > 0).tolist() == [0,0,0,1,0,0]
    assert (np.array(_k(w,(50,49))) > 0).tolist() == [0,0,0,0,1,0]
    assert (np.array(_k(w,(49,49))) > 0).tolist() == [0,0,0,1,1,0]
    assert (np.array(_k(w,(25,75))) > 0).tolist() == [0,0,0,0,0,0]
    assert (np.array(_k(w,(24,75))) > 0).tolist() == [1,0,0,0,0,0]
    assert (np.array(_k(w,(25,74))) > 0).tolist() == [0,1,0,0,0,0]
    assert (np.array(_k(w,(24,74))) > 0).tolist() == [1,1,0,0,0,0]
    assert (np.array(_k(w,(75,25))) > 0).tolist() == [0,0,0,0,0,0]
    assert (np.array(_k(w,(74,25))) > 0).tolist() == [0,0,0,0,1,0]
    assert (np.array(_k(w,(75,24))) > 0).tolist() == [0,0,0,0,0,1]
    assert (np.array(_k(w,(74,24))) > 0).tolist() == [0,0,0,0,1,1]
    # odd-length
    w = [4,2,4,2,4]
    assert _k(w) == w
    # odd-length clipping
    assert (np.array(_k(w,(10,10),clip=True)) > 0).tolist() == [1,1,1,1,1]
    assert (np.array(_k(w,(25,25),clip=True)) > 0).tolist() == [0,1,1,1,0]
    assert (np.array(_k(w,(50,50),clip=True)) > 0).tolist() == [0,0,1,0,0]
    assert (np.array(_k(w,(37.5,37.5),clip=True)) > 0).tolist() == [0,0,1,0,0]
    assert (np.array(_k(w,(37.4,37.5),clip=True)) > 0).tolist() == [0,1,1,0,0]
    assert (np.array(_k(w,(37.5,37.4),clip=True)) > 0).tolist() == [0,0,1,1,0]
    assert (np.array(_k(w,(37.4,37.4),clip=True)) > 0).tolist() == [0,1,1,1,0]
    assert (np.array(_k(w,(25,75),clip=True)) > 0).tolist() == [1,1,0,0,0]
    assert (np.array(_k(w,(24,75),clip=True)) > 0).tolist() == [1,0,0,0,0]
    assert (np.array(_k(w,(25,74),clip=True)) > 0).tolist() == [0,1,0,0,0]
    assert (np.array(_k(w,(24,74),clip=True)) > 0).tolist() == [1,1,0,0,0]
    assert (np.array(_k(w,(75,25),clip=True)) > 0).tolist() == [0,0,0,1,1]
    assert (np.array(_k(w,(74,25),clip=True)) > 0).tolist() == [0,0,0,1,0]
    assert (np.array(_k(w,(75,24),clip=True)) > 0).tolist() == [0,0,0,0,1]
    # odd-length trimming
    assert (np.array(_k(w,(10,10))) > 0).tolist() == [1,1,1,1,1]
    assert (np.array(_k(w,(25,25))) > 0).tolist() == [0,1,1,1,0]
    assert (np.array(_k(w,(50,50))) > 0).tolist() == [0,0,0,0,0]
    assert (np.array(_k(w,(37.5,37.5))) > 0).tolist() == [0,0,1,0,0]
    assert (np.array(_k(w,(37.4,37.5))) > 0).tolist() == [0,1,1,0,0]
    assert (np.array(_k(w,(37.5,37.4))) > 0).tolist() == [0,0,1,1,0]
    assert (np.array(_k(w,(37.4,37.4))) > 0).tolist() == [0,1,1,1,0]
    assert (np.array(_k(w,(25,75))) > 0).tolist() == [0,0,0,0,0]
    assert (np.array(_k(w,(24,75))) > 0).tolist() == [1,0,0,0,0]
    assert (np.array(_k(w,(25,74))) > 0).tolist() == [0,1,0,0,0]
    assert (np.array(_k(w,(24,74))) > 0).tolist() == [1,1,0,0,0]
    assert (np.array(_k(w,(75,25))) > 0).tolist() == [0,0,0,0,0]
    assert (np.array(_k(w,(74,25))) > 0).tolist() == [0,0,0,1,0]
    assert (np.array(_k(w,(75,24))) > 0).tolist() == [0,0,0,0,1]
    assert (np.array(_k(w,(74,24))) > 0).tolist() == [0,0,0,1,1]


# EOF
