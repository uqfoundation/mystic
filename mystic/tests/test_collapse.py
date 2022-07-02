#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

import mystic.collapse as ct
import numpy as np
import mystic as my
m = my.monitors._load('_log.py')
# cleanup *pyc
import os
try: os.remove('_log.pyc')
except OSError: pass

# at
x = ct.collapse_at(m, target=0.0, tolerance=0.05, mask=None)
assert x == set((1, 7, 8, 9, 10, 11, 12, 14))
x = ct.collapse_at(m, target=0.0, tolerance=0.05, mask=set((10,11,12)))
assert x == set((1, 7, 8, 9, 14))
try: ct.collapse_at(m, target=0.0, mask=10); raise RuntimeError()
except TypeError: pass
try: ct.collapse_at(m, target=0.0, mask=(10,11,12)); raise RuntimeError()
except TypeError: pass
try: ct.collapse_at(m, target=0.0, mask=[(10,11,12)])
except TypeError: pass
try: ct.collapse_at(m, target=0.0, mask=[(10,11,12),(13,14,15)]); raise RuntimeError()
except TypeError: pass
try: ct.collapse_at(m, target=0.0, mask=((10,11),(12,13))); raise RuntimeError()
except TypeError: pass
try: ct.collapse_at(m, target=0.0, mask=(np.array((10,11)),np.array((12,13)))); raise RuntimeError()
except TypeError: pass
try: ct.collapse_at(m, target=0.0, mask=set(((10,11),(12,13)))); raise RuntimeError()
except ValueError: pass
try: ct.collapse_at(m, target=0.0, mask={(10,11):(12,13)}); raise RuntimeError()
except TypeError: pass


# as
x = ct.collapse_as(m, tolerance=0.05, mask=None)
assert x == set(((10, 11), (7, 12), (10, 12), (8, 9), (11, 14), (7, 11), (1, 11), (16, 17), (8, 14), (1, 14), (8, 10), (9, 11), (7, 10), (1, 10), (7, 14), (9, 14), (12, 14), (8, 11), (9, 10), (1, 9), (11, 12), (7, 9), (1, 12), (8, 12), (3, 4), (1, 8), (10, 14), (6, 13), (1, 7), (7, 8), (9, 12)))
x = ct.collapse_as(m, tolerance=0.05, mask=set(((10,11),(7,12),(10,12))))
assert x == set(((8, 9), (11, 14), (7, 11), (1, 11), (16, 17), (8, 14), (1, 14), (8, 10), (9, 11), (7, 10), (1, 10), (7, 14), (9, 14), (12, 14), (8, 11), (9, 10), (1, 9), (11, 12), (7, 9), (1, 12), (8, 12), (3, 4), (1, 8), (10, 14), (6, 13), (1, 7), (7, 8), (9, 12)))
x = ct.collapse_as(m, tolerance=0.05, mask=set((10,12)))
assert x == set(((8, 9), (11, 14), (7, 11), (1, 11), (16, 17), (8, 14), (1, 14), (9, 11), (7, 14), (9, 14), (8, 11), (1, 9), (7, 9), (3, 4), (1, 8), (6, 13), (1, 7), (7, 8)))
x = ct.collapse_as(m, tolerance=0.05, mask=set((10,12,(9,8),(14,11))))
assert x == set(((7, 11), (1, 11), (16, 17), (8, 14), (1, 14), (9, 11), (7, 14), (9, 14), (8, 11), (1, 9), (7, 9), (3, 4), (1, 8), (6, 13), (1, 7), (7, 8)))
try: ct.collapse_as(m, mask=10); raise RuntimeError()
except TypeError: pass
try: ct.collapse_as(m, mask=(10,11)); raise RuntimeError()
except TypeError: pass
try: ct.collapse_as(m, mask=(10,11,12,13)); raise RuntimeError()
except TypeError: pass
try: ct.collapse_as(m, mask=set(((10,11,12),(14,8,9)))); raise RuntimeError()
except ValueError: pass
try: ct.collapse_as(m, mask=[(10,11,12,13)]); raise RuntimeError()
except TypeError: pass
try: ct.collapse_as(m, mask=[(11,12),(14,9)]); raise RuntimeError()
except TypeError: pass
try: ct.collapse_as(m, mask=(np.array([11,12]),np.array([14,9]))); raise RuntimeError()
except TypeError: pass
try: ct.collapse_as(m, mask=(np.array([10,11,12]),np.array([14,8,9]))); raise RuntimeError()
except TypeError: pass
try: ct.collapse_as(m, mask={11:9}); raise RuntimeError()
except TypeError: pass
try: ct.collapse_as(m, mask={(11,12):(14,9)}); raise RuntimeError()
except TypeError: pass


# weights
x = ct.collapse_weight(m, mask=None)
assert x == {0:set((1,)), 1:set((1,2)), 2:set((0,2))}
x = ct.collapse_weight(m, mask={0:set((1,)), 1:set((0,1))})
assert x == {1:set((2,)), 2:set((0,2))}
x = ct.collapse_weight(m, mask=((0,1),(1,2)))
assert x == ((1,2,2),(1,0,2))
x = ct.collapse_weight(m, mask=set(((0,1),(1,2),(2,0))))
assert x == set(((1,1),(2,2)))
try: ct.collapse_weight(m, mask=(None,None)); raise RuntimeError()
except TypeError: pass
try: ct.collapse_weight(m, mask=(0,None)); raise RuntimeError()
except TypeError: pass
try: ct.collapse_weight(m, mask=(None,2)); raise RuntimeError()
except TypeError: pass
try: ct.collapse_weight(m, mask=(1,2)); raise RuntimeError()
except TypeError: pass
try: ct.collapse_weight(m, mask=((1,2),None)); raise RuntimeError()
except TypeError: pass
try: ct.collapse_weight(m, mask=(None,(1,2))); raise RuntimeError()
except TypeError: pass
x = ct.collapse_weight(m, mask=((0,1,2),(0,1,2)))
assert x == ((0, 1, 2), (1, 2, 0))
try: ct.collapse_weight(m, mask=(slice(1,None,1),slice(1,None,1))); raise RuntimeError()
except TypeError: pass
try: ct.collapse_weight(m, mask=(slice(None,2,1),slice(None,2,1))); raise RuntimeError()
except TypeError: pass
try: ct.collapse_weight(m, mask=(None,set((0,1)))); raise RuntimeError()
except TypeError: pass
try: ct.collapse_weight(m, mask=(None,set(((0,1))))); raise RuntimeError()
except TypeError: pass
try: ct.collapse_weight(m, mask=(None,{0:1})); raise RuntimeError()
except TypeError: pass
try: ct.collapse_weight(m, mask=(None,{0:(1,)})); raise RuntimeError()
except TypeError: pass
try: ct.collapse_weight(m, mask=(None,[(1,2)])); raise RuntimeError()
except TypeError: pass
try: ct.collapse_weight(m, mask=(None,[(0,1),(1,2)])); raise RuntimeError()
except TypeError: pass
try: ct.collapse_weight(m, mask=([(1,2)],[(1,2)])); raise RuntimeError()
except TypeError: pass
try: ct.collapse_weight(m, mask=([(0,1),(1,2)],[(0,1),(1,2)])); raise RuntimeError()
except TypeError: pass
try: ct.collapse_weight(m, mask=set((None,))); raise RuntimeError()
except ValueError: pass
try: ct.collapse_weight(m, mask=set((((0,1),(1,2)),))); raise RuntimeError()
except ValueError: pass
try: ct.collapse_weight(m, mask=set((((0,1),(1,2)),((1,2),(2,2))))); raise RuntimeError()
except ValueError: pass
try: ct.collapse_weight(m, mask=set((0,))); raise RuntimeError() # error
except ValueError: pass
try: ct.collapse_weight(m, mask=set(((0,)))); raise RuntimeError()
except ValueError: pass
try: ct.collapse_weight(m, mask=set(((0,1,2)))); raise RuntimeError()
except ValueError: pass
try: ct.collapse_weight(m, mask=set((((0,1,1,2))))); raise RuntimeError()
except ValueError: pass
try: ct.collapse_weight(m, mask=set(([(0,1)],[(1,2)]))); raise RuntimeError()
except TypeError: pass
try: ct.collapse_weight(m, mask=set((((0,1),),((1,2),)))); raise RuntimeError()
except ValueError: pass
try: ct.collapse_weight(m, mask={None:None}); raise RuntimeError()
except ValueError: pass
try: ct.collapse_weight(m, mask={1:None}); raise RuntimeError()
except ValueError: pass
try: ct.collapse_weight(m, mask={None:2}); raise RuntimeError()
except ValueError: pass
try: ct.collapse_weight(m, mask={1:2}); raise RuntimeError()
except ValueError: pass
try: ct.collapse_weight(m, mask={None:(1,)}); raise RuntimeError()
except ValueError: pass
try: ct.collapse_weight(m, mask={1:(1,)}); raise RuntimeError()
except ValueError: pass
try: ct.collapse_weight(m, mask={0:(0,1,2)}); raise RuntimeError()
except ValueError: pass
x = ct.collapse_weight(m, mask={0:set((0,1,2))})
assert x == {1:set((1,2)), 2:set((0,2))}
try: ct.collapse_weight(m, mask={0:set(((0,1,2),))}); raise RuntimeError()
except ValueError: pass
try: ct.collapse_weight(m, mask={0:[(0,1,2)]}); raise RuntimeError()
except ValueError: pass
try: ct.collapse_weight(m, mask={0:[(0,1),(2,0)]}); raise RuntimeError()
except ValueError: pass
try: ct.collapse_weight(m, mask={None:[(0,1),(2,0)]}); raise RuntimeError()
except ValueError: pass
try: ct.collapse_weight(m, mask={(0,1):[(0,1),(2,0)]}); raise RuntimeError()
except ValueError: pass
try: ct.collapse_weight(m, mask={(1,):(1,)}); raise RuntimeError()
except ValueError: pass
try: ct.collapse_weight(m, mask={(1,2):(1,2)}); raise RuntimeError()
except ValueError: pass
try: ct.collapse_weight(m, mask={(1,0):(1,2)}); raise RuntimeError()
except ValueError: pass
try: ct.collapse_weight(m, mask={(0,1):(1,2)}); raise RuntimeError()
except ValueError: pass
try: ct.collapse_weight(m, mask={(0,1,2):(0,1,2)}); raise RuntimeError()
except ValueError: pass


# positions
x = ct.collapse_position(m, mask=None)
assert x == {0:set(((0,1),)), 1:set(((0,1),(1,2),(0,2)),), 2:set(((1,2),))}
x = ct.collapse_position(m, mask={0:set(((0,1),)), 1:set(((1,2),))})
assert x == {1:set(((0,1),(0,2)),), 2:set(((1,2),))}
x = ct.collapse_position(m, mask=((0,1,2),((0,1),(1,2),(0,2))))
assert x == ((1,1,2), ((0,1),(0,2),(1,2)))
x = ct.collapse_position(m, mask=set(((0,(0,1)),(1,(1,2)),(2,(2,0)))))
assert x == set(((1,(0,1)),(1,(0,2)),(2,(1,2))))
try: ct.collapse_position(m, mask=(None,None)); raise RuntimeError()
except TypeError: pass
try: ct.collapse_position(m, mask=(0,None)); raise RuntimeError()
except TypeError: pass
try: ct.collapse_position(m, mask=(None,2)); raise RuntimeError()
except TypeError: pass
try: ct.collapse_position(m, mask=(1,2)); raise RuntimeError()
except TypeError: pass
try: ct.collapse_position(m, mask=((1,2),None)); raise RuntimeError()
except TypeError: pass
try: ct.collapse_position(m, mask=(None,(1,2))); raise RuntimeError()
except TypeError: pass
try: ct.collapse_position(m, mask=((1,2),(1,2))); raise RuntimeError()
except TypeError: pass
try: ct.collapse_position(m, mask=((0,1),(0,1))); raise RuntimeError()
except TypeError: pass
try: ct.collapse_position(m, mask=((0,1,2),(0,1,2))); raise RuntimeError()
except TypeError: pass
try: ct.collapse_position(m, mask=(slice(1,None,1),slice(1,None,1))); raise RuntimeError()
except TypeError: pass
try: ct.collapse_position(m, mask=(slice(None,2,1),slice(None,2,1))); raise RuntimeError()
except TypeError: pass
try: ct.collapse_position(m, mask=(None,set((0,1)))); raise RuntimeError() # error
except TypeError: pass
try: ct.collapse_position(m, mask=(None,set(((0,1))))); raise RuntimeError()
except TypeError: pass
try: ct.collapse_position(m, mask=(None,{0:1})); raise RuntimeError()
except TypeError: pass
try: ct.collapse_position(m, mask=(None,{0:(1,)})); raise RuntimeError()
except TypeError: pass
try: ct.collapse_position(m, mask=(None,[(1,2)])); raise RuntimeError()
except TypeError: pass
try: ct.collapse_position(m, mask=(None,[(0,1),(1,2)])); raise RuntimeError()
except TypeError: pass
##### 4 CASES #####
x = ct.collapse_position(m, mask=((0,1,1), [(0,1),(0,1),(1,2)]))
assert x == ((1,2), ((0,2),(1,2)))
x = ct.collapse_position(m, mask=set(((0,(0,1)), (1,(0,1)), (1,(1,2)))))
assert x == set(((1,(0,2)), (2,(1,2))))
x = ct.collapse_position(m, mask={0:set(((0,1),)), 1:set(((0,1),(1,2)))})
assert x == {1:set(((0,2),)), 2:set(((1,2),))}
try: ct.collapse_position(m, mask={0:[(0,1)], 1:[(0,1),(1,2)]}); raise RuntimeError()
except ValueError: pass
###################
try: ct.collapse_position(m, mask=([(1,2)],[(1,2)])); raise RuntimeError()
except TypeError: pass
try: ct.collapse_position(m, mask=([(0,1),(1,2)],[(0,1),(1,2)])); raise RuntimeError()
except TypeError: pass
try: ct.collapse_position(m, mask=set((None,))); raise RuntimeError()
except ValueError: pass
try: ct.collapse_position(m, mask=set(((0,1)))); raise RuntimeError()
except ValueError: pass
try: ct.collapse_position(m, mask=set(((0,1),(1,2)))); raise RuntimeError()
except ValueError: pass
try: ct.collapse_position(m, mask=set((((0,1),(1,2))))); raise RuntimeError() # error
except ValueError: pass
try: ct.collapse_position(m, mask=set((((0,1),(1,2)),((1,2),(2,2))))); raise RuntimeError()
except ValueError: pass
try: ct.collapse_position(m, mask=set((0,))); raise RuntimeError()
except ValueError: pass
try: ct.collapse_position(m, mask=set(([(0,1)],[(1,2)]))); raise RuntimeError()
except TypeError: pass
try: ct.collapse_position(m, mask=set((((0,1),),((1,2),)))); raise RuntimeError()
except ValueError: pass
try: ct.collapse_position(m, mask={1:2})
except ValueError: pass
try: ct.collapse_position(m, mask={None:(1,)})
except ValueError: pass
try: ct.collapse_position(m, mask={1:(1,)})
except ValueError: pass
try: ct.collapse_position(m, mask={(1,):(1,)})
except ValueError: pass
try: ct.collapse_position(m, mask={0:set((0,1,2))}); raise RuntimeError()
except ValueError: pass
try: ct.collapse_position(m, mask={0:set(((0,1,2)))})
except ValueError: pass
try: ct.collapse_position(m, mask={0:[(0,1,2)]})
except ValueError: pass
try: ct.collapse_position(m, mask=set(((0,)))); raise RuntimeError()  # <wrong size>
except ValueError: pass
try: ct.collapse_position(m, mask=set(((0,1,2)))); raise RuntimeError()
except ValueError: pass
try: ct.collapse_position(m, mask=set((((0,1,1,2))))); raise RuntimeError()
except ValueError: pass
try: ct.collapse_position(m, mask={None:None})
except ValueError: pass
try: ct.collapse_position(m, mask={1:None}) # any
except ValueError: pass
try: ct.collapse_position(m, mask={None:2})
except ValueError: pass
try: ct.collapse_position(m, mask={0:(0,1,2)}) # only
except ValueError: pass
try: ct.collapse_position(m, mask={0:[(0,1),(1,2)]}) # <numpy>
except ValueError: pass
try: ct.collapse_position(m, mask={None:[(0,1),(1,2)]})
except ValueError: pass
try: ct.collapse_position(m, mask={(0,1):[(0,1)]})
except ValueError: pass
try: ct.collapse_position(m, mask={(1,2):[(1,2)]})
except ValueError: pass
try: ct.collapse_position(m, mask={(0,1):[(0,1),(1,2)]})
except ValueError: pass
try: ct.collapse_position(m, mask={(1,2):(1,2)})
except ValueError: pass
try: ct.collapse_position(m, mask={(1,0):(1,2)})
except ValueError: pass
try: ct.collapse_position(m, mask={(0,1):(1,2)})
except ValueError: pass
try: ct.collapse_position(m, mask={(0,1):(0,1)})
except ValueError: pass
try: ct.collapse_position(m, mask={(0,1,2):(0,1,2)})
except ValueError: pass



# EOF
