#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
helpers for compressed format for measures
"""

__oct2bin_lookup = {'0': '000', '1': '001', '2': '010', '3': '011',\
                    '4': '100', '5': '101', '6': '110', '7': '111'}
def binary(n):
  """converts an int to binary (returned as a string)
 Hence,  int(binary(x), base=2) == x
"""
  if n < 0:
    return "-" + binary(-n)
  s = [__oct2bin_lookup[oct] for oct in ("%o" % n)]
  s = "".join(s).lstrip("0")
  return s or "0"

def index2binary(index, npts=None):
  """convert a list of integers to a list of binary strings"""
  if npts is None: npts = max(index) + 1
  v = [binary(i) for i in index + [npts]]
  v = [( (len(v[-1]) - len(i))*'0' + i)[::-1] for i in v][:-1]
  return v

def differs_by_one(ith, binaries, all=True, index=True):
  """get the binary string that differs by exactly one index

  Inputs:
    ith   = the target index
    binaries = a list of binary strings
    all   = if False, return only the results for indices < i
    index = if True, return the index of the results (not results themselves)
"""
  from numpy import asarray, where
  v = asarray(binaries)  # a list of binary strings
  vi = v[ith]
  if not all: v = v[:ith]  #XXX: or ith+1 ???
  dif = [[int(j[i]) != int(vi[i]) for i in range(len(j))].count(1) for j in v]
  ind = 1 - asarray(dif) == 0
  if index: return list(where(ind == True)[0])
  return list(v[ind])

def binary2coords(binaries, positions, **kwds):
  """convert a list of binary strings to product measure coordinates"""
  reduce = True  # 'reduce' len=1 and len=0 results
  if 'reduce' in kwds: reduce = kwds['reduce']
  result = [tuple([j[0][j[1]] for j in \
                  zip(positions,[int(i) for i in vk])]) for vk in binaries]
  if len(result) > 1 or not reduce: return result
  if len(result) == 1: return result[0]  # [(1,2,3)] to (1,2,3)
  return                                 # [], 0, ... to None


if __name__ == '__main__':
  pass

#EOF
