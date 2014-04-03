#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2014 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE
"""
Tests functionality of misc. functions in scem.py
"""

# from mystic.scemtools import *
import numpy

def sort_complex(c, a):
    # this is dumb, because c (i.e., a, are almost sorted)
    # should sue the one below instead.
    D = zip(a,c)
    def mycmp(x,y):
        if x[0] < y[0]:
            return 1
        elif x[0] > y[0]:
            return -1
        else:
            return 0
    D.sort(cmp = mycmp)
    return numpy.array([x[1] for x in D]), [x[0] for x in D]

def myinsert(a, x):
    # a is in descending order
    from bisect import bisect
    return bisect([-i for i in a],-x)
    
def sort_complex2(c, a):
    """
- c and a are partially sorted (either the last one is bad, or the first one)
- pos : 0 (first one out of order)
       -1 (last one out of order)
    """
    from bisect import bisect
    # find where new key is relative to the rest
    if a[0] < a[1]:
        ax, cx = a[0], c[0]
        ind = myinsert(a[1:],ax)
        a[0:ind], a[ind] = a[1:1+ind], ax
        c[0:ind], c[ind] = c[1:1+ind], cx
    elif a[-1] > a[-2]:
        ax,cx = a[-1], c[0]
        ind = myinsert(a[0:],ax)
        a[ind+1:], a[ind] = a[ind:-1], ax
        c[ind+1:], c[ind] = c[ind:-1], cx
    return

print "Numpy Input" 

a = numpy.array([(i,i) for i in range(10)])*1.
c = [numpy.linalg.norm(x,2)  for x in a]

print a, c
a, c = sort_complex(a,c)
print a,c

print "List Input" 

a = numpy.array([(i,i) for i in range(10)])*1.
c = [numpy.linalg.norm(x,2)  for x in a]
a,c = list(a), list(c)
print a, c
a, c = sort_complex(a,c)
print a,c

print "update complex"
print a, c
b = [2.5, 2.5]
d = 5.6
update_complex(a,c,b,d,0)
print a,c

# end of file
