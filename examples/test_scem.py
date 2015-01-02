#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2015 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE
"""
Tests functionality of misc. functions in scem.py
"""

from mystic.scemtools import *
import numpy

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
