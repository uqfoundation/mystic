#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Tests functionality of misc. functions in scem.py
"""

from mystic.scemtools import *
import numpy

print("Numpy Input")

a = numpy.array([(i,i) for i in range(10)])*1.
c = [numpy.linalg.norm(x,2)  for x in a]

print("%s %s" % (a,c))
a, c = sort_complex(a,c)
print("%s %s" % (a,c))

print("List Input" )

a = numpy.array([(i,i) for i in range(10)])*1.
c = [numpy.linalg.norm(x,2)  for x in a]
a,c = list(a), list(c)
print("%s %s" % (a,c))
a, c = sort_complex(a,c)
print("%s %s" % (a,c))

print("update complex")
print("%s %s" % (a,c))
b = [2.5, 2.5]
d = 5.6
update_complex(a,c,b,d,0)
print("%s %s" % (a,c))

# end of file
