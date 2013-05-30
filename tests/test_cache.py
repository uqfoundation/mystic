"""
The decorator should produce the behavior as displayed in the following:

>>> s = Spam()
>>> s.eggs()
called
42
>>> s.eggs()
42
>>> s.eggs(1)
called
64
>>> s.eggs(1)
64
>>> s.eggs(1, bar='spam')
called
78
>>> s2 = Spam()
>>> s2.eggs(1, bar='spam')
78
"""

from mystic.cache.safe import inf_cache as memoized
#from mystic.cache import inf_cache as memoized
from mystic.cache.keymaps import picklemap
dumps = picklemap(flat=False)

class Spam(object):
    """A simple class with a memoized method"""

    @memoized(keymap=dumps)
    def eggs(self, *args, **kwds):
        print 'new:', args, kwds
        from random import random
        return int(100 * random())

s = Spam()
print s.eggs()
print s.eggs()
print s.eggs(1)
print s.eggs(1)
print s.eggs(1, bar='spam')
s2 = Spam() 
print s2.eggs(1, bar='spam')

print '=' * 30


# here caching saves time in a recursive function...
@memoized(keymap=dumps)
def fibonacci(n):
    "Return the nth fibonacci number."
    if n in (0, 1):
        return n
    print 'calculating %s' % n
    return fibonacci(n-1) + fibonacci(n-2)

print fibonacci(7)
print fibonacci(9)

print '=' * 30

from numpy import sum, asarray
@memoized(keymap=dumps, tol=3)
def add(*args):
    print 'new:', args
    return sum(args)

print add(1,2,3.0001)
# 6.0000999999999998
print add(1,2,3.00012)
# 6.0000999999999998
print add(1,2,3.0234)
# 6.0234000000000005
print add(1,2,3.023)
# 6.0234000000000005

print '=' * 30

def cost(x,y):
    print 'new: %s or %s' % (str(x), str(y))
    x = asarray(x)
    y = asarray(y)
    return sum(x**2 - y**2)

cost1 = memoized(keymap=dumps, tol=1)(cost)
cost0 = memoized(keymap=dumps, tol=0)(cost)
costD = memoized(keymap=dumps, tol=0, deep=True)(cost)

print "rounding to one decimals..."
print cost1([1,2,3.1234], 3.9876)
print cost1([1,2,3.1234], 3.9876)
print cost1([1,2,3.1234], 3.6789)
print cost1([1,2,3.4321], 3.6789)

print "\nrerun the above with rounding to zero decimals..."
print cost0([1,2,3.1234], 3.9876)
print cost0([1,2,3.1234], 3.9876)
print cost0([1,2,3.1234], 3.6789)
print cost0([1,2,3.4321], 3.6789)

print "\nrerun again with deep rounding to zero decimals..."
print costD([1,2,3.1234], 3.9876)
print costD([1,2,3.1234], 3.9876)
print costD([1,2,3.1234], 3.6789)
print costD([1,2,3.4321], 3.6789)
print ""


from mystic.cache.archives import archive_dict, db_archive 
import dill
@memoized(cache=archive_dict(archive=db_archive()))
def add(x,y):
    return x+y
add(1,2)
add(1,2)
add(1,3)
print "db_cache = %s" % add.__cache__()

@memoized(cache=dict())
def add(x,y):
    return x+y
add(1,2)
add(1,2)
add(1,3)
print "dict_cache = %s" % add.__cache__()

@memoized(cache=add.__cache__())
def add(x,y):
    return x+y
add(1,2)
add(2,2)
print "re_dict_cache = %s" % add.__cache__()

@memoized(keymap=dumps)
def add(x,y):
    return x+y
add(1,2)
add(1,2)
add(1,3)
print "pickle_dict_cache = %s" % add.__cache__()


# EOF
