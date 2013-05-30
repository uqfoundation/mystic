#from mystic.cache.safe import inf_cache as memoized
from mystic.cache import inf_cache as memoized
from mystic.cache.archives import file_archive
from timer import timed

# here caching saves time in a recursive function...
@memoized()
@timed()
def fibonacci(n):
    "Return the nth fibonacci number."
    if n in (0, 1):
        return n
    print 'calculating %s' % n
    return fibonacci(n-1) + fibonacci(n-2)

fibonacci.archive(file_archive('fibonacci.pkl'))
fibonacci.load()

print fibonacci(7)
print fibonacci(9)

fibonacci.dump()


# EOF
