#!/usr/bin/env python
"""
custom 'keymaps' for generating dictionary keys from function input signatures
"""

__all__ = ['SENTINEL','NOSENTINEL','keymap','hashmap','stringmap','picklemap']

class _Sentinel(object):
    """build a sentinel object for the SENTINEL singleton"""
    def __repr__(self):
        return "<SENTINEL>"
class _NoSentinel(object):
    """build a sentinel object for the NOSENTINEL singleton"""
    def __repr__(self):
        return "<NOSENTINEL>"

SENTINEL = _Sentinel()
NOSENTINEL = _NoSentinel()
# SENTINEL = object()
# NOSENTINEL = (SENTINEL,)  #XXX: use to indicate "don't use a sentinel" ?


class keymap(object):
    """tool for converting a function's input signature to an unique key

    This keymap does not serialize objects, but does do some formatting.
    Since the keys are stored as raw objects, there is no information loss,
    and thus it is easy to recover the original input signature.  However,
    to use an object as a key, the object must be hashable.
    """
    def __init__(self, typed=False, flat=True, sentinel=NOSENTINEL, **kwds):
        '''initialize the key builder

        typed: if True, include type information in the key
        flat: if True, flatten the key to a sequence; if False, use (args, kwds)
        sentinel: marker for separating args and kwds in flattened keys

        This keymap stores function args and kwds as (args, kwds) if flat=False,
        or a flattened (*args, zip(**kwds)) if flat=True.  If typed, then
        include a tuple of type information (args, kwds, argstypes, kwdstypes)
        in the generated key.  If a sentinel is given, the sentinel will be
        added to a flattened key to indicate the boundary between args, keys,
        argstypes, and kwdstypes. 
        '''
        self.typed = typed
        self.flat = flat
        self.sentinel = sentinel

        # some rare kwds that allow keymap customization
        self._fasttypes = kwds.get('fasttypes', set((int,str,frozenset,type(None))))
        self._sorted = kwds.get('sorted', sorted)
        self._tuple = kwds.get('tuple', tuple)
        self._type = kwds.get('type', type)
        self._len = kwds.get('len', len)
        return

    def __get_sentinel(self):
        if self._mark:
            return self._mark[0]
        return NOSENTINEL #XXX: or None?
    def __sentinel(self, mark):
        if mark != NOSENTINEL:
            self._mark = (mark,)
        else: self._mark = None

    def __call__(self, *args, **kwds):
        'generate a key from optionally typed positional and keyword arguments'
        if self.flat:
            return self.encode(*args, **kwds)
        return self.encrypt(*args, **kwds)

    def encrypt(self, *args, **kwds):
        """use a non-flat scheme for generating a key"""
        key = (args, kwds) #XXX: pickles larger, but is simpler to unpack
        if self.typed:
            sorted_items = self._sorted(kwds.items())
            key += (self._tuple(self._type(v) for v in args), \
                    self._tuple(self._type(v) for (k,v) in sorted_items))
        return key

    def encode(self, *args, **kwds):
        """use a flattened scheme for generating a key"""
        key = args
        if kwds:
            sorted_items = self._sorted(kwds.items())
            if self._mark: key += self._mark
            for item in sorted_items:
                key += item
        if self.typed: #XXX: 'mark' between each part, so easy to split
            if self._mark: key += self._mark
            key += self._tuple(self._type(v) for v in args)
            if kwds:
                if self._mark: key += self._mark
                key += self._tuple(self._type(v) for (k,v) in sorted_items)
        elif self._len(key) == 1 and self._type(key[0]) in self._fasttypes:
            return key[0]
        return key

    def decrypt(self, key):
        """recover the stored value directly from a generated (non-flat) key"""
        raise NotImplementedError, "Key decryption is not implemented"

    def decode(self, key):
        """recover the stored value directly from a generated (flattened) key"""
        raise NotImplementedError, "Key decoding is not implemented"

    def dumps(self, obj):
        """a more pickle-like interface for encoding a key"""
        return self.encode(obj)

    def loads(self, key):
        """a more pickle-like interface for decoding a key"""
        return self.decode(key)

    # interface
    sentinel = property(__get_sentinel, __sentinel)
    pass


class hashmap(keymap):
    """tool for converting a function's input signature to an unique key

    This keymap generates a hash for the given object.  Not all objects are
    hashable, and generating a hash incurrs some information loss.  Hashing
    is fast, however there is not a method to recover the input signature
    from a hash.
    """
    def encode(self, *args, **kwds):
        """use a flattened scheme for generating a key"""
        return hash(keymap.encode(self, *args, **kwds))
    def encrypt(self, *args, **kwds):
        """use a non-flat scheme for generating a key"""
        return hash(keymap.encrypt(self, *args, **kwds))

class stringmap(keymap):
    """tool for converting a function's input signature to an unique key

    This keymap serializes objects by converting __repr__ to a string.
    Converting to a string is slower than hashing, however will produce a
    key in all cases.  Some forms of archival storage, like a database,
    may require string keys.  There is not a method to recover the input
    signature from a string key that works in all cases, however this is
    possibe for any object where __repr__ effectively mimics __init__.
    """
   #def __init__(self, *args, **kwds):
   #    keymap.__init__(self, *args, **kwds)
   #    self.typed = False  #XXX: is always typed, so set typed=False
    def encode(self, *args, **kwds):
        """use a flattened scheme for generating a key"""
        return str(keymap.encode(self, *args, **kwds))
    def encrypt(self, *args, **kwds):
        """use a non-flat scheme for generating a key"""
        return str(keymap.encrypt(self, *args, **kwds))

import dill as pickle
class picklemap(keymap):
    """tool for converting a function's input signature to an unique key

    This keymap serializes objects by pickling the object.  Serializing an
    object with pickle is relatively slower, however will reliably produce a
    unique key for all picklable objects.  Also, pickling is a reversible
    operation, where the original input signature can be recovered from the
    generated key.
    """
   #def __init__(self, *args, **kwds):
   #    keymap.__init__(self, *args, **kwds)
   #    self.typed = False  #XXX: is always typed, so set typed=False
    def encode(self, *args, **kwds):
        """use a flattened scheme for generating a key"""
        return pickle.dumps(keymap.encode(self, *args, **kwds))
    def encrypt(self, *args, **kwds):
        """use a non-flat scheme for generating a key"""
        return pickle.dumps(keymap.encrypt(self, *args, **kwds))


# EOF
