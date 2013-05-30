#!/usr/bin/env python
"""
custom caching dict, which archives results to memory, file, or database
"""

__all__ = ['archive_dict', 'null_archive', 'file_archive', 'db_archive']

class archive_dict(dict):
    """dictionary augmented with an archive backend"""
    def __init__(self, *args, **kwds):
        """initialize a dictionary with an archive backend

    Additional Inputs:
        archive: instance of archive object
        """
        self.__swap__ = null_archive()
        self.__archive__ = kwds.pop('archive', null_archive())
        dict.__init__(self, *args, **kwds)
        return
    def load(self, *args):
        """load archive contents

    If arguments are given, only load the specified keys
        """
        if not args:
            self.update(self.archive.__asdict__())
        for arg in args:
            try:
                self.update({arg:self.archive[arg]})
            except KeyError:
                pass
        return
    def dump(self, *args):
        """dump contents to archive

    If arguments are given, only dump the specified keys
        """
        if not args:
            self.archive.update(self)
        for arg in args:
            if self.has_key(arg):
                self.archive.update({arg:self.__getitem__(arg)})
        return
    def archived(self, *on):
        """check if the dict is archived, or toggle archiving

    If on is True, turn on the archive; if on is False, turn off the archive
        """
        L = len(on)
        if not L:
            return not isinstance(self.archive, null_archive)
        if L > 1:
            raise TypeError, "archived expected at most 1 argument, got %s" % str(L+1)
        if bool(on[0]):
            if not isinstance(self.__swap__, null_archive):
                self.__swap__, self.__archive__ = self.__archive__, self.__swap__
            elif isinstance(self.__archive__, null_archive):
                raise ValueError, "no valid archive has been set"
        else:
            if not isinstance(self.__archive__, null_archive):
                self.__swap__, self.__archive__ = self.__archive__, self.__swap__
    def __get_archive(self):
       #if not isinstance(self.__archive__, null_archive):
       #    return
        return self.__archive__
    def __archive(self, archive):
        if not isinstance(self.__swap__, null_archive):
            self.__swap__, self.__archive__ = self.__archive__, self.__swap__
        self.__archive__ = archive
    # interface
    archive = property(__get_archive, __archive)
    pass


class null_archive(dict):
    """dictionary interface to nothing -- it's always empty"""
    def __init__(self):
        """initialize a permanently-empty dictionary"""
        dict.__init__(self)
        return
    def __asdict__(self):
        """build a dictionary containing the archive contents"""
        return self
    def __setitem__(self, key, value):
        pass
    __setitem__.__doc__ = dict.__setitem__.__doc__
    def update(self, adict, **kwds):
        pass
    update.__doc__ = dict.update.__doc__
    def setdefault(self, key, *value):
        return self.get(key, *value)
    setdefault.__doc__ = dict.setdefault.__doc__
    def __repr__(self):
        return "archive(NULL)"
    __repr__.__doc__ = dict.__repr__.__doc__
    pass


class file_archive(dict):
    """dictionary-style interface to a file"""
    def __init__(self, filename=None, serialized=True): # False
        """initialize a file with a synchronized dictionary interface

    Inputs:
        serialized: if True, pickle file contents; otherwise save python objects
        filename: name of the file backend [default: memo.pkl or memo.py]
        """
        import os
        """filename = full filepath"""
        if filename is None:
            if serialized: filename = 'memo.pkl' #FIXME: need better default
            else: filename = 'memo.py' #FIXME: need better default
        self._filename = filename
        self._serialized = serialized
        if not os.path.exists(filename):
            self.__save__({})
        return
    def __asdict__(self):
        """build a dictionary containing the archive contents"""
        if self._serialized:
            try:
                f = open(self._filename, 'rb')
                import dill as pickle
                memo = pickle.load(f)
                f.close()
            except:
                memo = {}
        else:
            import os
            file = os.path.basename(self._filename)
            root = os.path.realpath(self._filename).rstrip(file)[:-1]
            curdir = os.path.realpath(os.curdir)
            file = file.rstrip('.py') or file.rstrip('.pyc') \
                or file.rstrip('.pyo') or file.rstrip('.pyd')
            os.chdir(root)
            exec 'from %s import memo' % file #FIXME: unsafe
            os.chdir(curdir)
        return memo
    def __save__(self, memo=None):
        """create an archive from the given dictionary"""
        if memo == None: return
        if self._serialized:
            try:
                f = open(self._filename, 'wb')
                import dill as pickle
                pickle.dump(memo, f)
                f.close()
            except:
                pass  #XXX: warning? fail?
        else:
            open(self._filename, 'wb').write('memo = %s' % memo)
        return
    #FIXME: missing a bunch of __...__
    def __getitem__(self, key):
        memo = self.__asdict__()
        return memo[key]
    __getitem__.__doc__ = dict.__getitem__.__doc__
    def __iter__(self):
        return self.__asdict__().iterkeys()
    __iter__.__doc__ = dict.__iter__.__doc__
    def __repr__(self):
        return "archive(%s: %s)" % (self._filename, self.__asdict__())
    __repr__.__doc__ = dict.__repr__.__doc__
    def __setitem__(self, key, value):
        memo = self.__asdict__()
        memo[key] = value
        self.__save__(memo)
        return
    __setitem__.__doc__ = dict.__setitem__.__doc__
    def clear(self):
        self.__save__({})
        return
    clear.__doc__ = dict.clear.__doc__
    #FIXME: copy, fromkeys
    def get(self, key, value=None):
        memo = self.__asdict__()
        return memo.get(key, value)
    get.__doc__ = dict.get.__doc__
    def has_key(self, key):
        return key in self.__asdict__()
    has_key.__doc__ = dict.has_key.__doc__
    def items(self):
        return list(self.iteritems())
    items.__doc__ = dict.items.__doc__
    def iteritems(self):
        return self.__asdict__().iteritems()
    iteritems.__doc__ = dict.iteritems.__doc__
    iterkeys = __iter__
    def itervalues(self):
        return self.__asdict__().itervalues()
    itervalues.__doc__ = dict.itervalues.__doc__
    def keys(self):
        return list(self.__iter__())
    keys.__doc__ = dict.keys.__doc__
    def pop(self, key, *value):
        memo = self.__asdict__()
        res = memo.pop(key, *value)
        self.__save__(memo)
        return res
    pop.__doc__ = dict.pop.__doc__
    #FIXME: popitem
    def setdefault(self, key, *value):
        res = self.__asdict__().get(key, *value)
        self.__setitem__(key, res)
        return res
    setdefault.__doc__ = dict.setdefault.__doc__
    def update(self, adict, **kwds):
        memo = self.__asdict__()
        memo.update(adict, **kwds)
        self.__save__(memo)
        return
    update.__doc__ = dict.update.__doc__
    def values(self):
        return list(self.itervalues())
    values.__doc__ = dict.values.__doc__
    pass


#XXX: should inherit from object not dict ?
class db_archive(dict): #XXX: requires UTF-8 key
    """dictionary-style interface to a database"""
    def __init__(self, database=None, table=None):
        """initialize a database with a synchronized dictionary interface

    Inputs:
        database: url of the database backend [default: :memory:]
        table: name of the associated database table
        """
        if database is None: database = ':memory:'
        self._database = database
        if table is None: table = 'memo'
        self._table = table
        import sqlite3 as db
        self._conn = db.connect(database)
        self._curs = self._conn.cursor()
        sql = "create table if not exists %s(argstr, fval)" % table
        self._curs.execute(sql)
        return
    def __asdict__(self):
        """build a dictionary containing the archive contents"""
        sql = "select * from %s" % self._table
        res = self._curs.execute(sql)
        d = {}
        [d.update({k:v}) for (k,v) in res] # always get the last one
        return d
    #FIXME: missing a bunch of __...__
    def __getitem__(self, key):
        res = self._select_key_items(key)
        if res: return res[-1][-1] # always get the last one
        raise KeyError, key
    __getitem__.__doc__ = dict.__getitem__.__doc__
    def __iter__(self):
        sql = "select argstr from %s" % self._table
        return (k[-1] for k in set(self._curs.execute(sql)))
    __iter__.__doc__ = dict.__iter__.__doc__
    def __repr__(self):
        return "archive(%s: %s)" % (self._table, self.__asdict__())
    __repr__.__doc__ = dict.__repr__.__doc__
    def __setitem__(self, key, value): #XXX: maintains 'history' of values
        sql = "insert into %s values(?,?)" % self._table
        self._curs.execute(sql, (key,value))
        self._conn.commit()
        return
    __setitem__.__doc__ = dict.__setitem__.__doc__
    def clear(self):
        [self.pop(k) for k in self.keys()] # better delete table, add empty ?
        return
    clear.__doc__ = dict.clear.__doc__
    #FIXME: copy, fromkeys
    def get(self, key, value=None):
        res = self._select_key_items(key)
        if res: value = res[-1][-1]
        return value
    get.__doc__ = dict.get.__doc__
    def has_key(self, key):
        return bool(self._select_key_items(key))
    has_key.__doc__ = dict.has_key.__doc__
    def items(self):
       #return self.__asdict__().items()
        return list(self.iteritems())
    items.__doc__ = dict.items.__doc__
    def iteritems(self):
        return ((k,self.__getitem__(k)) for k in self.__iter__())
    iteritems.__doc__ = dict.iteritems.__doc__
    iterkeys = __iter__
    def itervalues(self):
        return (self.__getitem__(k) for k in self.__iter__())
    itervalues.__doc__ = dict.itervalues.__doc__
    def keys(self):
       #return self.__asdict__().keys()
        return list(self.__iter__())
    keys.__doc__ = dict.keys.__doc__
    def pop(self, key, *value):
        L = len(value)
        if L > 1:
            raise TypeError, "pop expected at most 2 arguments, got %s" % str(L+1)
        res = self._select_key_items(key)
        if res:
            _value = res[-1][-1]
        else:
            if not L: raise KeyError, key
            _value = value[0]
        sql = "delete from %s where argstr = ?" % self._table
        self._curs.execute(sql, (key,))
        self._conn.commit()
        return _value 
    pop.__doc__ = dict.pop.__doc__
    #FIXME: popitem
    def setdefault(self, key, *value):
        L = len(value)
        if L > 1:
            raise TypeError, "setvalue expected at most 2 arguments, got %s" % str(L+1)
        res = self._select_key_items(key)
        if res:
            _value = res[-1][-1]
        else:
            if not L: _value = None
            else: _value = value[0]
            self.__setitem__(key, _value)
        return _value
    setdefault.__doc__ = dict.setdefault.__doc__
    def update(self, adict, **kwds):
        _dict = adict.copy()
        _dict.update(**kwds)
        [self.__setitem__(k,v) for (k,v) in _dict.items()]
        return
    update.__doc__ = dict.update.__doc__
    def values(self):
       #return self.__asdict__().values()
        return list(self.itervalues())
    values.__doc__ = dict.values.__doc__
    def _select_key_items(self, key):
        '''Return a tuple of (key, value) pairs that match the specified key'''
        sql = "select * from %s where argstr = ?" % self._table
        return tuple(self._curs.execute(sql, (key,)))
    pass


# EOF
