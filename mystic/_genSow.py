#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                              Mike McKerns, Caltech
#                        (C) 1998-2009  All Rights Reserved
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

class genSow(object):
    def __init__(self,**kwds):
        self.__args = [] #NOTE: self.__args must be subset of kwds.keys()
        self.__dict = kwds
        self.__initlist = []
        self.__calllist = []
        self.__argslist = []
        self.__methlist = []
        self.__proplist = []
        for key,value in self.__dict.items():
            self.__initlist.append('self._%s = []' % key)
            self.__calllist.append('self._%s.append(kwds["%s"])' % (key,key))
            self.__argslist.append('self._%s.append(%s)' % (key,key))
            self.__methlist.append('def get_%s(self): return self._%s' % (key,key))
            self.__proplist.append('%s = property(get_%s, doc="%s")' % (key,key,value))
        return

    def __call__(self,*args):
        self.__args = [ i for i in args if self.__dict.has_key(i) ]
        exec(self._genClass()) # generates Sow()
        return Sow()
       #return self._genClass()

    def _genClass(self):
        code = self.__genHead() + \
               self.__genInit() + \
               self.__genCall() + \
               self.__genMeth() + \
               self.__genProp()
        return code

    def __genHead(self):
        code0 = """
class Sow(object):
"""
        return code0

    def __genInit(self):
        code1 = """
    def __init__(self):
"""
        for line in self.__initlist:
            code1 += "        %s\n" % line
        return code1

    def __genCall(self):
        code2 = """
    def __call__(self,"""
        for arg in self.__args:
            code2 += " %s," % arg
        code2 += """ **kwds):
"""
        for line in self.__calllist:
            code2 += "        try: %s\n" % line
            code2 += "        except: pass\n"
        for line in self.__argslist:
            code2 += "        try: %s\n" % line
            code2 += "        except: pass\n"
        return code2

    def __genMeth(self):
        code3 = """
"""
        for line in self.__methlist:
            code3 += "    %s\n" % line
        return code3

    def __genProp(self):
        code4 = """
"""
        for line in self.__proplist:
            code4 += "    %s\n" % line
        return code4


if __name__=='__main__':
    import doctest
    doctest.testmod(verbose=True)

# End of file
