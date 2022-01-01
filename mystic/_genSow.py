#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""a helper class for the CustomMonitor function"""

class genSow(object):
    """
a configurable Monitor generator
    """
    def __init__(self,**kwds):
        """
Build a Monitor, with channels for the provided properties
Takes **kwds of the form property="doc".
        """
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
        """
Takes string names of properties (given as *args), and sets the
corresponding properties as required inputs for the Monitor.
        """
        self.__args = [ i for i in args if i in self.__dict ]
        exec(self._genClass()) # generates Monitor() #FIXME: fail in python3.x
        return Monitor()
       #return self._genClass()

    def _genClass(self):
        """append the code blocks for the entire Monitor class"""
        code = self.__genHead() + \
               self.__genInit() + \
               self.__genCall() + \
               self.__genMeth() + \
               self.__genProp()
        return code

    def __genHead(self):
        """build the code block for the Monitor class header"""
        code0 = """
class Monitor(object):
"""
        return code0

    def __genInit(self):
        """build the code block for the Monitor class __init__ method"""
        code1 = """
    def __init__(self):
"""
        for line in self.__initlist:
            code1 += "        %s\n" % line
        return code1

    def __genCall(self):
        """build the code block for the Monitor class __call__ method"""
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
        """build the code block for the remaining Monitor class methods"""
        code3 = """
"""
        for line in self.__methlist:
            code3 += "    %s\n" % line
        return code3

    def __genProp(self):
        """build the code block for the Monitor class properties"""
        code4 = """
"""
        for line in self.__proplist:
            code4 += "    %s\n" % line
        return code4


if __name__=='__main__':
    import doctest
    doctest.testmod(verbose=True)

# End of file
