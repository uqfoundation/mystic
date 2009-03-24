#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                       Patrick Hung & Mike McKerns, Caltech
#                        (C) 1998-2008  All Rights Reserved
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

"""
Tools for prettifying help

Some of following code is taken from Ka-Ping Yee's pydoc module
"""

def commandfy(text):
    """Format a command string"""
    from string import join
    return join(map(lambda ch: ch + '\b' + ch, text), '')


def commandstring(text, BoldQ):
    "Bolds all lines in text that returns true by predicate BoldQ."
    s = text.split('\n') 
    o = []
    for line in s:
        try:
            if BoldQ(line):
            #if (line.lstrip()[0] == '#'):
                o.append(line)
            else:
                o.append(commandfy(line))
        except:
            o.append(line)
    return '\n'.join(o)

def paginate(text, BoldQ = lambda linein: linein.lstrip()[0] == '#'):
    "break printed content into pages"
    import pydoc
    pydoc.pager(commandstring(text, BoldQ))
             
test_string = """
# All strings that are comments should
# begin with a pound sign

# Strings that don't will be interpreted as a command:
is this bold or what ?

   # this is also a comment

"""

if __name__=='__main__':
    paginate(test_string)
    paginate('Now.. the opposite --- %s' %test_string, lambda linein: linein.lstrip()[0] != '#')


# End of file
