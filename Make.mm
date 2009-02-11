# -*- Makefile -*-

PROJECT = mystic

BUILD_DIRS = \
    mystic \
    models \

OTHER_DIRS =
    examples \
    examples_other \
    tests \

RECURSE_DIRS = $(BUILD_DIRS) $(OTHER_DIRS)

#--------------------------------------------------------------------------
#

all: 
	$(MM) recurse

clean::
	BLD_ACTION="clean" $(MM) recurse

distclean::
	BLD_ACTION="distclean" $(MM) recurse

tidy::
	BLD_ACTION="tidy" $(MM) recurse

#--------------------------------------------------------------------------
#


# End of file
