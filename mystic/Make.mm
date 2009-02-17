# -*- Makefile -*-

PROJECT = mystic
PACKAGE = mystic

BUILD_DIRS = \
	
RECURSE_DIRS = $(BUILD_DIRS)

#--------------------------------------------------------------------------
#

all: export
	BLD_ACTION="all" $(MM) recurse

release: tidy
	cvs release .

update: clean
	cvs update .

#--------------------------------------------------------------------------
#
# export

EXPORT_PYTHON_MODULES = \
    __init__.py \
    const.py \
    differential_evolution.py \
    scipy_optimize_fmin.py \
    nmtools.py \
    detools.py \
    tools.py \
    helputil.py \
    forward_model.py \
    filters.py \
    svmtools.py \
    svctools.py \
    metropolis.py \
    scemtools.py \


export:: export-python-modules

# End of file
