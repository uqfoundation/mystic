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
    _genSow.py \
    _scipy060optimize.py \
    abstract_map_solver.py \
    abstract_nested_solver.py \
    abstract_solver.py \
    const.py \
    differential_evolution.py \
    filters.py \
    forward_model.py \
    helputil.py \
    mystic_math.py \
    nested.py \
    python_map.py \
    scipy_optimize.py \
    strategy.py \
    termination.py \
    tools.py \
    metropolis.py \
    scemtools.py \
    svmtools.py \
    svctools.py \

export:: export-python-modules

# End of file
