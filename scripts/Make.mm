# -*- Makefile -*-

PROJECT = mystic
PACKAGE = scripts

PROJ_TIDY += *.log *.out *.txt
PROJ_CLEAN =

#--------------------------------------------------------------------------
#

all: export

#--------------------------------------------------------------------------
#

EXPORT_BINS = \
    mystic_log_reader.py \
    support_convergence.py \
    support_hypercube.py \
    support_hypercube_measures.py \

export:: export-binaries release-binaries

# End of file
