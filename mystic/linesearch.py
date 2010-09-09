#!/usr/bin/env python
""" local copy of scipy.optimize.linesearch """

def line_search(f, myfprime, xk, pk, gfk, old_fval, old_old_fval,
                args=(), c1=1e-4, c2=0.9, amax=50):

    try: #XXX: break dependency on scipy.optimize.linesearch
        from scipy.optimize import linesearch
        alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                         linesearch.line_search(f,myfprime,xk,pk,gfk,\
                         old_fval,old_old_fval,args,c1,c2,amax)
    except ImportError:
        alpha_k = None
        fc = 0
        gc = 0
        gfkp1 = gfk #XXX: or None ?

    return alpha_k, fc, gc, old_fval, old_old_fval, gfkp1

