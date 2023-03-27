"""
This module contains predicitons of
scaling of the critical point with system size.
"""

import numpy as np
#from config import hamiltonian

def _crit_free(params, idx_par, *args):
    """
    Free critical point for each system size
    For Heisenberg chain the sizes are even: 12,14,16,18,..
    For Ising chain the sizes are: 11,12,13,14,...
    """
    crit = np.array(args)
    return crit[idx_par]

def _crit_free_inv(params, idx_par, *args):
    """
    Free inversed critical point for each system size if critical points appears to be very small
    For Heisenberg chain the sizes are even: 12,14,16,18,..
    For Ising chain the sizes are: 11,12,13,14,...
    """
    crit = 1. / np.array(args)
    return crit[idx_par]

def _crit_const(size, x0):
    """
    Constant critical value
    """

    return x0


def _crit_lin(params, idx_par, x0, x1):
    """
    Linear scaling of critical point with size
    """

    return x0 + params[idx_par] * x1

def _crit_inv(params, idx_par, x0, x1, x2):
    """
    Scaling with inverse of system size
    """
    return x0 + 1.0 / float(x1 + x2 * params[idx_par])

def _crit_power_law(params, idx_par, x0, x1, nu):
    """
    Power law prediciton with system size
    (Only leading order taken)
    """

    return x0 + float(params[idx_par])**nu * x1

def _crit_log(params, idx_par, x0, x1):
    """
    Ansatz with logarithmic scaling with size
    """

    return x0 + x1 * np.log(params[idx_par])


def _crit_inv_log(params, idx_par, x0, x1):
    """
    Inverse of logarithmic scaling
    """
    return x0 + x1 / np.log(params[idx_par])
