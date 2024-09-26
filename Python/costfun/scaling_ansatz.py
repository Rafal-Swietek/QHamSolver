"""
Here the differeny types of scalings ansatzes are presented
to find the proper critical point
The names should be buolt as _rescale_*, where
the * should be replaced with a shoirt name characterizing the scaling
ansatz, like KT for Kosterlitz-Thouless
"""

import numpy as np
from scipy.special import binom

def _rescale_spacing(x, params, idx_par, crit_fun, *args):
    """Regular ansatz with power-law on L"""
    par_crit = crit_fun(params, idx_par, *args)
    return np.sign(x - par_crit) * np.abs(x - par_crit)

def _rescale_classic(x, params, idx_par, crit_fun, nu, *args):
    """Regular ansatz with power-law on L"""
    par_crit = crit_fun(params, idx_par, *args)
    return (x - par_crit) * params[idx_par]**(nu)


def _rescale_KT(x, params, idx_par, crit_fun, nu, *args):
    """ Kosterlitz Thouless type """
    par_crit = crit_fun(params, idx_par, *args)
    return  np.sign(x - par_crit) * params[idx_par] / np.exp(nu / np.sqrt(np.abs(x - par_crit )) )


def _rescale_FGR(x, params, idx_par, crit_fun, nu, *args):
    """Fermi Golden Rule"""
    par_crit = crit_fun(params, idx_par, *args)
    dim = 2**params[idx_par]
    return np.sign(x - par_crit) * np.abs(x - par_crit) * dim**(1. / nu)


def _rescale_RG(x, params, idx_par, crit_fun, nu, *args):
    """ RG-type arguments """
    par_crit = crit_fun(params, idx_par, *args)
    return np.sign(x - par_crit) * np.abs(x - par_crit)**nu * params[idx_par]


def _rescale_RGcorr(x, params, idx_par, crit_fun, nu, const, *args):
    """ RG-type arguments """
    par_crit = crit_fun(params, idx_par, *args)
    return np.sign(x - par_crit) / (1 + const * (x - par_crit) ) * np.abs(x - par_crit)**nu * params[idx_par]

def _rescale_exp(x, params, idx_par, crit_fun, nu, *args):
    """Exponential scaling of parameter"""
    par_crit = crit_fun(params, idx_par, *args)
    return np.sign(x - par_crit) * np.exp(nu * np.abs(x - par_crit)) * np.exp(np.log(2) / 2 * np.abs(params[idx_par]))