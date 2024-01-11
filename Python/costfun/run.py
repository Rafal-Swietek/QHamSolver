#!/usr/bin/env python

"""
Module to find collapse of input data or with input loader lambda

"""
import numpy as np
import os
import costfun.costfun as cost
import h5py

def find_collapse(x, y, vals, name, crit_fun = 'free', seed = None, wH = None):
    """
    Find collapse of input data by rescaling x appropriately
    
    Parameters:
    -----------
        x, y:   2D array
            x and y data for collapse with different vals (sizes) values
        
        vals:   1D array
            scaling parameters (i.e. system sizes)
        
        name:   string
            name of data to collapse (used as filename to save)
        
        crit_fun:   string
            ansatz for critical points (free, linear, power-law...)

        seed:   int
            random seed for collapse function (differential_evolution in scipy)

        wH:     2D array
            array of level mean spacings used for 'spacing' ansatz (by default not used)
    """
    if seed is None:
        seed = np.random.default_rng()

    print("seed = ", seed)

    dir = "CriticalParameters/raw/"
    try:
        os.mkdir(dir)
    except OSError as error:
        print(error)   

    def calculate_and_save(scaling_ansatz, crit_fun = 'free'):
        suffix = "_critfun=%s_ansatz=%s_seed=%d.hdf5"%(crit_fun, scaling_ansatz, seed)

        #-- calculate gap ratio collapse
        par, crit_pars, costfun, status = cost.get_crit_points(x=x, y=y, vals=vals, crit_fun=crit_fun, scaling_ansatz=scaling_ansatz, seed=seed, wH=wH)
        print(name, "\t", par, crit_pars, costfun, status)
        crit_pars = np.array(crit_pars)
        if costfun > 10: 
            status = "Failed"
            assert False, "Cost function too high: CF = %d"%costfun
        if status:
            filename = dir + os.sep + name + suffix
            
            hf = h5py.File(filename, 'w')
            hf.create_dataset('crit_pars',  crit_pars.shape, data = crit_pars)
            hf.create_dataset('vals',       vals.shape,      data = vals)
            hf['costfun']  = costfun
            hf['crit exp'] = par
            hf.close()

    calculate_and_save(scaling_ansatz='classic',    crit_fun=crit_fun)
    calculate_and_save(scaling_ansatz='KT',         crit_fun=crit_fun)
    calculate_and_save(scaling_ansatz='RG',         crit_fun=crit_fun)
    # calculate_and_save(scaling_ansatz='FGR',        crit_fun=crit_fun)
    # calculate_and_save(scaling_ansatz='spacing', crit_fun='free')
