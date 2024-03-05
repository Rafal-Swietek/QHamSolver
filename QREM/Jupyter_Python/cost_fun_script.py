#!/usr/bin/env python

"""



"""
import numpy as np
import pandas as pd
import copy
import sys
import os
import h5py
from tools import *

my_path = "../../Python/"
if my_path not in sys.path:
    sys.path.append(my_path)
 
# for place in sys.path: 
#     print(place)

from costfun import run as cost_run
# import costfun.run as cost_run
import importlib as imp
def reload_modules():
    imp.reload(cost_run)

if __name__ == '__main__':

    L_total=15
    J=1.0
    alfa=1.0
    h=0.0
    w=2.0
    zeta=0.0
    N=3
    gamma=1.0
    ini_ave=0
    
    nu = 0.5
    
    seed = int(sys.argv[1])

    sizes = np.arange(8, 15)

    perturbation = []
    gap_ratio = []
    for L in sizes:
        # L = L_total - N
        name = f"./collected data/results/nu={nu}/" + f'_L={L}.hdf5'
        if os.path.exists(name):
            with h5py.File(name, "r") as file:
                perturbation.append( np.array(file.get('perturbation')))
                gap_ratio.append(np.array(file.get('gap_ratio')))
        else:
            print(name)
    gap_ratio   = np.array(gap_ratio)

    thouless_cond = []
    for ii_L, L in enumerate(sizes):
        # L = L_total - N
        name = f"./thouless time/" + f'_L={L}.hdf5'
        if os.path.exists(name):
            with h5py.File(name, "r") as file:
                tH      = np.array(file.get('heisenberg time'))
                t_Th    = np.array(file.get('thouless time'))
                thouless_cond.append( np.log(tH / t_Th) )
        else:
            print(name)
    thouless_cond   = np.array(thouless_cond)

    print(sizes.shape)
    print(gap_ratio.shape)
    print(thouless_cond.shape)
    for crits in ['free']:
        cost_run.find_collapse(x = perturbation, y = gap_ratio,     vals = sizes, name = "GapRatio", crit_fun = crits, seed = seed )
        cost_run.find_collapse(x = perturbation, y = thouless_cond, vals = sizes, name = "Thouless", crit_fun = crits, seed = seed )