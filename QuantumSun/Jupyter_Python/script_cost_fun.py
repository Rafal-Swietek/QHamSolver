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
    
    nu = 500
    
    seed = int(sys.argv[1])
    if len(sys.argv) > 1:   N = int(sys.argv[2])
    else:                   N = 3

    sizes = np.arange(11, 17) - N
    r_min=0.42

    disorder = []
    gap_ratio = []
    S_p1 = []
    suffix = f',N={N},gamma={gamma},alfa={alfa}'
    for L in sizes:
        # L = L_total - N
        name = f"./collected data/results/nu={nu}/" + f'_L={L}{suffix}.hdf5'
        if os.path.exists(name):
            with h5py.File(name, "r") as file:
                r = np.array(file.get('gap_ratio'))

                S_p1.append( np.array(file.get('single_site_entropy'))[-2][r > r_min] )
                disorder.append( np.array(file.get('disorder'))[r > r_min] )
                gap_ratio.append( r[r > r_min] )
        else:
            print(name)
    gap_ratio   = np.array(gap_ratio)
    S_p1        = np.array(S_p1)
    print(sizes.shape)
    print(gap_ratio.shape)
    print(S_p1.shape)
    for crits in ['free', 'lin', 'power_law']:
        cost_run.find_collapse(x = disorder, y = gap_ratio, vals = sizes, name = "GapRatio" + suffix,    crit_fun = crits, seed = seed )
        cost_run.find_collapse(x = disorder, y = S_p1,      vals = sizes, name = "Entropy_p=1" + suffix, crit_fun = crits, seed = seed )