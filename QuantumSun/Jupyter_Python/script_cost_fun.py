#!/usr/bin/env python

"""



"""
import numpy as np
import pandas as pd
import copy
import sys
import os

my_path = "../../Python/"
if my_path not in sys.path:
    sys.path.append(my_path)
 
for place in sys.path: 
    print(place)

import costfun.costfun as cost
import importlib as imp
def reload_modules():
    imp.reload(cost)

if __name__ == '__main__':

    FOLDED = False
    L=14
    J=1.0
    alfa=1.00
    h=0.0
    w=2.0
    zeta=0.2
    M=3
    gamma=3.0
    ini_ave=1
    
    seed = int(sys.argv[1])

    COLLAPSE_WITH_GAMMA = 1
 
    params = None

    if COLLAPSE_WITH_GAMMA: 
        params = np.linspace(1.0, 4.0, 13) 
        L = int(sys.argv[2])
        
        _suffix = f"_L={L:d},M={M:d},alfa={alfa},zeta={zeta},ini_ave={ini_ave}"
        with open(f"results/disorder_L={L:d},M={M:d},alfa={alfa},zeta={zeta},ini_ave={ini_ave}_gamma=1.0,4.0,0.25.npy", 'rb') as file: w_plot = np.load(file, allow_pickle=True)
        with open(f"results/gap_ratio_L={L:d},M={M:d},alfa={alfa},zeta={zeta},ini_ave={ini_ave}_gamma=1.0,4.0,0.25.npy", 'rb') as file: gaps = np.load(file, allow_pickle=True)
        with open(f"results/thouless_L={L:d},M={M:d},alfa={alfa},zeta={zeta},ini_ave={ini_ave}_gamma=1.0,4.0,0.25.npy", 'rb') as file: taus = np.load(file, allow_pickle=True)
    else:
        params = np.array([10, 11, 12, 13, 14])
        gamma = float(sys.argv[2])
    
        _suffix = f'_gamma={gamma},M={M:d},alfa={alfa},zeta={zeta},ini_ave={ini_ave}'
        with open(f'results/disorder_gamma={gamma},M={M:d},alfa={alfa},zeta={zeta},ini_ave={ini_ave}_L=10_14.npy', 'rb') as file: w_plot = np.load(file, allow_pickle=True)
        with open(f'results/gap_ratio_gamma={gamma},M={M:d},alfa={alfa},zeta={zeta},ini_ave={ini_ave}_L=10_14.npy', 'rb') as file: gaps = np.load(file, allow_pickle=True)
        with open(f'results/thouless_gamma={gamma},M={M:d},alfa={alfa},zeta={zeta},ini_ave={ini_ave}_L=10_14.npy', 'rb') as file: taus = np.load(file, allow_pickle=True)
    
    idx = 0
    xdata = [wx[idx:] for wx in w_plot]

    def calculate_and_save(scaling_ansatz, crit_fun = 'free'):
        
        ydata = [r[idx:] for r in gaps]
        par, crit_pars, costfun, status = cost.get_crit_points(x=np.array(xdata), y=np.array(ydata), vals=params, crit_fun=crit_fun, scaling_ansatz=scaling_ansatz, seed=seed)
        print("GapRatio", scaling_ansatz, crit_fun, par, crit_pars, costfun)

        suffix = _suffix + "_critfun=%s_ansatz=%s_seed=%d"%(crit_fun, scaling_ansatz, seed)
        # if costfun > 3: 
        #     status = False
        #     assert False, "Cost function too high: CF = %d"%costfun
        if status:
            filename = "cost_fun_results/GapRatio" + suffix
            data = {
                "costfun": costfun,
                "crit exp": par
            }
            for i in range(len(crit_pars)):
                data["x_%d"%i] = crit_pars[i]
            np.savez(filename, **data)

        ydata = np.log10(1.0 / taus)
        ydata = [tTh[idx:] for tTh in ydata]
        par, crit_pars, costfun, status = cost.get_crit_points(x=np.array(xdata), y=np.array(ydata), vals=params, crit_fun=crit_fun, scaling_ansatz=scaling_ansatz, seed=seed)
        print("ThoulessTime", scaling_ansatz, crit_fun, par, crit_pars, costfun)
        
        # if costfun > 3: 
        #     status = False
        #     assert False, "Cost function too high: CF = %d"%costfun
        if status:
            filename = "cost_fun_results/ThoulessTime" + suffix
            data = {
                "costfun": costfun,
                "crit exp": par
            }
            for i in range(len(crit_pars)):
                data["x_%d"%i] = crit_pars[i]
            np.savez(filename, **data)

    for crits in ['free', 'power_law', 'lin']:
        for scales in ['KT', 'RG', 'classic']:
            calculate_and_save(scaling_ansatz = scales, crit_fun = crits)