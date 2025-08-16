#!/usr/bin/python -u

#--- MATPLOTLIB
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.markers import MarkerStyle
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd

import sys
my_path = "../../Python/"
if my_path not in sys.path:
    sys.path.append(my_path)
 
for place in sys.path: 
    print(place)

import costfun.costfun as cost
import utils.figures as fig_help

import importlib as imp
def reload_modules():
    imp.reload(cost)
    imp.reload(fig_help)

import itertools
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
latex_engine = 'xelatex'
latex_elements = {'preamble':r'\usepackage{physics}'}

colors_ls = (list(mcolors.TABLEAU_COLORS)[:120])
colors_ls_cyc = itertools.cycle(colors_ls)
markers_ls = ['o','s','v', 'D', '<', 'X', '^', '*', '+']
markers = itertools.cycle(markers_ls)

#--- NUMERICAL LIBS
import numpy as np
import itertools
import math
import random
from cmath import nan
import h5py   


# SCIPY LIBS
import scipy.stats as statistics
from scipy.special import binom
from scipy.special import erfinv
from scipy.special import digamma
from scipy.special import polygamma
from scipy.special import lambertw
from scipy.optimize import curve_fit as fit
from scipy.signal import savgol_filter
from scipy import integrate
from scipy import fft
 
# OTHER
import warnings
warnings.filterwarnings('ignore')
from joblib import Parallel, delayed
import copy
import os
from os import sep as kPSep
from os.path import exists


def order_of_magnitude2(a_value):
    #return 2
    if np.abs(a_value) < 1.0 and a_value != 0:
        m = np.abs(np.log10(np.abs(a_value)))
        return int(max(math.ceil(m) + 1., 2.))
    else: 
        return 2

def order_of_magnitude(a_value):
    x = a_value - int(a_value)
    x = np.round(x, 8)
    num_str = f"{x}"
    num_str = num_str[2:]
    _size = len(num_str)
    if num_str == "0":
        _size = 0;
    
    return _size
   
def info(L, g, ext='.hdf5', use_old=False):
    arr = [g]
    names = ['g']
    info = "_L=%d"%(L)
    for i, var in enumerate(arr):
        n = order_of_magnitude2(var) if use_old else order_of_magnitude(var)
        info += str(",%s={:.%df}"%(names[i], n)).format(round(var, n))
    return info + ext

from joblib import Parallel, delayed

if __name__ == '__main__':
    L=8
    g=1.0

    base_dir = "../results/PBC/"

    num_realis = 100

    sizes = np.arange(8, 17, 1)
    # w_vals = np.linspace(0.5, 35.5, 71)
    # g_vals = np.array([*np.linspace(0.05, 1, 20), *np.linspace(1, 10, 10)])
    g_vals = np.linspace(0.05, 1, 20)

    print(sizes, g_vals)

    num_of_points = 21

    #bins = np.linspace(0, 1, num_of_points)

    temperature = 0

    def loop_body(g):

        # num_of_points = 4 * (L) - 10
        bins = np.logspace(-3.5, 0.75, 2 * num_of_points) if temperature else np.linspace(0, 1, num_of_points)

        dim = 2**(L)

        E = np.zeros((dim))
        S = np.zeros((L + 1, dim))
        S_site = np.zeros((L + 1, dim))
        ratio = np.zeros((dim-2))
        counter = 0

        gaps_dens = np.zeros((bins.size - 1))
        ener_dens = np.zeros((bins.size - 1))
        entropy_dens = np.zeros((L + 1, bins.size - 1))
        entropy_site_dens = np.zeros((L + 1, bins.size - 1))
        g = np.round(g, 6)
        print(info(L=L, g=g, ext='.hdf5'))

        r1 = 0
        r2 = 0
        for real in range(num_realis):
            name = base_dir + 'Entropy/Eigenstate/realisation=%d/'%real + info(L=L, g=g, ext='.hdf5')
            #name = base_dir + 'Entropy/Eigenstate/' + info(L=L, N=N, J=J, gamma=gamma, zeta=zeta, alfa=alfa, h=h, w=w, ext='') + '_jobid=%d.hdf5'%real
            if exists(name):
                with h5py.File(name, "r") as file:
                    energies = np.array(file.get('energies')[0])
                    entropies = np.array(file.get('entropy'))
                    single_site_entropy = np.array(file.get('single_site_entropy'))

                    if entropies.shape != S.shape:
                        print("Shit", name)
                    else:
                        counter += 1
                        E += energies
                        S += entropies
                        S_site += single_site_entropy
                        
                        E_av = np.mean(energies)
                        index_meanE = min(range(len(energies)), key=lambda i: abs(energies[i] - E_av))
                        
                        gaps = np.diff(energies)
                        ratio_tmp = np.minimum(gaps[:-1], gaps[1:]) / np.maximum(gaps[:-1], gaps[1:])

                        r1 += np.mean(ratio_tmp[index_meanE - 250 : index_meanE + 250])
                        r2 += np.mean(ratio_tmp[int(index_meanE - 0.25*dim) : int(index_meanE + 0.25*dim)])

                        E_rescaled = (E - np.min(E)) / (np.max(E) - np.min(E))

                        for k in range(0, bins.size - 1):
                            if temperature:
                                meanE1 = np.sum(energies * np.exp(-bins[k+1] * (energies - energies[0]))) / np.sum(np.exp(-bins[k+1] * (energies - energies[0])))
                                meanE2 = np.sum(energies * np.exp(-bins[k] * (energies - energies[0]))) / np.sum(np.exp(-bins[k] * (energies - energies[0])))

                                idx1 = min(range(len(energies)), key=lambda i: abs(energies[i] - meanE1))
                                idx2 = min(range(len(energies)), key=lambda i: abs(energies[i] - meanE2))
                            else:
                                idx1 = min(range(len(E_rescaled)), key=lambda i: abs(E_rescaled[i] - bins[k]))
                                idx2 = min(range(len(E_rescaled)), key=lambda i: abs(E_rescaled[i] - bins[k + 1]))

                            if np.abs(idx2 - idx1) > 2:
                                gaps_dens[k] += np.mean(ratio_tmp[idx1 : min(idx2, len(ratio_tmp)) ])
                                ener_dens[k] += np.mean(E_rescaled[idx1 : min(idx2, len(E_rescaled)) ])

                                for jj in range(L + 1):
                                    entropy_dens[jj, k]      += np.mean(entropies[jj][idx1 : min(idx2, len(entropies[jj])) ])
                                    entropy_site_dens[jj, k] += np.mean(single_site_entropy[jj][idx1 : min(idx2, len(single_site_entropy[jj])) ])



                        ratio += ratio_tmp
            
        print(L, g, counter)
        if counter > 0:
            name = base_dir + 'Entropy/Eigenstate/' + info(L=L, g=g, ext='_beta.hdf5' if temperature else '.hdf5')

            hf = h5py.File(name, 'w')
            hf.create_dataset('realisations',(1,), data = [counter])
            hf.create_dataset('mean energies',(dim,), data = E / counter)
            hf.create_dataset('entropies',(L + 1,dim), data = S / counter)
            hf.create_dataset('single_site_entropy',(L + 1,dim), data = S_site / counter)
            hf.create_dataset('gap ratio',(dim-2,), data = ratio / counter)
            hf.create_dataset('gap ratio 500',(1,), data = [r1 / counter])
            hf.create_dataset('gap ratio D/2',(1,), data = [r2 / counter])

            hf.create_dataset('gap ratio density',    gaps_dens.shape,          data = gaps_dens / counter)
            hf.create_dataset('energy density',       ener_dens.shape,          data = ener_dens / counter)
            hf.create_dataset('bins',                 bins.shape,               data = bins)
            hf.create_dataset('entropy density',      entropy_dens.shape,       data = entropy_dens / counter)
            hf.create_dataset('entropy site density', entropy_site_dens.shape,  data = entropy_site_dens / counter)

            hf.close()


    for L in sizes:
        dim = 2**(L)

        # Parallel(n_jobs=31)(delayed(loop_body)(gx) for gx in g_vals)
        for gx in g_vals: loop_body(gx)