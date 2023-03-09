#import user modules
#--- MATPLOTLIB
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.markers import MarkerStyle
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd

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

base_dir = "../results/"

def order_of_magnitude(a_value):
    #return 2
    if np.abs(a_value) < 1.0 and a_value != 0:
        m = np.abs(np.log10(np.abs(a_value)))
        return int(max(math.ceil(m) + 1., 2.))
    else: 
        return 2

def info_raw(L, M, J, zeta, alfa, h, w, ini_ave = 0):
    arr = [J, zeta, alfa, h, w]
    names = ['J', 'zeta', 'alfa', 'h', 'w']
    info = "_L=%d,M=%d"%(L,M)
    for i, var in enumerate(arr):
        n = order_of_magnitude(var)
        info += str(",%s={:.%df}"%(names[i], n)).format(round(var, n))
    info += ",ini_ave=%d"%ini_ave
    return info

def info(L, M, J, zeta, alfa, h, w, ini_ave = 0):
    return info_raw(L, M, J, zeta, alfa, h, w, ini_ave) + ".dat"


def GOE(x : np.array):
    """
    GOE shape of sff in thermodynamic limit
    
    Parameters:
    -----------------
        x : np.array
            numpy array with datapoints (times defined for unfolded data)
    """
    return np.array([2 * a - a * np.log(1 + 2 * a) if a < 1 else 2 - a * np.log( (2 * a + 1) / (2 * a - 1) ) for a in x])

def remove_fluctuations(data, bucket_size=10):
    new_data = data;
    half_bucket = int(bucket_size / 2)
    for k in range(half_bucket, len(data) - half_bucket):
        average = np.sum(data[k - half_bucket : k + half_bucket])
        new_data[k - half_bucket] = average / bucket_size
    return new_data


J=1.0
alfa=0.75
h=0.5
w=1.0
zeta=0.2
M=3
ini_ave=0

if __name__ == '__main__':
    alfa_vals = np.linspace(0.7, 0.85, 16)

    num_of_points = 35
    gap_ratio = []
    ener = []
    L = 15

    sizes2 = np.arange(10, 16, 1)
    for L in sizes2:
        gap_ratio_L = []
        ener_L = []

        dir_E = base_dir + 'DIAGONALIZATION/'
        for alfa in alfa_vals:
            r_mean = np.zeros((num_of_points))
            ener_L = np.zeros((num_of_points))

            counter = 0
            for realis in range(5000):
                filename = dir_E + info_raw(L, M, J, zeta, alfa, h, w, ini_ave) + "_real=%d"%realis + ".hdf5"

                if exists(filename):
                    counter += 1
                    with h5py.File(filename, "r") as file:
                        E = np.array(file.get('eigenvalues/dataset')[0])
                        dim = E.size

                        gaps = np.diff(E)
                        ratio = np.minimum(gaps[:-1], gaps[1:]) / np.maximum(gaps[:-1], gaps[1:])

                        E_rescaled = (E - np.min(E)) / (np.max(E) - np.min(E))
                        for k in range(0, num_of_points):
                            r_mean[k] += np.mean(ratio[ k * (dim // num_of_points) : (k+1) * (dim // num_of_points) ])
                            ener_L[k] += np.mean(E_rescaled[ k * (dim // num_of_points) : (k+1) * (dim // num_of_points) ])
            
            gap_ratio_L.append(r_mean / counter)
            ener_L /= counter
            
            print(L, alfa, counter)
            
        with open(f'rescaled_energy_L={L}.npy', 'wb') as file:  np.save(file, ener_L, allow_pickle=True)
        with open(f'gap_ratio_L={L}.npy', 'wb') as file:        np.save(file, np.array(gap_ratio_L), allow_pickle=True)

        gap_ratio.append( np.array(gap_ratio_L) )
        ener.append( np.array(ener_L) )

