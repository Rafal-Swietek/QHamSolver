"""
Module containing tools for Quadratic models
"""

import numpy as np

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
    
def info_raw(L, J, w, g, model = 'Anderson', use_old = False):
    
    names = ['J', 'w', 'g']
    arr = [J]
    if model == 'Anderson' or model == 'AubryAndre': arr.append(w)
    if model == 'AubryAndre': arr.append(g)
    
    info = "_L=%d"%(L)
    for i, var in enumerate(arr):
        n = order_of_magnitude2(var) if use_old else order_of_magnitude(var)
        info += str(",%s={:.%df}"%(names[i], n)).format(round(var, n))
    return info

def info(L, J=1.0, w=0.0, g=0.0, use_old = False, model = 'Anderson', ext = '.dat'):
    return info_raw(L, J, w, g, model, use_old) + ext
