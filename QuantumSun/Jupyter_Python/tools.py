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
    
def info_raw(L, N, J, gamma, zeta, alfa, h, w, ini_ave = 0, use_old = False):
    arr = [J, gamma, zeta, alfa, h, w] if alfa < 1.0 else [J, gamma, alfa, h, w]
    names = ['J', 'g', 'zeta', 'alfa', 'h', 'w'] if alfa < 1.0 else ['J', 'g', 'alfa', 'h', 'w']
    info = "_L=%d,N=%d"%(L,N)
    for i, var in enumerate(arr):
        n = order_of_magnitude2(var) if use_old else order_of_magnitude(var)
        info += str(",%s={:.%df}"%(names[i], n)).format(round(var, n))
    if ini_ave: info += ",ini_ave"
    return info

def info(L, N, J, gamma, zeta, alfa, h, w, ini_ave = 0, use_old = False, ext = '.dat'):
    return info_raw(L, N, J, gamma, zeta, alfa, h, w, ini_ave, use_old) + ext


def legend_label(name, value):
    n = order_of_magnitude(value)
    return str("$%s={:.%df}$"%(name, n)).format(round(value, n))