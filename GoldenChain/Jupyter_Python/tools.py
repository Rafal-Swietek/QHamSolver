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
    
def info_raw(L, J, c, k, p):
    arr = [J, c, k, p] if c == 0 and (k == 0 or float(k) == (L/2.0)) else [J, c, k]
    names = ['J', 'c', 'k', 'p'] if c == 0 and (k == 0 or float(k) == (L/2.0)) else ['J', 'c', 'k']
    info = "_L=%d"%(L)
    for i, var in enumerate(arr):
        n = order_of_magnitude(var)# if not use_old else order_of_magnitude2(var)
        info += str(",%s={:.%df}"%(names[i], n)).format(round(var, n))
    return info

def info(L, J, c, k, p, ext = '.dat'):
    return info_raw(L, J, c, k, p) + ext


def legend_label(name, value):
    n = order_of_magnitude(value)
    return str("$%s={:.%df}$"%(name, n)).format(round(value, n))


def Fibonacci(n):

    # Check if input is 0 then it will
    # print incorrect input
    if n < 0:
        print("Incorrect input")

    # Check if n is 0
    # then it will return 0
    elif n == 0:
        return 0

    # Check if n is 1,2
    # it will return 1
    elif n == 1 or n == 2:
        return 1

    else:
        return Fibonacci(n-1) + Fibonacci(n-2)
