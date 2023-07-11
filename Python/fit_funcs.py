import numpy as np

def xover_from_below(input_array, cut):
    """Find index of value crossing threshold from below"""
    x = input_array < cut
    return x.argmax() if x.any() else -1

def xover_from_above(input_array, cut):
    """Find index of value crossing threshold from above"""
    x = input_array > cut
    return x.argmin() if x.any() else -1

def exp_fit(x, mu, a):
    """Exponential function to fit and find decay rate"""
    return a * np.exp(- x / mu)

def exp_fit2(x, mu, a):
    """Exponential function to fit and find decay rate"""
    return a * np.exp(x / mu)

def exp_fit_offset(x, mu, a, b):
    """Exponential function to fit and find decay rate"""
    return a * np.exp(- x / mu) + b

def exp_fit2_offset(x, mu, a, b):
    """Exponential function to fit and find decay rate"""
    return a * np.exp(x / mu) + b


def lin_fit(x, mu, a):
    """Linear function to fit and find decay rate from ln( f(t) )"""
    return -x / mu + a

def lin_fit2(x, a, b):
    """Linear function to fit and find decay rate from ln( f(t) )"""
    return a * x + b

def lin_fit_nooffset(x, a):
    """Linear function to fit with no offset"""
    return a * x

def lin_fit_inv(x, a, b):
    """Linear function to fit and find decay rate from ln( f(t) )"""
    return a / x + b

def lin_fit_inv2(x, a):
    """Linear function to fit and find decay rate from ln( f(t) ) without offset"""
    return a / x 

def power_law(x, alfa, a):
    """ Power-law fit function """
    return a * x**alfa

def power_law_offset(x, alfa, a, b):
    """ Power-law fit function """
    return a * x**alfa + b


def power_law_inv(x, alfa, a):
    """ Power-law fit function """
    return a / x**alfa
    
def stretch_exp(x, alfa, a, b):
    """ Ffit function for exponential decay with arbitrary exponent """
    return a * np.exp(b * x**alfa)


def log_fit(x, a, b):
    """
    Logarithmic fit function without shift
    """

    return a + b * np.log(x)

def log_fit2(x, b):
    """
    Logarithmic fit function without shift
    """

    return b * np.log(x)


def log_fit_inv(size, a, b):
    """
    Inversed logarithmic fit function without shift
    """
    return a + b / np.log(size)