import numpy as np
from scipy.special import digamma
from scipy.special import polygamma
from scipy.special import binom

def kronecker_delta(x, x0):
    return np.heaviside(x - x0, 1) + np.heaviside(x0 - x, 1) - 1

# -------------------------------------------------- CHAOTIC/ERGODIC PREDICTIONS --------------------------------------------------
def page(LA, LB):
    """
    Calculate Page's prediction for linear chain split into L = LA + LB sites
    ----------------
    LA:    (int)
        subsystem size
    LB:    (int)
        environment size (sites to trace out)
    """
    # if LB > LA: x = LA;     LA = LB;    LB = x
    d_a = 2**LA
    d_b = 2**LB
    return digamma(d_a * d_b + 1) - digamma(max(d_a, d_b) + 1) - (min(d_a, d_b) - 1) / (2 * max(d_a, d_b))

def page_asy(LA, LB):
    """
    Calculate Page's prediction for linear chain split into L = LA + LB sites
    ----------------
    LA:    (int)
        subsystem size
    LB:    (int)
        environment size (sites to trace out)
    """
    # if LB > LA: x = LA;     LA = LB;    LB = x
    L = LA + LB
    f = LA / L
    return f * L * np.log(2) - 2**(-np.abs(1-2*f)*L - 1)

def page_var(LA, LB):
    """
    Calculate Page's prediction for linear chain split into L = LA + LB sites
    ----------------
    LA:    (int)
        subsystem size
    LB:    (int)
        environment size (sites to trace out)
    """
    # if LB > LA: x = LA;     LA = LB;    LB = x
    d_a = 2**LA
    d_b = 2**LB
    return (d_a + d_b) / (d_a*d_b+1) * polygamma(1, d_b+1) - polygamma(1, d_a*d_b+1) - (d_a-1)*(d_a+2*d_b-1) / (4 * d_b**2 * (d_a*d_b+1))

def page_var_asy(LA, LB):
    """
    Calculate Page's prediction for linear chain split into L = LA + LB sites
    ----------------
    LA:    (int)
        subsystem size
    LB:    (int)
        environment size (sites to trace out)
    """
    # if LB > LA: x = LA;     LA = LB;    LB = x
    L = LA + LB
    f = LA / L
    return ( 1/2 - 1/4*kronecker_delta(f, 1/2) ) * 2**(-L - np.abs(1-2*f)*L)

# ----------------------------------------------------------------
def page_U1(LA, LB, N):
    """
    Calculate Page's prediction for linear chain split into L = LA + LB sites with U(1) symmetry
    ----------------
    LA:    (int)
        subsystem size
    LB:    (int)
        environment size (sites to trace out)
    """
    # if LB > LA: x = LA;     LA = LB;    LB = x
    value = 0
    d_N = binom(LA+LB, N)
    for NA in np.arange(0, min([N, LA]), 1):
        d_a = binom(LA, NA)
        d_b = binom(LB, N - NA)
        page_NA = digamma(d_a * d_b + 1) - digamma(max(d_a, d_b) + 1) - (min(d_a, d_b) - 1) / (2 * max(d_a, d_b))
        value += d_a*d_b / d_N * (page_NA + digamma(d_N+1) - digamma(d_a*d_b+1))
    return value

def page_U1_asy(LA, LB, N):
    """
    Calculate Page's prediction for linear chain split into L = LA + LB sites with U(1) symmetry
    ----------------
    LA:    (int)
        subsystem size
    LB:    (int)
        environment size (sites to trace out)
    """
    # if LB > LA: x = LA;     LA = LB;    LB = x
    L = LA + LB
    f = LA / L
    n = N / L
    
    leading_term = ( (n-1)*np.log(1-n) - n*np.log(n) ) * f * L
    subleading = np.sqrt( n*(1-n) / (2*np.pi) ) * np.abs( np.log(1/n - 1) ) * kronecker_delta(f, 1/2) * np.sqrt(L)
    mean_field = (f + np.log(1-f)) / 2
    return leading_term - subleading + mean_field - 1/2 * kronecker_delta(f,1/2) * kronecker_delta(n,1/2)


# -------------------------------------------------- GAUSSIAN PREDICTIONS --------------------------------------------------
def gaussian_page(LA, LB):
    """
    Calculate Page's prediction for Gaussian states split into L = LA + LB sites
    ----------------
    LA:    (int)
        subsystem size
    LB:    (int)
        environment size (sites to trace out)
    """
    # if LB > LA: x = LA;     LA = LB;    LB = x
    L = LA + LB
    return (L-1/2) * digamma(2*L) + (1/2-LB) * digamma(2*LB) + (1/4-LA) * digamma(L) - 1/4 * digamma(LB) - LA

def gaussian_page_asy(LA, LB):
    """
    Calculate Page's prediction for Gaussian states split into L = LA + LB sites
    ----------------
    LA:    (int)
        subsystem size
    LB:    (int)
        environment size (sites to trace out)
    """
    if LB > LA: x = LA;     LA = LB;    LB = x
    L = LA + LB
    f = LA / L
    return L * ( (np.log(2)-1)*f + (f-1)*np.log(1-f) ) + f/2 + np.log(1-f)/4

def gaussian_page_var(LA, LB):
    """
    Calculate Page's prediction for variance over Gaussian states split into L = LA + LB sites
    ----------------
    LA:    (int)
        subsystem size
    LB:    (int)
        environment size (sites to trace out)
    N:    (int)
        number of particles in entire system
    """
    # if LB > LA: x = LA;     LA = LB;    LB = x
    L = LA + LB
    f = LA / L
    return (f + f**2 + np.log(1-f)) / 2.

# ----------------------------------------------------------------
def gaussian_page_U1(LA, LB, N):
    """
    Calculate Page's prediction for Gaussian states split into L = LA + LB sites with U(1) conservation (particle number)
    ----------------
    LA:    (int)
        subsystem size
    LB:    (int)
        environment size (sites to trace out)
    N:    (int)
        number of particles in entire system
    """
    # if LB > LA: x = LA;     LA = LB;    LB = x
    L = LA + LB
    return 1 - LA / L * (1 + L) - N * LA / L * digamma(N) + L * digamma(L) + LA*(N-L) / L * digamma(L-N) - LB * digamma(LB + 1)

def gaussian_page_U1_asy(LA, LB, N):
    """
    Calculate Page's prediction for Gaussian states split into L = LA + LB sites with U(1) conservation (particle number)
    ----------------
    LA:    (int)
        subsystem size
    LB:    (int)
        environment size (sites to trace out)
    N:    (int)
        number of particles in entire system
    """
    # if LB > LA: x = LA;     LA = LB;    LB = x
    L = LA + LB
    f = LA / L
    n = N / L
    term1 = ( (f-1)*np.log(1-f) + f*( (n-1)*np.log(1-n) - n*np.log(n) - 1) ) * L
    term2 = f * (1-f+n*(1-n)) / ( 12*(1-f)*(1-n)*n ) * 1/L
    return term1 + term2

def gaussian_page_U1_var(LA, LB, N):
    """
    Calculate Page's prediction for variance over Gaussian states split into L = LA + LB sites with U(1) conservation (particle number)
    ----------------
    LA:    (int)
        subsystem size
    LB:    (int)
        environment size (sites to trace out)
    N:    (int)
        number of particles in entire system
    """
    # if LB > LA: x = LA;     LA = LB;    LB = x
    L = LA + LB
    f = LA / L
    n = N / L
    term1 = np.log(1-f) + f + f**2 + f**2 * (2*n-1)*np.log(1/n-1)
    term2 = f*(f-1)*(n-1)*n*np.log(1/n-1)**2
    return term1 + term2


# -------------------------------------------------- FREE FERMION PREDICTIONS --------------------------------------------------
def fermion_page(LA, LB):
    """
    Calculate Page's prediction for Gaussian states split into L = LA + LB sites
    ----------------
    LA:    (int)
        subsystem size
    LB:    (int)
        environment size (sites to trace out)
    """
    # if LB > LA: x = LA;     LA = LB;    LB = x
    L = LA + LB
    return (L-1/2) * digamma(2*L) + (1/2-LB) * digamma(2*LB) + (1/4-LA) * digamma(L) - 1/4 * digamma(LB) - LA
