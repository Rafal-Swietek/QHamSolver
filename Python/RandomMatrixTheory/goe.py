import tenpy.linalg.random_matrix as rmt
import numpy as np
from scipy.special import iv as BesselI



def spectral_form_factor(x : np.array):
    """
    GOE shape of sff in thermodynamic limit
    
    Parameters:
    -----------------
        x : np.array
            numpy array with datapoints (times defined for unfolded data)
    """
    return np.array([2 * a - a * np.log(1 + 2 * a) if a < 1 else 2 - a * np.log( (2 * a + 1) / (2 * a - 1) ) for a in x])



def eigs(N, normalize = True):

    """
    Calculates the energy levels of normalized / not-normalized GOE matrix

    Parameters:
    --------------
    N: int
        artificial system size, sets matrix siz to 2^N

    normalize: boolean
        normalize the matrix to get unit variance?
    """

    dim = 2**N
    A = np.array(rmt.GOE( (dim, dim) )) * np.sqrt(2) # rmt.GOE returns (A + A.T)/2, that why *sqrt(2)
    
    if normalize: 
        A /= np.sqrt(dim + 1)
    E = np.linalg.eigvalsh(A)
    return E


def partition_function(beta, N):
    """
    Analytical prediction for partition function for large N
    
    Parameters:
    --------------
    beta (float):   inverse temperature
        
    N (int):        artificial system size, sets matrix siz to 2^N
    """
    return (2**N - 1/2.) * BesselI(1, 2 * beta) / beta - 1/2. * BesselI(2, 2 * beta) + 1/2. * np.cosh(2 * beta)


def energy_mean(beta, N):
    """
    Analytical prediction for the mean energy (first moment) for large N
    
    Parameters:
    --------------
    beta (float):   inverse temperature
        
    N (int):        artificial system size, sets matrix siz to 2^N
    """
    nominator = 2**(N+2) * BesselI(0, 2*beta) - 2*(2**(N+1) + beta**2) * BesselI(1, 2*beta) / beta + 2 * beta * np.sinh(2*beta)
    return -nominator / ( (2**(N+1) - 1) * BesselI(1, 2*beta) + beta * (np.cosh(2*beta) - BesselI(2, 2*beta)))


def energy_variance(beta, N):
    """
    Analytical prediction for partition function for large N
    
    Parameters:
    --------------
    beta (float):   inverse temperature
        
    N (int):        artificial system size, sets matrix siz to 2^N
    """
    nominator = 2**(N) / (2**N+1) * ((-2 * beta * (3 * 2**N + beta**2) * BesselI(0, 2 * beta) + (beta**2 + 2**(1 + N) * (3 + 2 * beta**2)) * BesselI(1, 2 * beta) ) / beta**3 + 2 * np.cosh(2 * beta))
    return nominator / ((2**N - 1/2.) * BesselI(1, 2 * beta) / beta - 1/2. * BesselI(2, 2 * beta) + 1./2 * np.cosh(2*beta))


def heat_capacity(beta, N):
    """
    Analytical prediction for partition function for large N
    
    Parameters:
    --------------
    beta (float):   inverse temperature
        
    N (int):        artificial system size, sets matrix siz to 2^N
    """
    return (energy_variance(beta, N) - energy_mean(beta, N)**2) * beta**2 / N


def thermal_entropy(beta, N, E0 = 0):
    """
    Analytical prediction for partition function for large N
    
    Parameters:
    --------------
    beta (float):   inverse temperature
        
    N (int):        artificial system size, sets matrix siz to 2^N

    E0 (float):     ground state energy
    """
    return (np.log(partition_function(beta, N)) + (energy_mean(beta, N) - E0) * beta) / N