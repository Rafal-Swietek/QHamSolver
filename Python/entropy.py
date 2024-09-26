
from scipy.special import digamma
def page(L_a, L_b):
    """
    Calculate Page's prediction for linear chain split into L = L_a + L_b sites
    ----------------
    L_a:    (int)
        subsystem size
    L_b:    (int)
        environment size (sites to trace out)
    """
    # return L_a * np.log(2)
    d_a = 2**L_a
    d_b = 2**L_b
    # print(d_a, L_a)
    # print(d_b, L_b)
    return digamma(d_a * d_b + 1) - digamma(max(d_a, d_b) + 1) - (min(d_a, d_b) - 1) / (2 * max(d_a, d_b))

def page_U1(L_a, L_b):
    """
    Calculate Page's prediction for linear chain split into L = L_a + L_b sites with U(1) symmetry
    ----------------
    L_a:    (int)
        subsystem size
    L_b:    (int)
        environment size (sites to trace out)
    """
    # return L_a * np.log(2)
    d_a = 2**L_a
    d_b = 2**L_b
    # print(d_a, L_a)
    # print(d_b, L_b)
    return digamma(d_a * d_b + 1) - digamma(max(d_a, d_b) + 1) - (min(d_a, d_b) - 1) / (2 * max(d_a, d_b))