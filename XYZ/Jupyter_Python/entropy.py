
from scipy.special import digamma
from scipy.special import polygamma

def page(LA, LB):
    dA = 2**LA
    dB = 2**LB
    return digamma(dA * dB + 1) - digamma(dB + 1) - (dA - 1) / (2*dB)

def page_var(LA, LB):
    dA = 2**LA
    dB = 2**LB
    return (dA + dB) / (dA * dB + 1) * polygamma(n=1, x=dB + 1) - polygamma(n=1, x=dA * dB + 1) - (dA - 1) * (dA + 2 * dB - 1) / (4*dB**2 * (dA * dB + 1) )