import decimal
import numpy as np

def format_number(num):
    """
    Get number of zeros I think
    """
    try:
        dec = decimal.Decimal(num)
    except:
        return 'bad'
    tup = dec.as_tuple()
    delta = len(tup.digits) + tup.exponent
    digits = ''.join(str(d) for d in tup.digits)
    if delta <= 0:
        zeros = abs(tup.exponent) - len(tup.digits)
        val = '0.' + ('0'*zeros) + digits
    else:
        val = digits[:delta] + ('0'*tup.exponent) + '.' + digits[delta:]
    val = val.rstrip('0')
    if val[-1] == '.':
        val = val[:-1]
    if tup.sign:
        return '-' + val
    return val

def order_of_magnitude(a_value):
    """
    Get the order of magnitude of input number (0.003 -> -3)
    """
    a_value = np.round(a_value, 6)
    if a_value - int(a_value) != 0:
        a_str = format_number(f'{a_value}')
        a_str = a_str.split(".")[1]
        return len(a_str)
    else:
        return 0

parity_sectors = [-1, 1]
def get_sectors(L, BOUNDARY_COND = 'PBC', mu = 0):
    translation_real_sectors = [0, L // 2] if BOUNDARY_COND == 'PBC' and L % 2 == 0 else [0]
    translation_imag_sectors = range(1, L // 2 + L % 2)
    
    spin_flip_X_sectors = [-1, 1] if mu == 0 and (L % 2 == 0) else [1]

    real_sectors = [[ks, ps, zx] for ks in translation_real_sectors for ps in parity_sectors for zx in spin_flip_X_sectors]
    imag_sectors = [[kx, 1, zx] for kx in translation_imag_sectors for zx in spin_flip_X_sectors]

    return real_sectors, imag_sectors


def info_base(L, N=0, t1=0, t2=0, V1=0, V2=0, mu=0):
    """
    Main body of file names with model parameters as input
    """
    arr = [N, t1, t2, V1, V2]
    names = ['N', 't1', 't2','V1', 'V2']
    info = "_L=%d"%L
    for i, var in enumerate(arr):
        n = order_of_magnitude(var)
        info += str(",%s={:.%df}"%(names[i], n)).format(round(var, n))
    return info


def info(L, N = 1, t1=0, t2=0, V1=0, V2=0, mu=0, k=0, p=1, zx=1, BOUNDARY_COND = 'PBC'):
    """
    Main body of file names with model parameters as input for symmetric model with symmetry sectors as input
    """
    info = info_base(L, N, t1, t2, V1, V2, mu)
    if BOUNDARY_COND == 'PBC':                          info += ",k=%d"%k
    if BOUNDARY_COND == 'OBC' or (k==0 or k==L/2):      info += ",p=%d"%p
    # if not(BOUNDARY_COND == 'PBC') or N == L//2 and mu==0:                             info += ",zx=%d"%zx
    return info