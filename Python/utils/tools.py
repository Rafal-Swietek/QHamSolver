import numpy as np
import scipy
from scipy.optimize import fsolve


def typical(data):
    """
    Calculate the typical value of given dataset
    """
    # return np.prod(np.array(data))**(1.0 / data.size)
    return np.exp(np.mean(np.log(np.array(data))))

#--------------- FLUCTUATIONS FROM DATA
def remove_fluctuations(data, bucket_size=10, type = 'mean'):
    """
    Remove fluctuations from data and extract the running mean
    -------------
    data:           Input data to average points
    bucket_size:    Number of elements in average for new point in running mean (size of window)
    """
    new_data = np.zeros(data.shape);
    half_bucket = int(bucket_size // 2)
    for k in range(half_bucket, len(data) - half_bucket):
        average = 0
        if type == 'mean':     average = np.mean(  data[k - half_bucket : k + half_bucket + 1])
        elif type == 'typ':    average = np.median(data[k - half_bucket : k + half_bucket + 1])
        else:                  average = 0
         
        new_data[k] = average
    return new_data


def get_fluctuations(data, bucket_size=10, type = 'mean'):
    """
    Get fluctuations from data and remove the running mean
    -------------
    data:           Input data to find fluctuations
    bucket_size:    Number of elements in average for new point in running mean (size of window)
    """
    new_data = np.zeros(data.shape);
    half_bucket = int(bucket_size // 2)
    for k in range(half_bucket, len(data) - half_bucket):
        average = 0
        if type == 'mean':     average = np.mean(  data[k - half_bucket : k + half_bucket + 1])
        elif type == 'typ':    average = np.median(data[k - half_bucket : k + half_bucket + 1])
        else:                  average = 0
         
        new_data[k] = data[k] - average
    return new_data

class RootFinder:
    """
    Class to find roots of function within a given interval
    """
    def __init__(self, start, stop, step=0.01, root_dtype="float64", xtol=1e-9):

        self.start = start
        self.stop = stop
        self.step = step
        self.xtol = xtol
        self.roots = np.array([], dtype=root_dtype)

    def add_to_roots(self, x):

        if (x < self.start) or (x > self.stop):
            return  # outside range
        if any(abs(self.roots - x) < self.xtol):
            return  # root already found.

        self.roots = np.append(self.roots, x)

    def find(self, f, *args):
        current = self.start

        for x0 in np.arange(self.start, self.stop + self.step, self.step):
            if x0 < current:
                continue
            x = self.find_root(f, x0, *args)
            if x is None:  # no root found.
                continue
            current = x
            self.add_to_roots(x)

        return self.roots

    def find_root(self, f, x0, *args):

        x, _, ier, _ = fsolve(f, x0=x0, args=args, full_output=True, xtol=self.xtol)
        if ier == 1:
            return x[0]
        return None