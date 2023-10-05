import numpy as np


def typical(data):
    """
    Calculate the typical value of given dataset
    """
    return np.prod(np.array(data))**(1.0 / data.size)
    # return np.exp(np.mean(np.log(np.array(data))))

#--------------- FLUCTUATIONS FROM DATA
def remove_fluctuations(data, bucket_size=10):
    """
    Remove fluctuations from data and extract the running mean
    -------------
    data:           Input data to average points
    bucket_size:    Number of elements in average for new point in running mean (size of window)
    """
    new_data = data;
    half_bucket = int(bucket_size / 2)
    for k in range(half_bucket, len(data) - half_bucket):
        average = np.sum(data[k - half_bucket : k + half_bucket])
        new_data[k - half_bucket] = average / bucket_size
    return new_data


def get_fluctuations(data, bucket_size=10, type = 'mean'):
    """
    Get fluctuations from data and remove the running mean
    -------------
    data:           Input data to find fluctuations
    bucket_size:    Number of elements in average for new point in running mean (size of window)
    """
    new_data = np.zeros(data.shape);
    half_bucket = int(bucket_size / 2)
    for k in range(half_bucket, len(data) - half_bucket):
        average = 0
        if type == 'mean':     average = np.mean(data[k - half_bucket : k + half_bucket])
        elif type == 'typ':    average = np.median(data[k - half_bucket : k + half_bucket])
        else:                  average = 0
         
        new_data[k] = data[k] - average
    return new_data