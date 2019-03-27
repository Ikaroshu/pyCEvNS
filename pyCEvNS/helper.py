"""
some helper classes and functions
"""


import numpy as np
from scipy.special import xlogy, gammaln


class LinearInterp:
    """
    linear interpolation that deals with numbers
    for better performance than interp1d
    """
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x.copy()
        self.y = y.copy()
        ind = np.argsort(self.x)
        self.x = self.x[ind]
        self.y = self.y[ind]

    def __call__(self, xv):
        if xv < self.x[0] or xv > self.x[-1]:
            return 0
        elif xv == self.x[0]:
            return self.y[0]
        ind = np.searchsorted(self.x, xv)
        slope = (self.y[ind] - self.y[ind-1]) / (self.x[ind] - self.x[ind-1])
        return slope * (xv - self.x[ind-1]) + self.y[ind-1]


def _poisson(k, l):
    """
    poisson distribution
    :param k: observed number
    :param l: mean number
    :return: probability
    """
    return np.exp(xlogy(k, l) - gammaln(k+1) - l)


def _gaussian(x, mu, sigma):
    """
    gaussian distribution
    :param x: number
    :param mu: mean
    :param sigma: standard deviation
    :return: probability density
    """
    return np.exp(-(x-mu)**2/(2*sigma**2)) / np.sqrt(2*np.pi*sigma**2)
