"""
some helper classes and functions
"""


import numpy as np
from scipy.integrate import quad
from scipy.special import xlogy, gammaln


class LinearInterp:
    """
    linear interpolation that deals with numbers
    for better performance than interp1d
    """
    def __init__(self, x: np.ndarray, y: np.ndarray, extend=False):
        self.x = x.copy()
        self.y = y.copy()
        ind = np.argsort(self.x)
        self.x = self.x[ind]
        self.y = self.y[ind]
        self.extend = extend

    def __call__(self, xv):
        if self.extend and xv < self.x[0]:
            return self.y[0]
        if self.extend and xv > self.x[-1]:
            return self.y[-1]
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


class TimeDistribution:
    def __init__(self, kind, initializer):
        if kind not in ['binned', 'histogram', 'pdf']:
            raise Exception('only support binned, histogram, and pdf as input')
        self.kind = kind
        self.initializer = initializer
        self.bin_centers = None
        self.bin_probs = None
        self.bin_widths = None
        if self.kind == 'binned':
            self.bin_centers = initializer[:, 0]
            self.bin_probs = initializer[:, 1]

    def generate_binned_probability(self, bin_centers, bin_widths):
        """
        :param bin_centers: list of bin centers
        :param bin_widths: the width of the bins, scalar or array
        :return: None
        """
        self.bin_centers = bin_centers.copy()
        self.bin_probs = np.zeros_like(bin_centers)
        self.bin_widths = bin_widths
        if self.kind == 'binned':
            raise Exception('You cannot regenerate binned probability if binned data.')
        elif self.kind == 'histogram':
            for i in range(self.initializer.shape[0]):
                if len(self.initializer.shape) > 1:
                    weight = self.initializer[:, 1]
                    self.bin_probs += np.where(self.bin_centers - self.initializer[i, 0] < bin_widths, weight, 0)
                else:
                    self.bin_probs += np.where(self.bin_centers - self.initializer[i] < bin_widths, 1, 0)
            if len(self.initializer.shape) > 1:
                self.bin_probs /= np.sum(self.initializer[:, 1])
            else:
                self.bin_probs /= self.initializer.shape[0]
        elif self.kind == 'pdf':
            for i in range(bin_centers.shape[0]):
                if isinstance(bin_widths, np.ndarray):
                    self.bin_probs[i] += quad(self.initializer, bin_centers[i]-bin_widths[i]/2, bin_centers+bin_widths[i]/2)[0]
                else:
                    self.bin_probs[i] += quad(self.initializer, bin_centers-bin_widths/2, bin_centers+bin_widths/2)[0]

    def binned_probability(self, bin_center):
        if self.bin_centers is None:
            raise Exception('binned probability distribution is not generated yet. please call generate_binned_probability method.')
        idx = np.argwhere(self.bin_centers == bin_center)
        if idx.shape[0] > 0:
            return self.bin_probs[idx[0, 0]]
        else:
            return 0

    def change_parameters(self, bin_centers=None, bin_widths=None, initializer=None):
        self.bin_centers = bin_centers if bin_centers is not None else self.bin_centers
        self.bin_widths = bin_widths if bin_widths is not None else self.bin_widths
        self.initializer = initializer if initializer is not None else self.initializer
        self.generate_binned_probability(self.bin_centers, self.bin_widths)


def polar_to_cartesian(r):
    """
    convert polar direction (zenith and azimuth to unit vector)
    :param r: 2d array, first azimuth second cos(zenith)
    :return: unit vector in cartesian coordinate system
    """
    return np.array([np.sqrt(1-r[1]**2)*np.cos(r[0]), np.sqrt(1-r[1]**2)*np.sin(r[0]), r[1]])


def lorentz_boost(momentum, v):
    """
    Lorentz boost momentum to a new frame with velocity v
    :param momentum: four vector
    :param v: velocity of new frame, 3-dimention
    :return: boosted momentum
    """
    n = v/np.sqrt(np.sum(v**2))
    beta = np.sqrt(np.sum(v**2))
    gamma = 1/np.sqrt(1-beta**2)
    mat = np.array([[gamma, -gamma*beta*n[0], -gamma*beta*n[1], -gamma*beta*n[2]],
                    [-gamma*beta*n[0], 1+(gamma-1)*n[0]*n[0], (gamma-1)*n[0]*n[1], (gamma-1)*n[0]*n[2]],
                    [-gamma*beta*n[1], (gamma-1)*n[1]*n[0], 1+(gamma-1)*n[1]*n[1], (gamma-1)*n[1]*n[2]],
                    [-gamma*beta*n[2], (gamma-1)*n[2]*n[0], (gamma-1)*n[2]*n[1], 1+(gamma-1)*n[2]*n[2]]])
    return mat @ momentum
