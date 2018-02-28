"""
functions for calculating likelihood
"""


from numpy import *
from scipy.special import gammaln


def loglike(nui, ni, theta, theta_un):
    """
    calculating log-likelihood, following Poisson distribution
    it represents the probability (up to a normalization) of nui when ni
    :param nui: expected number of events (including background), it may be a function of theta
    :param ni: measured number of events
    :param theta: list of nuisance parameters, normalized around 1, using numpy.array
    :param theta_un: list of uncertainty of each nuisance parameter, using numpy.array
    :return: log of the likelihood
    """
    return sum(-nui + ni * log(nui) - gammaln(ni + 1)) - sum((theta - 1) ** 2 / (2 * (theta_un ** 2)))
