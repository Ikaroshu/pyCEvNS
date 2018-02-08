from numpy import *
from scipy.special import gammaln


def loglike(nui, ni, theta, theta_un):
    return sum(-nui + ni * log(nui) - gammaln(ni + 1)) - sum((theta - 1) ** 2 / (2 * (theta_un ** 2)))
