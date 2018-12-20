"""
functions for fitting data using pyMultinest
"""


import pymultinest
from numpy import *
from scipy.integrate import quad, dblquad
from scipy.special import xlogy, gammaln


def _poisson(k, l):
    """
    poisson distribution
    :param k: observed number
    :param l: mean number
    :return: probability
    """
    return exp(xlogy(k, l) - gammaln(k+1) - l)


def _gaussian(x, mu, sigma):
    """
    gaussian distribution
    :param x: number
    :param mu: mean
    :param sigma: standard deviation
    :return: probability density
    """
    return exp(-(x-mu)**2/(2*sigma**2)) / sqrt(2*pi*sigma**2)


def fit_without_background_uncertainty(events_generator, n_params, n_bg, n_obs, sigma, prior, out_put_dir,
                                       resume=False, verbose=True, n_live_points=1500,
                                       evidence_tolerance=0.1, sampling_efficiency=0.3, **kwargs):
    """
    fitting data using provided loglikelihood,
    assumming no uncertainty on background,
    the default of each parameter ranges from 0 to 1 uniformly,
    use prior to modify the range or distribution of parameters
    :param events_generator: functions to generate predicted number of events
    :param n_params: number of parameters to be fitted
    :param n_bg: background data, 1d array
    :param n_obs: experiment observed data, 1d array
    :param sigma: systemmatic uncertainty on signal
    :param prior: prior for each parameters
    :param out_put_dir: out put directories
    :param resume: multinest parameter, default is False
    :param verbose: multinest parameter, default is True
    :param n_live_points: multinest parameter, default is 1500
    :param evidence_tolerance: multinest parameter, default is 0.1
    :param sampling_efficiency: multinest parameter, default is 0.3
    :param kwargs other parameters for multinest
    """
    # pymultinest requires ndim and nparams in loglikelihood and prior, it seems we don't need it here
    def lgl(cube, ndim, nparams):
        n_signal = events_generator(cube)
        likelihood = zeros(n_obs.shape[0])
        for i in range(n_obs.shape[0]):
            likelihood[i] = quad(lambda a: _poisson(n_obs[i], (1 + a) * n_signal[i] + n_bg[i]) * _gaussian(a, 0, sigma),
                                 -3 * sigma, 3 * sigma)[0]
        prod_like = prod(likelihood)
        return log(prod_like) if prod_like > 0 else -inf

    def prr(cube, ndim, nparams):
        prior(cube)

    pymultinest.run(lgl, prr, n_params, outputfiles_basename=out_put_dir+'_',
                    resume=resume, verbose=verbose, n_live_points=n_live_points,
                    evidence_tolerance=evidence_tolerance, sampling_efficiency=sampling_efficiency,
                    **kwargs)


def fit_with_background_uncertainty(events_generator, n_params, n_bg, n_obs, sigma, sigma_bg, prior, out_put_dir,
                                    resume=False, verbose=True, n_live_points=1500,
                                    evidence_tolerance=0.1, sampling_efficiency=0.3, **kwargs):
    """
       fitting data using provided loglikelihood,
       assumming uncertainties on background,
       the default of each parameter ranges from 0 to 1 uniformly,
       use prior to modify the range or distribution of parameters
       :param events_generator: functions to generate predicted number of events
       :param n_params: number of parameters to be fitted
       :param n_bg: background data, 1d array
       :param n_obs: experiment observed data, 1d array
       :param sigma: systemmatic uncertainty on signal
       :param sigma_bg: systemmatic uncertianty on baground, if 0, use poisson distribution
       :param prior: prior for each parameters
       :param out_put_dir: out put directories
       :param resume: multinest parameter, default is False
       :param verbose: multinest parameter, default is True
       :param n_live_points: multinest parameter, default is 1500
       :param evidence_tolerance: multinest parameter, default is 0.1
       :param sampling_efficiency: multinest parameter, default is 0.3
       :param kwargs other parameters for multinest
       """
    # pymultinest requires ndim and nparams in loglikelihood and prior, it seems we don't need it here
    def lgl(cube, ndim, nparams):
        n_signal = events_generator(cube)
        likelihood = zeros(n_obs.shape[0])
        for i in range(n_obs.shape[0]):
            if sigma_bg == 0:
                n_bg_list = arange(max(0, int(n_bg[i] - 2*sqrt(n_bg[i]))), max(10, int(n_bg[i] + 2*sqrt(n_bg[i]))))
                for nbgi in n_bg_list:
                    likelihood[i] += quad(lambda a: _poisson(n_obs[i], (1 + a) * n_signal[i] + nbgi) *
                                          _gaussian(a, 0, sigma), -3 * sigma, 3 * sigma)[0] * _poisson(n_bg[i], nbgi)
            else:
                likelihood[i] = dblquad(lambda nbg, a: _poisson(n_obs[i], (1 + a) * n_signal[i] + nbg) *
                                        _gaussian(a, 0, sigma) * _gaussian(n_bg[i], nbg, sigma_bg),
                                        -3*sigma, 3*sigma, lambda aa: n_bg[i]-3*sigma_bg, lambda aa: n_bg[i]+3*sigma_bg)[0]
        prod_like = prod(likelihood)
        return log(prod_like) if prod_like > 0 else -inf

    def prr(cube, ndim, nparams):
        prior(cube)

    pymultinest.run(lgl, prr, n_params, outputfiles_basename=out_put_dir+'_',
                    resume=resume, verbose=verbose, n_live_points=n_live_points,
                    evidence_tolerance=evidence_tolerance, sampling_efficiency=sampling_efficiency,
                    **kwargs)
