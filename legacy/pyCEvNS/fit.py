"""
functions for fitting data using pyMultinest
"""


import numpy as np
import pymultinest
from scipy.integrate import quad

from .helper import _gaussian, _poisson


def fit(events_generator, n_params, n_obs, n_bg, sigma, prior, out_put_dir, bg_model='vary',
        resume=False, verbose=True, n_live_points=5000, evidence_tolerance=0.1, sampling_efficiency=0.3, **kwargs):
    """
    fitting data using multinest package
    :param events_generator: function to generate predicted number of events, it takes n_params paramters to generate events
    :param n_params: number of parameters to be fitted
    :param n_obs: experiment observed data, 1d array
    :param n_bg: background data, 1d array
    :param sigma: theoretical/systemmatic uncertainty of the predecred number of events
    :param prior: prior for each parameters
    :param out_put_dir: out put directories
    :param bg_model: can be 'vary', 'shape', 'fixed', corresponding to poisson distribution on each bin,
                        poisson distribution on total number of events, fixed background
    :param resume: multinest parameter, default is False
    :param verbose: multinest parameter, default is True
    :param n_live_points: multinest parameter, default is 5000
    :param evidence_tolerance: multinest parameter, default is 0.1
    :param sampling_efficiency: multinest parameter, default is 0.3
    :param kwargs: other parameters for multinest
    :return: None, see output at the ouput directory
    """
    if bg_model == 'vary':
        def lgl(cube, ndim, nparams):
            n_signal = events_generator(cube)
            likelihood = np.zeros(n_obs.shape[0])
            for i in range(n_obs.shape[0]):
                n_bg_list = np.arange(max(0, int(n_bg[i] - 2*np.sqrt(n_bg[i]))), max(10, int(n_bg[i] + 2*np.sqrt(n_bg[i]))))
                for nbgi in n_bg_list:
                    likelihood[i] += quad(lambda a: _poisson(n_obs[i], (1 + a) * n_signal[i] + nbgi) *
                                          _gaussian(a, 0, sigma), -3 * sigma, 3 * sigma)[0] * _poisson(n_bg[i], nbgi)
            prod_like = np.prod(likelihood)
            return np.log(prod_like) if prod_like > 0 else -np.inf
    elif bg_model == 'shape':
        def lgl(cube, ndim, nparams):
            n_signal = events_generator(cube)
            prod_like = 0
            nbg_total = sum(n_bg)
            for nbg in np.arange(int(nbg_total - 2*np.sqrt(nbg_total)), int(nbg_total + 2*np.sqrt(nbg_total))):
                likelihood = np.zeros(n_obs.shape[0])
                for i in range(n_obs.shape[0]):
                    if n_bg[i] <= 0:
                        likelihood[i] = 1
                        continue
                    likelihood[i] += quad(lambda a: _poisson(n_obs[i], (1 + a)*n_signal[i] + n_bg[i]*nbg/nbg_total) *
                                          _gaussian(a, 0, sigma), -3 * sigma, 3 * sigma)[0]
                prod_like += np.prod(likelihood) * _poisson(nbg_total, nbg)
            return np.log(prod_like) if prod_like > 0 else -np.inf
    elif bg_model == 'fixed':
        def lgl(cube, ndim, nparams):
            n_signal = events_generator(cube)
            likelihood = np.zeros(n_obs.shape[0])
            for i in range(n_obs.shape[0]):
                if n_bg[i] <= 0:
                    likelihood[i] = 1
                    continue
                likelihood[i] = quad(lambda a: _poisson(n_obs[i], (1 + a) * n_signal[i] + n_bg[i]) *
                                     _gaussian(a, 0, sigma), -3 * sigma, 3 * sigma)[0]
            return np.sum(np.log(likelihood))
    else:
        raise Exception('background model not implemented!')

    def prr(cube, ndim, nparams):
        prior(cube)

    pymultinest.run(lgl, prr, n_params, outputfiles_basename=out_put_dir+'_',
                    resume=resume, verbose=verbose, n_live_points=n_live_points,
                    evidence_tolerance=evidence_tolerance, sampling_efficiency=sampling_efficiency,
                    **kwargs)


class PoissonLoglike:
    def __init__(self, systematic_uncertainty, kind='vary'):
        self.sigma = systematic_uncertainty
        background_models = {'vary': self.vary, 'shape': self.shape}
        self.__call__ = background_models[kind]

    def vary(self, n_obs, n_bg, n_pred):
        likelihood = np.zeros(n_obs.shape[0])
        for i in range(n_obs.shape[0]):
            n_bg_list = np.arange(max(0, int(n_bg[i] - 2 * np.sqrt(n_bg[i]))),
                                  max(10, int(n_bg[i] + 2 * np.sqrt(n_bg[i]))))
            for nbgi in n_bg_list:
                likelihood[i] += quad(lambda a: _poisson(n_obs[i], (1 + a) * n_pred[i] + nbgi) *
                                      _gaussian(a, 0, self.sigma), -3 * self.sigma, 3 * self.sigma)[0] * _poisson(n_bg[i], nbgi)
        prod_like = np.prod(likelihood)
        return np.log(prod_like) if prod_like > 0 else -np.inf

    def shape(self, n_obs, n_bg, n_pred):
        prod_like = 0
        nbg_total = sum(n_bg)
        for nbg in np.arange(int(nbg_total - 2 * np.sqrt(nbg_total)), int(nbg_total + 2 * np.sqrt(nbg_total))):
            likelihood = np.zeros(n_obs.shape[0])
            for i in range(n_obs.shape[0]):
                if n_bg[i] <= 0:
                    likelihood[i] = 1
                    continue
                likelihood[i] += quad(lambda a: _poisson(n_obs[i], (1 + a) * n_pred[i] + n_bg[i] * nbg / nbg_total) *
                                      _gaussian(a, 0, self.sigma), -3 * self.sigma, 3 * self.sigma)[0]
            prod_like += np.prod(likelihood) * _poisson(nbg_total, nbg)
        return np.log(prod_like) if prod_like > 0 else -np.inf

    def fixed(self, n_obs, n_bg, n_pred):
        likelihood = np.zeros(n_obs.shape[0])
        for i in range(n_obs.shape[0]):
            if n_bg[i] <= 0:
                likelihood[i] = 1
                continue
            likelihood[i] = quad(lambda a: _poisson(n_obs[i], (1 + a) * n_pred[i] + n_bg[i]) *
                                           _gaussian(a, 0, self.sigma), -3 * self.sigma, 3 * self.sigma)[0]
        return np.sum(np.log(likelihood))


class Fit:
    def __init__(self, n_obs, n_bg, loglike, output_dir, **kwargs):
        self.n_obs = n_obs
        self.n_bg = n_bg
        self.loglike = loglike
        self.output_dir = output_dir
        self.kwargs = kwargs

    def __call__(self, generator, prior, nparams):
        def lgl(cube, ndim, nparams):
            n_pred = generator(cube)
            return self.loglike(self.n_obs, self.n_bg, n_pred)
        def prr(cube, ndim, nparams):
            prior(cube)
        pymultinest.run(lgl, prr, nparams, outputfiles_basename=self.output_dir + '_', **self.kwargs)
