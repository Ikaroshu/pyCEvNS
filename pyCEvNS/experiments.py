"""
provide a uniform API for creating experiments
"""


from .events import *   # pylint: disable=W0401, W0614, W0622


class Experiment(Detector, Flux):
    """
    class for certain type of experiment
    privide methods to calculate number of events in this experiment
    """
    def __init__(self, fx_ty, det_ty, coherence_type, exposure, nbins=1, bins=None):
        Flux.__init__(self, fx_ty)
        Detector.__init__(self, det_ty)
        self.expo = exposure
        self.nbins = nbins
        self.ctype = coherence_type
        if not bins:
            self.bins = linspace(self.er_min, self.er_max, nbins+1)
        else:
            self.bins = bins

    def rates(self, er, nsip: NSIparameters, flavor='e', op=None, r=0.05):
        """
        predicted rates based on input NSI parameters
        :param er: recoil energy
        :param flavor: flux flavor
        :param nsip: nsi parameters
        :param op: oscillation parameters
        :param r: position where neutrino is produced in the sun, for solar neutrino only
        :return: rates in units of dru
        """
        if self.ctype == 'nucleus':
            return rates_nucleus(er, self, self, nsip, flavor, op, r) * \
                   1 * mev_per_kg * 24 * 60 * 60 / dot(self.m, self.frac) * 1e-3
        elif self.ctype == 'electron':
            return rates_electron(er, self, self, nsip, flavor, op, r) * \
                   1 * mev_per_kg * 24 * 60 * 60 / dot(self.m, self.frac) * 1e-3
        else:
            raise Exception('''Experiment can either be "nucleus" or "electron"!''')