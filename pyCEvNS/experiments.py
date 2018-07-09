"""
provide a uniform API for creating experiments
"""


from .events import *   # pylint: disable=W0401, W0614, W0622


class Experiment(Detector, Flux):
    """
    class for certain type of experiment
    privide methods to calculate number of events in this experiment
    support faster calculation for events
    """
    def __init__(self, fl_ty, det_ty, coherence_type, exposure, nbins=1, bins=None, fl_file=None):
        """
        initailize experiment
        :param fl_ty: flux type
        :param det_ty: detector type
        :param coherence_type: nucleus or electron
        :param exposure: exposure in kg*days
        :param nbins: number of bins, this will devide the default energy range of detector into nbins evenly
        :param bins: program will use this bins if user provided it
        :param fl_file: user provided flux data
        """
        Flux.__init__(self, fl_ty, fl_file)
        Detector.__init__(self, det_ty)
        self.expo = exposure
        self.nbins = nbins
        self.ctype = coherence_type
        if not bins:
            self.bins = linspace(self.er_min, self.er_max, nbins+1)
        else:
            self.bins = bins
        erl = linspace(self.bins[0], self.bins[self.bins.shape[0]-1], 100)
        if coherence_type == 'nucleus':
            finte = zeros(shape=(erl.shape[0], self.m.shape[0]))
            fintinve = zeros(shape=(erl.shape[0], self.m.shape[0]))
            fintinvse = zeros(shape=(erl.shape[0], self.m.shape[0]))
            fintm = zeros(shape=(erl.shape[0], self.m.shape[0]))
            fintinvm = zeros(shape=(erl.shape[0], self.m.shape[0]))
            fintinvsm = zeros(shape=(erl.shape[0], self.m.shape[0]))
            fintt = zeros(shape=(erl.shape[0], self.m.shape[0]))
            fintinvt = zeros(shape=(erl.shape[0], self.m.shape[0]))
            fintinvst = zeros(shape=(erl.shape[0], self.m.shape[0]))
            for i in range(erl.shape[0]):
                finte[i] = self.fint(erl[i], self.m, flavor='e')

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