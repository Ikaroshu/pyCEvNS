"""
parameter classes
"""


import numpy as np
import pkg_resources
from scipy.interpolate import interp1d

from .constants import *


class NSIparameters:
    r"""
    nsi parameter class,
    g = g_\nu*g_f
    mz = mediator mass
    for scattering, it is more convenient to use g
    for oscillation, it is more convenient to use epsilon
    it contains L and R couplings of electron scattering,
    and vector couplings of quarks
    """
    def __init__(self, mz=0):
        """
        initializing all nsi == 0
        """
        self.mz = mz
        self.gel = {'ee': 0, 'mm': 0, 'tt': 0, 'em': 0, 'et': 0, 'mt': 0}
        self.ger = {'ee': 0, 'mm': 0, 'tt': 0, 'em': 0, 'et': 0, 'mt': 0}
        self.gu = {'ee': 0, 'mm': 0, 'tt': 0, 'em': 0, 'et': 0, 'mt': 0}
        self.gd = {'ee': 0, 'mm': 0, 'tt': 0, 'em': 0, 'et': 0, 'mt': 0}
        self.epel = {'ee': 0, 'mm': 0, 'tt': 0, 'em': 0, 'et': 0, 'mt': 0}
        self.eper = {'ee': 0, 'mm': 0, 'tt': 0, 'em': 0, 'et': 0, 'mt': 0}
        self.epe = {'ee': 0, 'mm': 0, 'tt': 0, 'em': 0, 'et': 0, 'mt': 0}
        self.epu = {'ee': 0, 'mm': 0, 'tt': 0, 'em': 0, 'et': 0, 'mt': 0}
        self.epd = {'ee': 0, 'mm': 0, 'tt': 0, 'em': 0, 'et': 0, 'mt': 0}

    def ee(self):
        """
        :return: matrix of nsi for electron
        """
        if self.mz != 0:
            for i in self.epe:
                self.epe[i] = (self.gel[i]+self.ger[i]) / (2*np.sqrt(2)*gf*self.mz**2)
        return np.array([[self.epe['ee'], self.epe['em'], self.epe['et']],
                        [np.conj(self.epe['em']), self.epe['mm'], self.epe['mt']],
                        [np.conj(self.epe['et']), np.conj(self.epe['mt']), self.epe['tt']]])

    def eu(self):
        """
        :return: matrix of nsi for u quark
        """
        if self.mz != 0:
            for i in self.epu:
                self.epu[i] = self.gu[i] / (2*np.sqrt(2)*gf*self.mz**2)
        return np.array([[self.epu['ee'], self.epu['em'], self.epu['et']],
                        [np.conj(self.epu['em']), self.epu['mm'], self.epu['mt']],
                        [np.conj(self.epu['et']), np.conj(self.epu['mt']), self.epu['tt']]])

    def ed(self):
        """
        :return: matrix of nsi for d quark
        """
        if self.mz != 0:
            for i in self.epd:
                self.epd[i] = self.gu[i] / (2*np.sqrt(2)*gf*self.mz**2)
        return np.array([[self.epd['ee'], self.epd['em'], self.epd['et']],
                        [np.conj(self.epd['em']), self.epd['mm'], self.epd['mt']],
                        [np.conj(self.epd['et']), np.conj(self.epd['mt']), self.epd['tt']]])

    def eel(self):
        if self.mz != 0:
            for i in self.epel:
                self.epel[i] = (self.gel[i]) / (2*np.sqrt(2)*gf*self.mz**2)
        return np.array([[self.epel['ee'], self.epel['em'], self.epel['et']],
                        [np.conj(self.epel['em']), self.epel['mm'], self.epel['mt']],
                        [np.conj(self.epel['et']), np.conj(self.epel['mt']), self.epel['tt']]])

    def eer(self):
        if self.mz != 0:
            for i in self.eper:
                self.eper[i] = (self.ger[i]) / (2*np.sqrt(2)*gf*self.mz**2)
        return np.array([[self.eper['ee'], self.eper['em'], self.eper['et']],
                        [np.conj(self.eper['em']), self.eper['mm'], self.eper['mt']],
                        [np.conj(self.eper['et']), np.conj(self.eper['mt']), self.eper['tt']]])


def oscillation_parameters(t12=0.5763617589722192,
                           t13=0.14819001778459273,
                           t23=0.7222302630963306,
                           delta=1.35*np.pi,
                           d21=7.37e-17,
                           d31=2.5e-15+3.685e-17):
    r"""
    creating a list of oscillation parameter, default: LMA solution
    :param t12: \theta_12
    :param t23: \theta_23
    :param t13: \theta_13
    :param delta: \delta
    :param d21: \Delta m^{2}_{21} in MeV^2
    :param d31: \Delta m^{2}_{31} in MeV^2
    :return: list of oscillation parameter
    """
    return {'t12': t12, 't13': t13, 't23': t23, 'delta': delta, 'd21': d21, 'd31': d31}


class OSCparameters:
    """
    oscillation parameter class
    """
    def __init__(self, t12=0.5763617589722192,
                 t13=0.14819001778459273,
                 t23=0.7222302630963306,
                 delta=1.35*np.pi,
                 d21=7.37e-17,
                 d31=2.5e-15+3.685e-17):
        r"""
        creating a list of oscillation parameter, default: LMA solution
        :param t12: \theta_12
        :param t23: \theta_23
        :param t13: \theta_13
        :param delta: \delta
        :param d21: \Delta m^{2}_{21} in MeV^2
        :param d31: \Delta m^{2}_{31} in MeV^2
        :return: list of oscillation parameter
        """
        self.t12 = t12
        self.t13 = t13
        self.t23 = t23
        self.delta = delta
        self.d21 = d21
        self.d31 = d31

    def __getitem__(self, item):
        if item == 't12':
            return self.t12
        if item == 't13':
            return self.t13
        if item == 't23':
            return self.t23
        if item == 'delta':
            return self.delta
        if item == 'd21':
            return self.d21
        if item == 'd31':
            return self.d31

    def copy(self):
        return oscillation_parameters(self.t12, self.t13, self.t23, self.delta, self.d21, self.d31)


class Density:
    """
    solar number density
    """
    def __init__(self):
        """
        initializing with SSM
        """
        fpath = pkg_resources.resource_filename(__name__, 'data/density_data.txt')
        density = np.genfromtxt(fpath, delimiter='  ')
        rs = density[:, 1]
        rho = density[:, 3]
        npd = rho * (density[:, 6] / massofh + density[:, 7] / massof4he * 2 + density[:, 8] / massof3he * 2 +
                     density[:, 9] / massof12c * 6 + density[:, 10] / massof14n * 7 + density[:, 11] / massof16o * 8) *\
            1e6 * (meter_by_mev ** 3)
        nnd = rho * (density[:, 7] / massof4he * 2 + density[:, 8] / massof3he * 1 +
                     density[:, 9] / massof12c * 6 + density[:, 10] / massof14n * 7 + density[:, 11] / massof16o * 8) *\
            1e6 * (meter_by_mev ** 3)
        nud = 2 * npd + nnd
        ndd = npd + 2 * nnd
        self.__nud_interp = interp1d(rs, nud)
        self.__ndd_interp = interp1d(rs, ndd)

    def nu(self, r):
        """
        u quark number density
        :param r: distance from the center in terms of sorlar radius
        :return: u quark number density
        """
        return self.__nud_interp(r)[()]

    def nd(self, r):
        """
        d quark number density
        :param r: distance from the center in terms of sorlar radius
        :return: d quark number density
        """
        return self.__ndd_interp(r)[()]

    def ne(self, r):
        """
        electron number density
        :param r: distance from the center in terms of sorlar radius
        :return: electron number density
        """
        return (2 * self.nu(r) - self.nd(r)) / 3
