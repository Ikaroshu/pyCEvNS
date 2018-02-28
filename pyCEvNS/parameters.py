"""
parameter classes
"""


import pkg_resources
from numpy import *
from scipy.interpolate import interp1d

from .constants import *


class Epsilon:
    """
    nsi parameter class,
    it contains L and R couplings of electron scattering,
    and vector couplings of quarks
    """
    def __init__(self):
        """
        initializing all nsi == 0
        """
        self.epel = {'ee': 0, 'mm': 0, 'tt': 0, 'em': 0, 'et': 0, 'mt': 0}
        self.eper = {'ee': 0, 'mm': 0, 'tt': 0, 'em': 0, 'et': 0, 'mt': 0}
        self.epu = {'ee': 0, 'mm': 0, 'tt': 0, 'em': 0, 'et': 0, 'mt': 0}
        self.epd = {'ee': 0, 'mm': 0, 'tt': 0, 'em': 0, 'et': 0, 'mt': 0}

    def ee(self):
        """
        :return: matrix of nsi for electron
        """
        epe = {'ee': 0, 'mm': 0, 'tt': 0, 'em': 0, 'et': 0, 'mt': 0}
        for i in epe:
            epe[i] = self.epel[i] + self.eper[i]
        return matrix([[epe['ee'], epe['em'], epe['et']],
                       [conj(epe['em']), epe['mm'], epe['mt']],
                       [conj(epe['et']), conj(epe['mt']), epe['tt']]]) + diag(array([1, 0, 0]))

    def eu(self):
        """
        :return: matrix of nsi for u quark
        """
        return matrix([[self.epu['ee'], self.epu['em'], self.epu['et']],
                       [conj(self.epu['em']), self.epu['mm'], self.epu['mt']],
                       [conj(self.epu['et']), conj(self.epu['mt']), self.epu['tt']]])

    def ed(self):
        """
        :return: matrix of nsi for d quark
        """
        return matrix([[self.epd['ee'], self.epd['em'], self.epd['et']],
                       [conj(self.epd['em']), self.epd['mm'], self.epd['mt']],
                       [conj(self.epd['et']), conj(self.epd['mt']), self.epd['tt']]])


def ocsillation_parameters(t12=0.5763617589722192,
                           t13=0.14819001778459273,
                           t23=0.7222302630963306,
                           delta=1.35 * pi,
                           d21=7.37e-17,
                           d31=2.5e-15+3.685e-17):
    """
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


class Density:
    """
    solar number density
    """
    def __init__(self):
        """
        initializing with SSM
        """
        fpath = pkg_resources.resource_filename(__name__, 'data/density_data.txt')
        density = genfromtxt(fpath, delimiter='  ')
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
