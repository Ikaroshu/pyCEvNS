"""
CEvNS events
"""

from abc import ABC, abstractmethod

from scipy.special import spherical_jn

from .detectors import *
from .flux import *


class EventGen(ABC):
    """
    abstract class for events generator,
    the inherited class must implement rates and events function
    """
    @abstractmethod
    def rates(self, er, **kwargs):
        """
        :param er: recoil energy in MeV
        :return: dN/dE_r
        """
        pass

    @abstractmethod
    def events(self, ea, eb, **kwargs):
        """
        :return: number of events in [ea, eb]
        """
        pass


def formfsquared(q, rn=5.5):
    """
    form factor squared
    1810.05606
    :param q: momentum transfered
    :param rn: neutron skin radius
    :return: form factor squared
    """
    r = rn * (10 ** -15) / meter_by_mev
    s = 0.9 * (10 ** -15) / meter_by_mev
    r0 = np.sqrt(5 / 3 * (r ** 2) - 5 * (s ** 2))
    return (3 * spherical_jn(1, q * r0) / (q * r0) * np.exp((-(q * s) ** 2) / 2)) ** 2


def eff_coherent(er):
    pe_per_mev = 0.0878 * 13.348 * 1000
    pe = er * pe_per_mev
    a = 0.6655
    k = 0.4942
    x0 = 10.8507
    f = a / (1 + np.exp(-k * (pe - x0)))
    if pe < 5:
        return 0
    if pe < 6:
        return 0.5 * f
    return f


def rates_nucleus(er, det: Detector, fx: Flux, efficiency=None, f=None, nsip=NSIparameters(), flavor='e',
                  op=oscillation_parameters(), ffs=formfsquared, q2=False, **kwargs):
    """
    calculating scattering rates per nucleus
    :param er: recoil energy
    :param det: detector
    :param fx: flux
    :param f: oscillation function
    :param efficiency: efficiency function
    :param flavor: flux flavor
    :param nsip: nsi parameters
    :param op: oscillation parameters
    :param ffs: custom formfactor sqared function
    :param q2: whether to include q^2 formfactor
    :return: scattering rates per nucleus
    """
    deno = 2 * np.sqrt(2) * gf * (2 * det.m * er + nsip.mz ** 2)
    # radiative corrections,
    # Barranco, 2005
    # is it necessary?
    rho = 1.0086
    knu = 0.9978
    lul = -0.0031
    ldl = -0.0025
    ldr = 7.5e-5
    lur = ldr / 2
    q2fact = 1.0
    if q2:
        q2fact = 2 * det.m * er
    if nsip.mz != 0:
        if flavor[0] == 'e':
            qvs = (0.5 * det.z * (rho * (0.5 - 2 * knu * ssw) + 2 * lul + 2 * lur + ldl + ldr +
                                  2 * q2fact * nsip.gu['ee'] / deno + q2fact * nsip.gd['ee'] / deno) +
                   0.5 * det.n * (-0.5 * rho + lul + lur + 2 * ldl + 2 * ldr +
                                  q2fact * nsip.gu['ee'] / deno + 2 * q2fact * nsip.gd['ee'] / deno)) ** 2 + \
                  np.abs(0.5 * det.z * (2 * nsip.gu['em'] / deno + nsip.gd['em'] / deno) +
                         0.5 * det.n * (nsip.gu['em'] / deno + 2 * nsip.gd['em'] / deno)) ** 2 + \
                  np.abs(0.5 * det.z * (2 * nsip.gu['et'] / deno + nsip.gd['et'] / deno) +
                         0.5 * det.n * (nsip.gu['et'] / deno + 2 * nsip.gd['et'] / deno)) ** 2
        elif flavor[0] == 'm':
            qvs = (0.5 * det.z * (rho * (0.5 - 2 * knu * ssw) + 2 * lul + 2 * lur + ldl + ldr +
                                  2 * q2fact * nsip.gu['mm'] / deno + q2fact * nsip.gd['mm'] / deno) +
                   0.5 * det.n * (-0.5 * rho + lul + lur + 2 * ldl + 2 * ldr +
                                  q2fact * nsip.gu['mm'] / deno + 2 * q2fact * nsip.gd['mm'] / deno)) ** 2 + \
                  np.abs(0.5 * det.z * (2 * nsip.gu['em'] / deno + nsip.gd['em'] / deno) +
                         0.5 * det.n * (nsip.gu['em'] / deno + 2 * nsip.gd['em'] / deno)) ** 2 + \
                  np.abs(0.5 * det.z * (2 * nsip.gu['mt'] / deno + nsip.gd['mt'] / deno) +
                         0.5 * det.n * (nsip.gu['mt'] / deno + 2 * nsip.gd['mt'] / deno)) ** 2
        elif flavor[0] == 't':
            qvs = (0.5 * det.z * (rho * (0.5 - 2 * knu * ssw) + 2 * lul + 2 * lur + ldl + ldr +
                                  2 * q2fact * nsip.gu['tt'] / deno + q2fact * nsip.gd['tt'] / deno) +
                   0.5 * det.n * (-0.5 * rho + lul + lur + 2 * ldl + 2 * ldr +
                                  q2fact * nsip.gu['tt'] / deno + 2 * q2fact * nsip.gd['tt'] / deno)) ** 2 + \
                  np.abs(0.5 * det.z * (2 * nsip.gu['et'] / deno + nsip.gd['et'] / deno) +
                         0.5 * det.n * (nsip.gu['et'] / deno + 2 * nsip.gd['et'] / deno)) ** 2 + \
                  np.abs(0.5 * det.z * (2 * nsip.gu['mt'] / deno + nsip.gd['mt'] / deno) +
                         0.5 * det.n * (nsip.gu['mt'] / deno + 2 * nsip.gd['mt'] / deno)) ** 2
        else:
            raise Exception('No such neutrino flavor!')
    else:
        if flavor[0] == 'e':
            qvs = (0.5 * det.z * (rho * (0.5 - 2 * knu * ssw) + 2 * lul + 2 * lur + ldl + ldr +
                                  2 * q2fact * nsip.epu['ee'] + q2fact * nsip.epd['ee']) +
                   0.5 * det.n * (-0.5 * rho + lul + lur + 2 * ldl + 2 * ldr +
                                  q2fact * nsip.epu['ee'] + 2 * q2fact * nsip.epd['ee'])) ** 2 + \
                  np.abs(0.5 * det.z * (2 * nsip.epu['em'] + nsip.epd['em']) +
                         0.5 * det.n * (nsip.epu['em'] + 2 * nsip.epd['em'])) ** 2 + \
                  np.abs(0.5 * det.z * (2 * nsip.epu['et'] + nsip.epd['et']) +
                         0.5 * det.n * (nsip.epu['et'] + 2 * nsip.epd['et'])) ** 2
        elif flavor[0] == 'm':
            qvs = (0.5 * det.z * (rho * (0.5 - 2 * knu * ssw) + 2 * lul + 2 * lur + ldl + ldr +
                                  2 * q2fact * nsip.epu['mm'] + q2fact * nsip.epd['mm']) +
                   0.5 * det.n * (-0.5 * rho + lul + lur + 2 * ldl + 2 * ldr +
                                  q2fact * nsip.epu['mm'] + 2 * q2fact * nsip.epd['mm'])) ** 2 + \
                  np.abs(0.5 * det.z * (2 * nsip.epu['em'] + nsip.epd['em']) +
                         0.5 * det.n * (nsip.epu['em'] + 2 * nsip.epd['em'])) ** 2 + \
                  np.abs(0.5 * det.z * (2 * nsip.epu['mt'] + nsip.epd['mt']) +
                         0.5 * det.n * (nsip.epu['mt'] + 2 * q2fact * nsip.epd['mt'])) ** 2
        elif flavor[0] == 't':
            qvs = (0.5 * det.z * (rho * (0.5 - 2 * knu * ssw) + 2 * lul + 2 * lur + ldl + ldr +
                                  2 * q2fact * nsip.epu['tt'] + q2fact * nsip.epd['tt']) +
                   0.5 * det.n * (-0.5 * rho + lul + lur + 2 * ldl + 2 * ldr +
                                  q2fact * nsip.epu['tt'] + 2 * q2fact * nsip.epd['tt'])) ** 2 + \
                  np.abs(0.5 * det.z * (2 * nsip.epu['et'] + nsip.epd['et']) +
                         0.5 * det.n * (nsip.epu['et'] + 2 * nsip.epd['et'])) ** 2 + \
                  np.abs(0.5 * det.z * (2 * nsip.epu['mt'] + nsip.epd['mt']) +
                         0.5 * det.n * (nsip.epu['mt'] + 2 * nsip.epd['mt'])) ** 2
        else:
            raise Exception('No such neutrino flavor!')
    if efficiency is not None:
        return np.dot(2 / np.pi * (gf ** 2) * (2 * fx.fint(er, det.m, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs) -
                                         2 * er * fx.fintinv(er, det.m, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs) +
                                         er * er * fx.fintinvs(er, det.m, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs) -
                                         det.m * er * fx.fintinvs(er, det.m, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs)) *
                   det.m * qvs * ffs(np.sqrt(2 * det.m * er), **kwargs), det.frac) * efficiency(er)
    else:
        return np.dot(2 / np.pi * (gf ** 2) * (2 * fx.fint(er, det.m, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs) -
                                         2 * er * fx.fintinv(er, det.m, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs) +
                                         er * er * fx.fintinvs(er, det.m, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs) -
                                         det.m * er * fx.fintinvs(er, det.m, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs)) *
                   det.m * qvs * ffs(np.sqrt(2 * det.m * er), **kwargs), det.frac)


def rates_electron(er, det: Detector, fx: Flux, efficiency=None, f=None, nsip=NSIparameters(), flavor='e',
                   op=oscillation_parameters(), **kwargs):
    """
    calculating electron scattering rates per nucleus
    :param er: recoil energy
    :param det: detector
    :param fx: flux
    :param f: oscillation function
    :param efficiency: efficiency function
    :param flavor: flux flavor
    :param nsip: nsi parameters
    :param op: oscillation parameters
    :return: scattering rates per nucleus
    """
    deno = 2 * np.sqrt(2) * gf * (2 * me * er + nsip.mz ** 2)
    if flavor[0] == 'e':
        epls = (0.5 + ssw + nsip.gel['ee'] / deno) ** 2 + (nsip.gel['em'] / deno) ** 2 + (nsip.gel['et'] / deno) ** 2
        eprs = (ssw + nsip.ger['ee'] / deno) ** 2 + (nsip.ger['em'] / deno) ** 2 + (nsip.ger['et'] / deno) ** 2
        eplr = (0.5 + ssw + nsip.gel['ee'] / deno) * (ssw + nsip.ger['ee'] / deno) + \
            0.5 * (np.real(nsip.gel['em'] / deno) * np.real(nsip.ger['em'] / deno) + np.imag(nsip.gel['em'] / deno) * np.imag(nsip.ger['em'] / deno)) + \
            0.5 * (np.real(nsip.gel['et'] / deno) * np.real(nsip.ger['et'] / deno) + np.imag(nsip.gel['et'] / deno) * np.imag(nsip.ger['et'] / deno))
    elif flavor[0] == 'm':
        epls = (-0.5 + ssw + nsip.gel['mm'] / deno) ** 2 + (nsip.gel['em'] / deno) ** 2 + (nsip.gel['mt'] / deno) ** 2
        eprs = (ssw + nsip.ger['mm'] / deno) ** 2 + (nsip.ger['em'] / deno) ** 2 + (nsip.ger['mt'] / deno) ** 2
        eplr = (-0.5 + ssw + nsip.gel['mm'] / deno) * (ssw + nsip.ger['mm'] / deno) + \
            0.5 * (np.real(nsip.gel['em'] / deno) * np.real(nsip.ger['em'] / deno) + np.imag(nsip.gel['em'] / deno) * np.imag(nsip.ger['em'] / deno)) + \
            0.5 * (np.real(nsip.gel['mt'] / deno) * np.real(nsip.ger['mt'] / deno) + np.imag(nsip.gel['mt'] / deno) * np.imag(nsip.ger['mt'] / deno))
    elif flavor[0] == 't':
        epls = (-0.5 + ssw + nsip.gel['tt'] / deno) ** 2 + (nsip.gel['mt'] / deno) ** 2 + (nsip.gel['et'] / deno) ** 2
        eprs = (ssw + nsip.ger['tt'] / deno) ** 2 + (nsip.ger['mt'] / deno) ** 2 + (nsip.ger['et'] / deno) ** 2
        eplr = (-0.5 + ssw + nsip.gel['tt'] / deno) * (ssw + nsip.ger['tt'] / deno) + \
            0.5 * (np.real(nsip.gel['et'] / deno) * np.real(nsip.ger['et'] / deno) + np.imag(nsip.gel['et'] / deno) * np.imag(nsip.ger['et'] / deno)) + \
            0.5 * (np.real(nsip.gel['mt'] / deno) * np.real(nsip.ger['mt'] / deno) + np.imag(nsip.gel['mt'] / deno) * np.imag(nsip.ger['mt'] / deno))
    else:
        raise Exception('No such neutrino flavor!')
    if flavor[-1] == 'r':
        temp = epls
        epls = eprs
        eprs = temp
    if efficiency is not None:
        return np.dot(2 / np.pi * (gf ** 2) * me * det.z *
                   (epls * fx.fint(er, me, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs) +
                    eprs * (fx.fint(er, me, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs) -
                            2 * er * fx.fintinv(er, me, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs) +
                            (er ** 2) * fx.fintinvs(er, me, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs)) -
                    eplr * me * er * fx.fintinvs(er, me, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs)), det.frac) * efficiency(er)
    else:
        return np.dot(2 / np.pi * (gf ** 2) * me * det.z *
                   (epls * fx.fint(er, me, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs) +
                    eprs * (fx.fint(er, me, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs) -
                            2 * er * fx.fintinv(er, me, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs) +
                            (er ** 2) * fx.fintinvs(er, me, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs)) -
                    eplr * me * er * fx.fintinvs(er, me, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs)), det.frac)


def binned_events_nucleus(era, erb, expo, det: Detector, fx: Flux, nsip: NSIparameters, efficiency=None, f=None,
                          flavor='e', op=oscillation_parameters(), q2=False, **kwargs):
    """
    :return: number of nucleus recoil events in the bin [era, erb]
    """
    def rates(er):
        return rates_nucleus(er, det, fx, efficiency=efficiency, f=f, nsip=nsip, flavor=flavor, op=op, q2=q2, **kwargs)
    return quad(rates, era, erb)[0] * \
        expo * mev_per_kg * 24 * 60 * 60 / np.dot(det.m, det.frac)


def binned_events_electron(era, erb, expo, det: Detector, fx: Flux, nsip: NSIparameters, efficiency=None, f=None,
                           flavor='e', op=oscillation_parameters(), **kwargs):
    """
    :return: number of electron recoil events in the bin [era, erb]
    """
    def rates(er):
        return rates_electron(er, det, fx, efficiency=efficiency, f=f, nsip=nsip, flavor=flavor, op=op, **kwargs)
    return quad(rates, era, erb)[0] * \
        expo * mev_per_kg * 24 * 60 * 60 / np.dot(det.m, det.frac)


class NSIEventsGen(EventGen):
    def __init__(self, flux: Flux, detector: Detector, expo: float, target='nucleus', nsi_param=NSIparameters(),
                 osci_param=oscillation_parameters(), osci_func=None, formfactsq=formfsquared, q2form=False, efficiency=None):
        self.flux = flux
        self.detector = detector
        self.expo = expo
        self.target = target
        self.nsi_param = nsi_param
        self.osci_param = osci_param
        self.formfactsq = formfactsq
        self.q2form = q2form
        self.efficiency = efficiency
        self.osci_func = osci_func

    def rates(self, er, flavor='e', **kwargs):
        if self.target == 'nucleus':
            return rates_nucleus(er, self.detector, self.flux, efficiency=self.efficiency, f=self.osci_func,
                                 nsip=self.nsi_param, flavor=flavor, op=self.osci_param, ffs=self.formfactsq, q2=self.q2form, **kwargs)
        elif self.target == 'electron':
            return rates_electron(er, self.detector, self.flux, efficiency=self.efficiency, f=self.osci_func,
                                  nsip=self.nsi_param, flavor=flavor, op=self.osci_param, **kwargs)
        else:
            raise Exception('Target should be either nucleus or electron!')

    def events(self, ea, eb, flavor='e', **kwargs):
        if self.target == 'nucleus':
            return binned_events_nucleus(ea, eb, self.expo, self.detector, self.flux, nsip=self.nsi_param, flavor=flavor,
                                         efficiency=self.efficiency, f=self.osci_func, op=self.osci_param, q2=self.q2form, **kwargs)
        elif self.target == 'electron':
            return binned_events_electron(ea, eb, self.expo, self.detector, self.flux, nsip=self.nsi_param,
                                          flavor=flavor, op=self.osci_param, efficiency=self.efficiency, **kwargs)
        else:
            return Exception('Target should be either nucleus or electron!')


