"""
CEvNS events
"""

from scipy.special import spherical_jn, erf

from .detectors import *  # pylint: disable=W0401, W0614, W0622
from .flux import *  # pylint: disable=W0401, W0614, W0622


def formfsquared(q, a):
    """
    form factor squared
    Engel, 1991
    :param q: momentum transfered
    :param a: number of nucleons
    :return: form factor squared
    """
    r = 1.2 * (10 ** -15) * (a ** (1 / 3)) / meter_by_mev
    s = 0.5 * (10 ** -15) / meter_by_mev
    r0 = sqrt(r ** 2 - 5 * (s ** 2))
    return (3 * spherical_jn(1, q * r0) / (q * r0) * exp((-(q * s) ** 2) / 2)) ** 2


def eff_coherent(er):
    return 0.331 * (1 + erf(0.248 * (er * 1e3 - 9.22)))


def rates_nucleus(er, det: Detector, fx: Flux, efficiency=None, f=None, nsip=NSIparameters(), flavor='e',
                  op=oscillation_parameters(), **kwargs):
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
    :return: scattering rates per nucleus
    """
    deno = 2 * sqrt(2) * gf * (2 * det.m * er + nsip.mz ** 2)
    # radiative corrections,
    # Barranco, 2005
    # is it necessary?
    rho = 1.0086
    knu = 0.9978
    lul = -0.0031
    ldl = -0.0025
    ldr = 7.5e-5
    lur = ldr / 2
    if nsip.mz != 0:
        if flavor[0] == 'e':
            qvs = (0.5 * det.z * (rho * (0.5 - 2 * knu * ssw) + 2 * lul + 2 * lur + ldl + ldr +
                                  2 * nsip.gu['ee'] / deno + nsip.gd['ee'] / deno) +
                   0.5 * det.n * (-0.5 * rho + lul + lur + 2 * ldl + 2 * ldr +
                                  nsip.gu['ee'] / deno + 2 * nsip.gd['ee'] / deno)) ** 2 + \
                  abs(0.5 * det.z * (2 * nsip.gu['em'] / deno + nsip.gd['em'] / deno) +
                      0.5 * det.n * (nsip.gu['em'] / deno + 2 * nsip.gd['em'] / deno)) ** 2 + \
                  abs(0.5 * det.z * (2 * nsip.gu['et'] / deno + nsip.gd['et'] / deno) +
                      0.5 * det.n * (nsip.gu['et'] / deno + 2 * nsip.gd['et'] / deno)) ** 2
        elif flavor[0] == 'm':
            qvs = (0.5 * det.z * (rho * (0.5 - 2 * knu * ssw) + 2 * lul + 2 * lur + ldl + ldr +
                                  2 * nsip.gu['mm'] / deno + nsip.gd['mm'] / deno) +
                   0.5 * det.n * (-0.5 * rho + lul + lur + 2 * ldl + 2 * ldr +
                                  nsip.gu['mm'] / deno + 2 * nsip.gd['mm'] / deno)) ** 2 + \
                  abs(0.5 * det.z * (2 * nsip.gu['em'] / deno + nsip.gd['em'] / deno) +
                      0.5 * det.n * (nsip.gu['em'] / deno + 2 * nsip.gd['em'] / deno)) ** 2 + \
                  abs(0.5 * det.z * (2 * nsip.gu['mt'] / deno + nsip.gd['mt'] / deno) +
                      0.5 * det.n * (nsip.gu['mt'] / deno + 2 * nsip.gd['mt'] / deno)) ** 2
        elif flavor[0] == 't':
            qvs = (0.5 * det.z * (rho * (0.5 - 2 * knu * ssw) + 2 * lul + 2 * lur + ldl + ldr +
                                  2 * nsip.gu['tt'] / deno + nsip.gd['tt'] / deno) +
                   0.5 * det.n * (-0.5 * rho + lul + lur + 2 * ldl + 2 * ldr +
                                  nsip.gu['tt'] / deno + 2 * nsip.gd['tt'] / deno)) ** 2 + \
                  abs(0.5 * det.z * (2 * nsip.gu['et'] / deno + nsip.gd['et'] / deno) +
                      0.5 * det.n * (nsip.gu['et'] / deno + 2 * nsip.gd['et'] / deno)) ** 2 + \
                  abs(0.5 * det.z * (2 * nsip.gu['mt'] / deno + nsip.gd['mt'] / deno) +
                      0.5 * det.n * (nsip.gu['mt'] / deno + 2 * nsip.gd['mt'] / deno)) ** 2
        else:
            raise Exception('No such neutrino flavor!')
    else:
        if flavor[0] == 'e':
            qvs = (0.5 * det.z * (rho * (0.5 - 2 * knu * ssw) + 2 * lul + 2 * lur + ldl + ldr +
                                  2 * nsip.epu['ee'] + nsip.epd['ee']) +
                   0.5 * det.n * (-0.5 * rho + lul + lur + 2 * ldl + 2 * ldr +
                                  nsip.epu['ee'] + 2 * nsip.epd['ee'])) ** 2 + \
                  abs(0.5 * det.z * (2 * nsip.epu['em'] + nsip.epd['em']) +
                      0.5 * det.n * (nsip.epu['em'] + 2 * nsip.epd['em'])) ** 2 + \
                  abs(0.5 * det.z * (2 * nsip.epu['et'] + nsip.epd['et']) +
                      0.5 * det.n * (nsip.epu['et'] + 2 * nsip.epd['et'])) ** 2
        elif flavor[0] == 'm':
            qvs = (0.5 * det.z * (rho * (0.5 - 2 * knu * ssw) + 2 * lul + 2 * lur + ldl + ldr +
                                  2 * nsip.epu['mm'] + nsip.epd['mm']) +
                   0.5 * det.n * (-0.5 * rho + lul + lur + 2 * ldl + 2 * ldr +
                                  nsip.epu['mm'] + 2 * nsip.epd['mm'])) ** 2 + \
                  abs(0.5 * det.z * (2 * nsip.epu['em'] + nsip.epd['em']) +
                      0.5 * det.n * (nsip.epu['em'] + 2 * nsip.epd['em'])) ** 2 + \
                  abs(0.5 * det.z * (2 * nsip.epu['mt'] + nsip.epd['mt']) +
                      0.5 * det.n * (nsip.epu['mt'] + 2 * nsip.epd['mt'])) ** 2
        elif flavor[0] == 't':
            qvs = (0.5 * det.z * (rho * (0.5 - 2 * knu * ssw) + 2 * lul + 2 * lur + ldl + ldr +
                                  2 * nsip.epu['tt'] + nsip.epd['tt']) +
                   0.5 * det.n * (-0.5 * rho + lul + lur + 2 * ldl + 2 * ldr +
                                  nsip.epu['tt'] + 2 * nsip.epd['tt'])) ** 2 + \
                  abs(0.5 * det.z * (2 * nsip.epu['et'] + nsip.epd['et']) +
                      0.5 * det.n * (nsip.epu['et'] + 2 * nsip.epd['et'])) ** 2 + \
                  abs(0.5 * det.z * (2 * nsip.epu['mt'] + nsip.epd['mt']) +
                      0.5 * det.n * (nsip.epu['mt'] + 2 * nsip.epd['mt'])) ** 2
        else:
            raise Exception('No such neutrino flavor!')
    if efficiency is not None:
        return dot(2 / pi * (gf ** 2) * (2 * fx.fint(er, det.m, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs) -
                                         2 * er * fx.fintinv(er, det.m, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs) +
                                         er * er * fx.fintinvs(er, det.m, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs) -
                                         det.m * er * fx.fintinvs(er, det.m, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs)) *
                   det.m * qvs * formfsquared(sqrt(2 * det.m * er), det.z + det.n), det.frac) * efficiency(er)
    else:
        return dot(2 / pi * (gf ** 2) * (2 * fx.fint(er, det.m, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs) -
                                         2 * er * fx.fintinv(er, det.m, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs) +
                                         er * er * fx.fintinvs(er, det.m, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs) -
                                         det.m * er * fx.fintinvs(er, det.m, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs)) *
                   det.m * qvs * formfsquared(sqrt(2 * det.m * er), det.z + det.n), det.frac)


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
    deno = 2 * sqrt(2) * gf * (2 * me * er + nsip.mz ** 2)
    if flavor[0] == 'e':
        epls = (0.5 + ssw + nsip.gel['ee'] / deno) ** 2 + (nsip.gel['em'] / deno) ** 2 + (nsip.gel['et'] / deno) ** 2
        eprs = (ssw + nsip.ger['ee'] / deno) ** 2 + (nsip.ger['em'] / deno) ** 2 + (nsip.ger['et'] / deno) ** 2
        eplr = (0.5 + ssw + nsip.gel['ee'] / deno) * (ssw + nsip.ger['ee'] / deno) + \
            0.5 * (real(nsip.gel['em'] / deno) * real(nsip.ger['em'] / deno) + imag(nsip.gel['em'] / deno) * imag(nsip.ger['em'] / deno)) + \
            0.5 * (real(nsip.gel['et'] / deno) * real(nsip.ger['et'] / deno) + imag(nsip.gel['et'] / deno) * imag(nsip.ger['et'] / deno))
    elif flavor[0] == 'm':
        epls = (-0.5 + ssw + nsip.gel['mm'] / deno) ** 2 + (nsip.gel['em'] / deno) ** 2 + (nsip.gel['mt'] / deno) ** 2
        eprs = (ssw + nsip.ger['mm'] / deno) ** 2 + (nsip.ger['em'] / deno) ** 2 + (nsip.ger['mt'] / deno) ** 2
        eplr = (-0.5 + ssw + nsip.gel['mm'] / deno) * (ssw + nsip.ger['mm'] / deno) + \
            0.5 * (real(nsip.gel['em'] / deno) * real(nsip.ger['em'] / deno) + imag(nsip.gel['em'] / deno) * imag(nsip.ger['em'] / deno)) + \
            0.5 * (real(nsip.gel['mt'] / deno) * real(nsip.ger['mt'] / deno) + imag(nsip.gel['mt'] / deno) * imag(nsip.ger['mt'] / deno))
    elif flavor[0] == 't':
        epls = (-0.5 + ssw + nsip.gel['tt'] / deno) ** 2 + (nsip.gel['mt'] / deno) ** 2 + (nsip.gel['et'] / deno) ** 2
        eprs = (ssw + nsip.ger['tt'] / deno) ** 2 + (nsip.ger['mt'] / deno) ** 2 + (nsip.ger['et'] / deno) ** 2
        eplr = (-0.5 + ssw + nsip.gel['tt'] / deno) * (ssw + nsip.ger['tt'] / deno) + \
            0.5 * (real(nsip.gel['et'] / deno) * real(nsip.ger['et'] / deno) + imag(nsip.gel['et'] / deno) * imag(nsip.ger['et'] / deno)) + \
            0.5 * (real(nsip.gel['mt'] / deno) * real(nsip.ger['mt'] / deno) + imag(nsip.gel['mt'] / deno) * imag(nsip.ger['mt'] / deno))
    else:
        raise Exception('No such neutrino flavor!')
    if flavor[-1] == 'r':
        temp = epls
        epls = eprs
        eprs = temp
    if efficiency is not None:
        return dot(2 / pi * (gf ** 2) * me * det.z *
                   (epls * fx.fint(er, me, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs) +
                    eprs * (fx.fint(er, me, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs) -
                            2 * er * fx.fintinv(er, me, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs) +
                            (er ** 2) * fx.fintinvs(er, me, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs)) -
                    eplr * me * er * fx.fintinvs(er, me, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs)), det.frac) * efficiency(er)
    else:
        return dot(2 / pi * (gf ** 2) * me * det.z *
                   (epls * fx.fint(er, me, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs) +
                    eprs * (fx.fint(er, me, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs) -
                            2 * er * fx.fintinv(er, me, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs) +
                            (er ** 2) * fx.fintinvs(er, me, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs)) -
                    eplr * me * er * fx.fintinvs(er, me, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs)), det.frac)


def binned_events_nucleus(era, erb, expo, det: Detector, fx: Flux, nsip: NSIparameters, efficiency=None, f=None,
                          flavor='e', op=oscillation_parameters(), **kwargs):
    """
    :return: number of nucleus recoil events in the bin [era, erb]
    """
    def rates(er):
        return rates_nucleus(er, det, fx, efficiency=efficiency, f=f, nsip=nsip, flavor=flavor, op=op, **kwargs)
    return quad(rates, era, erb)[0] * \
        expo * mev_per_kg * 24 * 60 * 60 / dot(det.m, det.frac)


def binned_events_electron(era, erb, expo, det: Detector, fx: Flux, nsip: NSIparameters, efficiency=None, f=None,
                           flavor='e', op=oscillation_parameters(), **kwargs):
    """
    :return: number of electron recoil events in the bin [era, erb]
    """
    def rates(er):
        return rates_electron(er, det, fx, efficiency=efficiency, f=f, nsip=nsip, flavor=flavor, op=op, **kwargs)
    return quad(rates, era, erb)[0] * \
        expo * mev_per_kg * 24 * 60 * 60 / dot(det.m, det.frac)
