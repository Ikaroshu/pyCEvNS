"""
CEvNS events
"""

from scipy.special import spherical_jn
from .flux import *    # pylint: disable=W0401, W0614, W0622
from .detectors import *    # pylint: disable=W0401, W0614, W0622


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


def rates_nucleus(er, det: Detector, fx: Flux, nsip: NSIparameters, flavor='e', op=None, r=0.05):
    """
    calculating scattering rates per nucleus
    :param er: recoil energy
    :param det: detector
    :param fx: flux
    :param flavor: flux flavor
    :param nsip: nsi parameters
    :param op: oscillation parameters
    :param r: position where neutrino is produced in the sun, for solar neutrino only
    :return: scattering rates per nucleus
    """
    if op is None:
        op = ocsillation_parameters()
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
    if flavor == 'e':
        qvs = (0.5 * det.z * (rho * (0.5 - 2 * knu * ssw) + 2 * lul + 2 * lur + ldl + ldr +
                              2 * nsip.gu['ee'] / deno + nsip.gd['ee'] / deno) +
               0.5 * det.n * (-0.5 * rho + lul + lur + 2 * ldl + 2 * ldr +
                              nsip.gu['ee'] / deno + 2 * nsip.gd['ee'] / deno)) ** 2 + \
              (0.5 * det.z * (2 * nsip.gu['em'] / deno + nsip.gd['em'] / deno) +
               0.5 * det.n * (nsip.gu['em'] / deno + 2 * nsip.gd['em'] / deno)) ** 2 + \
              (0.5 * det.z * (2 * nsip.gu['et'] / deno + nsip.gd['et'] / deno) +
               0.5 * det.n * (nsip.gu['et'] / deno + 2 * nsip.gd['et'] / deno)) ** 2
    elif flavor == 'mu':
        qvs = (0.5 * det.z * (rho * (0.5 - 2 * knu * ssw) + 2 * lul + 2 * lur + ldl + ldr +
                              2 * nsip.gu['mm'] / deno + nsip.gd['mm'] / deno) +
               0.5 * det.n * (-0.5 * rho + lul + lur + 2 * ldl + 2 * ldr +
                              nsip.gu['mm'] / deno + 2 * nsip.gd['mm'] / deno)) ** 2 + \
              (0.5 * det.z * (2 * nsip.gu['em'] / deno + nsip.gd['em'] / deno) +
               0.5 * det.n * (nsip.gu['em'] / deno + 2 * nsip.gd['em'] / deno)) ** 2 + \
              (0.5 * det.z * (2 * nsip.gu['mt'] / deno + nsip.gd['mt'] / deno) +
               0.5 * det.n * (nsip.gu['mt'] / deno + 2 * nsip.gd['mt'] / deno)) ** 2
    elif flavor == 'tau':
        qvs = (0.5 * det.z * (rho * (0.5 - 2 * knu * ssw) + 2 * lul + 2 * lur + ldl + ldr +
                              2 * nsip.gu['tt'] / deno + nsip.gd['tt'] / deno) +
               0.5 * det.n * (-0.5 * rho + lul + lur + 2 * ldl + 2 * ldr +
                              nsip.gu['tt'] / deno + 2 * nsip.gd['tt'] / deno)) ** 2 + \
              (0.5 * det.z * (2 * nsip.gu['et'] / deno + nsip.gd['et'] / deno) +
               0.5 * det.n * (nsip.gu['et'] / deno + 2 * nsip.gd['et'] / deno)) ** 2 + \
              (0.5 * det.z * (2 * nsip.gu['mt'] / deno + nsip.gd['mt'] / deno) +
               0.5 * det.n * (nsip.gu['mt'] / deno + 2 * nsip.gd['mt'] / deno)) ** 2
    else:
        raise Exception('No such neutrino flavor!')
    return dot(2 / pi * (gf ** 2) * (2 * fx.fint(er, det.m, flavor, epsi=nsip, op=op, r=r) -
                                     2 * er * fx.fintinv(er, det.m, flavor, epsi=nsip, op=op, r=r) +
                                     er * er * fx.fintinvs(er, det.m, flavor, epsi=nsip, op=op, r=r) -
                                     det.m * er * fx.fintinvs(er, det.m, flavor, epsi=nsip, op=op, r=r)) *
               det.m * qvs * formfsquared(sqrt(2 * det.m * er), det.z + det.n), det.frac)


def rates_electron(er, det: Detector, fx: Flux, nsip: NSIparameters, flavor='e', op=None, r=0.05):
    """
    calculating electron scattering rates per nucleus
    :param er: recoil energy
    :param det: detector
    :param fx: flux
    :param flavor: flux flavor
    :param nsip: nsi parameters
    :param op: oscillation parameters
    :param r: position where neutrino is produced in the sun, for solar neutrino only
    :return: scattering rates per nucleus
    """
    if op is None:
        op = ocsillation_parameters()
    deno = 2 * sqrt(2) * gf * (2 * me * er + nsip.mz ** 2)
    if flavor == 'e':
        epls = (0.5 + ssw + nsip.gel['ee'] / deno) ** 2 + (nsip.gel['em'] / deno) ** 2 + (nsip.gel['et'] / deno) ** 2
        eprs = (ssw + nsip.ger['ee'] / deno) ** 2 + (nsip.ger['em'] / deno) ** 2 + (nsip.ger['et'] / deno) ** 2
        eplr = (1 + (-0.5 + ssw) + nsip.gel['ee'] / deno) * (ssw + nsip.ger['ee'] / deno) + \
               (nsip.gel['em'] / deno) * (nsip.ger['em'] / deno) + \
               (nsip.gel['et'] / deno) * (nsip.ger['et'] / deno)
    elif flavor == 'mu':
        epls = (-0.5 + ssw + nsip.gel['mm'] / deno) ** 2 + (nsip.gel['em'] / deno) ** 2 + (nsip.gel['mt'] / deno) ** 2
        eprs = (ssw + nsip.ger['mm'] / deno) ** 2 + (nsip.ger['em'] / deno) ** 2 + (nsip.ger['mt'] / deno) ** 2
        eplr = (1 + (-0.5 + ssw) + nsip.gel['mm'] / deno) * (ssw + nsip.ger['mm'] / deno) + \
               (nsip.gel['em'] / deno) * (nsip.ger['em'] / deno) + \
               (nsip.gel['mt'] / deno) * (nsip.ger['mt'] / deno)
    elif flavor == 'tau':
        epls = (-0.5 + ssw + nsip.gel['tt'] / deno) ** 2 + (nsip.gel['mt'] / deno) ** 2 + (nsip.gel['et'] / deno) ** 2
        eprs = (ssw + nsip.ger['tt'] / deno) ** 2 + (nsip.ger['mt'] / deno) ** 2 + (nsip.ger['et'] / deno) ** 2
        eplr = (1 + (-0.5 + ssw) + nsip.gel['tt'] / deno) * (ssw + nsip.ger['tt'] / deno) + \
               (nsip.gel['mt'] / deno) * (nsip.ger['mt'] / deno) + \
               (nsip.gel['et'] / deno) * (nsip.ger['et'] / deno)
    else:
        raise Exception('No such neutrino flavor!')
    return dot(2 / pi * (gf ** 2) * me * det.z *
               (epls * fx.fint(er, me, flavor, epsi=nsip, op=op, r=r) +
                eprs * (fx.fint(er, me, flavor, epsi=nsip, op=op, r=r) -
                        2 * er * fx.fintinv(er, me, flavor, epsi=nsip, op=op, r=r) +
                        (er ** 2) * fx.fintinvs(er, me, flavor, epsi=nsip, op=op, r=r)) -
                eplr * me * er * fx.fintinvs(er, me, flavor, epsi=nsip, op=op, r=r)), det.frac)


def binned_events_nucleus(era, erb, expo, det: Detector, fx: Flux, nsip: NSIparameters, flavor='e', op=None, r=0.05):
    """
    :return: number of nucleus recoil events in the bin [era, erb]
    """
    return quad(rates_nucleus, era, erb, args=(det, fx, nsip, flavor, op, r))[0] * \
           expo * mev_per_kg * 24 * 60 * 60 / dot(det.m, det.frac)


def binned_events_electron(era, erb, expo, det: Detector, fx: Flux, nsip: NSIparameters, flavor='e', op=None, r=0.05):
    """
    :return: number of electron recoil events in the bin [era, erb]
    """
    return quad(rates_electron, era, erb, args=(det, fx, nsip, flavor, op, r))[0] * \
           expo * mev_per_kg * 24 * 60 * 60 / dot(det.m, det.frac)