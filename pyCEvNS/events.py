"""
CEvNS events
"""

from scipy.special import spherical_jn

from .detectors import *
from .flux import *
from .helper import _poisson


def formfsquared(q, rn=5.5, **kwargs):
    """
    form factor squared
    1810.05606
    :param q: momentum transfered
    :param rn: neutron skin radius
    :param kwargs: this is for legacy compatibility
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


class NSIEventsGen:
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


def rates_dm(er, det: Detector, fx: DMFlux, mediator_mass=None, epsilon=None, efficiency=None, smear=False,  **kwargs):
    """
    calculating dark matter scattering rates per nucleus
    :param er: recoil energy in MeV
    :param det: detector
    :param fx: dark matter flux
    :param mediator_mass: mediator mass in MeV
    :param epsilon: mediator to quark coupling multiply by mediator to dark matter coupling
    :param efficiency: efficiency function
    :return: dark matter scattering rates per nucleus
    """
    if mediator_mass is None:
        mediator_mass = fx.dp_mass
    if epsilon is None:
        epsilon = fx.epsi_quark

    def rates(err):
        if efficiency is not None:
            return np.dot(det.frac, e_charge**4 * epsilon**2 * det.z**2 *
                      (2*det.m*fx.fint2(err, det.m) -
                       (err+(det.m**2*err-fx.dm_mass**2*err)/(2*det.m))*2*det.m*fx.fint(err, det.m) +
                       err**2*det.m*fx.fint(err, det.m)) / (4*np.pi*(2*det.m*err+mediator_mass**2)**2) *
                      formfsquared(np.sqrt(2*err*det.m), **kwargs)) * efficiency(err)
        else:
            return np.dot(det.frac, e_charge ** 4 * epsilon ** 2 * det.z ** 2 *
                          (2 * det.m * fx.fint2(err, det.m) -
                           (err + (det.m ** 2 * err - fx.dm_mass ** 2 * err) / (2 * det.m)) * 2 * det.m * fx.fint(err, det.m) +
                           err**2 * det.m * fx.fint(err, det.m)) / (4 * np.pi * (2 * det.m * err + mediator_mass**2)**2) *
                          formfsquared(np.sqrt(2 * err * det.m), **kwargs))

    if not smear:
        return rates(er)
    else:
        def func(pep):
            pe_per_mev = 0.0878 * 13.348 * 1000
            return rates(pep/pe_per_mev) * _poisson(er*pe_per_mev, pep)
        return quad(func, 0, 60)[0]


def binned_events_dm(era, erb, expo, det: Detector, fx: DMFlux, mediator_mass=None, epsilon=None, efficiency=None, smear=False, **kwargs):
    """
    :return: number of nucleus recoil events in the bin [era, erb]
    """
    def rates(er):
        return rates_dm(er, det, fx, mediator_mass, epsilon, efficiency, smear, **kwargs)
    return quad(rates, era, erb)[0] * expo * mev_per_kg * 24 * 60 * 60 / np.dot(det.m, det.frac)


class DmEventsGen:
    """
    Dark matter events generator for COHERENT
    """
    def __init__(self, dark_photon_mass, life_time, dark_matter_mass, expo=4466, detector_type='csi',
                 detector_distance=19.3, pot_mu=0.75, pot_sigma=0.25, size=100000, smear=False, rn=5.5):
        self.dp_mass = dark_photon_mass
        self.tau = life_time
        self.dm_mass = dark_matter_mass
        self.det_dist = detector_distance
        self.mu = pot_mu
        self.sigma = pot_sigma
        self.size = size
        self.det = Detector(detector_type)
        self.fx = None
        self.expo = expo
        self.smear = smear
        self.rn = rn
        self.generate_flux()

    def generate_flux(self):
        self.fx = DMFlux(self.dp_mass, self.tau, 1, self.dm_mass, self.det_dist, self.mu, self.sigma, self.size)

    def set_dark_photon_mass(self, dark_photon_mass):
        self.dp_mass = dark_photon_mass
        self.generate_flux()

    def set_life_time(self, life_time):
        self.tau = life_time
        self.generate_flux()

    def set_dark_matter_mass(self, dark_matter_mass):
        self.dm_mass = dark_matter_mass
        self.generate_flux()

    def set_detector_distance(self, detector_distance):
        self.det_dist = detector_distance
        self.generate_flux()

    def set_pot_mu(self, pot_mu):
        self.mu = pot_mu
        self.generate_flux()

    def set_pot_sigma(self, pot_sigma):
        self.sigma = pot_sigma
        self.generate_flux()

    def set_size(self, size):
        self.size = size
        self.generate_flux()

    def events(self, mediator_mass, epsilon, n_meas):
        """
        generate events according to the time and energy in measured data
        :param mediator_mass: mediator mass
        :param epsilon: mediator coupling to quark multiply by mediator coupling to dark matter
        :param n_meas: measured data
        :return: predicted number of event according to the time and energy in the measured data
        """
        pe_per_mev = 0.0878 * 13.348 * 1000
        n_dm = np.zeros(n_meas.shape[0])
        tmin = n_meas[:, 1].min()
        plist = np.zeros(int((n_meas[:, 1].max()-tmin)/0.5)+1)
        for tt in self.fx.timing:
            if int((tt-tmin+0.25)/0.5) < plist.shape[0]:
                plist[int((tt-tmin+0.25)/0.5)] += 1
        plist /= self.fx.timing.shape[0]
        for i in range(n_meas.shape[0]):
            pe = n_meas[i, 0]
            t = n_meas[i, 1]
            n_dm[i] = binned_events_dm((pe - 1)/pe_per_mev, (pe + 1)/pe_per_mev, self.expo,
                                       self.det, self.fx, mediator_mass, epsilon, eff_coherent, self.smear, rn=self.rn) * plist[int((t-tmin)/0.5)]
        return n_dm


class HelmFormFactor:
    """
    square of the form factor!
    """
    def __init__(self, rn=4.7):
        self.rn = rn

    def __call__(self, q):
        r = self.rn * (10 ** -15) / meter_by_mev
        s = 0.9 * (10 ** -15) / meter_by_mev
        r0 = np.sqrt(5 / 3 * (r ** 2) - 5 * (s ** 2))
        return (3 * spherical_jn(1, q * r0) / (q * r0) * np.exp((-(q * s) ** 2) / 2)) ** 2

    def change_parameters(self, rn=None):
        self.rn = rn if rn is not None else self.rn


def _inv(ev):
    return 1/ev


def _invs(ev):
    return 1/ev**2


class NeutrinoNucleusElasticVector:
    def __init__(self, nsi_parameters: NSIparameters, form_factor_square=HelmFormFactor()):
        self.nsi_parameters = nsi_parameters
        self.form_factor_square = form_factor_square

    def rates(self, er, flavor, flux: NeutrinoFlux, detector: Detector):
        rho = 1.0086
        knu = 0.9978
        lul = -0.0031
        ldl = -0.0025
        ldr = 7.5e-5
        lur = ldr / 2
        epu = self.nsi_parameters.eu()
        epd = self.nsi_parameters.ed()
        scale = 1
        if self.nsi_parameters.mz != 0:
            scale = self.nsi_parameters.mz**2 / (self.nsi_parameters.mz**2 + 2*detector.m*er)
        qvs = 0
        if flavor[0] == 'e':
            qvs = (0.5 * detector.z * (rho*(0.5 - 2*knu*ssw)+2*lul+2*lur+ldl+ldr+2*epu[0, 0]*scale+epd[0, 0]*scale) +
                   0.5*detector.n*(-0.5*rho + lul + lur + 2*ldl + 2*ldr + epu[0, 0]*scale + 2*epd[0, 0]*scale)) ** 2 + \
                np.abs(0.5*detector.z*(2*epu[0, 1]*scale + epd[0, 1]*scale) + 0.5*detector.n*(epu[0, 1]*scale + 2*epd[0, 1]*scale)) ** 2 + \
                np.abs(0.5*detector.z*(2*epu[0, 2]*scale + epd[0, 2]*scale) + 0.5*detector.n*(epu[0, 2]*scale + 2*epd[0, 2]*scale)) ** 2
        if flavor[0] == 'm':
            qvs = (0.5 * detector.z * (rho*(0.5 - 2*knu*ssw)+2*lul+2*lur+ldl+ldr+2*epu[1, 1]*scale+epd[1, 1]*scale) +
                   0.5*detector.n*(-0.5*rho + lul + lur + 2*ldl + 2*ldr + epu[1, 1]*scale + 2*epd[1, 1]*scale)) ** 2 + \
                np.abs(0.5*detector.z*(2*epu[1, 0]*scale + epd[1, 0]*scale) + 0.5*detector.n*(epu[1, 0]*scale + 2*epd[1, 0]*scale)) ** 2 + \
                np.abs(0.5*detector.z*(2*epu[1, 2]*scale + epd[1, 2]*scale) + 0.5*detector.n*(epu[1, 2]*scale + 2*epd[1, 2]*scale)) ** 2
        if flavor[0] == 't':
            qvs = (0.5 * detector.z * (rho*(0.5 - 2*knu*ssw)+2*lul+2*lur+ldl+ldr+2*epu[2, 2]*scale+epd[2, 2]*scale) +
                   0.5*detector.n*(-0.5*rho + lul + lur + 2*ldl + 2*ldr + epu[2, 2]*scale + 2*epd[2, 2]*scale)) ** 2 + \
                np.abs(0.5*detector.z*(2*epu[2, 0]*scale + epd[2, 0]*scale) + 0.5*detector.n*(epu[2, 0]*scale + 2*epd[2, 0]*scale)) ** 2 + \
                np.abs(0.5*detector.z*(2*epu[2, 1]*scale + epd[2, 1]*scale) + 0.5*detector.n*(epu[2, 1]*scale + 2*epd[2, 1]*scale)) ** 2
        fint = np.zeros(detector.iso)
        fintinv = np.zeros(detector.iso)
        fintinvs = np.zeros(detector.iso)
        emin = 0.5 * (np.sqrt(er ** 2 + 2 * er * detector.m) + er)
        for i in range(detector.iso):
            fint[i] = flux.integrate(emin[i], flux.ev_max, flavor)
            fintinv[i] = flux.integrate(emin[i], flux.ev_max, flavor, weight_function=_inv)
            fintinvs[i] = flux.integrate(emin[i], flux.ev_max, flavor, weight_function=_invs)
        res = np.dot(2 / np.pi * (gf ** 2) * (2 * fint - 2 * er * fintinv + er * er * fintinvs - detector.m * er * fintinvs) *
                     detector.m * qvs * self.form_factor_square(np.sqrt(2 * detector.m * er)), detector.frac)
        if detector.detectoin_efficiency is not None:
            res *= detector.detectoin_efficiency(er)
        return res

    def events(self, ea, eb, flavor, flux: NeutrinoFlux, detector: Detector, exposure):
        def func(er):
            return self.rates(er, flavor, flux, detector)
        return quad(func, ea, eb)[0] * exposure * mev_per_kg * 24 * 60 * 60 / np.dot(detector.m, detector.frac)

    def change_parameters(self):
        pass


class NeutrinoElectronElasticVector:
    def __init__(self, nsi_parameters: NSIparameters):
        self.nsi_parameters = nsi_parameters

    def rates(self, er, flavor, flux: NeutrinoFlux, detector: Detector):
        epel = self.nsi_parameters.eel()
        eper = self.nsi_parameters.eer()
        scale = 1
        if self.nsi_parameters.mz != 0:
            scale = self.nsi_parameters.mz**2 / (self.nsi_parameters.mz**2 + 2*me*er)
        epls = 0
        eprs = 0
        eplr = 0
        if flavor[0] == 'e':
            epls = (0.5 + ssw + epel[0, 0] * scale) ** 2 + np.abs(epel[0, 1] * scale) ** 2 + np.abs(epel[0, 2] * scale) ** 2
            eprs = (ssw + eper[0, 0] * scale) ** 2 + np.abs(eper[0, 1] * scale) ** 2 + np.abs(eper[0, 2] * scale) ** 2
            eplr = (0.5 + ssw + epel[0, 0] * scale) * (ssw + eper[0, 0] * scale) + \
                0.5 * (np.real(epel[0, 1] * scale) * np.real(eper[0, 1] * scale) +
                       np.imag(epel[0, 1] * scale) * np.imag(eper[0, 1] * scale)) + \
                0.5 * (np.real(epel[0, 2] * scale) * np.real(eper[0, 2] * scale) +
                       np.imag(epel[0, 2] * scale) * np.imag(eper[0, 2] * scale))
        elif flavor[0] == 'm':
            epls = (-0.5 + ssw + epel[1, 1] * scale) ** 2 + np.abs(epel[1, 0] * scale) ** 2 + np.abs(epel[1, 2] * scale) ** 2
            eprs = (ssw + eper[1, 1] * scale) ** 2 + np.abs(eper[1, 0] * scale) ** 2 + np.abs(eper[1, 2] * scale) ** 2
            eplr = (-0.5 + ssw + epel[1, 1] * scale) * (ssw + eper[1, 1] * scale) + \
                0.5 * (np.real(epel[1, 0] * scale) * np.real(eper[1, 0] * scale) +
                       np.imag(epel[1, 0] * scale) * np.imag(eper[1, 0] * scale)) + \
                0.5 * (np.real(epel[1, 2] * scale) * np.real(eper[1, 2] * scale) +
                       np.imag(epel[1, 2] * scale) * np.imag(eper[1, 2] * scale))
        elif flavor[0] == 't':
            epls = (-0.5 + ssw + epel[2, 2] * scale) ** 2 + np.abs(epel[2, 1] * scale) ** 2 + np.abs(epel[2, 0] * scale) ** 2
            eprs = (ssw + eper[2, 2] * scale) ** 2 + np.abs(eper[2, 1] * scale) ** 2 + np.abs(eper[2, 0] * scale) ** 2
            eplr = (-0.5 + ssw + epel[2, 2] * scale) * (ssw + eper[2, 2] * scale) + \
                0.5 * (np.real(epel[2, 0] * scale) * np.real(eper[2, 0] * scale) +
                       np.imag(epel[2, 0] * scale) * np.imag(eper[2, 0] * scale)) + \
                0.5 * (np.real(epel[2, 1] * scale) * np.real(eper[2, 1] * scale) +
                       np.imag(epel[2, 1] * scale) * np.imag(eper[2, 1] * scale))
        emin = 0.5 * (np.sqrt(er ** 2 + 2 * er * me) + er)
        fint = flux.integrate(emin, flux.ev_max, flavor)
        fintinv = flux.integrate(emin, flux.ev_max, flavor, weight_function=_inv)
        fintinvs = flux.integrate(emin, flux.ev_max, flavor, weight_function=_invs)
        if flavor[-1] == 'r':
            tmp = epls
            epls = eprs
            eprs = tmp
        res = np.dot(2 / np.pi * (gf ** 2) * me * detector.z *
                     (epls * fint + eprs * (fint - 2 * er * fintinv + (er ** 2) * fintinvs) - eplr * me * er * fintinvs), detector.frac)
        if detector.detectoin_efficiency is not None:
            res *= detector.detectoin_efficiency(er)
        return res

    def events(self, ea, eb, flavor, flux: NeutrinoFlux, detector: Detector, exposure):
        def func(er):
            return self.rates(er, flavor, flux, detector)
        return quad(func, ea, eb)[0] * exposure * mev_per_kg * 24 * 60 * 60 / np.dot(detector.m, detector.frac)

    def change_parameters(self):
        pass


# Charged Current Quasi-Elastic (CCQE) cross-section, assuming no CC NSI. Follows Bodek, Budd, Christy [1106.0340].
class NeutrinoNucleonCCQE:
    def __init__(self, flavor, flux: NeutrinoFlux):
        self.flavor = flavor
        self.flux = flux
        self.FastXS = np.vectorize(self.rates)

    def rates(self, ev, flavor='e', masq=axial_mass**2):
        m_lepton = me
        m_nucleon = massofn
        xi = 4.706  # Difference between proton and neutron magnetic moments.
        sign = -1

        if flavor == "mu" or flavor == "mubar":
            m_lepton = mmu
        if flavor == "tau" or flavor == "taubar":
            m_lepton = mtau

        if flavor == "ebar" or flavor == "mubar" or flavor == "taubar":
            sign = 1
            m_nucleon = massofp

        def dsigma(qsq):
            tau = qsq / (4 * m_nucleon ** 2)
            GD = (1 / (1 + qsq / 710000) ** 2)  # Dipole form factor with vector mass.
            TE = np.sqrt(1 + (6e-6 * qsq) * np.exp(-qsq / 350000))  # Transverse Enhancement of the magnetic dipole.

            FA = -1.267 / (1 + (qsq / masq)) ** 2  # Axial form factor.
            Fp = (2 * FA * (m_nucleon) ** 2) / (massofpi0 ** 2 + qsq)  # Pion dipole form factor (only relevant for low ev).
            F1 = GD * ((1 + xi * tau * TE) / (1 + tau))  # First nuclear form factor in dipole approximation.
            F2 = GD * (xi * TE - 1) / (1 + tau)  # Second nuclear form factor in dipole approximation.

            # A, B, and C are the vector, pseudoscalar, and axial vector terms, respectively.
            A = ((m_lepton ** 2 + qsq) / m_nucleon ** 2) * (
                    (1 + tau) * FA ** 2 - (1 - tau) * F1 ** 2 + tau * (1 - tau) * (F2) ** 2 + 4 * tau * F1 * F2
                    - 0.25 * ((m_lepton / m_nucleon) ** 2) * ((F1 + F2) ** 2 + (FA + 2 * Fp) ** 2
                                                              - 4 * (tau + 1) * Fp ** 2))
            B = 4 * tau * (F1 + F2) * FA
            C = 0.25 * (FA ** 2 + F1 ** 2 + tau * (F2) ** 2)

            return ((1 / (8 * np.pi)) * (gf * cabibbo * m_nucleon / ev) ** 2) * \
                   (A + sign * B * ((4 * m_nucleon * ev - qsq - m_lepton ** 2) / (m_nucleon) ** 2)
                    + C * ((4 * m_nucleon * ev - qsq - m_lepton ** 2) / (m_nucleon) ** 2) ** 2)

        sqts = np.sqrt(m_nucleon ** 2 + 2 * m_nucleon * ev)
        E_l = (sqts ** 2 + m_lepton ** 2 - m_nucleon ** 2) / (2 * sqts)
        if E_l ** 2 < m_lepton ** 2:
            return 0
        q2min = -m_lepton ** 2 + (sqts ** 2 - m_nucleon ** 2) / (sqts) * \
                (E_l - np.sqrt(E_l ** 2 - m_lepton ** 2))
        q2max = -m_lepton ** 2 + (sqts ** 2 - m_nucleon ** 2) / (sqts) * \
                (E_l + np.sqrt(E_l ** 2 - m_lepton ** 2))

        return quad(dsigma, q2min, q2max)[0]


    def events(self, eva, evb, detector: Detector, exposure):
        nucleons = detector.z  # convert the per-nucleon cross section into total cross section.
        if self.flavor == 'ebar' or self.flavor == 'mubar' or self.flavor == 'taubar':
            nucleons = detector.n

        return nucleons * self.flux.integrate(eva, evb, self.flavor, weight_function=self.FastXS) * \
               exposure * mev_per_kg * 24 *60 * 60 / np.dot(detector.m, detector.frac)


    def change_parameters(self):
        pass
