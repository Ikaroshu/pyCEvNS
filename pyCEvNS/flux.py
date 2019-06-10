"""
flux related class and functions
"""


from scipy.integrate import quad

from .helper import *
from .oscillation import *


class Flux:
    """
    flux class,
    flux at source
    """
    def __init__(self, fl_name, delimiter=',', fl_unc=0):
        """
        initializing flux, can take in user provided flux
        restrictions: user provided data must have 7 columns,
        first column is neutrino energy in MeV,
        other columns are neutrino flux in cm^2/s/MeV, they are enu, munu, taunu, enubar, munubar, taunubar
        :param fl_name: name of the flux or path to the file or array of neutrino flux
        :param delimiter: delimiter of the input file, default is ','
        :param fl_unc: uncertainty of flux
        """
        if isinstance(fl_name, str):
            self.fl_name = fl_name.lower()
        else:
            self.fl_name = 'default'
        if self.fl_name == 'reactor':
            self.evMin = 0.0
            self.evMax = 30  # MeV
            self.flUn = 0.02
            fpers = 3.0921 * (10 ** 16)  # antineutrinos per fission
            nuperf = 6.14102
            self.__nuflux1m = nuperf * fpers / (4 * np.pi) * (meter_by_mev ** 2)
        elif self.fl_name in ['sns', 'prompt', 'delayed']:
            self.evMin = 0
            self.evMax = 52  # MeV
            self.flUn = 0.1
            self.__norm = 1.13 * (10 ** 11) * (meter_by_mev ** 2)
        elif self.fl_name in ['solar', 'b8', 'f17', 'n13', 'o15', 'pp', 'hep']:
            f = np.genfromtxt(pkg_resources.resource_filename(__name__, 'data/' + self.fl_name + '.csv'), delimiter=',')
            self.flUn = 0
            self.evMin = f[0, 0]
            self.evMax = f[-1, 0]
            self.__nue = LinearInterp(f[:, 0], f[:, 1] * ((100 * meter_by_mev) ** 2))
        else:
            if isinstance(fl_name, np.ndarray):
                f = fl_name
            else:
                f = np.genfromtxt(fl_name, delimiter=delimiter)
            self.evMin = np.amin(f[:, 0])
            self.evMax = np.amax(f[:, 0])
            self.flUn = fl_unc
            self.__nue = LinearInterp(f[:, 0], f[:, 1] * ((100 * meter_by_mev) ** 2))
            self.__numu = LinearInterp(f[:, 0], f[:, 2] * ((100 * meter_by_mev) ** 2))
            self.__nutau = LinearInterp(f[:, 0], f[:, 3] * ((100 * meter_by_mev) ** 2))
            self.__nuebar = LinearInterp(f[:, 0], f[:, 4] * ((100 * meter_by_mev) ** 2))
            self.__numubar = LinearInterp(f[:, 0], f[:, 5] * ((100 * meter_by_mev) ** 2))
            self.__nutaubar = LinearInterp(f[:, 0], f[:, 6] * ((100 * meter_by_mev) ** 2))

    def flux(self, ev, flavor='e', f=None, **kwargs):
        """
        differential neutrino flux at the detector, unit MeV^-3*s^-1
        :param ev: nuetrino energy
        :param flavor: nuetrino flavor
        :param f: function that convolves with neutrino flux, typically neutrino oscillation,
                the first argument must be neutrino energy,
                the last two arguments must be input flavor nui and out put flavor nuf
        :param kwargs: parameters with keys that goes into function f
        :return: neutrino flux
        """
        if self.fl_name == 'reactor':
            # Phys.Rev.D39, 11 Vogel
            # 5.323608902707208 = Integrate[Exp[.870 - .16*e - .091*e^2], {e, 0, 10}]
            # reactor neutrino is actually anti-neutrino, this may cause problem when doing electron scattering
            if flavor == 'ebar':
                if f is not None:
                    return np.exp(0.87 - 0.16 * ev - 0.091 * (ev ** 2)) / 5.323608902707208 * \
                           f(ev, nui='ebar', nuf=flavor, **kwargs)
                return np.exp(0.87 - 0.16 * ev - 0.091 * (ev ** 2)) / 5.323608902707208 * self.__nuflux1m
            elif flavor[-1] == 'r':
                if f is not None:
                    return np.exp(0.87 - 0.16 * ev - 0.091 * (ev ** 2)) / 5.323608902707208 * \
                           f(ev, nui='ebar', nuf=flavor, **kwargs)
                return 0
            else:
                return 0
        elif self.fl_name in ['sns', 'delayed']:
            if flavor[-1] != 'r':
                if f is not None:
                    return (3 * ((ev / (2 / 3 * 52)) ** 2) - 2 * ((ev / (2 / 3 * 52)) ** 3)) / 29.25 * self.__norm * \
                           f(ev, nui='e', nuf=flavor, **kwargs)
                return (3 * ((ev / (2 / 3 * 52)) ** 2) - 2 * ((ev / (2 / 3 * 52)) ** 3)) / 29.25 * self.__norm \
                    if flavor == 'e' else 0
            else:
                if f is not None:
                    return (3 * ((ev / 52) ** 2) - 2 * ((ev / 52) ** 3)) / 26 * self.__norm * \
                           f(ev, nui='mubar', nuf=flavor, **kwargs)
                return (3 * ((ev / 52) ** 2) - 2 * ((ev / 52) ** 3)) / 26 * self.__norm if flavor == 'mubar' else 0
        elif self.fl_name == 'prompt':
            return 0
        elif self.fl_name in ['solar', 'b8', 'f17', 'n13', 'o15', 'pp', 'hep']:
            if flavor[-1] != 'r':
                if f is None:
                    f = survival_solar
                return self.__nue(ev) * f(ev, nui='e', nuf=flavor, **kwargs)
            return 0
        else:
            if flavor[-1] != 'r':
                if f is None:
                    if flavor == 'e':
                        return self.__nue(ev)
                    elif flavor == 'mu':
                        return self.__numu(ev)
                    elif flavor == 'tau':
                        return self.__nutau(ev)
                    else:
                        return 0
                return self.__nue(ev) * f(ev, nui='e', nuf=flavor, **kwargs) + \
                    self.__numu(ev) * f(ev, nui='mu', nuf=flavor, **kwargs) + \
                    self.__nutau(ev) * f(ev, nui='tau', nuf=flavor, **kwargs)
            else:
                if f is None:
                    if flavor == 'ebar':
                        return self.__nuebar(ev)
                    elif flavor == 'mubar':
                        return self.__numubar(ev)
                    elif flavor == 'taubar':
                        return self.__nutaubar(ev)
                    else:
                        return 0
                return self.__nuebar(ev) * f(ev, nui='ebar', nuf=flavor, **kwargs) + \
                    self.__numubar(ev) * f(ev, nui='mubar', nuf=flavor, **kwargs) + \
                    self.__nutaubar(ev) * f(ev, nui='taubar', nuf=flavor, **kwargs)

    def fint(self, er, m, flavor='e', f=None, **kwargs):
        """
        flux integration over the range that can produce a recoil energy er
        :param er: recoil energy
        :param m: mass of the target, it can be an array
        :param flavor: neutrino flavor
        :param f: function that convolves with neutrino flux, typically neutrino oscillation,
                the first argument must be neutrino energy,
                the last two arguments must be input flavor nui and out put flavor nuf
        :param kwargs: parameters with keys that goes into function f
        :return: the result of integration, it can be an array
        """
        emin = 0.5 * (np.sqrt(er ** 2 + 2 * er * m) + er)

        def fx(ev):
            return self.flux(ev, flavor, f, **kwargs)

        if not isinstance(emin, np.ndarray):
            res = quad(fx, emin, self.evMax)[0]  # no need to check range, because outside evMin and evMax are 0
            if self.fl_name == 'solar':
                if f is None:
                    f = survival_solar
                # pep
                res += 1.44e8 * ((100 * meter_by_mev) ** 2) * f(1.439, nui='e', nuf=flavor, **kwargs) \
                    if emin < 1.439 else 0
                # be7
                res += 5e9 * ((100 * meter_by_mev) ** 2) * f(0.8613, nui='e', nuf=flavor, **kwargs) \
                    if emin < 0.8613 else 0
            elif self.fl_name in ['sns', 'prompt']:
                if f is None and flavor == 'mu':
                    # prompt neutrino
                    res += self.__norm if emin <= 29 else 0
                elif f is not None and flavor[-1] != 'r':
                    res += self.__norm * f(29, nui='mu', nuf=flavor, **kwargs) if emin <= 29 else 0
        else:
            res = np.zeros_like(emin)
            for i in range(emin.shape[0]):
                res[i] = quad(fx, emin[i], self.evMax)[0]
                if self.fl_name == 'solar':
                    if f is None:
                        f = survival_solar
                    # pep
                    res[i] += 1.44e8 * ((100 * meter_by_mev) ** 2) * f(1.439, nui='e', nuf=flavor, **kwargs) \
                        if emin[i] < 1.439 else 0
                    # be7
                    res[i] += 5e9 * ((100 * meter_by_mev) ** 2) * f(0.8613, nui='e', nuf=flavor, **kwargs) \
                        if emin[i] < 0.8613 else 0
                elif self.fl_name in ['sns', 'prompt']:
                    if f is None and flavor == 'mu':
                        # prompt neutrino
                        res[i] += self.__norm if emin[i] <= 29 else 0
                    elif f is not None and flavor[-1] != 'r':
                        res[i] += self.__norm * f(29, nui='mu', nuf=flavor, **kwargs) if emin[i] <= 29 else 0
        return res

    def fintinv(self, er, m, flavor='e', f=None, **kwargs):
        """
        flux/ev integration over the range that can produce a recoil energy er
        :param er: recoil energy
        :param m: mass of the target, it can be an array
        :param flavor: neutrino flavor
        :param f: function that convolves with neutrino flux, typically neutrino oscillation,
                the first argument must be neutrino energy,
                the last two arguments must be input flavor nui and out put flavor nuf
        :param kwargs: parameters with keys that goes into function f
        :return: the result of integration, it can be an array
        """
        emin = 0.5 * (np.sqrt(er ** 2 + 2 * er * m) + er)

        def finv(ev):
            """
            flux/ev
            """
            return self.flux(ev, flavor, f, **kwargs) / ev

        if not isinstance(emin, np.ndarray):
            res = quad(finv, emin, self.evMax)[0]
            if self.fl_name == 'solar':
                if f is None:
                    f = survival_solar
                # pep
                res += 1.44e8 * ((100 * meter_by_mev) ** 2) * f(1.439, nui='e', nuf=flavor, **kwargs) / 1.439 \
                    if emin < 1.439 else 0
                # be7
                res += 5e9 * ((100 * meter_by_mev) ** 2) * f(0.8613, nui='e', nuf=flavor, **kwargs) / 0.8613 \
                    if emin < 0.8613 else 0
            elif self.fl_name in ['sns', 'prompt']:
                if f is None and flavor == 'mu':
                    # prompt neutrino
                    res += self.__norm / 29 if emin <= 29 else 0
                elif f is not None and flavor[-1] != 'r':
                    res += self.__norm / 29 * f(29, nui='mu', nuf=flavor, **kwargs) if emin <= 29 else 0
        else:
            res = np.zeros_like(emin)
            for i in range(emin.shape[0]):
                res[i] = quad(finv, emin[i], self.evMax)[0]
                if self.fl_name == 'solar':
                    if f is None:
                        f = survival_solar
                    # pep
                    res[i] += 1.44e8 * ((100 * meter_by_mev) ** 2) * f(1.439, nui='e', nuf=flavor, **kwargs) / \
                        1.439 if emin[i] < 1.439 else 0
                    # be7
                    res[i] += 5e9 * ((100 * meter_by_mev) ** 2) * f(0.8613, nui='e', nuf=flavor, **kwargs) / \
                        0.8613 if emin[i] < 0.8613 else 0
                elif self.fl_name in ['sns', 'prompt']:
                    if f is None and flavor == 'mu':
                        # prompt neutrino
                        res[i] += self.__norm / 29 if emin[i] <= 29 else 0
                    elif f is not None and flavor[-1] != 'r':
                        res[i] += self.__norm / 29 * f(29, nui='mu', nuf=flavor, **kwargs) \
                            if emin[i] <= 29 else 0
        return res

    def fintinvs(self, er, m, flavor='e', f=None, **kwargs):
        """
        flux/ev^2 integration over the range that can produce a recoil energy er
        :param er: recoil energy
        :param m: mass of the target, it can be an array
        :param flavor: neutrino flavor
        :param f: function that convolves with neutrino flux, typically neutrino oscillation,
                the first argument must be neutrino energy,
                the last two arguments must be input flavor nui and out put flavor nuf
        :param kwargs: parameters with keys that goes into function f
        :return: the result of integration, it can be an array
        """
        emin = 0.5 * (np.sqrt(er ** 2 + 2 * er * m) + er)

        def finvs(ev):
            """
            flux/ev^2
            """
            return self.flux(ev, flavor, f, **kwargs) / (ev ** 2)

        if not isinstance(emin, np.ndarray):
            res = quad(finvs, emin, self.evMax)[0]
            if self.fl_name == 'solar':
                if f is None:
                    f = survival_solar
                # pep
                res += 1.44e8 * ((100 * meter_by_mev) ** 2) * f(1.439, nui='e', nuf=flavor, **kwargs) / 1.439**2\
                    if emin < 1.439 else 0
                # be7
                res += 5e9 * ((100 * meter_by_mev) ** 2) * f(0.8613, nui='e', nuf=flavor, **kwargs) / 0.8613**2 \
                    if emin < 0.8613 else 0
            elif self.fl_name in ['sns', 'prompt']:
                if f is None and flavor == 'mu':
                    # prompt neutrino
                    res += self.__norm / 29**2 if emin <= 29 else 0
                elif f is not None and flavor[-1] != 'r':
                    res += self.__norm / 29**2 * f(29, nui='mu', nuf=flavor, **kwargs) if emin <= 29 else 0
        else:
            res = np.zeros_like(emin)
            for i in range(emin.shape[0]):
                res[i] = quad(finvs, emin[i], self.evMax)[0]
                if self.fl_name == 'solar':
                    if f is None:
                        f = survival_solar
                    # pep
                    res[i] += 1.44e8 * ((100 * meter_by_mev) ** 2) * f(1.439, nui='e', nuf=flavor, **kwargs) / \
                        1.439**2 if emin[i] < 1.439 else 0
                    # be7
                    res[i] += 5e9 * ((100 * meter_by_mev) ** 2) * f(0.8613, nui='e', nuf=flavor, **kwargs) / \
                        0.8613**2 if emin[i] < 0.8613 else 0
                elif self.fl_name in ['sns', 'prompt']:
                    if f is None and flavor == 'mu':
                        # prompt neutrino
                        res[i] += self.__norm / 29**2 if emin[i] <= 29 else 0
                    elif f is not None and flavor[-1] != 'r':
                        res[i] += self.__norm / 29**2 * f(29, nui='mu', nuf=flavor, **kwargs) \
                            if emin[i] <= 29 else 0
        return res


class DMFlux:
    """
    Dark matter flux at COHERENT
    """
    def __init__(self, dark_photon_mass, life_time, coupling_quark, dark_matter_mass,
                 detector_distance=19.3, pot_mu=0.75, pot_sigma=0.25, size=100000):
        """
        initialize and generate flux
        :param dark_photon_mass: dark photon mass
        :param life_time: life time of dark photon in rest frame, unit in micro second
        :param coupling_quark: dark photon coupling to quarks
        :param dark_matter_mass: mass of dark matter, unit in MeV
        :param detector_distance: distance from the detector to the Hg target
        :param pot_mu: mean of guassian distribution of proton on target, unit in micro second
        :param pot_sigma: std of guassian distribution of proton on target, unit in micro second
        :param size: size of sampling dark photons
        """
        self.dp_mass = dark_photon_mass
        self.dm_mass = dark_matter_mass
        self.epsi_quark = coupling_quark
        self.det_dist = detector_distance / meter_by_mev
        self.dp_life = life_time * 1e-6 * c_light / meter_by_mev
        self.mu = pot_mu * 1e-6 * c_light / meter_by_mev
        self.sigma = pot_sigma * 1e-6 * c_light / meter_by_mev
        self.timing, self.energy = self._generate(size)
        self.ed_min = self.energy.min()
        self.ed_max = self.energy.max()
        self.dm_norm = self.epsi_quark**2*0.23*1e20 / (4*np.pi*(detector_distance**2)*24*3600) * (meter_by_mev**2) * \
            self.timing.shape[0] * 2 / size

    def _generate(self, size=1000000):
        """
        generate dark matter flux at COHERENT
        :param size: size of sampling dark photons
        :return: time and energy histogram of dark matter
        """
        dp_m = self.dp_mass
        dp_e = ((massofpi+massofp)**2 - massofn**2 + dp_m**2)/(2*(massofpi+massofp))
        dp_p = np.sqrt(dp_e ** 2 - dp_m ** 2)
        dp_v = dp_p / dp_e
        gamma = dp_e / dp_m
        tau = self.dp_life * gamma
        tf = np.random.normal(self.mu, self.sigma, size)  # POT
        t = np.random.exponential(tau, size)  # life time of each dark photon
        cs = np.random.uniform(-1, 1, size)  # direction of each dark photon
        # in rest frame
        estar = dp_m / 2
        pstar = np.sqrt(estar ** 2 - self.dm_mass ** 2)
        pstarx = pstar * cs
        pstary = pstar * np.sqrt(1 - cs ** 2)
        # boost to lab frame
        elab = gamma * (estar + dp_v * pstarx)
        plabx = gamma * (pstarx + dp_v * estar)
        plaby = pstary
        vx = plabx / elab
        vy = plaby / elab
        timing = []
        energy = []
        for i in range(size):
            a = vx[i] ** 2 + vy[i] ** 2
            b = 2 * vx[i] * t[i] * dp_v
            cc = dp_v ** 2 * t[i] ** 2 - self.det_dist ** 2
            if b ** 2 - 4 * a * cc >= 0:
                if (-b - np.sqrt(b ** 2 - 4 * a * cc)) / (2 * a) > 0:
                    timing.append((-b - np.sqrt(b ** 2 - 4 * a * cc)) / (2 * a) + t[i] + tf[i])
                    energy.append(elab[i])
                if (-b + np.sqrt(b ** 2 - 4 * a * cc)) / (2 * a) > 0:
                    timing.append((-b + np.sqrt(b ** 2 - 4 * a * cc)) / (2 * a) + t[i] + tf[i])
                    energy.append(elab[i])
        return np.array(timing) / c_light * meter_by_mev * 1e6, np.array(energy)

    def flux(self, ev):
        """
        dark matter flux
        :param ev: dark matter energy
        :return: dark matter flux
        """
        return 1/(self.ed_max-self.ed_min)*self.dm_norm if self.ed_min <= ev <= self.ed_max else 0

    def fint(self, er, m, **kwargs):
        """
        flux/(ex^2-mx^2) integration
        :param er: recoil energy in MeV
        :param m: target nucleus mass in MeV
        :param kwargs: other argument
        :return: flux/(ex^2-mx^2) integration
        """
        emin = 0.5 * (np.sqrt((er**2*m+2*er*m**2+2*er*self.dm_mass**2+4*m*self.dm_mass**2)/m) + er)

        def integrand(ex):
            return self.flux(ex)/(ex**2 - self.dm_mass**2)

        if not isinstance(emin, np.ndarray):
            res = quad(integrand, emin, self.ed_max)[0]
        else:
            res = np.zeros_like(emin)
            for i in range(emin.shape[0]):
                res[i] = quad(integrand, emin[i], self.ed_max)[0]
        return res

    def fint1(self, er, m, **kwargs):
        """
        flux*ex/(ex^2-mx^2) integration
        :param er: recoil energy in MeV
        :param m: target nucleus mass in MeV
        :param kwargs: other argument
        :return: flux*ex/(ex^2-mx^2) integration
        """
        emin = 0.5 * (np.sqrt((er**2*m+2*er*m**2+2*er*self.dm_mass**2+4*m*self.dm_mass**2)/m) + er)

        def integrand(ex):
            return self.flux(ex) * ex / (ex ** 2 - self.dm_mass ** 2)

        if not isinstance(emin, np.ndarray):
            res = quad(integrand, emin, self.ed_max)[0]
        else:
            res = np.zeros_like(emin)
            for i in range(emin.shape[0]):
                res[i] = quad(integrand, emin[i], self.ed_max)[0]
        return res

    def fint2(self, er, m, **kwargs):
        """
        flux*ex^2/(ex^2-mx^2) integration
        :param er: recoil energy in MeV
        :param m: target nucleus mass in MeV
        :param kwargs: other argument
        :return: flux*ex^2/(ex^2-mx^2) integration
        """
        emin = 0.5 * (np.sqrt((er**2*m+2*er*m**2+2*er*self.dm_mass**2+4*m*self.dm_mass**2)/m) + er)

        def integrand(ex):
            return self.flux(ex) * ex**2 / (ex ** 2 - self.dm_mass ** 2)

        if not isinstance(emin, np.ndarray):
            res = quad(integrand, emin, self.ed_max)[0]
        else:
            res = np.zeros_like(emin)
            for i in range(emin.shape[0]):
                res[i] = quad(integrand, emin[i], self.ed_max)[0]
        return res

