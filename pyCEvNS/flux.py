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
            self.__nuflux1m = nuperf * fpers / (4 * pi) * (meter_by_mev ** 2)
        elif self.fl_name in ['sns', 'prompt', 'delayed']:
            self.evMin = 0
            self.evMax = 52  # MeV
            self.flUn = 0.1
            self.__norm = 1.05 * (10 ** 11) * (meter_by_mev ** 2)
        elif self.fl_name in ['solar', 'b8', 'f17', 'n13', 'o15', 'pp', 'hep']:
            f = genfromtxt(pkg_resources.resource_filename(__name__, 'data/' + self.fl_name + '.csv'), delimiter=',')
            self.flUn = 0
            self.evMin = f[0, 0]
            self.evMax = f[-1, 0]
            self.__nue = LinearInterp(f[:, 0], f[:, 1] * ((100 * meter_by_mev) ** 2))
        else:
            if isinstance(fl_name, ndarray):
                f = fl_name
            else:
                f = genfromtxt(fl_name, delimiter=delimiter)
            self.evMin = amin(f[:, 0])
            self.evMax = amax(f[:, 0])
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
                    return exp(0.87 - 0.16 * ev - 0.091 * (ev ** 2)) / 5.323608902707208 * \
                           f(ev, nui='ebar', nuf=flavor, **kwargs)
                return exp(0.87 - 0.16 * ev - 0.091 * (ev ** 2)) / 5.323608902707208 * self.__nuflux1m
            elif flavor[-1] == 'r':
                if f is not None:
                    return exp(0.87 - 0.16 * ev - 0.091 * (ev ** 2)) / 5.323608902707208 * \
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
        emin = 0.5 * (sqrt(er ** 2 + 2 * er * m) + er)

        def fx(ev):
            return self.flux(ev, flavor, f, **kwargs)

        if not isinstance(emin, ndarray):
            res = quad(fx, emin, self.evMax)[0]
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
            res = zeros_like(emin)
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
        emin = 0.5 * (sqrt(er ** 2 + 2 * er * m) + er)

        def finv(ev):
            """
            flux/ev
            """
            return self.flux(ev, flavor, f, **kwargs) / ev

        if not isinstance(emin, ndarray):
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
            res = zeros_like(emin)
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
        emin = 0.5 * (sqrt(er ** 2 + 2 * er * m) + er)

        def finvs(ev):
            """
            flux/ev^2
            """
            return self.flux(ev, flavor, f, **kwargs) / (ev ** 2)

        if not isinstance(emin, ndarray):
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
            res = zeros_like(emin)
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
