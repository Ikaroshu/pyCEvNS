"""
flux related class and functions
"""


from scipy.integrate import quad

from .oscillation import *


class Flux:
    """
    flux class
    """
    def __init__(self, fl_type, fl_file=None, epsi=None, op=None, r=0.05, lenth=0):
        """
        initializing flux, can take in user provided flux
        restrictions: user provided data must be .csv, it must have 4 columns,
        first column is neutrino energy in MeV,
        second column is electron neutrino flux in cm^2/s
        third column is muon neutrino flux in cm^2/s
        fourth column is tau neutrino flux in cm^2/s
        :param fl_type: name of the flux
        :param fl_file: file name of user provided data
        """
        self.fl_type = fl_type
        self.epsi = epsi
        self.op = op
        self.r = r
        self.lenth = lenth
        if fl_type.lower() == 'reactor':
            self.evMin = 0.0
            self.evMax = 30  # MeV
            self.flUn = 0.02
            fpers = 3.0921 * (10 ** 16)  # antineutrinos per fission
            nuperf = 6.14102
            self.__nuflux1m = nuperf * fpers / (4 * pi) * (meter_by_mev ** 2)
        elif fl_type.lower() == 'sns' or fl_type.lower() == 'prompt' or fl_type.lower() == 'delayed':
            self.evMin = 0
            self.evMax = 52  # MeV
            self.flUn = 0.1
            self.__norm = 1.05 * (10 ** 11) * (meter_by_mev ** 2)
        elif fl_type.lower() == 'solar':
            b8 = genfromtxt(pkg_resources.resource_filename(__name__, 'data/b8.csv'), delimiter=',')
            self.__b8x = b8[:, 0]
            self.__b8y = b8[:, 1] * ((100 * meter_by_mev) ** 2) * 5.58e6
            self.__b8interp = interp1d(self.__b8x, self.__b8y)
            f17 = genfromtxt(pkg_resources.resource_filename(__name__, 'data/f17.csv'), delimiter=',')
            self.__f17x = f17[:, 0]
            self.__f17y = f17[:, 1] * ((100 * meter_by_mev) ** 2) * 5.52e6
            self.__f17interp = interp1d(self.__f17x, self.__f17y)
            hep = genfromtxt(pkg_resources.resource_filename(__name__, 'data/hep.csv'), delimiter=',')
            self.__hepx = hep[:, 0]
            self.__hepy = hep[:, 1] * ((100 * meter_by_mev) ** 2) * 8.04e3
            self.__hepinterp = interp1d(self.__hepx, self.__hepy)
            n13 = genfromtxt(pkg_resources.resource_filename(__name__, 'data/n13.csv'), delimiter=',')
            self.__n13x = n13[:, 0]
            self.__n13y = n13[:, 1] * ((100 * meter_by_mev) ** 2) * 2.96e8
            self.__n13interp = interp1d(self.__n13x, self.__n13y)
            o15 = genfromtxt(pkg_resources.resource_filename(__name__, 'data/o15.csv'), delimiter=',')
            self.__o15x = o15[:, 0]
            self.__o15y = o15[:, 1] * ((100 * meter_by_mev) ** 2) * 2.23e8
            self.__o15interp = interp1d(self.__o15x, self.__o15y)
            pp = genfromtxt(pkg_resources.resource_filename(__name__, 'data/pp.csv'), delimiter=',')
            self.__ppx = pp[:, 0]
            self.__ppy = pp[:, 1] * ((100 * meter_by_mev) ** 2) * 5.98e10
            self.__ppinterp = interp1d(self.__ppx, self.__ppy)
            self.evMin = 0.003464
            self.evMax = 20
        elif not fl_file:
            f = genfromtxt(fl_file)
            self.__ev = f[:, 0]
            self.evMin = amin(self.__ev)
            self.evMax = amax(self.__ev)
            self.__nue = f[:, 1] * ((100 * meter_by_mev) ** 2)
            self.__num = f[:, 2] * ((100 * meter_by_mev) ** 2)
            self.__nut = f[:, 3] * ((100 * meter_by_mev) ** 2)
            self.__nueinterp = interp1d(self.__ev, self.__nue)
            self.__numinterp = interp1d(self.__ev, self.__num)
            self.__nutinterp = interp1d(self.__ev, self.__nut)
        else:
            raise Exception("No such flux in code or file yet.")

    def flux(self, ev, flavor='e', epsi=NSIparameters(), op=oscillation_parameters(), r=0.05):
        """
        differential neutrino flux, unit MeV^-3*s^-1
        :param ev: nuetrino energy
        :param flavor: nuetrino flavor
        :param r: distance where solor neutrino is produced
        :param epsi: NSI parameters for solar neutrino
        :param op: oscillation parametes
        :return: e neutrino flux
        """
        if self.fl_type == 'reactor':
            # Phys.Rev.D39, 11 Vogel
            # 5.323608902707208 = Integrate[Exp[.870 - .16*e - .091*e^2], {e, 0, 10}]
            # reactor neutrino is actually anti-neutrino, this may cause problem when doing electron scattering
            if flavor == 'e':
                if self.lenth != 0:
                    return exp(0.87 - 0.16 * ev - 0.091 * (ev ** 2)) / 5.323608902707208 * self.__nuflux1m / \
                           self.lenth**2 * \
                           survival_probability(ev, self.lenth, self.epsi, nui=0, nuf=0, anti=True, op=self.op)
                return exp(0.87 - 0.16 * ev - 0.091 * (ev ** 2)) / 5.323608902707208 * self.__nuflux1m
            elif flavor == 'mu':
                if self.lenth != 0:
                    return exp(0.87 - 0.16 * ev - 0.091 * (ev ** 2)) / 5.323608902707208 * self.__nuflux1m / \
                           self.lenth**2 * \
                           survival_probability(ev, self.lenth, self.epsi, nui=0, nuf=1, anti=True, op=self.op)
                return 0
            elif flavor == 'tau':
                if self.lenth != 0:
                    return exp(0.87 - 0.16 * ev - 0.091 * (ev ** 2)) / 5.323608902707208 * self.__nuflux1m / \
                           self.lenth**2 * \
                           survival_probability(ev, self.lenth, self.epsi, nui=0, nuf=2, anti=True, op=self.op)
                return 0
            else:
                return 0
        elif self.fl_type == 'sns' or self.fl_type == 'delayed':
            if flavor == 'e':
                if self.lenth != 0:
                    return (3 * ((ev / (2 / 3 * 52)) ** 2) - 2 * ((ev / (2 / 3 * 52)) ** 3)) / 29.25 * self.__norm * \
                           (20 / self.lenth)**2 * \
                           survival_probability(ev, self.lenth, epsi=self.epsi, op=self.op, nui=0, nuf=0) + \
                           (3 * ((ev / 52) ** 2) - 2 * ((ev / 52) ** 3)) / 26 * self.__norm * (20 / self.lenth)**2 * \
                           survival_probability(ev, self.lenth, epsi=self.epsi, op=self.op, nui=1, nuf=0, anti=True)
                return (3 * ((ev / (2 / 3 * 52)) ** 2) - 2 * ((ev / (2 / 3 * 52)) ** 3)) / 29.25 * self.__norm
            elif flavor == 'mu':
                if self.lenth != 0:
                    return (3 * ((ev / (2 / 3 * 52)) ** 2) - 2 * ((ev / (2 / 3 * 52)) ** 3)) / 29.25 * self.__norm * \
                           (20 / self.lenth) ** 2 * \
                           survival_probability(ev, self.lenth, epsi=self.epsi, op=self.op, nui=0, nuf=1) + \
                           (3 * ((ev / 52) ** 2) - 2 * ((ev / 52) ** 3)) / 26 * self.__norm * (20 / self.lenth) ** 2 * \
                           survival_probability(ev, self.lenth, epsi=self.epsi, op=self.op, nui=1, nuf=1, anti=True)
                return (3 * ((ev / 52) ** 2) - 2 * ((ev / 52) ** 3)) / 26 * self.__norm
            elif flavor == 'tau':
                return 0
        elif self.fl_type == 'prompt':
            return 0
        elif self.fl_type == 'solar':
            res = 0
            res += self.__b8interp(ev)[()] if self.__b8x[0] <= ev <= self.__b8x[self.__b8x.shape[0] - 1] else 0
            res += self.__f17interp(ev)[()] if self.__f17x[0] <= ev <= self.__f17x[self.__f17x.shape[0] - 1] else 0
            res += self.__hepinterp(ev)[()] if self.__hepx[0] <= ev <= self.__hepx[self.__hepx.shape[0] - 1] else 0
            res += self.__n13interp(ev)[()] if self.__n13x[0] <= ev <= self.__n13x[self.__n13x.shape[0] - 1] else 0
            res += self.__o15interp(ev)[()] if self.__o15x[0] <= ev <= self.__o15x[self.__o15x.shape[0] - 1] else 0
            res += self.__ppinterp(ev)[()] if self.__ppx[0] <= ev <= self.__ppx[self.__ppx.shape[0] - 1] else 0
            if flavor == 'e':
                res *= survp(ev, r, epsi, 0, op)
            elif flavor == 'mu':
                res *= survp(ev, r, epsi, 1, op)
            elif flavor == 'tau':
                res *= survp(ev, r, epsi, 2, op)
            else:
                raise Exception('No such neutrino flavor!')
            return res
        else:
            if flavor == 'e':
                return self.__nueinterp(ev)[()]
            elif flavor == 'mu':
                return self.__numinterp(ev)[()]
            elif flavor == 'tau':
                return self.__nutinterp(ev)[()]
            else:
                raise Exception('No such neutrino flavor!')

    def fint(self, er, m, flavor='e', epsi=None, op=None, r=0.05):
        """
        flux integration over the range that can produce a recoil energy er
        :param er: recoil energy
        :param m: mass of the target, it can be an array
        :param flavor: neutrino flavor
        :param r: distance where solar neutrino is produced
        :param epsi: NSI parameters for solar neutrino
        :param op: oscillation parametes
        :return: the result of integration, it can be an array
        """
        emin = 0.5 * (sqrt(er ** 2 + 2 * er * m) + er)
        if not isinstance(emin, ndarray):
            res = quad(self.flux, emin, self.evMax, args=(flavor, epsi, op, r))[0]
            if self.fl_type == 'solar':
                if flavor == 'e':
                    flav = 0
                elif flavor == 'mu':
                    flav = 1
                else:
                    flav = 2
                # pep
                res += 1.44e8 * ((100 * meter_by_mev) ** 2) * survp(1.439, r, epsi, flav, op) if emin < 1.439 else 0
                # be7
                res += 5e9 * ((100 * meter_by_mev) ** 2) * survp(0.8613, r, epsi, flav, op) if emin < 0.8613 else 0
            if (self.fl_type == 'sns' or self.fl_type == 'prompt') and flavor == 'mu':
                # prompt neutrino
                res += self.__norm if emin <= 29 else 0
        else:
            res = zeros_like(emin)
            for i in range(emin.shape[0]):
                res[i] = quad(self.flux, emin[i], self.evMax, args=(flavor, epsi, op, r))[0]
                if self.fl_type == 'solar':
                    if flavor == 'e':
                        flav = 0
                    elif flavor == 'mu':
                        flav = 1
                    else:
                        flav = 2
                    # pep
                    res[i] += 1.44e8 * ((100 * meter_by_mev) ** 2) * survp(1.439, r, epsi, flav, op) \
                        if emin[i] < 1.439 else 0
                    # be7
                    res[i] += 5e9 * ((100 * meter_by_mev) ** 2) * survp(0.8613, r, epsi, flav, op) \
                        if emin[i] < 0.8613 else 0
                if (self.fl_type == 'sns' or self.fl_type == 'prompt') and flavor == 'mu' and self.lenth == 0:
                    # prompt neutrino
                    res[i] += self.__norm if emin[i] <= 29 else 0
                if self.fl_type == 'sns' and self.lenth != 0:
                    if flavor == 'e':
                        res[i] += self.__norm * (20 / self.lenth)**2 * \
                                  survival_probability(29, self.lenth, epsi=self.epsi, op=self.op, nui=1, nuf=0) \
                            if emin[i] <= 29 else 0
                    elif flavor == 'mu':
                        res[i] += self.__norm * (20 / self.lenth) ** 2 * \
                                  survival_probability(29, self.lenth, epsi=self.epsi, op=self.op, nui=1, nuf=1) \
                            if emin[i] <= 29 else 0
                    elif flavor == 'tau':
                        res[i] += self.__norm * (20 / self.lenth) ** 2 * \
                                  survival_probability(29, self.lenth, epsi=self.epsi, op=self.op, nui=1, nuf=2) \
                            if emin[i] <= 29 else 0
        return res

    def fintinv(self, er, m, flavor='e', epsi=None, op=None, r=0.05):
        """
        flux/ev integration over the range that can produce a recoil energy er
        :param er: recoil energy
        :param m: mass of the target, it can be an array
        :param flavor: neutrino flavor
        :param r: distance where solar neutrino is produced
        :param epsi: NSI parameters for solar neutrino
        :param op: oscillation parametes
        :return: the result of integration, it can be an array
        """
        emin = 0.5 * (sqrt(er ** 2 + 2 * er * m) + er)

        def finv(ev):
            """
            flux/ev
            """
            return self.flux(ev, flavor, epsi, op, r) / ev

        if not isinstance(emin, ndarray):
            res = quad(finv, emin, self.evMax)[0]
            if self.fl_type == 'solar':
                if flavor == 'e':
                    flav = 0
                elif flavor == 'mu':
                    flav = 1
                else:
                    flav = 2
                # pep
                res += 1.44e8 * ((100 * meter_by_mev) ** 2) * survp(1.439, r, epsi, flav, op) / 1.439 \
                    if emin < 1.439 else 0
                # be7
                res += 5e9 * ((100 * meter_by_mev) ** 2) * survp(0.8613, r, epsi, flav, op) / 0.8613 \
                    if emin < 0.8613 else 0
            if (self.fl_type == 'sns' or self.fl_type == 'prompt') and flavor == 'mu':
                # prompt neutrino
                res += self.__norm / 29 if emin <= 29 else 0
        else:
            res = zeros_like(emin)
            for i in range(emin.shape[0]):
                res[i] = quad(finv, emin[i], self.evMax)[0]
                if self.fl_type == 'solar':
                    if flavor == 'e':
                        flav = 0
                    elif flavor == 'mu':
                        flav = 1
                    else:
                        flav = 2
                    # pep
                    res[i] += 1.44e8 * ((100 * meter_by_mev) ** 2) * survp(1.439, r, epsi, flav, op) / 1.439 \
                        if emin[i] < 1.439 else 0
                    # be7
                    res[i] += 5e9 * ((100 * meter_by_mev) ** 2) * survp(0.8613, r, epsi, flav, op) / 0.8613 \
                        if emin[i] < 0.8613 else 0
                if (self.fl_type == 'sns' or self.fl_type == 'prompt') and flavor == 'mu' and self.lenth == 0:
                    # prompt neutrino
                    res[i] += self.__norm / 29 if emin[i] <= 29 else 0
                if self.fl_type == 'sns' and self.lenth != 0:
                    if flavor == 'e':
                        res[i] += self.__norm / 29 * (20 / self.lenth)**2 * \
                                  survival_probability(29, self.lenth, epsi=self.epsi, op=self.op, nui=1, nuf=0) \
                            if emin[i] <= 29 else 0
                    elif flavor == 'mu':
                        res[i] += self.__norm / 29 * (20 / self.lenth) ** 2 * \
                                  survival_probability(29, self.lenth, epsi=self.epsi, op=self.op, nui=1, nuf=1) \
                            if emin[i] <= 29 else 0
                    elif flavor == 'tau':
                        res[i] += self.__norm / 29 * (20 / self.lenth) ** 2 * \
                                  survival_probability(29, self.lenth, epsi=self.epsi, op=self.op, nui=1, nuf=2) \
                            if emin[i] <= 29 else 0
        return res

    def fintinvs(self, er, m, flavor='e', epsi=None, op=None, r=0.05):
        """
        flux/ev^2 integration over the range that can produce a recoil energy er
        :param er: recoil energy
        :param m: mass of the target, it can be an array
        :param flavor: neutrino flavor
        :param r: distance where solar neutrino is produced
        :param epsi: NSI parameters for solar neutrino
        :param op: oscillation parametes
        :return: the result of integration, it can be an array
        """
        emin = 0.5 * (sqrt(er ** 2 + 2 * er * m) + er)

        def finvs(ev):
            """
            flux/ev^2
            """
            return self.flux(ev, flavor, epsi, op, r) / (ev ** 2)

        if not isinstance(emin, ndarray):
            res = quad(finvs, emin, self.evMax)[0]
            if self.fl_type == 'solar':
                if flavor == 'e':
                    flav = 0
                elif flavor == 'mu':
                    flav = 1
                else:
                    flav = 2
                # pep
                res += 1.44e8 * ((100 * meter_by_mev) ** 2) * survp(1.439, r, epsi, flav, op) / (1.439 ** 2) \
                    if emin < 1.439 else 0
                # be7
                res += 5e9 * ((100 * meter_by_mev) ** 2) * survp(0.8613, r, epsi, flav, op) / (0.8613 ** 2) \
                    if emin < 0.8613 else 0
            if (self.fl_type == 'sns' or self.fl_type == 'prompt') and flavor == 'mu':
                # prompt neutrino
                res += self.__norm / (29 ** 2) if emin <= 29 else 0
        else:
            res = zeros_like(emin)
            for i in range(emin.shape[0]):
                res[i] = quad(finvs, emin[i], self.evMax)[0]
                if self.fl_type == 'solar':
                    if flavor == 'e':
                        flav = 0
                    elif flavor == 'mu':
                        flav = 1
                    else:
                        flav = 2
                    # pep
                    res[i] += 1.44e8 * ((100 * meter_by_mev) ** 2) * survp(1.439, r, epsi, flav, op) / (1.439 ** 2) \
                        if emin[i] < 1.439 else 0
                    # be7
                    res[i] += 5e9 * ((100 * meter_by_mev) ** 2) * survp(0.8613, r, epsi, flav, op) / (0.8613 ** 2) \
                        if emin[i] < 0.8613 else 0
                if (self.fl_type == 'sns' or self.fl_type == 'prompt') and flavor == 'mu' and self.lenth == 0:
                    # prompt neutrino
                    res[i] += self.__norm / (29 ** 2) if emin[i] <= 29 else 0
                if self.fl_type == 'sns' and self.lenth != 0:
                    if flavor == 'e':
                        res[i] += self.__norm / (29 ** 2) * (20 / self.lenth)**2 * \
                                  survival_probability(29, self.lenth, epsi=self.epsi, op=self.op, nui=1, nuf=0) \
                            if emin[i] <= 29 else 0
                    elif flavor == 'mu':
                        res[i] += self.__norm / (29 ** 2) * (20 / self.lenth) ** 2 * \
                                  survival_probability(29, self.lenth, epsi=self.epsi, op=self.op, nui=1, nuf=1) \
                            if emin[i] <= 29 else 0
                    elif flavor == 'tau':
                        res[i] += self.__norm / (29 ** 2) * (20 / self.lenth) ** 2 * \
                                  survival_probability(29, self.lenth, epsi=self.epsi, op=self.op, nui=1, nuf=2) \
                            if emin[i] <= 29 else 0
        return res
