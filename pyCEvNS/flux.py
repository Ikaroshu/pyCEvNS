"""
flux related class and functions
"""


from scipy.integrate import quad

from .parameters import *


def survp(ev, r=0.05, epsi=Epsilon(), nuf=0, op=ocsillation_parameters()):
    """
    calculating survival/transitional probability of solar neutrino
    :param ev: neutrino energy in MeV
    :param r: position where netrino is produced, in terms solar radius
    :param epsi: nsi parameters
    :param nuf: final state, 0: electron neutrino, 1: muon neutrino, 2: tau neutrino
    :param op: oscillation parameters
    :return: survival/transitional probability
    """
    dp = Density()
    o23 = matrix([[1, 0, 0],
                  [0, cos(op['t23']), sin(op['t23'])],
                  [0, -sin(op['t23']), cos(op['t23'])]])
    u13 = matrix([[cos(op['t13']), 0, sin(op['t13']) * (e ** (- op['delta'] * 1j))],
                  [0, 1, 0],
                  [-sin(op['t13'] * (e ** (op['delta'] * 1j))), 0, cos(op['t13'])]])
    o12 = matrix([[cos(op['t12']), sin(op['t12']), 0],
                  [-sin(op['t12']), cos(op['t12']), 0],
                  [0, 0, 1]])
    umix = o23 * u13 * o12
    m = diag(array([0, op['d21'] / (2 * ev), op['d31'] / (2 * ev)]))
    v = sqrt(2) * gf * (dp.ne(r) * epsi.ee() + dp.nu(r) * epsi.eu() + dp.nd(r) * epsi.ed())
    hvac = umix * m * umix.H

    def sorteig(w, vec):
        minindex = 0
        maxindex = 0
        for j in range(3):
            if w[minindex] > w[j]:
                minindex = j
        for j in range(3):
            if w[maxindex] < w[j]:
                maxindex = j
        midindex = 3 - minindex - maxindex
        avec = array(vec)
        return matrix([avec[:, minindex], avec[:, midindex], avec[:, maxindex]]).T

    wr, vecr = linalg.eigh(hvac + v)
    utr = sorteig(wr, vecr)
    ws, vecs = linalg.eigh(hvac)
    uts = sorteig(ws, vecs)
    res = 0
    for i in range(3):
        res += conj(utr[0, i]) * utr[0, i] * conj(uts[nuf, i]) * uts[nuf, i]
    return real(res)


class Flux:
    """
    flux class
    """
    def __init__(self, fl_type):
        """
        initializing flux
        :param fl_type: name of the flux
        """
        self.fl_type = fl_type
        if fl_type.lower() == 'reactor':
            self.evMin = 0.0
            self.evMax = 10  # MeV
            self.flUn = 0.02
            fpers = 3.0921 * (10 ** 16)  # antineutrinos per fission
            nuperf = 6.14102
            self.__nuflux1m = nuperf * fpers / (4 * pi) * (meter_by_mev ** 2)
        elif fl_type.lower() == 'sns':
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
        else:
            raise Exception("No such flux in code yet.")

    def flux(self, ev, flavor='e', r=0.05, epsi=None, op=None):
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
                return exp(0.87 - 0.16 * ev - 0.091 * (ev ** 2)) / 5.323608902707208 * self.__nuflux1m
            else:
                return 0
        elif self.fl_type == 'sns':
            if flavor == 'e':
                return (3 * ((ev / (2 / 3 * 52)) ** 2) - 2 * ((ev / (2 / 3 * 52)) ** 3)) / 29.25 * self.__norm
            elif flavor == 'mu':
                return (3 * ((ev / 52) ** 2) - 2 * ((ev / 52) ** 3)) / 26 * self.__norm
            elif flavor == 'tau':
                return 0
        elif self.fl_type == 'solar':
            if not epsi:
                raise Exception("missing parameter epsi: NSI parameters")
            if not op:
                raise Exception("missing parameter op: oscillation parametes")
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
                return 0
            return res
        else:
            return Exception("No such flux in code yet.")

    def fint(self, er, m, flavor='e', r=0.05, epsi=None, op=None):
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
            res = quad(self.flux, emin, self.evMax, args=(flavor, r, epsi, op))[0]
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
            if self.fl_type == 'sns' and flavor == 'mu':
                # prompt neutrino
                res += self.__norm if emin <= 29 else 0
        else:
            res = zeros_like(emin)
            for i in range(emin.shape[0]):
                res[i] = quad(self.flux, emin[i], self.evMax, args=(flavor, r, epsi, op))[0]
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
                if self.fl_type == 'sns' and flavor == 'mu':
                    # prompt neutrino
                    res[i] += self.__norm if emin[i] <= 29 else 0
        return res

    def fintinv(self, er, m, flavor='e', r=0.05, epsi=None, op=None):
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
            return self.flux(ev, flavor, r, epsi, op) / ev

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
            if self.fl_type == 'sns' and flavor == 'mu':
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
                if self.fl_type == 'sns' and flavor == 'mu':
                    # prompt neutrino
                    res[i] += self.__norm / 29 if emin[i] <= 29 else 0
        return res

    def fintinvs(self, er, m, flavor='e', r=0.05, epsi=None, op=None):
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
            return self.flux(ev, flavor, r, epsi, op) / (ev ** 2)

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
            if self.fl_type == 'sns' and flavor == 'mu':
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
                if self.fl_type == 'sns' and flavor == 'mu':
                    # prompt neutrino
                    res[i] += self.__norm / (29 ** 2) if emin[i] <= 29 else 0
        return res
