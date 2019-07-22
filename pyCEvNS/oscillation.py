"""
neutrino oscillation related funtions
"""

from .parameters import *

# solar number density at r=0.05 solar radius, unit is MeV^3 (natural unit)
__ne_solar = 4.163053492437814e-07
__nu_solar = 1.0053941490424488e-06
__nd_solar = 7.618722503535536e-07


def survival_solar(ev, epsi=NSIparameters(), op=oscillation_parameters(), nui='e', nuf='e'):
    """
    calculating survival/transitional probability of solar neutrino
    :param ev: neutrino energy in MeV
    :param epsi: nsi parameters
    :param nui: intial state
    :param nuf: final state, 0: electron neutrino, 1: muon neutrino, 2: tau neutrino
    :param op: oscillation parameters
    :return: survival/transitional probability
    """
    op = op.copy()
    dic = {'e': 0, 'mu': 1, 'tau': 2}
    fi = dic[nui]
    ff = dic[nuf]
    o23 = np.array([[1, 0, 0],
                   [0, np.cos(op['t23']), np.sin(op['t23'])],
                   [0, -np.sin(op['t23']), np.cos(op['t23'])]])
    u13 = np.array([[np.cos(op['t13']), 0, np.sin(op['t13']) * (np.exp(- op['delta'] * 1j))],
                   [0, 1, 0],
                   [-np.sin(op['t13'] * (np.exp(op['delta'] * 1j))), 0, np.cos(op['t13'])]])
    o12 = np.array([[np.cos(op['t12']), np.sin(op['t12']), 0],
                   [-np.sin(op['t12']), np.cos(op['t12']), 0],
                   [0, 0, 1]])
    umix = o23 @ u13 @ o12
    m = np.diag(np.array([0, op['d21'] / (2 * ev), op['d31'] / (2 * ev)]))
    v = np.sqrt(2) * gf * (__ne_solar * (epsi.ee() + np.diag(np.array([1, 0, 0]))) + __nu_solar * epsi.eu() + __nd_solar * epsi.ed())
    hvac = umix @ m @ umix.conj().T

    def sorteig(w, vec):
        """
        sort the eigenstates to make the resultant eigenvalue continuous
        """
        minindex = 0
        maxindex = 0
        for j in range(3):
            if w[minindex] > w[j]:
                minindex = j
        for j in range(3):
            if w[maxindex] < w[j]:
                maxindex = j
        midindex = 3 - minindex - maxindex
        avec = np.array(vec)
        return np.array([avec[:, minindex], avec[:, midindex], avec[:, maxindex]]).T

    wr, vecr = np.linalg.eigh(hvac + v)
    utr = sorteig(wr, vecr)
    ws, vecs = np.linalg.eigh(hvac)
    uts = sorteig(ws, vecs)
    res = 0
    for i in range(3):
        res += np.conj(utr[0, i]) * utr[0, i] * np.conj(uts[ff, i]) * uts[ff, i]
    return np.real(res)


def survival_solar_amp(ev, epsi=NSIparameters(), op=oscillation_parameters(), nui='e', nuf='e', **kwargs):
    """
    calculating survival/transitional amplitude of solar neutrino, this is just hack, not real amplitude!
    :param ev: neutrino energy in MeV
    :param epsi: nsi parameters
    :param nui: intial state
    :param nuf: final state, 0: electron neutrino, 1: muon neutrino, 2: tau neutrino
    :param op: oscillation parameters
    :return: survival/transitional probability
    """
    op = op.copy()
    dic = {'e': 0, 'mu': 1, 'tau': 2}
    fi = dic[nui]
    ff = dic[nuf]
    o23 = np.array([[1, 0, 0],
                   [0, np.cos(op['t23']), np.sin(op['t23'])],
                   [0, -np.sin(op['t23']), np.cos(op['t23'])]])
    u13 = np.array([[np.cos(op['t13']), 0, np.sin(op['t13']) * (np.exp(- op['delta'] * 1j))],
                   [0, 1, 0],
                   [-np.sin(op['t13'] * (np.exp(op['delta'] * 1j))), 0, np.cos(op['t13'])]])
    o12 = np.array([[np.cos(op['t12']), np.sin(op['t12']), 0],
                   [-np.sin(op['t12']), np.cos(op['t12']), 0],
                   [0, 0, 1]])
    umix = o23 @ u13 @ o12
    m = np.diag(np.array([0, op['d21'] / (2 * ev), op['d31'] / (2 * ev)]))
    v = np.sqrt(2) * gf * (__ne_solar * (epsi.ee() + np.diag(np.array([1, 0, 0]))) + __nu_solar * epsi.eu() + __nd_solar * epsi.ed())
    hvac = umix @ m @ umix.conj().T

    def sorteig(w, vec):
        """
        sort the eigenstates to make the resultant eigenvalue continuous
        """
        minindex = 0
        maxindex = 0
        for j in range(3):
            if w[minindex] > w[j]:
                minindex = j
        for j in range(3):
            if w[maxindex] < w[j]:
                maxindex = j
        midindex = 3 - minindex - maxindex
        avec = np.array(vec)
        return np.array([avec[:, minindex], avec[:, midindex], avec[:, maxindex]]).T

    wr, vecr = np.linalg.eigh(hvac + v)
    utr = sorteig(wr, vecr)
    ws, vecs = np.linalg.eigh(hvac)
    uts = sorteig(ws, vecs)
    res = 0
    for i in range(3):
        res += np.conj(utr[fi, i]) * utr[fi, i] * np.conj(uts[ff, i]) * uts[ff, i]
    return np.sqrt(np.real(res))

# using Caylay-Hamilton theorem to calculate survival probability, it has probems at transitsion probabilities
#
# def survival_probability(ev, lenth, epsi=NSIparameters(), nui=0, nuf=0,
#                          op=ocsillation_parameters(), ne=2.2*6.02e23*(100*meter_by_mev)**3):
#     o23 = np.matrix([[1, 0, 0],
#                   [0, np.cos(op['t23']), np.sin(op['t23'])],
#                   [0, -np.sin(op['t23']), np.cos(op['t23'])]])
#     u13 = np.matrix([[np.cos(op['t13']), 0, np.sin(op['t13']) * (np.exp(- op['delta'] * 1j))],
#                   [0, 1, 0],
#                   [-np.sin(op['t13'] * (np.exp(op['delta'] * 1j))), 0, np.cos(op['t13'])]])
#     o12 = np.matrix([[np.cos(op['t12']), np.sin(op['t12']), 0],
#                   [-np.sin(op['t12']), np.cos(op['t12']), 0],
#                   [0, 0, 1]])
#     umix = o23 * u13 * o12
#     m = np.diag(np.array([0, op['d21'] / (2 * ev), op['d31'] / (2 * ev)]))
#     vf = np.sqrt(2) * gf * ne * (epsi.ee() + 3 * epsi.eu() + 3 * epsi.ed())
#     hf = umix * m * umix.H + vf
#     w, v = np.linalg.eigh(hf)
#     # print(w)
#     b = e**(-1j*w*lenth)
#     # print(b)
#     a = np.array([[1, 1, 1], -1j * lenth * w, -lenth**2 * w**2]).T
#     # print(a)
#     x = np.linalg.solve(a, b)
#     tnp.matrix = x[0] + -1j * lenth * x[1] * hf - lenth**2 * x[2] * hf.dot(hf)
#     # print(tnp.matrix)
#     return abs(tnp.matrix[nui, nuf])**2


def survival_const(ev, lenth=0.0, epsi=NSIparameters(), op=oscillation_parameters(),
                   ne=2.2 * 6.02e23 * (100 * meter_by_mev) ** 3, nui='e', nuf='e'):
    """
    survival/transitional probability with constant matter density
    :param ev: nuetrino energy in MeV
    :param lenth: oscillation lenth in meters
    :param epsi: epsilons
    :param nui: initail flavor
    :param nuf: final flavor
    :param op: oscillation parameters
    :param ne: electron number density in MeV^3
    :return: survival/transitional probability
    """
    op = op.copy()
    dic = {'e': 0, 'mu': 1, 'tau': 2, 'ebar': 0, 'mubar': 1, 'taubar': 2}
    fi = dic[nui]
    ff = dic[nuf]
    lenth = lenth / meter_by_mev
    if nuf[-1] == 'r':
        op['delta'] = -op['delta']
    o23 = np.array([[1, 0, 0],
                   [0, np.cos(op['t23']), np.sin(op['t23'])],
                   [0, -np.sin(op['t23']), np.cos(op['t23'])]])
    u13 = np.array([[np.cos(op['t13']), 0, np.sin(op['t13']) * (np.exp(- op['delta'] * 1j))],
                   [0, 1, 0],
                   [-np.sin(op['t13'] * (np.exp(op['delta'] * 1j))), 0, np.cos(op['t13'])]])
    o12 = np.array([[np.cos(op['t12']), np.sin(op['t12']), 0],
                   [-np.sin(op['t12']), np.cos(op['t12']), 0],
                   [0, 0, 1]])
    umix = o23 @ u13 @ o12
    m = np.diag(np.array([0, op['d21'] / (2 * ev), op['d31'] / (2 * ev)]))
    vf = np.sqrt(2) * gf * ne * ((epsi.ee() + np.diag(np.array([1, 0, 0]))) + 3 * epsi.eu() + 3 * epsi.ed())
    if nuf[-1] == 'r':
        hf = umix @ m @ umix.conj().T - np.conj(vf)
    else:
        hf = umix @ m @ umix.conj().T + vf
    w, v = np.linalg.eigh(hf)
    res = 0.0
    for i in range(3):
        for j in range(3):
            theta = (w[i]-w[j]) * lenth
            res += v[ff, i] * np.conj(v[fi, i]) * np.conj(v[ff, j]) * v[fi, j] * (np.cos(theta) - 1j * np.sin(theta))
    return np.real(res)


def survival_const_amp(ev, lenth=0.0, epsi=NSIparameters(), op=oscillation_parameters(),
                   ne=2.2 * 6.02e23 * (100 * meter_by_mev) ** 3, nui='e', nuf='e'):
    """
    survival/transitional amplitude with constant matter density
    :param ev: nuetrino energy in MeV
    :param lenth: oscillation lenth in meters
    :param epsi: epsilons
    :param nui: initail flavor
    :param nuf: final flavor
    :param op: oscillation parameters
    :param ne: electron number density in MeV^3
    :return: survival/transitional probability
    """
    op = op.copy()
    dic = {'e': 0, 'mu': 1, 'tau': 2, 'ebar': 0, 'mubar': 1, 'taubar': 2}
    fi = dic[nui]
    ff = dic[nuf]
    lenth = lenth / meter_by_mev
    if nuf[-1] == 'r':
        op['delta'] = -op['delta']
    o23 = np.array([[1, 0, 0],
                   [0, np.cos(op['t23']), np.sin(op['t23'])],
                   [0, -np.sin(op['t23']), np.cos(op['t23'])]])
    u13 = np.array([[np.cos(op['t13']), 0, np.sin(op['t13']) * (np.exp(- op['delta'] * 1j))],
                   [0, 1, 0],
                   [-np.sin(op['t13'] * (np.exp(op['delta'] * 1j))), 0, np.cos(op['t13'])]])
    o12 = np.array([[np.cos(op['t12']), np.sin(op['t12']), 0],
                   [-np.sin(op['t12']), np.cos(op['t12']), 0],
                   [0, 0, 1]])
    umix = o23 @ u13 @ o12
    m = np.diag(np.array([0, op['d21'] / (2 * ev), op['d31'] / (2 * ev)]))
    vf = np.sqrt(2) * gf * ne * (epsi.ee() + np.diag(np.array([1, 0, 0])) + 3 * epsi.eu() + 3 * epsi.ed())
    if nuf[-1] == 'r':
        hf = umix @ m @ umix.conj().T - np.conj(vf)
    else:
        hf = umix @ m @ umix.conj().T + vf
    w, v = np.linalg.eigh(hf)
    res = 0.0
    for i in range(3):
        # for j in range(3):
        theta = (w[i]) * lenth
        res += v[ff, i] * np.conj(v[fi, i]) * (np.cos(theta) - 1j * np.sin(theta))
    return res


def survival_average(ev, epsi=NSIparameters(), op=oscillation_parameters(),
                     ne=2.2 * 6.02e23 * (100 * meter_by_mev) ** 3, nui='e', nuf='e'):
    dic = {'e': 0, 'mu': 1, 'tau': 2, 'ebar': 0, 'mubar': 1, 'taubar': 2}
    op = op.copy()
    fi = dic[nui]
    ff = dic[nuf]
    if nuf[-1] == 'r':
        op['delta'] = -op['delta']
    o23 = np.array([[1, 0, 0],
                   [0, np.cos(op['t23']), np.sin(op['t23'])],
                   [0, -np.sin(op['t23']), np.cos(op['t23'])]])
    u13 = np.array([[np.cos(op['t13']), 0, np.sin(op['t13']) * (np.exp(- op['delta'] * 1j))],
                   [0, 1, 0],
                   [-np.sin(op['t13'] * (np.exp(op['delta'] * 1j))), 0, np.cos(op['t13'])]])
    o12 = np.array([[np.cos(op['t12']), np.sin(op['t12']), 0],
                   [-np.sin(op['t12']), np.cos(op['t12']), 0],
                   [0, 0, 1]])
    umix = o23 @ u13 @ o12
    m = np.diag(np.array([0, op['d21'] / (2 * ev), op['d31'] / (2 * ev)]))
    vf = np.sqrt(2) * gf * ne * ((epsi.ee() + np.diag(np.array([1, 0, 0]))) + 3 * epsi.eu() + 3 * epsi.ed())
    if nuf[-1] == 'r':
        hf = umix @ m @ umix.conj().T - np.conj(vf)
    else:
        hf = umix @ m @ umix.conj().T + vf
    w, v = np.linalg.eigh(hf)
    res = 0.0
    for i in range(3):
        res += v[ff, i] * np.conj(v[fi, i]) * np.conj(v[ff, i]) * v[fi, i]
    return np.real(res)


def survial_atmos(ev, zenith=1.0, epsi=NSIparameters(), op=oscillation_parameters(), nui='e', nuf='e'):
    """
    survival probability of atmospherical neutrino,
    assuming 2 layers of the earth,
    and eath is perfect sphere,
    it depends on zenith angle
    :param ev: nuetrino energy in MeV
    :param zenith: cosine of zenith angle respect to the detector, upward is positive
    :param epsi: NSI parameters
    :param nui: initial flavor
    :param nuf: final flavor
    :param op: oscillation parameters
    :return: survival probability in this direction
    """
    op = op.copy()
    n_core = 11850.56/1.672621898e-27/2*(meter_by_mev**3)
    n_mantle = 4656.61/1.672621898e-27/2*(meter_by_mev**3)
    r_core = 3480000
    r_mantle = 6368000
    cos_th = -np.sqrt(r_mantle**2 - r_core**2) / r_mantle
    if zenith >= 0:
        return 1 if nui == nuf else 0
    elif zenith >= cos_th:
        lenth = -r_mantle * zenith * 2
        return survival_const(ev, lenth, epsi=epsi, nui=nui, nuf=nuf, op=op, ne=n_mantle)
    else:
        vert = r_mantle * np.sqrt(1 - zenith**2)
        l_core = 2 * np.sqrt(r_core**2 - vert**2)
        l_mantle_half = -r_mantle * zenith - l_core / 2
        res = 0
        if nuf[-1] == 'r':
            f_list = ['ebar', 'mubar', 'taubar']
        else:
            f_list = ['e', 'mu', 'tau']
        for i in f_list:
            for j in f_list:
                res += survival_const_amp(ev, l_mantle_half, epsi=epsi, nui=nui, nuf=i, op=op, ne=n_mantle) * \
                       survival_const_amp(ev, l_core, epsi=epsi, nui=i, nuf=j, ne=n_core) * \
                       survival_const_amp(ev, l_mantle_half, epsi=epsi, nui=j, nuf=nuf, ne=n_mantle)
        return np.real(res * np.conj(res))


def survial_atmos_amp(ev, zenith=1.0, epsi=NSIparameters(), op=oscillation_parameters(), nui='e', nuf='e'):
    """
    survival amplitude of atmospherical neutrino
    assuming 2 layers of the earth,
    and eath is perfect sphere,
    it depends on zenith angle
    :param ev: nuetrino energy in MeV
    :param zenith: cosine of zenith angle respect to the detector, upward is positive
    :param epsi: NSI parameters
    :param nui: initial flavor
    :param nuf: final flavor
    :param op: oscillation parameters
    :return: survival probability in this direction
    """
    op = op.copy()
    n_core = 11850.56/1.672621898e-27/2*(meter_by_mev**3)
    n_mantle = 4656.61/1.672621898e-27/2*(meter_by_mev**3)
    r_core = 3480000
    r_mantle = 6368000
    cos_th = -np.sqrt(r_mantle**2 - r_core**2) / r_mantle
    if zenith >= 0:
        return 1 if nui == nuf else 0
    elif zenith >= cos_th:
        lenth = -r_mantle * zenith * 2
        return survival_const(ev, lenth, epsi=epsi, nui=nui, nuf=nuf, op=op, ne=n_mantle)
    else:
        vert = r_mantle * np.sqrt(1 - zenith**2)
        l_core = 2 * np.sqrt(r_core**2 - vert**2)
        l_mantle_half = -r_mantle * zenith - l_core / 2
        res = 0
        if nuf[-1] == 'r':
            f_list = ['ebar', 'mubar', 'taubar']
        else:
            f_list = ['e', 'mu', 'tau']
        for i in f_list:
            for j in f_list:
                res += survival_const_amp(ev, l_mantle_half, epsi=epsi, nui=nui, nuf=i, op=op, ne=n_mantle) * \
                       survival_const_amp(ev, l_core, epsi=epsi, nui=i, nuf=j, ne=n_core) * \
                       survival_const_amp(ev, l_mantle_half, epsi=epsi, nui=j, nuf=nuf, ne=n_mantle)
        return res


class Oscillator:
    def __init__(self, layers, nsi_parameter: NSIparameters, oscillation_parameter: OSCparameters, **kwargs):
        """
        init
        :param layers:
        :param nsi_parameter:
        :param oscillation_parameter:
        :param kwargs: the parameters that goes into each layer
        """
        self.layers = layers
        self.nsi_parameter = nsi_parameter
        self.oscillation_paramter = oscillation_parameter
        self.kwargs = kwargs

    def _dfs(self, ev, amplist, inter, cur_layer, cur_value, nui, nuf):
        if cur_layer == len(self.layers)-1:
            cur_value *= self.layers[cur_layer](ev, nui=nui, nuf=nuf, epsi=self.nsi_parameter, op=self.oscillation_paramter, **self.kwargs)
            amplist.append(cur_value)
            return
        for internu in inter:
            cv = cur_value * self.layers[cur_layer](ev, nui=nui, nuf=internu, epsi=self.nsi_parameter, op=self.oscillation_paramter, **self.kwargs)
            self._dfs(ev, amplist, inter, cur_layer+1, cv, internu, nuf)

    def transition_probability(self, ev, nui, nuf):
        if (nui[-1] == 'r' and nuf[-1] != 'r') or (nui[-1] != 'r' and nuf[-1] == 'r'):
            return 0
        inter = ['e', 'mu', 'tau']
        if nui[-1] == 'r':
            inter = ['ebar', 'mubar', 'taubar']
        amplist = []
        self._dfs(ev, amplist, inter, 0, 1, nui, nuf)
        amp = sum(amplist)
        return np.real(amp * np.conj(amp))

    def transform(self, flux):
        if flux.nu is None:
            nu = None
        else:
            nu = {'ev': flux.ev}
            for flavor in ['e', 'mu', 'tau']:
                if flux.nu[flavor] is not None:
                    if 'e' not in nu:
                        nu['e'] = np.zeros_like(flux.ev)
                        nu['mu'] = np.zeros_like(flux.ev)
                        nu['tau'] = np.zeros_like(flux.ev)
                    for i in range(flux.ev.shape[0]):
                        nu['e'][i] += flux.nu[flavor][i] * self.transition_probability(flux.ev[i], flavor, 'e')
                        nu['mu'][i] += flux.nu[flavor][i] * self.transition_probability(flux.ev[i], flavor, 'mu')
                        nu['tau'][i] += flux.nu[flavor][i] * self.transition_probability(flux.ev[i], flavor, 'tau')
            for flavor in ['ebar', 'mubar', 'taubar']:
                if flux.nu[flavor] is not None:
                    if 'ebar' not in nu:
                        nu['ebar'] = np.zeros_like(flux.ev)
                        nu['mubar'] = np.zeros_like(flux.ev)
                        nu['taubar'] = np.zeros_like(flux.ev)
                    for i in range(flux.ev.shape[0]):
                        nu['ebar'][i] += flux.nu[flavor][i] * self.transition_probability(flux.ev[i], flavor, 'ebar')
                        nu['mubar'][i] += flux.nu[flavor][i] * self.transition_probability(flux.ev[i], flavor, 'mubar')
                        nu['taubar'][i] += flux.nu[flavor][i] * self.transition_probability(flux.ev[i], flavor, 'taubar')
        if flux.delta_nu is None:
            dnu = None
        else:
            dnu = {}
            for flavor in ['e', 'mu', 'tau']:
                if flux.delta_nu[flavor] is not None:
                    if 'e' not in dnu:
                        dnu['e'] = []
                        dnu['mu'] = []
                        dnu['tau'] = []
                    for d in flux.delta_nu[flavor]:
                        dnu['e'].append((d[0], d[1]*self.transition_probability(d[0], flavor, 'e')))
                        dnu['mu'].append((d[0], d[1]*self.transition_probability(d[0], flavor, 'mu')))
                        dnu['tau'].append((d[0], d[1]*self.transition_probability(d[0], flavor, 'tau')))
            for flavor in ['ebar', 'mubar', 'taubar']:
                if flux.delta_nu[flavor] is not None:
                    if 'ebar' not in dnu:
                        dnu['ebar'] = []
                        dnu['mubar'] = []
                        dnu['taubar'] = []
                    for d in flux.delta_nu[flavor]:
                        dnu['ebar'].append((d[0], d[1]*self.transition_probability(d[0], flavor, 'ebar')))
                        dnu['mubar'].append((d[0], d[1]*self.transition_probability(d[0], flavor, 'mubar')))
                        dnu['taubar'].append((d[0], d[1]*self.transition_probability(d[0], flavor, 'taubar')))
        from .flux import NeutrinoFlux
        return NeutrinoFlux(continuous_fluxes=nu, delta_fluxes=dnu, norm=flux.norm/((100 * meter_by_mev) ** 2))

    def change_parameters(self, **kwargs):
        for k, v in kwargs.items():
            self.kwargs[k] = v


class OscillatorFactory:
    def __init__(self):
        self.oscillator_list = ['solar', 'atmospheric']

    def print_available(self):
        print(self.oscillator_list)

    def get(self, oscillator_name, **kwargs):
        if oscillator_name not in self.oscillator_list:
            raise Exception('such oscillator not in factory yet, consider build your own.')
        if oscillator_name == 'solar':
            return Oscillator([survival_solar_amp], NSIparameters(), OSCparameters())
        if oscillator_name == 'atmospheric':
            if 'zenith' not in kwargs:
                raise Exception('please specify zenith angle')
            return Oscillator([survial_atmos_amp], NSIparameters(), OSCparameters(), **kwargs)
