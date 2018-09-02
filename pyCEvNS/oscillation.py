"""
neutrino oscillation related funtions
"""

from .parameters import *

# solar number density at r=0.05 solar radius, unit is MeV^3 (natural unit)
__ne_solar = 4.163053492437814e-07
__nu_solar = 1.0053941490424488e-06
__nd_solar = 7.618722503535536e-07


def surv_solar(ev, epsi=NSIparameters(), op=oscillation_parameters(), nui='e', nuf='e'):
    """
    calculating survival/transitional probability of solar neutrino
    :param ev: neutrino energy in MeV
    :param epsi: nsi parameters
    :param nui: intial state
    :param nuf: final state, 0: electron neutrino, 1: muon neutrino, 2: tau neutrino
    :param op: oscillation parameters
    :return: survival/transitional probability
    """
    opt = op.copy()
    dic = {'e': 0, 'mu': 1, 'tau': 2}
    fi = dic[nui]
    ff = dic[nuf]
    o23 = matrix([[1, 0, 0],
                  [0, cos(opt['t23']), sin(opt['t23'])],
                  [0, -sin(opt['t23']), cos(opt['t23'])]])
    u13 = matrix([[cos(opt['t13']), 0, sin(opt['t13']) * (e ** (- opt['delta'] * 1j))],
                  [0, 1, 0],
                  [-sin(opt['t13'] * (e ** (opt['delta'] * 1j))), 0, cos(opt['t13'])]])
    o12 = matrix([[cos(opt['t12']), sin(opt['t12']), 0],
                  [-sin(opt['t12']), cos(opt['t12']), 0],
                  [0, 0, 1]])
    umix = o23 * u13 * o12
    m = diag(array([0, opt['d21'] / (2 * ev), op['d31'] / (2 * ev)]))
    v = sqrt(2) * gf * (__ne_solar * epsi.ee() + __nu_solar * epsi.eu() + __nd_solar * epsi.ed())
    hvac = umix * m * umix.H

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
        avec = array(vec)
        return matrix([avec[:, minindex], avec[:, midindex], avec[:, maxindex]]).T

    wr, vecr = linalg.eigh(hvac + v)
    utr = sorteig(wr, vecr)
    ws, vecs = linalg.eigh(hvac)
    uts = sorteig(ws, vecs)
    res = 0
    for i in range(3):
        res += conj(utr[0, i]) * utr[0, i] * conj(uts[ff, i]) * uts[ff, i]
    return real(res)

# using Caylay-Hamilton theorem to calculate survival probability, it has probems at transitsion probabilities
#
# def survival_probability(ev, lenth, epsi=NSIparameters(), nui=0, nuf=0,
#                          op=ocsillation_parameters(), ne=2.2*6.02e23*(100*meter_by_mev)**3):
#     o23 = matrix([[1, 0, 0],
#                   [0, cos(op['t23']), sin(op['t23'])],
#                   [0, -sin(op['t23']), cos(op['t23'])]])
#     u13 = matrix([[cos(op['t13']), 0, sin(op['t13']) * (e ** (- op['delta'] * 1j))],
#                   [0, 1, 0],
#                   [-sin(op['t13'] * (e ** (op['delta'] * 1j))), 0, cos(op['t13'])]])
#     o12 = matrix([[cos(op['t12']), sin(op['t12']), 0],
#                   [-sin(op['t12']), cos(op['t12']), 0],
#                   [0, 0, 1]])
#     umix = o23 * u13 * o12
#     m = diag(array([0, op['d21'] / (2 * ev), op['d31'] / (2 * ev)]))
#     vf = sqrt(2) * gf * ne * (epsi.ee() + 3 * epsi.eu() + 3 * epsi.ed())
#     hf = umix * m * umix.H + vf
#     w, v = linalg.eigh(hf)
#     # print(w)
#     b = e**(-1j*w*lenth)
#     # print(b)
#     a = array([[1, 1, 1], -1j * lenth * w, -lenth**2 * w**2]).T
#     # print(a)
#     x = linalg.solve(a, b)
#     tmatrix = x[0] + -1j * lenth * x[1] * hf - lenth**2 * x[2] * hf.dot(hf)
#     # print(tmatrix)
#     return abs(tmatrix[nui, nuf])**2


def surv_const(ev, lenth, epsi=NSIparameters(), op=oscillation_parameters(),
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
    dic = {'e': 0, 'mu': 1, 'tau': 2, 'ebar': 0, 'mubar': 1, 'taubar': 2}
    fi = dic[nui]
    ff = dic[nuf]
    lenth = lenth / meter_by_mev
    opt = op.copy()
    if nuf[-1] == 'r':
        opt['delta'] = -opt['delta']
    o23 = matrix([[1, 0, 0],
                  [0, cos(opt['t23']), sin(opt['t23'])],
                  [0, -sin(opt['t23']), cos(opt['t23'])]])
    u13 = matrix([[cos(opt['t13']), 0, sin(opt['t13']) * (e ** (- opt['delta'] * 1j))],
                  [0, 1, 0],
                  [-sin(opt['t13'] * (e ** (opt['delta'] * 1j))), 0, cos(opt['t13'])]])
    o12 = matrix([[cos(opt['t12']), sin(opt['t12']), 0],
                  [-sin(opt['t12']), cos(opt['t12']), 0],
                  [0, 0, 1]])
    umix = o23 * u13 * o12
    m = diag(array([0, opt['d21'] / (2 * ev), opt['d31'] / (2 * ev)]))
    vf = sqrt(2) * gf * ne * (epsi.ee() + 3 * epsi.eu() + 3 * epsi.ed())
    if nuf[-1] == 'r':
        hf = umix * m * umix.H - conj(vf)
    else:
        hf = umix * m * umix.H + vf
    w, v = linalg.eigh(hf)
    res = 0.0
    for i in range(3):
        for j in range(3):
            theta = (w[i]-w[j]) * lenth
            res += v[ff, i] * conj(v[fi, i]) * conj(v[ff, j]) * v[fi, j] * (cos(theta) - 1j * sin(theta))
    return real(res)


def survival_average(ev, epsi=NSIparameters(), op=oscillation_parameters(),
                     ne=2.2 * 6.02e23 * (100 * meter_by_mev) ** 3, nui='e', nuf='e'):
    opt = op.copy()
    dic = {'e': 0, 'mu': 1, 'tau': 2, 'ebar': 0, 'mubar': 1, 'taubar': 2}
    fi = dic[nui]
    ff = dic[nuf]
    if nuf[-1] == 'r':
        opt['delta'] = -opt['delta']
    o23 = matrix([[1, 0, 0],
                  [0, cos(opt['t23']), sin(opt['t23'])],
                  [0, -sin(opt['t23']), cos(opt['t23'])]])
    u13 = matrix([[cos(opt['t13']), 0, sin(opt['t13']) * (e ** (- opt['delta'] * 1j))],
                  [0, 1, 0],
                  [-sin(opt['t13'] * (e ** (opt['delta'] * 1j))), 0, cos(opt['t13'])]])
    o12 = matrix([[cos(opt['t12']), sin(opt['t12']), 0],
                  [-sin(opt['t12']), cos(opt['t12']), 0],
                  [0, 0, 1]])
    umix = o23 * u13 * o12
    m = diag(array([0, opt['d21'] / (2 * ev), opt['d31'] / (2 * ev)]))
    vf = sqrt(2) * gf * ne * (epsi.ee() + 3 * epsi.eu() + 3 * epsi.ed())
    if nuf[-1] == 'r':
        hf = umix * m * umix.H - conj(vf)
    else:
        hf = umix * m * umix.H + vf
    w, v = linalg.eigh(hf)
    res = 0.0
    for i in range(3):
        res += v[ff, i] * conj(v[fi, i]) * conj(v[ff, i]) * v[fi, i]
    return real(res)


def survial_atmos(ev, zenith, epsi=NSIparameters(), op=oscillation_parameters(), nui='e', nuf='e'):
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
    n_core = 11850.56/1.672621898e-27/2*(meter_by_mev**3)
    n_mantle = 4656.61/1.672621898e-27/2*(meter_by_mev**3)
    r_core = 3480000 / meter_by_mev
    r_mantle = 6368000 / meter_by_mev
    cos_th = -sqrt(r_mantle**2 - r_core**2) / r_core
    if zenith >= 0:
        return 1 if nui == nuf else 0
    elif zenith >= cos_th:
        lenth = -r_mantle * zenith * 2
        return surv_const(ev, lenth, epsi=epsi, nui=nui, nuf=nuf, op=op, ne=n_mantle)
    else:
        vert = r_mantle * sqrt(1 - zenith**2)
        l_core = 2 * sqrt(r_core**2 - vert**2)
        l_mantle_half = -r_mantle * zenith - l_core / 2
        res = 0
        if nuf[-1] == 'r':
            f_list = ['ebar', 'mubar', 'taubar']
        else:
            f_list = ['e', 'mu', 'tau']
        for i in f_list:
            for j in f_list:
                res += surv_const(ev, l_mantle_half, epsi=epsi, nui=nui, nuf=i, op=op, ne=n_mantle) *\
                    surv_const(ev, l_core, epsi=epsi, nui=i, nuf=j, ne=n_core) *\
                    surv_const(ev, l_mantle_half, epsi=epsi, nui=j, nuf=nuf, ne=n_mantle)
        return res
