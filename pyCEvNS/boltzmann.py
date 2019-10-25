from .constants import *
from .helper import *
from scipy.integrate import solve_ivp


def gammaaff(epsilon, ma, mf=me):
    return epsilon**2*e_charge**2*ma / (12 * np.pi) * (1-4*mf**2/ma**2)**0.5 * (1+2*mf**2/ma**2)


def sigmav(epsilon, ma, z, alphad=0.5, mf=me, mchi_ratio=3):
    mchi = ma/mchi_ratio
    t = z*mchi
    def distri(v):
        return np.exp(-mchi*v**2/t) * 4 * np.pi * v**2
    def func(v):
        echi = mchi / np.sqrt(1-v**2)
        return np.sum(4*alphad/6*epsilon**2*e_charge**2 * np.sqrt(echi**2-mf**2)*(2*echi**2+mchi**2)*(2*echi**2+mf**2) / \
               (echi**3 * (4*echi**2-ma**2)**2 + (ma*gammaaff(epsilon, ma, mf))**2) * distri(v))
    vmax = 1
    return quad(func, 0, vmax)[0]/quad(distri, 0, vmax)[0]


mpl = 1.22e22
mp = mpl/np.sqrt(8*np.pi) # reduced planck mass
t0 = 2.75/1.1605e10
tss = 4/1.1605e10
h0 = 2.2e-18/c_light*meter_by_mev
rho0 = 3*h0**2*mp**2

gdata = np.genfromtxt('./Gstar1.dat')
x = gdata[13010:, 0]*1e3
y = gdata[13010:, 1]
d = 2 + 7/8*(6)*4/11. - gdata[0, 1]
geff = LinearInterp(gdata[:, 0]*1e3, gdata[:, 1], extend=True)
xl = np.array([0.2, 0.1, 0.01])
yl = np.array([geff(0.2)+0.2*d, geff(0.1)+d, geff(0.01)+d])
x = np.hstack((x, xl))
y = np.hstack((y, yl))
geffs = LinearInterp(x, y, extend=True)


# def rho_tilde(t):
#     return np.pi**2/30*geff(t)*t**4
# def hgr(t):
#     return np.sqrt(rho_tilde(t)/3)
# def hh(t):
#     return hgr(t)/mp
#
#
# gp = 4
# def neq(x, m):
#     return gp*(m**2/(2*np.pi))**(1.5)*x**(-1.5)*np.exp(-x)
# def gamma_eq(epsilon, ma, z, alphad=0.5, mf=me, mchi_ratio=3):
#     m = ma/mchi_ratio
#     x = 1/z
#     return sigmav(epsilon, ma, z, alphad, mf, mchi_ratio)*neq(x, m)
# def ss(x, m):
#     t = m/x
#     return 2*np.pi**2/45*geffs(t)*t**3
# def yeq(x, m):
#     return neq(x, m)/ss(x, m)


# this function solves bolzmann equation for DM relic abundance
def boltzmann(epsilon, ma, alphad=0.5, mf=me, mchi_ratio=3, xinit=1, xfin=10000):
    m = ma/mchi_ratio
    def func(x, y):
        return -1/x**2*sigmav(epsilon, ma,1/x, alphad, mf, mchi_ratio) * (y**2-(4*0.192*mp*m*x**1.5*np.exp(-x))**2)
    return solve_ivp(func, (xinit, xfin), [4*0.192*mp*m*np.exp(-1)], method='BDF')
