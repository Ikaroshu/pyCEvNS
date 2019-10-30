from .constants import *
from .helper import *


class Axion:
    def __init__(self, photon_rates, axion_mass, axion_coupling, target_mass, target_z, target_photon_cross, detector_distance, min_decay_length):
        self.photon_rates = photon_rates # per second
        self.axion_mass = axion_mass # MeV
        self.axion_coupling = axion_coupling # MeV^-1
        self.target_mass = target_mass # MeV
        self.target_z = target_z
        self.target_photon_cross = target_photon_cross # cm^2
        self.detector_distance = detector_distance # meter
        self.min_decay_length = min_decay_length
        self.photon_energy = []
        self.photon_weight = []
        self.axion_energy = []
        self.axion_weight = []
        self.simulate()

    def form_factor(self):
        # place holder for now, included in the cross section
        return 1

    def photon_axion_cross(self, pgamma):
        # Primakoff
        # def func(cs):
        #     t = self.axion_mass**2 + 2*(-pgamma**2+pgamma*np.sqrt(pgamma**2-self.axion_mass**2)*cs)
        #     return (1-cs**2)/t**2
        # return 1/4*self.axion_coupling**2*1/137*self.target_z**2*(pgamma**2-self.axion_mass**2)**2*quad(func, -1, 1)[0]*self.form_factor()
        ma = self.axion_mass
        it = 1/(ma**2*pgamma**2-pgamma**4)+(ma**2-2*pgamma**2)*np.arctanh(2*pgamma*np.sqrt(-ma**2+pgamma**2)/(ma**2-2*pgamma**2))/(2*pgamma**3*(-ma**2+pgamma**2)**1.5)
        return 1/4*self.axion_coupling**2*1/137*self.target_z**2*(pgamma**2-self.axion_mass**2)**2*it*self.form_factor()

    def axion_probability(self, pgamma):
        # target_n_gamma is in cm^2
        # assuming that target is thick enough and photon cross section is large enough that all photon is converted / absorbed
        cross_prim = self.photon_axion_cross(pgamma)
        return cross_prim / (cross_prim + (self.target_photon_cross/(100*meter_by_mev)**2))

    def simulate_single(self, energy, rate, nsamplings=1000):
        pgamma=energy
        ma = self.axion_mass
        if energy <= 1.5*self.axion_mass or np.abs(2*pgamma*np.sqrt(-ma**2+pgamma**2)/(ma**2-2*pgamma**2))>=1:
            return
        prob = self.axion_probability(energy)
        axion_count = 0
        axion_p = np.sqrt(energy ** 2 - self.axion_mass ** 2)
        axion_v = axion_p / energy
        axion_boost = energy / self.axion_mass
        tau = 64 * np.pi / (self.axion_coupling ** 2 * self.axion_mass ** 3) * axion_boost
        axion_decay = np.random.exponential(tau, nsamplings)
        axion_pos = axion_decay * axion_v
        photon_cs = np.random.uniform(-1, 1, nsamplings)
        for i in range(nsamplings):
            photon1_momentum = np.array([self.axion_mass/2, self.axion_mass/2*np.sqrt(1-photon_cs[i]**2), 0, self.axion_mass/2*photon_cs[i]])
            photon1_momentum = lorentz_boost(photon1_momentum, np.array([0, 0, axion_v]))
            photon2_momentum = np.array([self.axion_mass/2, -self.axion_mass/2*np.sqrt(1-photon_cs[i]**2), 0, -self.axion_mass/2*photon_cs[i]])
            photon2_momentum = lorentz_boost(photon2_momentum, np.array([0, 0, axion_v]))
            r = axion_pos[i]
            pos = np.array([0, 0, r])
            if r > self.detector_distance / meter_by_mev:
                axion_count += 1
                threshold = np.sqrt(r**2 - (self.detector_distance / meter_by_mev)**2) / r
                cs1 = np.sum(-photon1_momentum[1:] * pos) / np.sqrt(np.sum(photon1_momentum[1:] ** 2) * r ** 2)
                cs2 = np.sum(-photon2_momentum[1:] * pos) / np.sqrt(np.sum(photon2_momentum[1:] ** 2) * r ** 2)
                if cs1 >= threshold:
                    self.photon_energy.append(photon1_momentum[0])
                    self.photon_weight.append(rate * prob / nsamplings)
                if cs2 >= threshold:
                    self.photon_energy.append(photon2_momentum[0])
                    self.photon_weight.append(rate * prob / nsamplings)
            elif r > self.min_decay_length / meter_by_mev:
                self.photon_energy.append(photon1_momentum[0])
                self.photon_weight.append(rate*prob/nsamplings)
                self.photon_energy.append(photon2_momentum[0])
                self.photon_weight.append(rate*prob/nsamplings)
        self.axion_energy.append(energy)
        self.axion_weight.append(axion_count/nsamplings*rate*prob)

    def simulate(self, nsamplings=1000):
        self.photon_energy = []
        self.photon_weight = []
        self.axion_energy = []
        self.axion_weight = []
        for f in self.photon_rates:
            self.simulate_single(f[0], f[1], nsamplings)

    def photon_events(self, detector_area, detection_time, threshold):
        res = 0
        for i in range(len(self.photon_energy)):
            if self.photon_energy[i] >= threshold:
                res += self.photon_weight[i]
        return res * detection_time * detector_area / (4*np.pi*self.detector_distance**2)

    def scatter_events(self, detector_number, detector_z, detection_time, threshold):
        res = 0
        for i in range(len(self.axion_energy)):
            if self.axion_energy[i] >= threshold:
                res += self.axion_weight[i] * self.photon_axion_cross(self.axion_energy[i])
        return res * meter_by_mev**2 * detection_time * detector_number / (4 * np.pi * self.detector_distance ** 2) * detector_z**2 / self.target_z**2
