"""
detector related class and functions
"""

import json

import numpy as np
import pkg_resources


class Detector:
    """
    detector class
    """
    def __init__(self, det_type):
        """
        initializing Detector,
        it reads ./det_init.json for detector information,
        if not found, asking for inputing detector information
        :param det_type: name of the detector
        """
        self.det_type = det_type
        fpath = pkg_resources.resource_filename(__name__, 'data/det_init.json')
        f = open(fpath, 'r')
        det_file = json.load(f)
        f.close()
        if det_type.lower() in det_file:
            det_info = det_file[det_type.lower()]
            self.iso = det_info['iso']
            self.z = np.array(det_info['z'])
            self.n = np.array(det_info['n'])
            self.m = np.array(det_info['m'])
            self.frac = np.array(det_info['frac'])
            self.er_min = det_info['er_min']
            self.er_max = det_info['er_max']
            self.bg = det_info['bg']
            self.bg_un = det_info['bg_un']
        else:
            try:
                f = open('./det_init.json', 'x+')
            except FileExistsError:
                f = open('./det_init.json', 'r+')
            if f.read() == '':
                f.write('{}')
            f.seek(0)
            det_file = json.load(f)
            f.close()
            if det_type.lower() in det_file:
                det_info = det_file[det_type.lower()]
                self.iso = det_info['iso']
                self.z = np.array(det_info['z'])
                self.n = np.array(det_info['n'])
                self.m = np.array(det_info['m'])
                self.frac = np.array(det_info['frac'])
                self.er_min = det_info['er_min']
                self.er_max = det_info['er_max']
                self.bg = det_info['bg']
                self.bg_un = det_info['bg_un']
            else:
                answer = \
                    input("There isn't such detector in det_init.json. Would you like to create one? (y or n)\n")
                if answer == 'y':
                    print('Please entering the following information')
                    self.iso = int(input('Number of isotope: '))
                    while True:
                        self.z = np.array(list(map(int, input('Z for each isotope: ').split(' '))))
                        self.n = np.array(list(map(int, input('N for each isotope: ').split(' '))))
                        self.m = np.array(list(map(float, input('Mass for each isotop (in MeV): ').split(' '))))
                        self.frac = np.array(list(map(float, input('Fraction of each isotope: ').split(' '))))
                        if self.iso == self.z.shape[0] == self.n.shape[0] == self.m.shape[0] == self.frac.shape[0]:
                            break
                        else:
                            print('The number of iso dosen\'t match, please try again.\n')
                    self.er_min = float(input('Minimum detecting energy (in MeV): '))
                    self.er_max = float(input('Maximum detecting energy (in MeV): '))
                    self.bg = float(input('Background (in dru): '))
                    self.bg_un = float(input('Background uncertainty: '))
                    det = {"iso": self.iso,
                           "z": self.z.tolist(), "n": self.n.tolist(), "m": self.m.tolist(), "frac": self.frac.tolist(),
                           "er_min": self.er_min, "er_max": self.er_max, "bg": self.bg, "bg_un": self.bg_un}
                    det_file[det_type.lower()] = det
                    with open('./det_init.json', 'w') as f:
                        json.dump(det_file, f)
                else:
                    raise Exception("No such detector in det_init.json.")
