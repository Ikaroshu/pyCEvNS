from .constants import *
import json


class Detector:
    def __init__(self, det_type):
        try:
            with open('./det_init.json', 'a+') as f:
                det_file = json.load(f)
                if det_type.lower() in det_file:
                    det_info = det_file[det_type.lower()]
                    self.iso = det_info['iso']
                    self.z = array(det_info['z'])
                    self.n = array(det_info['n'])
                    self.m = array(det_info['m'])
                    self.frac = array(det_info['frac'])
                    self.er_min = det_info['er_min']
                    self.er_max = det_info['er_max']
                    self.bg = det_info['bg']
                    self.bg_un = det_info['bg_un']
                else:
                    answer = \
                        input("There isn't such detector in det_init.json. Would you like to create one? (y or n)\n")
                    if answer == 'y':
                        self.detector_creator(f)
                    else:
                        raise Exception("No such detector in det_init.json.")
        except FileNotFoundError as fnfe:
            raise fnfe
            # answer = input("There no det_init.json. Would you like to create one? (y or n)\n")
            # if answer == 'y':
            #     self.detector_creator()
            # elif answer == 'n':
            #     print('You can create a det_init.json file in the current directory,\n'
            #           'please refer to: ...\n')
            #     raise fnfe
            # else:
            #     raise fnfe

    def detector_creator(self, f):
        print('Please entering the following information')
        self.iso = input('Number of isotope: ')
        self.z = array(map(int, input('Z for each isotope: ').split(' ')))
        self.n = array(map(int, input('N for each isotope: ').split(' ')))
        self.m = array(map(int, input('Mass for each isotop (in MeV): ').split(' ')))
        self.frac = array(map(float, input('Fraction of each isotope: ').split(' ')))
        self.er_min = float(input('Minimum detecting energy: '))
        self.er_max = float(input('Maximum detecting energy: '))
        self.bg = float(input('Background (in dru): '))
        self.bg_un = float(input('Background uncertainty: '))
