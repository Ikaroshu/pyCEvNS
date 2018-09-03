"""
some helper classes and functions
"""


from numpy import *


class LinearInterp:
    """
    linear interpolation that deals with numbers
    for better performance than interp1d
    """
    def __init__(self, x: ndarray, y: ndarray):
        self.x = x.copy()
        self.y = y.copy()
        ind = argsort(self.x)
        self.x = self.x[ind]
        self.y = self.y[ind]

    def __call__(self, xv):
        if xv < self.x[0] or xv > self.x[-1]:
            return 0
        elif xv == self.x[0]:
            return self.y[0]
        ind = searchsorted(self.x, xv)
        slope = (self.y[ind] - self.y[ind-1]) / (self.x[ind] - self.x[ind-1])
        return slope * (xv - self.x[ind-1]) + self.y[ind-1]
