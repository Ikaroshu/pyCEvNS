"""
dealing with multinest out put
"""

from matplotlib.pyplot import subplots
from numpy import *


class CrediblePlot:
    """
    class for plotting results from multinest
    """
    def __init__(self, file_path: str):
        """
        read .txt file
        :param file_path: path to .txt file
        """
        self.ftxt = genfromtxt(file_path)
        if abs(sum(self.ftxt[:, 0])-1) > 1e-3:
            raise Exception("Invalid file!")

    def credible_1d(self, idx: int, credible_level=(0.6827, 0.9545), nbins=80):
        fig, ax = subplots()
        minx = amin(self.ftxt[:, idx+2])
        maxx = amax(self.ftxt[:, idx+2])
        binw = (maxx - minx) / nbins
        binx = linspace(minx + binw/2, maxx - binw/2, nbins)
        biny = zeros_like(binx)
        for i in range(self.ftxt.shape[0]):
            pos = int((self.ftxt[i, idx+2] - minx) / binw)
            if pos < nbins:
                biny[pos] += self.ftxt[i, 0]
            else:
                biny[pos-1] += self.ftxt[i, 0]
        cl = sort(credible_level)[::-1]
        ax.bar(binx, biny, width=binw, alpha=0.1, color='b')
        sorted_idx = argsort(biny)[::-1]
        al = linspace(0.2, 0.3, cl.shape[0])
        for ic in range(cl.shape[0]):
            s = 0
            cxl = []
            cyl = []
            for i in sorted_idx:
                s += biny[i]
                cyl.append(biny[i])
                cxl.append(binx[i])
                if s > cl[ic]:
                    break
            ax.bar(cxl, cyl, width=binw, alpha=al[ic], color='b')
        return fig, ax
