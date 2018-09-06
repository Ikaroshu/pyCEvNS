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
        """
        plot binned parameter v.s. its probability on the bin
        :param idx: which parameter should be plotted? index starts from 0
        :param credible_level: color different credible level, default is 1\sigma and 2\sigma
        :param nbins: number of bins
        :return: figure and axes object for further fine tuning the plot
        """
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

    def credible_2d(self, idx: tuple, credible_level=(0.6827, 0.9545), nbins=80):
        fig, ax = subplots()
        minx = amin(self.ftxt[:, idx[0]+2])
        miny = amin(self.ftxt[:, idx[1]+2])
        maxx = amax(self.ftxt[:, idx[0]+2])
        maxy = amax(self.ftxt[:, idx[1]+2])
        binxw = (maxx - minx) / nbins
        binyw = (maxy - miny) / nbins
        binx = linspace(minx + binxw/2, maxx - binxw/2, nbins)
        biny = linspace(miny + binyw/2, maxy - binyw/2, nbins)
        xv, yv = meshgrid(binx, biny)
        zv = zeros_like(xv)
        for i in range(self.ftxt.shape[0]):
            posx = int((self.ftxt[i, idx[0]+2] - minx) / binxw)
            posy = int((self.ftxt[i, idx[1]+2] - miny) / binyw)
            if posx < nbins and posy < nbins:
                zv[posx, posy] += self.ftxt[i, 0]
            elif posx < nbins:
                zv[posx, posy-1] += self.ftxt[i, 0]
            elif posy < nbins:
                zv[posx-1, posy] += self.ftxt[i, 0]
            else:
                zv[posx-1, posy-1] += self.ftxt[i, 0]
        sorted_idx = unravel_index(argsort(zv, axis=None)[::-1], zv.shape)
        cl = sort(credible_level)[::-1]
        al = linspace(0.2, 0.3, cl.shape[0])
        for ic in range(cl.shape[0]):
            cz = zeros_like(zv)
            s = 0
            for i in range(sorted_idx[0].shape[0]):
                s += zv[sorted_idx[0][i], sorted_idx[1][i]]
                cz[sorted_idx[0][i], sorted_idx[1][i]] = zv[sorted_idx[0][i], sorted_idx[1][i]]
                if s > cl[ic]:
                    break
            ax.contourf(xv, yv, cz, alpha=al[ic])
        return fig, ax
