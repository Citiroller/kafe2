#!/usr/bin/env python3
"""
This example wants to illustrate when unbinned fits are used and when histogram fits are used
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from multiprocessing import Pool

from kafe2 import UnbinnedFit
from kafe2 import HistContainer, HistFit


np.random.seed(1009131719)


def pdf(x, mu=0, sigma=1):
    return norm(mu, sigma).pdf(x)


def cdf(x, mu=0, sigma=1):
    return norm(mu, sigma).cdf(x)


class Fitters:
    def __init__(self, size, low, high, steps):
        self.data = self.gen_data(size)
        self.steps = self.gen_steps(low, high, steps)
        self.borders = (np.min(self.data), np.max(self.data))
        self.n_bins = None

    @staticmethod
    def gen_data(size=100000):
        return np.random.standard_normal(size)

    @staticmethod
    def gen_steps(low, high, size, log=True):
        if log:
            low = np.log10(low)
            high = np.log10(high)
            return np.logspace(low, high, num=size, dtype=int)-1
        else:
            return np.linspace(low, high, num=size, dtype=int)-1

    def do_unbinned(self, step):
        data = self.data[0:step]
        fit = UnbinnedFit(data, model_density_function=pdf)
        fit.do_fit()
        params = fit.parameter_values
        errors = fit.parameter_errors
        return [params, errors]

    def do_hist(self, step):
        hist_cont = HistContainer(n_bins=self.n_bins, bin_range=self.borders, fill_data=self.data[0:step])
        hist_fit = HistFit(hist_cont, model_density_function=pdf, model_density_antiderivative=cdf)
        hist_fit.do_fit()
        params = hist_fit.parameter_values
        errors = hist_fit.parameter_errors
        return [params, errors]

    def do_fits(self):
        p = Pool(processes=10)
        _result = [p.map(self.do_unbinned, [i for i in self.steps])]
        for n in [10, 50, 100]:
            p = Pool(processes=10)
            self.n_bins = n
            _result.append(p.map(self.do_hist, [i for i in self.steps]))
        return np.array(_result)


if __name__ == '__main__':
    fitters = Fitters(10000, 4, 10000, 50)
    result = fitters.do_fits()
    mu_lim = [-0.25, 0.25]
    sig_lim = [0.3, 1.1]
    fig, axs = plt.subplots(nrows=4, ncols=2, sharex=True)
    ax = axs[0, 0]
    ax.errorbar(fitters.steps, result[1, :, 0, 0], yerr=result[1, :, 1, 0], fmt='o')
    ax.plot(fitters.steps, np.zeros(len(fitters.steps)), 'r--')
    ax.set_ylim(mu_lim[0], mu_lim[1])
    ax.set_title(r'10 Bins $\mu$')
    ax = axs[0, 1]
    ax.errorbar(fitters.steps, result[1, :, 0, 1], yerr=result[1, :, 1, 1], fmt='o')
    ax.plot(fitters.steps, np.ones(len(fitters.steps)), 'r--')
    ax.set_ylim(sig_lim[0], sig_lim[1])
    ax.set_title(r'10 Bins $\sigma$')
    ax = axs[1, 0]
    ax.errorbar(fitters.steps, result[2, :, 0, 0], yerr=result[2, :, 1, 0], fmt='o')
    ax.plot(fitters.steps, np.zeros(len(fitters.steps)), 'r--')
    ax.set_ylim(mu_lim[0], mu_lim[1])
    ax.set_title(r'50 Bins $\mu$')
    ax = axs[1, 1]
    ax.errorbar(fitters.steps, result[2, :, 0, 1], yerr=result[2, :, 1, 1], fmt='o')
    ax.plot(fitters.steps, np.ones(len(fitters.steps)), 'r--')
    ax.set_ylim(sig_lim[0], sig_lim[1])
    ax.set_title(r'50 Bins $\sigma$')
    ax = axs[2, 0]
    ax.errorbar(fitters.steps, result[3, :, 0, 0], yerr=result[3, :, 1, 0], fmt='o')
    ax.plot(fitters.steps, np.zeros(len(fitters.steps)), 'r--')
    ax.set_ylim(mu_lim[0], mu_lim[1])
    ax.set_title(r'100 Bins $\mu$')
    ax = axs[2, 1]
    ax.errorbar(fitters.steps, result[3, :, 0, 1], yerr=result[3, :, 1, 1], fmt='o')
    ax.plot(fitters.steps, np.ones(len(fitters.steps)), 'r--')
    ax.set_ylim(sig_lim[0], sig_lim[1])
    ax.set_title(r'100 Bins $\sigma$')
    ax = axs[3, 0]
    ax.errorbar(fitters.steps, result[0, :, 0, 0], yerr=result[0, :, 1, 0], fmt='o')
    ax.plot(fitters.steps, np.zeros(len(fitters.steps)), 'r--')
    ax.set_ylim(mu_lim[0], mu_lim[1])
    ax.set_title(r'Unbinned $\mu$')
    ax = axs[3, 1]
    ax.errorbar(fitters.steps, result[0, :, 0, 1], yerr=result[0, :, 1, 1], fmt='o')
    ax.plot(fitters.steps, np.ones(len(fitters.steps)), 'r--')
    ax.set_ylim(sig_lim[0], sig_lim[1])
    ax.set_title(r'Unbinned $\sigma$')
    plt.xscale('log')
    plt.show()


"""
improvements:
scale: (ref-estimate)/error
dashed line around ref, 0 if scaled
parallelize fits
save fit results, load when plotting
check if empty bins have an influence on result: n_bins -> inf likelihood?
"""