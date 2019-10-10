#!/usr/bin/env python3
"""
This example wants to illustrate when unbinned fits are used and when histogram fits are used
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from multiprocessing import Pool

from kafe2 import UnbinnedFit
from kafe2 import HistContainer, HistFit, HistCostFunction_Chi2


np.random.seed(1009131719)


def pdf(x, mu=0, sigma=1):
    return norm(mu, sigma).pdf(x)


def cdf(x, mu=0, sigma=1):
    return norm(mu, sigma).cdf(x)


class Fitters:
    def __init__(self, size, low, high, steps):
        self.data = self.gen_data(int(size))
        self.steps = self.gen_steps(low, high, steps)
        self.borders = (np.min(self.data), np.max(self.data))
        self.n_bins = None
        self.minimizer = "iminuit"

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
        fit = UnbinnedFit(data, model_density_function=pdf, minimizer=self.minimizer)
        fit.do_fit()
        params = fit.parameter_values
        errors = fit.parameter_errors
        return [params, errors]

    def do_hist(self, step):
        hist_cont = HistContainer(n_bins=self.n_bins, bin_range=self.borders, fill_data=self.data[0:step])
        hist_fit = HistFit(hist_cont, model_density_function=pdf, model_density_antiderivative=cdf,
                           minimizer=self.minimizer)
        #                   cost_function=HistCostFunction_Chi2(errors_to_use=None))
        hist_fit.do_fit()
        params = hist_fit.parameter_values
        errors = hist_fit.parameter_errors
        return [params, errors]

    def do_fits(self):
        p = Pool(processes=10)
        _result = [p.map(self.do_unbinned, [i for i in self.steps])]
        for n in [3, 6, 10, 50]:
            p = Pool(processes=10)
            self.n_bins = n
            _result.append(p.map(self.do_hist, [i for i in self.steps]))
        return np.array(_result)


if __name__ == '__main__':
    fitters = Fitters(1e4, 4, 1e4, 50)
    result = fitters.do_fits()
    # mu_lim = [-0.25, 0.25]
    # sig_lim = [0.2, 1.1]
    helper_dict = {3: {'index': 1, 'loc': 0, 'title': '3 Bins'}, 6: {'index': 2, 'loc': 1, 'title': '6 Bins'},
                   10: {'index': 3, 'loc': 2, 'title': '10 Bins'}, 50: {'index': 4, 'loc': 3, 'title': '50 Bins'},
                   0: {'index': 0, 'loc': 4, 'title': 'Unbinned'}}
    fig, axs = plt.subplots(nrows=5, ncols=2, sharex=True)
    fig.set_size_inches(8.26, 11.7)
    fig.set_dpi(300)
    for key, params in helper_dict.items():
        index = params['index']
        loc = params['loc']
        title = params['title']
        ax = axs[loc, 0]
        # ax.errorbar(fitters.steps, result[index, :, 0, 0], yerr=result[index, :, 1, 0], fmt='o')
        ax.scatter(fitters.steps, (0-result[index, :, 0, 0])/result[index, :, 1, 0])
        ax.plot(fitters.steps, np.zeros(len(fitters.steps)), 'r--')
        # ax.set_ylim(mu_lim[0], mu_lim[1])
        ax.set_title(title+' $\\mu$')
        ax = axs[loc, 1]
        # ax.errorbar(fitters.steps, result[index, :, 0, 1], yerr=result[index, :, 1, 1], fmt='o')
        ax.scatter(fitters.steps, (0 - result[index, :, 0, 1]) / result[index, :, 1, 1])
        ax.plot(fitters.steps, np.ones(len(fitters.steps)), 'r--')
        # ax.set_ylim(sig_lim[0], sig_lim[1])
        ax.set_title(title+r' $\sigma$')
    plt.xscale('log')
    plt.savefig('out1.png')


"""
improvements:
scale: (ref-estimate)/error
dashed line around ref, 0 if scaled
parallelize fits
save fit results, load when plotting
check if empty bins have an influence on result: n_bins -> inf likelihood?
"""