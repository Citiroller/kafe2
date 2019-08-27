#!/usr/bin/env python3
"""
This example wants to illustrate when unbinned fits are used and when histogram fits are used
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm

from kafe2 import UnbinnedFit, UnbinnedPlotContainer
from kafe2 import HistContainer, HistFit, HistPlotContainer
from kafe2.fit.util.function_library import normal_distribution_pdf

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle


def pdf(x, mu=0, sigma=1):
    return norm(mu, sigma).pdf(x)


def cdf(x, mu=0, sigma=1):
    return norm(mu, sigma).cdf(x)


size = 100000
start = 4
step = 100
np.random.seed(1009131719)
sample = np.random.standard_normal(size+1)

# get parameters from fits
u_fit = UnbinnedFit(sample, model_density_function=pdf, minimizer='scipy')
h_data = []
bins = [10, 50, 100]
for i in bins:
    data = HistContainer(n_bins=i, bin_range=(-3, 5), fill_data=sample[0:10])
    h_data.append(data)
h_fits = []
for data in h_data:
    fit = HistFit(data, model_density_function=pdf, model_density_antiderivative=cdf, minimizer='scipy')
    h_fits.append(fit)

param_results = []
error_results = []
n_points = np.logspace(0.5, 5, num=step, dtype=int)
#n_points = range(start-1, size, step)
for j, i in enumerate(n_points):
    print("Run {} of {}".format(j, len(n_points)))
    u_fit.data = sample[0:i]
    u_fit.do_fit()
    params = [u_fit.parameter_values]
    errors = [u_fit.parameter_errors]
    for data, fit, n in zip(h_data, h_fits, bins):
        data = HistContainer(n_bins=n, bin_range=(-3, 3), fill_data=sample[0:i])
        fit.data = data
        fit.do_fit()
        params.append(fit.parameter_values)
        errors.append(fit.parameter_errors)
    param_results.append(params)
    error_results.append(errors)
param_results = np.array(param_results)
error_results = np.array(error_results)
results = [n_points]
for i in range(len(param_results)):
    results.append(param_results[:, i, 0])  # mu
    results.append(error_results[:, i, 0])  # mu error
    results.append(param_results[:, i, 1])  # sigma
    results.append(error_results[:, i, 1])  # sigma error
np.save('sample_fits', results)
"""

results = np.load('sample_fits.npy', allow_pickle=True)
n_points = results[0]
param_results = results[1]
error_results = results[2]
print(error_results)
plt.xscale('log')
plt.scatter(n_points, param_results[:, 0, 0]/error_results[:, 0, 0], label=r'$\mu$ Unbinned')
plt.scatter(n_points, param_results[:, 1, 0]/error_results[:, 1, 0], label=r'$\mu$ Binned 10')
plt.scatter(n_points, param_results[:, 2, 0]/error_results[:, 2, 0], label=r'$\mu$ Binned 50')
plt.scatter(n_points, param_results[:, 3, 0]/error_results[:, 3, 0], label=r'$\mu$ Binned 100')
plt.legend()
plt.show()
"""
"""
improvements:
scale: (ref-estimate)/error
dashed line around ref, 0 if scaled
parallelize fits
save fit results, load when plotting
check if empty bins have an influence on result: n_bins -> inf likelihood?
"""

'''
fig = p.figure
ims = []
for i in range(100, len(data)):
    f.data = data[:i]
    f.do_fit()
    print(f.parameter_values)
    p.plot()
    # p.show_fit_info_box()
    artists = list(p._artist_store[0].values())
    im = []
    # flatten the array
    for artist in artists:
        try:
            artist = artist[0]
        except TypeError:  # except indexing error for already flattened elements
            pass
        artist.set_animated(True)
        im.append(artist)
    ims.append(im)

anim = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=100)
plt.show()
'''
