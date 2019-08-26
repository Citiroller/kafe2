import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from kafe2 import UnbinnedFit, UnbinnedPlot, HistContainer, HistFit, HistPlot


def pdf_mod(x, tau, fbg, k, a, omega, delta):
    pdf1 = np.exp(-x/tau)*(1-a*np.cos(omega*x+delta))
    return k * pdf1 + fbg

def pdf_mod_antidiv(x, tau, fbg, k, a, omega, delta):
    pdf1 = a/2*np.exp(-x/tau)*(np.exp((omega*x+delta)*1j)/(omega*1j-1/tau)-np.exp(-(omega*x+delta)*1j)/(omega*1j+1/tau))
    return k*pdf1 + fbg*x

def pdf(x, tau=2.0, fbg=0.1, a=1.5, b=20):
    pdf1 = np.exp(-x / tau) / tau / (np.exp(-a / tau) - np.exp(-b / tau))
    pdf2 = 1. / (b - a)
    return (1 - fbg) * pdf1 + fbg * pdf2


# data_tau = data_tau[data_tau > 1.5]  # 316840 events left, already saved to tau_lande.dat
data = np.loadtxt('tau_lande.dat', delimiter=',')
data_limits = np.min(data), np.max(data)
data_short = data[0:20]
data_short_limits = np.min(data_short), np.max(data_short)

fit = UnbinnedFit(data=data_short, model_density_function=pdf)
fit.fix_parameter('a', data_short_limits[0])
fit.fix_parameter('b', data_short_limits[1])
fit.do_fit()
fit.report()

# crosscheck if integral from min to max equals 1
integral, error = integrate.quad(pdf, data_short_limits[0], data_short_limits[1], tuple(fit.parameter_values))
assert(np.allclose(1, integral, atol=error))

plot = UnbinnedPlot(fit)
plot.show_fit_info_box()
plot.plot()
plt.show()

"""
hist_data = HistContainer(100, (1.5, 19), fill_data=data)
hist_fit = HistFit(hist_data, model_density_function=pdf)
hist_fit.fix_parameter('a', data_limits[0])
hist_fit.fix_parameter('b', data_limits[1])
hist_fit.do_fit()
hist_fit.report()

# crosscheck if integral from min to max equals 1
integral, error = integrate.quad(pdf, data_limits[0], data_limits[1], tuple(hist_fit.parameter_values))
assert(np.allclose(1, integral, atol=error))

hist_plot = HistPlot(hist_fit)
hist_plot.show_fit_info_box()
hist_plot.plot()
plt.show()
"""
