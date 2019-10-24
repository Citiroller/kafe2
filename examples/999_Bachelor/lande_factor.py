import numpy as np
import matplotlib.pyplot as plt
from kafe2 import HistFit, HistContainer, Plot, MultiFit

def events_top(x, tau=2.2, k_top=1.0, a_bar_top=1.0, omega=1.0, delta=1.0, f_top=0.1):
    return k_top*np.exp(-x/tau)+a_bar_top*np.exp(-x/tau)*np.cos(omega*x+delta)+f_top

def events_bot(x, tau=2.2, k_bot=1.0, a_bar_bot=1.0, omega=1.0, delta=1.0, f_bot=0.1):
    return k_bot*np.exp(-x/tau)-a_bar_bot*np.exp(-x/tau)*np.cos(omega*x+delta)+f_bot

def decay_top(x, tau=2.2, k_top=1.0, f_top=0.1):
    return k_top*np.exp(-x/tau)+f_top

def decay_bot(x, tau=2.2, k_bot=1.0, f_bot=0.1):
    return k_bot*np.exp(-x/tau)+f_bot


class Lande:
    def __init__(self, fname="dpFilt_190325-0148.dat", low_limit=2, high_limit=16):
        self.top_events = None
        self.bot_events = None
        self.hist_top = None
        self.hist_bot = None
        self.fit_top = None
        self.fit_bot = None
        self._load_data(fname)
        self.gen_fits(low_limit, high_limit)

    def _load_data(self, fname):
        _data = np.loadtxt(fname, delimiter=',')
        print("Data size before strip: {}".format(len(_data)))
        # remove physically impossible events
        delete = []
        for i, event in enumerate(_data):
            if (event[3] > 0 or event[4] > 0) and event[5] > 0:
                delete.append(i)
        _data = np.delete(_data, delete, 0)
        print("Data size after strip: {}".format(len(_data)))
        self.top_events = np.array(_data[:, 3])
        self.bot_events = np.array(_data[:, 5])

    def _gen_data(self, size):
        from scipy.constants import e
        from scipy.constants import physical_constants
        g = 2.0023318418  # lande factor of the myon
        tau = 2.1969811e-6  # mean decay time
        b = 5e-6  # magnetic field in T
        m = physical_constants["muon mass"][0]
        omega = g * e * b / (2 * m)
        print("Expected omega ist {}".format(omega))
        k = 0.1  # amount of decay
        a = 0.01  # amount of precession
        delta = np.pi  # phase delay
        f = 0.01  # background
        x = np.linspace(self._low_limit, self._high_limit, 100)
        _max = np.amax(events_top(x, tau, k, a, omega, delta, f))
        self.top_events = np.zeros(size, dtype=float)
        self.bot_events = np.zeros(int(size / 2), dtype=float)
        random = np.random.RandomState(42)
        i = 0
        while i < size:
            r = random.random() * (self._high_limit - self._low_limit) + self._low_limit
            y = random.random() * _max
            if events_top(r, tau, k, a, omega, delta, f) > y:
                self.top_events[i] = r
                i += 1

    def gen_fits(self, low_limit, high_limit):
        self.hist_top = HistContainer(n_bins=100, bin_range=(low_limit, high_limit),
                                      fill_data=self.top_events)
        self.hist_bot = HistContainer(n_bins=100, bin_range=(low_limit, high_limit),
                                      fill_data=self.bot_events)
        self.fit_top = HistFit(self.hist_top, events_top, cost_function='nllr')
        self.fit_bot = HistFit(self.hist_bot, events_bot, cost_function='nllr')
        self.fit_multi = MultiFit((self.fit_top, self.fit_bot), minimizer='iminuit')
        self.decay_fit_top = HistFit(self.hist_top, decay_top)
        self.decay_fit_bot = HistFit(self.hist_bot, decay_bot)
        self.decay_fit_multi = MultiFit((self.decay_fit_top, self.decay_fit_bot), minimizer='iminuit')

    def decay_do_fit(self):
        self.decay_fit_multi.do_fit()
        # self.decay_fit_multi.report()

    def do_fit(self):
        self.fit_multi.do_fit()
        # self.fit_multi.report()

    def plot(self):
        _plot = Plot(self.fit_multi, separate_figures=True)
        # _plot = Plot((self.fit_top, self.fit_bot), separate_figures=True)
        _plot.plot()


if __name__ == '__main__':
    high_limits = np.linspace(12, 14, 20)
    cost = np.zeros((len(high_limits), 2))
    params = np.array([])
    lande = Lande(low_limit=2, high_limit=19)
    for i, lim in enumerate(high_limits):
        lande.gen_fits(low_limit=2, high_limit=lim)
        lande.decay_do_fit()  # do decay fit for better starting values of  the decay params
        for _par_name, _par_val, _par_err in zip(lande.decay_fit_multi.parameter_names,
                                                 lande.decay_fit_multi.parameter_values,
                                                 lande.decay_fit_multi.parameter_errors):
            print('%s: %.3E +- %.3E' % (_par_name, _par_val, _par_err))
        lande.fit_multi.set_parameter_values(**lande.decay_fit_multi.parameter_name_value_dict)
        start_values = {'omega': 3, 'delta': np.pi, 'a_bar_top': 0.01, 'a_bar_bot': 0.01}
        lande.fit_multi.set_parameter_values(**start_values)
        lande.fit_multi.limit_parameter('delta', (0, 2 * np.pi))
        print(lande.fit_multi.parameter_name_value_dict)
        lande.do_fit()
        print()
        for _par_name, _par_val, _par_err in zip(lande.fit_multi.parameter_names, lande.fit_multi.parameter_values,
                                                 lande.fit_multi.parameter_errors):
            print('%s: %.3E +- %.3E' % (_par_name, _par_val, _par_err))
        params = np.append(params, list(zip(lande.fit_multi.parameter_names, lande.fit_multi.parameter_values,
                                            lande.fit_multi.parameter_errors)))
        for j, fit in enumerate(lande.fit_multi.fits):
            print("cost/ndf: {}".format(fit.cost_function_value / fit._cost_function.ndf))
            cost[i][j] = fit.cost_function_value / fit._cost_function.ndf

    np.save("cost_check_around_min_min", [high_limits, params, cost])
    plt.plot(high_limits, np.sum(cost, axis=1))
    plt.show()
