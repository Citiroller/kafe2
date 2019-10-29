import numpy as np
import matplotlib.pyplot as plt
from kafe2 import HistFit, HistContainer, Plot, MultiFit
from six import iteritems

def events_top(x, tau=2.2, k_top=1.0, a_bar_top=1.0, omega=1.0, delta=1.0, f_top=0.1):
    return k_top*np.exp(-x/tau)+a_bar_top*np.exp(-x/tau)*np.cos(omega*x+delta)+f_top

def events_bot(x, tau=2.2, k_bot=1.0, a_bar_bot=1.0, omega=1.0, delta=1.0, f_bot=0.1):
    return k_bot*np.exp(-x/tau)-a_bar_bot*np.exp(-x/tau)*np.cos(omega*x+delta)+f_bot

def decay_top(x, tau=2.2, k_top=1.0, f_top=0.1):
    return k_top*np.exp(-x/tau)+f_top

def decay_bot(x, tau=2.2, k_bot=1.0, f_bot=0.1):
    return k_bot*np.exp(-x/tau)+f_bot


class Lande:
    def __init__(self, low_limit=2, high_limit=16, b=5e-3, gen_data=False, fname="dpFilt_190325-0148.dat"):
        self.b = b
        self.top_events = None
        self.bot_events = None
        self.hist_top = None
        self.hist_bot = None
        self.fit_top = None
        self.fit_bot = None
        if gen_data:
            self._gen_data(1e6, low_limit, high_limit)
        else:
            self._load_data(fname)
        self.gen_fits(low_limit, high_limit)

    def _load_data(self, fname):
        _data = np.loadtxt(fname, delimiter=',')
        print("Data size before strip: {}".format(len(_data)))
        # remove physically impossible events
        delete = []
        for i, event in enumerate(_data):
            if (event[3] > 0 or event[4] > 0) and (event[5] > 0):
                delete.append(i)
        _data = np.delete(_data, delete, 0)
        print("Data size after strip: {}".format(len(_data)))
        self.top_events = np.array(_data[:, 3])
        self.bot_events = np.array(_data[:, 5])

    def _gen_data(self, size, low, high, pars_top=None, pars_bot=None, rand_seed=42):
        from scipy.constants import e, physical_constants
        g = 2.0023318418  # lande factor of the myon
        tau = 2.1969811e-6  # mean decay time
        b = self.b  # magnetic field in T
        m = physical_constants["muon mass"][0]
        omega = g * e * b / (2 * m)
        print("Expected omega ist {}".format(omega))
        delta = 9.85  # phase delay
        if pars_top is None:
            pars_top = {'tau': tau*1e6, 'k_top': 0.8, 'a_bar_top': 1e-3, 'omega': omega*1e-6, 'delta': delta,
                        'f_top': 2.6e-3}
        if pars_bot is None:
            pars_bot = {'tau': tau*1e6, 'k_bot': 0.7, 'a_bar_bot': 1e-2, 'omega': omega*1e-6, 'delta': delta,
                        'f_bot': 1.5e-3}
        self.top_events = np.zeros(int(size), dtype=float)
        self.bot_events = np.zeros(int(size / 2), dtype=float)
        random = np.random.RandomState(rand_seed)

        def gen(arr, fcn, low, high, **pars):
            x = np.linspace(low, high, 100)
            _max = np.amax(fcn(x, **pars))
            i = 0
            while i < len(arr):
                r = random.rand() * (high - low) + low
                y = random.rand() * _max
                if fcn(r, **pars) > y:
                    arr[i] = r
                    i += 1
            return arr

        print("Generating top data...")
        self.top_events = gen(self.top_events, events_top, low, high, **pars_top)
        print("Generating bottom data...")
        self.bot_events = gen(self.bot_events, events_bot, low, high, **pars_bot)
        print("Done generating data.")

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

    def do_fit(self, pre_fit=True, limits=None, **kwargs):
        self.decay_do_fit()
        if pre_fit:
            self.fit_multi.do_fit()
            self.fit_multi.set_parameter_values(**self.decay_fit_multi.parameter_name_value_dict)
        self.fit_multi.set_parameter_values(**kwargs)
        for key, value in iteritems(limits):
            self.fit_multi.limit_parameter(key, value)
        print(self.fit_multi.parameter_name_value_dict)
        self.fit_multi.do_fit()
        # self.fit_multi.report()

    def plot(self):
        _plot = Plot(self.fit_multi, separate_figures=False)
        _plot.customize(plot_type='data', keyword='markersize', values=[5, 5])
        # _plot = Plot((self.fit_top, self.fit_bot), separate_figures=True)
        _plot.plot()

def compare_limits():
    high_limits = np.linspace(12, 14, 20)
    cost = np.zeros((len(high_limits), 2))
    params = np.array([])
    lande = Lande(low_limit=2, high_limit=19)
    for i, lim in enumerate(high_limits):
        lande.gen_fits(low_limit=2, high_limit=lim)
        lande.do_fit(pre_fit=True, limits={'delta': (0, 2*np.pi)},
                     **{'omega': 3, 'delta': np.pi, 'a_bar_top': 0.01, 'a_bar_bot': 0.01})
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

def fit(lande):
    starting_values = {'omega': 3, 'delta': np.pi, 'a_bar_top': 1e-4, 'a_bar_bot': 1e-4}
    limits = {'a_bar_top': (1e-20, 1), 'a_bar_bot': (1e-20, 1)}
    lande.do_fit(pre_fit=True, limits=limits, **starting_values)
    print()
    for _par_name, _par_val, _par_err in zip(lande.fit_multi.parameter_names, lande.fit_multi.parameter_values,
                                             lande.fit_multi.parameter_errors):
        print('%s: %.3E +- %.3E' % (_par_name, _par_val, _par_err))
    lande.plot()
    from scipy.constants import e, physical_constants
    omega = lande.fit_multi.parameter_name_value_dict['omega'] * 1e6
    g_ref = 2.0023318418  # lande factor of the myon
    m = physical_constants["muon mass"][0]
    b = lande.b
    g = 2 * m * omega / (e * b)  # calculate g-factor with fit result for omega
    print("Ref g: {}\nFit g: {}".format(g_ref, g))
    plt.show()

def data_fit():
    low_limit, high_limit = 2, 13  # best limits according to prefits
    lande = Lande(low_limit, high_limit, b=3.6e-3)
    fit(lande)

def toy_data_fit():
    lande = Lande(2, 13, gen_data=True, b=5e-3)
    fit(lande)


if __name__ == '__main__':
    toy_data_fit()
