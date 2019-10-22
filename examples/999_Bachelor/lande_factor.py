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
    def __init__(self, fname="dpFilt_190325-0148.dat", low_limit=2, high_limit=19):
        self._low_limit = low_limit
        self._high_limit = high_limit
        self.top_events = None
        self.bot_events = None
        self.hist_top = None
        self.hist_bot = None
        self.fit_top = None
        self.fit_bot = None
        self._load_data(fname)
        self._gen_fits()

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
        self.hist_top = HistContainer(n_bins=100, bin_range=(self._low_limit, self._high_limit),
                                      fill_data=self.top_events)
        self.hist_bot = HistContainer(n_bins=100, bin_range=(self._low_limit, self._high_limit),
                                      fill_data=self.bot_events)

        def _gen_data(self, size):
            from scipy.constants import e, m_e
            g = 2.0023318418  # lande factor of the myon
            tau = 2.1969811e-6  # mean decay time
            b = 5e-6  # magnetic field in T
            omega = g * e * b / (2 * m_e)
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

    def _gen_fits(self):
        self.fit_top = HistFit(self.hist_top, events_top)
        self.fit_bot = HistFit(self.hist_bot, events_bot)
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
    lande = Lande()
    lande.decay_do_fit()  # do decay fit for better starting values of  the decay params
    for _par_name, _par_val, _par_err in zip(
            lande.decay_fit_multi.parameter_names, lande.decay_fit_multi.parameter_values, lande.decay_fit_multi.parameter_errors):
        print('%s: %.3E +- %.3E' % (_par_name, _par_val, _par_err))
    lande.fit_multi.set_parameter_values(**lande.decay_fit_multi.parameter_name_value_dict)
    start_values = {'omega': 1, 'delta': 1e-8, 'a_bar_top': 0.01, 'a_bar_bot': 0.01}
    lande.fit_multi.set_parameter_values(**start_values)
    print(lande.fit_multi.parameter_name_value_dict)
    lande.do_fit()
    lande.plot()
    print()
    for _par_name, _par_val, _par_err in zip(
            lande.fit_multi.parameter_names, lande.fit_multi.parameter_values, lande.fit_multi.parameter_errors):
        print('%s: %.3E +- %.3E' % (_par_name, _par_val, _par_err))
    plt.show()
