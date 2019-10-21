import numpy as np
import matplotlib.pyplot as plt
from kafe2 import HistFit, HistContainer, Plot, MultiFit

def events_top(x, tau=2.2, k_top=1.0, a_bar_top=1.0, omega=1.0, delta=1.0, f_top=0.1):
    return k_top*np.exp(-x/tau)+a_bar_top*np.exp(-x/tau)*np.cos(omega*x+delta)+f_top

def events_bot(x, tau=2.2, k_bot=1.0, a_bar_bot=1.0, omega=1.0, delta=1.0, f_bot=0.1):
    return k_bot*np.exp(-x/tau)-a_bar_bot*np.exp(-x/tau)*np.cos(omega*x+delta)+f_bot

def decay(x, tau=2.2, k=1.0, f=0.1):
    return k*np.exp(-x/tau)+f

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
        self.hist_top = HistContainer(n_bins=100, bin_range=(self._low_limit, self._high_limit), fill_data=self.top_events)
        self.hist_bot = HistContainer(n_bins=100, bin_range=(self._low_limit, self._high_limit), fill_data=self.bot_events)
        self.fit_top = HistFit(self.hist_top, events_top)
        self.fit_bot = HistFit(self.hist_bot, events_bot)
        # self.fit_top.limit_parameter("delta", (0, 2 * np.pi))
        # self.fit_bot.limit_parameter("delta", (0, 2 * np.pi))
        # self.fit_top.limit_parameter('tau', (1, 3))
        # self.fit_bot.limit_parameter('tau', (1, 3))
        self.fit_multi = MultiFit((self.fit_top, self.fit_bot))

    def do_fit(self):
        self.fit_multi.do_fit()
        self.fit_multi.report()

    def plot(self):
        _plot = Plot(self.fit_multi, separate_figures=True)
        # _plot = Plot((self.fit_top, self.fit_bot), separate_figures=True)
        _plot.plot()


if __name__ == '__main__':
    lande = Lande()
    lande.do_fit()
    lande.plot()
    plt.show()
