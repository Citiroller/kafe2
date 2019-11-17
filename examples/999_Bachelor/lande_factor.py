import numpy as np
import matplotlib.pyplot as plt
from kafe2 import HistFit, HistContainer, Plot, MultiFit
from six import iteritems


class DataGenerator:
    def __init__(self, fcn, limits=(0, 1), size=1000, **pars):
        """Generates a Dataset which follows the distribution of the input function.

        :param fcn: Distribution function to follow
        :type fcn: callable
        :param limits: low and high limits for generated datapoints
        :type limits: tuple
        :param size: amount of datapoints to be generated
        :type size: int
        :param pars: parameters for the distribution function
        """
        self.function = fcn
        self.limits = limits
        self.size = size
        self.pars = pars

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, size):
        self._size = int(size)
        self.events = np.zeros(int(size), dtype=float)

    @property
    def limits(self):
        return self._limits

    @limits.setter
    def limits(self, limits):
        self._limits = tuple(limits)

    def gen_data(self, rand_seed=42):
        random = np.random.RandomState(rand_seed)
        low, high = self.limits
        _x = np.linspace(low, high, 100)
        _max = 1.1*np.amax(self.function(_x, **self.pars))
        arr = np.zeros(self.size)
        i = 0
        while i < len(arr):
            r = random.rand() * (high - low) + low
            y = random.rand() * _max
            if self.function(r, **self.pars) > y:
                arr[i] = r
                i += 1
        return arr


def events_top(x, tau=2.2, k_top=1.0, a_bar_top=1.0, omega=1.0, delta=1.0, f_top=0.1):
    return k_top*np.exp(-x/tau)*(1+a_bar_top*np.cos(omega*x+delta))+f_top


def events_bot(x, tau=2.2, k_bot=1.0, a_bar_bot=1.0, omega=1.0, delta=1.0, f_bot=0.1):
    return k_bot*np.exp(-x/tau)*(1-a_bar_bot*np.cos(omega*x+delta))+f_bot


def decay_top(x, tau=2.2, k_top=1.0, f_top=0.1):
    return k_top*np.exp(-x/tau)+f_top


def decay_bot(x, tau=2.2, k_bot=1.0, f_bot=0.1):
    return k_bot*np.exp(-x/tau)+f_bot


class Lande:
    def __init__(self, data, limits=(2, 16)):
        self.limits = limits
        self.top_events = data[0]
        self.bot_events = data[1]
        self.gen_fits()

    def gen_fits(self):
        _hist_top = HistContainer(n_bins=100, bin_range=self.limits,
                                  fill_data=self.top_events)
        _hist_bot = HistContainer(n_bins=100, bin_range=self.limits,
                                  fill_data=self.bot_events)
        _fit_top = HistFit(_hist_top, events_top, cost_function='nllr')
        _fit_bot = HistFit(_hist_bot, events_bot, cost_function='nllr')
        shared_par_dict = {'x': 't', 'tau': r'\tau', 'omega': r'\omega', 'delta': r'\delta'}
        top_par_names = {'k_top': r'K_0', 'a_bar_top': r'\bar{A}_0', 'f_top': 'f_0'}
        bot_par_names = {'k_bot': r'K_1', 'a_bar_bot': r'\bar{A}_1', 'f_bot': 'f_1'}
        top_par_names.update(shared_par_dict)
        bot_par_names.update(shared_par_dict)
        _fit_top.assign_model_function_latex_name("N_0")
        _fit_top.assign_parameter_latex_names(**top_par_names)
        _fit_top._model_function._formatter._latex_x_name = '{t}'  # workaround, because it's currently hardcoded
        _fit_top.assign_model_function_latex_expression(r'{k_top}\cdot\exp{{\left(-\frac{x}{tau}\right)}}'
                                                        r'\left(1+{a_bar_top}\cdot\cos\left({omega}{x}+{delta}\right)\right)'
                                                        r'+{f_top}')
        _fit_bot.assign_model_function_latex_name("N_1")
        _fit_bot.assign_parameter_latex_names(**bot_par_names)
        _fit_bot._model_function._formatter._latex_x_name = '{t}'  # workaround, because it's currently hardcoded
        _fit_bot.assign_model_function_latex_expression(r'{k_bot}\cdot\exp{{\left(-\frac{x}{tau}\right)}}'
                                                        r'\left(1+{a_bar_bot}\cdot\cos\left({omega}{x}+{delta}\right)\right)'
                                                        r'+{f_bot}')
        self.fit_multi = MultiFit((_fit_top, _fit_bot), minimizer='iminuit')
        _decay_fit_top = HistFit(_hist_top, decay_top)
        _decay_fit_bot = HistFit(_hist_bot, decay_bot)
        self.decay_fit_multi = MultiFit((_decay_fit_top, _decay_fit_bot), minimizer='iminuit')

    def do_fit(self, starting_values=dict(), par_limits=None, pre_fit=True):
        if pre_fit:
            # guess some good starting values for speeding up the pre-fit
            self.decay_fit_multi.set_parameter_values(**dict(tau=2.2, k_top=1, f_top=1e-2, k_bot=1, f_bot=1e-2))
            self.decay_fit_multi.do_fit()
            starting_values.update(self.decay_fit_multi.parameter_name_value_dict)
        print('Final Fit starting values are:', starting_values)
        self.fit_multi.set_parameter_values(**starting_values)
        for key, value in iteritems(par_limits):
            self.fit_multi.limit_parameter(key, value)
        self.fit_multi.do_fit()
        # self.fit_multi.report()

    def plot(self):
        _plot = Plot(self.fit_multi, separate_figures=False)
        _data_upper_kw = dict(label='Upper Detector', markersize=3)
        _data_lower_kw = dict(label='Lower Detector', markersize=3)
        _model_upper_kw = dict(label='Upper Model')
        _model_lower_kw = dict(label='Lower Model')
        _density_upper_kw = dict(label='Upper Model Density')
        _density_lower_kw = dict(label='Lower Model Density')
        _plot.set_keywords('data', [_data_upper_kw, _data_lower_kw])
        _plot.set_keywords('model', [_model_upper_kw, _model_lower_kw])
        _plot.set_keywords('model_density', [_density_upper_kw, _density_lower_kw])
        _plot.plot()
        fig = plt.gcf()
        fig.xlabel = r'$t$ [$\mu$s]'


if __name__ == '__main__':
    from scipy.constants import e, physical_constants
    g_ref = 2.0023318418  # lande factor of the myon
    tau_ref = 2.1969811e-6  # mean decay time
    b = 5e-3  # magnetic field in T
    m = physical_constants["muon mass"][0]
    omega_ref = g_ref * e * b / (2 * m)
    print("Expected omega ist {}".format(omega_ref))
    delta = 9.85  # phase delay
    gen_pars_top = {'tau': tau_ref * 1e6, 'k_top': 0.8, 'a_bar_top': 0.00125, 'omega': omega_ref * 1e-6, 'delta': delta,
                    'f_top': 2e-2}
    gen_pars_bot = {'tau': tau_ref * 1e6, 'k_bot': 0.7, 'a_bar_bot': 0.015, 'omega': omega_ref * 1e-6, 'delta': delta,
                    'f_bot': 2e-2}
    limits = (2, 13)

    print('Generating Data...')
    top_gen = DataGenerator(events_top, limits, size=int(1e6), **gen_pars_top)
    bot_gen = DataGenerator(events_bot, limits, size=int(4e5), **gen_pars_bot)
    data = np.array([top_gen.gen_data(), bot_gen.gen_data()])

    print('Performing fit...')
    lande = Lande(data, limits)
    starting_values = {'omega': 3, 'delta': np.pi, 'a_bar_top': 1e-4, 'a_bar_bot': 1e-4}
    par_limits = {'a_bar_top': (1e-20, 1), 'a_bar_bot': (1e-20, 1)}
    lande.do_fit(starting_values=starting_values, par_limits=par_limits, pre_fit=True)
    print()
    for _par_name, _par_val, _par_err in zip(lande.fit_multi.parameter_names, lande.fit_multi.parameter_values,
                                             lande.fit_multi.parameter_errors):
        print('%s: %.3E +- %.3E' % (_par_name, _par_val, _par_err))
    lande.plot()

    omega = lande.fit_multi.parameter_name_value_dict['omega'] * 1e6
    g = 2 * m * omega / (e * b)  # calculate g-factor with fit result for omega
    print("Ref g: {}\nFit g: {}".format(g_ref, g))

    plt.show()
