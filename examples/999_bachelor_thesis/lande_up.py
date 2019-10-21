import numpy as np
import matplotlib.pyplot as plt
from kafe2 import HistContainer, HistFit, HistPlot, XYMultiFit, XYMultiPlot, XYCostFunction_NegLogLikelihood


LOW_LIMIT = 2
HIGH_LIMIT = 19

DATA = None

def events_top(x, tau=2.2, k_top=1.0, a_bar_top=1.0, omega=1.0, delta_top=1.0, f_top=0.1):
    return k_top*np.exp(-x/tau)+a_bar_top*np.exp(-x/tau)*np.cos(omega*x+delta_top)+f_top

def events_bot(x, tau=2.2, k_bot=1.0, a_bar_bot=1.0, omega=1.0, delta_bot=1.0, f_bot=0.1):
    return k_bot*np.exp(-x/tau)+a_bar_bot*np.exp(-x/tau)*np.cos(omega*x+delta_bot)+f_bot

def decay(x, tau=2.2, k=1.0, f=0.1):
    return k*np.exp(-x/tau)+f

def get_data(fname=None):
    print("Loading data...")
    if fname is None:
        fname = "dpFilt_190325-0148.dat"
    _data = np.loadtxt(fname, delimiter=',')
    print("Data size before strip: {}".format(len(_data)))
    # strip strange events
    delete = []
    for i, event in enumerate(_data):
        if (event[3] > 0 or event[4] > 0) and event[5] > 0:
            delete.append(i)
    _data = np.delete(_data, delete, 0)
    print("Data size after strip: {}".format(len(_data)))
    return _data

def get_top_data():
    top_data = DATA[:, 3]
    return top_data[top_data > LOW_LIMIT]

def get_bottom_data():
    bot_data = DATA[:, 5]
    return bot_data[bot_data > LOW_LIMIT]

def get_diff():
    top_data, bins_top, _ = plt.hist(get_top_data(), bins=100, range=(LOW_LIMIT, HIGH_LIMIT), density=1, alpha=0.5)
    bot_data, bins_bot, _ = plt.hist(get_bottom_data(), bins=100, range=(LOW_LIMIT, HIGH_LIMIT), density=1, alpha=0.5)
    # check if bins are equal
    for top, bot in zip(bins_top, bins_bot):
        if top != bot:
            print("Different edges: {}, {}".format(top. bot))
    data = top_data - bot_data
    binc = (bins_top[1:]-bins_top[:-1])/2.0 + bins_top[:-1]
    top_data = np.array(top_data, dtype=float)
    bot_data = np.array(bot_data, dtype=float)
    plt.show()
    plt.gcf().clear()
    plt.plot(binc, data)
    plt.show()
    """
    fit = XYMultiFit(xy_data=[[binc, top_data], [binc, bot_data]], model_function=[events_top, events_bot])
                     #cost_function=XYCostFunction_NegLogLikelihood('poisson'))
    fit.limit_parameter('tau', (2.0, 3))
    fit.limit_parameter('omega', (1.0, 1.9))
    fit.do_fit()
    fit.report()
    plot = XYMultiPlot(fit)
    plot.show_fit_info_box()
    plt.show()
    """

def fit_simple(data):
    hist_data = HistContainer(n_bins=100, bin_range=(LOW_LIMIT, HIGH_LIMIT), fill_data=data)
    hist_fit = HistFit(hist_data, decay)
    # hist_fit.limit_parameter('tau', (2.0, 2.4))
    # hist_fit.limit_parameter('k', (0.5, 0.8))
    # hist_fit.limit_parameter('f', (0.01, 0.02))
    # assign latex names for the parameters for nicer display
    hist_fit.assign_parameter_latex_names(tau=r'\tau', k='k', f='f')
    # assign a latex expression for the fit function for nicer display
    hist_fit.assign_model_function_latex_expression("{k}e^{{-{x}/{tau}}}+{f}")
    hist_fit.do_fit()
    binc = hist_data.bin_centers
    binh = hist_data.data
    bine = hist_data.bin_edges
    model = hist_fit.model
    params = hist_fit.parameter_values
    points = binh - model
    plt.plot(binc, points)
    plt.show()
    """    
    hist_fit.report()
    hist_plot = HistPlot(hist_fit)
    hist_plot.show_fit_info_box()
    hist_plot.plot()
    plt.show()"""


def fit_advanced(data):
    hist_data = HistContainer(n_bins=100, bin_range=(LOW_LIMIT, HIGH_LIMIT), fill_data=data)
    hist_fit = HistFit(hist_data, events)
    hist_fit.set_all_parameter_values([2.429, 0.504, -0.0375, 1.168, 0.0, 0.01854])
    #hist_fit.limit_parameter('tau', (2.0, 3))
    #hist_fit.limit_parameter('k', (0.5, 0.8))
    #hist_fit.limit_parameter('f', (0.01, 0.02))
    # hist_fit.fix_parameter('delta', 0)
    #hist_fit.limit_parameter('delta', (-0.01*np.pi, 0.01*np.pi))
    #hist_fit.limit_parameter('a_bar', (-0.1, 0.1))
    hist_fit.limit_parameter('omega', (1.0, 1.9))
    # assign latex names for the parameters for nicer display
    hist_fit.assign_parameter_latex_names(tau=r'\tau', k='k', f='f', a_bar=r'\bar{{A}}')
    # assign a latex expression for the fit function for nicer display
    hist_fit.assign_model_function_latex_expression("{k}e^{{-{x}/{tau}}}+{f}")
    hist_fit.do_fit()
    hist_fit.report()
    hist_plot = HistPlot(hist_fit)
    hist_plot.show_fit_info_box()
    hist_plot.plot()
    plt.show()


if __name__ == "__main__":
    DATA = get_data()
    get_diff()
