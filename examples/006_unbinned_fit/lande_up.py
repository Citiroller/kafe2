import numpy as np
import matplotlib.pyplot as plt
from kafe2 import HistContainer, HistFit, HistPlot


LOW_LIMIT = 2
HIGH_LIMIT = 19

def events(x, tau=2.2, k=1.0, a_bar=1.0, omega=1.0, delta=1.0, f=0.1):
    return k*np.exp(-x/tau)+a_bar*np.exp(-x/tau)*np.cos(omega*x+delta)+f

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
    top_data = get_data()[:, 3]
    return top_data[top_data > LOW_LIMIT]

def get_bottom_data():
    bot_data = get_data()[:, 5]
    return bot_data[bot_data > LOW_LIMIT]

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
    hist_fit.fix_parameter('delta', 0)
    #hist_fit.limit_parameter('delta', (-0.01*np.pi, 0.01*np.pi))
    #hist_fit.limit_parameter('a_bar', (-0.1, 0.1))
    hist_fit.limit_parameter('omega', (1.0, 1.9))
    # assign latex names for the parameters for nicer display
    # hist_fit.assign_parameter_latex_names(tau=r'\tau', k='k', f='f', a_bar=r'\bar{{A}}')
    # assign a latex expression for the fit function for nicer display
    # hist_fit.assign_model_function_latex_expression("{k}e^{{-{x}/{tau}}}+{f}")
    hist_fit.do_fit()
    hist_fit.report()
    hist_plot = HistPlot(hist_fit)
    hist_plot.show_fit_info_box()
    hist_plot.plot()
    plt.show()


if __name__ == "__main__":
    fit_advanced(get_top_data())
