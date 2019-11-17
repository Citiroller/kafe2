import numpy as np
import matplotlib.pyplot as plt
from kafe2 import UnbinnedFit, Plot, ContoursProfiler


def simple_decay(x, tau=2.2, fbg=0.1, a=1., b=9.75):
    """Probability density function for the decay time of a myon. The pdf is normalized to the interval (a, b).

    :param x: decay time
    :param fbg: background
    :param tau: expected mean of the decay time
    :param a: lower limit of normalization
    :param b: upper limit of normalization
    :return: probability for decay time x
    """
    pdf1 = np.exp(-x / tau) / tau / (np.exp(-a / tau) - np.exp(-b / tau))
    pdf2 = 1. / (b - a)
    return (1 - fbg) * pdf1 + fbg * pdf2


if __name__ == '__main__':
    data = np.loadtxt("dpFilt_190325-0148.dat", delimiter=',')[:, 2]  # just load double pulses
    limits = (2, 15)
    # only use delta t in between the limits, to avoid underground
    data = data[(data >= limits[0]) & (data <= limits[1])]
    print(data)
    fit = UnbinnedFit(data[0:200], simple_decay)  # only use first 200 events
    # fix the parameters for normalizing the distribution function
    fit.fix_parameter('a', limits[0])
    fit.fix_parameter('b', limits[1])
    fit.do_fit()
    fit.report()
    # assign latex names for the parameters for nicer display
    fit.assign_parameter_latex_names(tau=r'\tau', fbg='f', a='a', b='b')
    # assign a latex expression for the fit function for nicer display
    fit.assign_model_function_latex_expression("(1-{fbg}) \\frac{{e^{{-{x}/{tau}}}}}"
                                               "{{{tau}(e^{{-{a}/{tau}}}-e^{{-{b}/{tau}}})}}"
                                               "+ {fbg} \\frac{{1}}{{{b}-{a}}}")
    # plot the fit results
    plot = Plot(fit)
    plot.plot()  # add with_asymmetric_parameter_errors=True when this is fixed. For the time being see contours
    # create contours
    cpf = ContoursProfiler(fit)
    cpf.plot_profiles_contours_matrix(parameters=['tau', 'fbg'], label_ticks_in_sigma=False)
    plt.show()
