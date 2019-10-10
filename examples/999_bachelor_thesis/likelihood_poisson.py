import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from kafe2 import UnbinnedFit, ContoursProfiler

LOW, HIGH = 0, 20  # limits for generating data
P = 4  # parameter of poisson distribution for generating data
np.random.seed(1578839424)  # fix seed for consistent reproduction


def poisson(x, p):
    return p**x/factorial(x)*np.exp(-p)


def gen_data(length=100):
    data = []
    while len(data) < length:
        x = np.random.randint(LOW, HIGH)
        y = np.random.rand()
        if y <= poisson(x, P):
            data.append(x)
    print("Mean of data is {:3.2f}".format(np.mean(data)))
    return data


def main():
    x = gen_data(200)
    fit = UnbinnedFit(x, poisson)
    fit.assign_parameter_latex_names(p=r'\lambda')
    fit.do_fit()
    fit.report(asymmetric_parameter_errors=True)
    # create likelihood Plot
    fig = plt.figure(figsize=(8, 8))
    ax = plt.gca()
    cpf = ContoursProfiler(fit, profile_subtract_min=True)
    cpf.plot_profile('p', label_ticks_in_sigma=False, target_axes=ax)
    # cpf.plot_profiles_contours_matrix(show_grid_for='all', show_error_span_profiles=True, label_ticks_in_sigma=False)
    # plt.subplots_adjust(left=0.5)
    plt.tight_layout()
    plt.show()  # show the plot


if __name__ == "__main__":
    main()
