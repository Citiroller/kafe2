import warnings
import numpy as np

from .._base import (CostFunction, CostFunction_Chi2, CostFunction_NegLogLikelihood,
                     CostFunctionException)

__all__ = [
    "XYCostFunction_Chi2",
    "XYCostFunction_NegLogLikelihood"
]


class XYCostFunction_Chi2(CostFunction_Chi2):
    def __init__(self, errors_to_use='covariance', fallback_on_singular=True, axes_to_use='xy'):
        """
        Built-in least-squares cost function for *xy* data.

        :param errors_to_use: which errors to use when calculating :math:`\chi^2`
        :type errors_to_use: ``'covariance'``, ``'pointwise'`` or ``None``
        :param axes_to_use: take into account errors for which axes
        :type axes_to_use: ``'y'`` or ``'xy'``
        """
        self._DATA_NAME = "y_data"
        self._MODEL_NAME = "y_model"
        if axes_to_use.lower() == 'y':
            self._COV_MAT_INVERSE_NAME = "y_total_cov_mat_inverse"
            self._ERROR_NAME = "y_total_error"
        elif axes_to_use.lower() == 'xy':
            self._COV_MAT_INVERSE_NAME = "total_cov_mat_inverse"
            self._ERROR_NAME = "total_error"
        else:
            raise CostFunctionException(
                "Unknown value '%s' for 'axes_to_use': must be one of ('xy', 'y')")
        super(XYCostFunction_Chi2, self).__init__(
            errors_to_use=errors_to_use, fallback_on_singular=fallback_on_singular)


class XYCostFunction_NegLogLikelihood(CostFunction_NegLogLikelihood):
    def __init__(self, data_point_distribution="poisson", ratio=False, axes_to_use="xy"):
        self._DATA_NAME = "y_data"
        self._MODEL_NAME = "y_model"
        if axes_to_use.lower() == 'y':
            self._ERROR_NAME = "y_total_error"
        elif axes_to_use.lower() == 'xy':
            self._ERROR_NAME = "total_error"
        else:
            raise CostFunctionException(
                "Unknown value '%s' for 'axes_to_use': must be one of ('xy', 'y')")
        super(XYCostFunction_NegLogLikelihood, self).__init__(
            data_point_distribution=data_point_distribution, ratio=ratio)


STRING_TO_COST_FUNCTION = {
    'chi2': (XYCostFunction_Chi2, {}),
    'chi_2': (XYCostFunction_Chi2, {}),
    'chisquared': (XYCostFunction_Chi2, {}),
    'chi_squared': (XYCostFunction_Chi2, {}),
    'nll': (XYCostFunction_NegLogLikelihood, {"ratio": False}),
    'nll-poisson': (XYCostFunction_NegLogLikelihood,
                    {"data_point_distribution": "poisson", "ratio": False}),
    'nll_poisson': (XYCostFunction_NegLogLikelihood,
                    {"data_point_distribution": "poisson", "ratio": False}),
    'nllpoisson': (XYCostFunction_NegLogLikelihood,
                    {"data_point_distribution": "poisson", "ratio": False}),
    'nll-gaussian': (XYCostFunction_NegLogLikelihood,
                    {"data_point_distribution": "gaussian", "ratio": False}),
    'nll_gaussian': (XYCostFunction_NegLogLikelihood,
                    {"data_point_distribution": "gaussian", "ratio": False}),
    'nllgaussiann': (XYCostFunction_NegLogLikelihood,
                    {"data_point_distribution": "gaussian", "ratio": False}),
    'negloglikelihood': (XYCostFunction_NegLogLikelihood, {"ratio": False}),
    'neg_log_likelihood': (XYCostFunction_NegLogLikelihood, {"ratio": False}),
    'nllr': (CostFunction_NegLogLikelihood, {"ratio": True}),
    'nllr-poisson': (XYCostFunction_NegLogLikelihood,
                    {"data_point_distribution": "poisson", "ratio": True}),
    'nllr_poisson': (XYCostFunction_NegLogLikelihood,
                    {"data_point_distribution": "poisson", "ratio": True}),
    'nllrpoisson': (XYCostFunction_NegLogLikelihood,
                    {"data_point_distribution": "poisson", "ratio": True}),
    'nllr-gaussian': (XYCostFunction_NegLogLikelihood,
                    {"data_point_distribution": "gaussian", "ratio": True}),
    'nllr_gaussian': (XYCostFunction_NegLogLikelihood,
                    {"data_point_distribution": "gaussian", "ratio": True}),
    'nllrgaussian': (XYCostFunction_NegLogLikelihood,
                    {"data_point_distribution": "gaussian", "ratio": True}),
    'negloglikelihoodratio': (XYCostFunction_NegLogLikelihood, {"ratio": True}),
    'neg_log_likelihood_ratio': (XYCostFunction_NegLogLikelihood, {"ratio": True}),
}
