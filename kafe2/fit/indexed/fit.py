from collections import OrderedDict
from copy import deepcopy

import numpy as np
import sys
import six
import textwrap

from ...tools import print_dict_as_table
from ...config import kc
from ...core import NexusFitter, Nexus
from ...core.fitters.nexus import Parameter, Alias, NexusError
from .._base import FitException, FitBase, DataContainerBase, CostFunctionBase
from .container import IndexedContainer
from .cost import IndexedCostFunction_Chi2, IndexedCostFunction_UserDefined, STRING_TO_COST_FUNCTION
from .model import IndexedParametricModel, IndexedModelFunction
from ..util import function_library, add_in_quadrature, collect, invert_matrix


__all__ = ["IndexedFit"]


class IndexedFitException(FitException):
    pass


class IndexedFit(FitBase):
    CONTAINER_TYPE = IndexedContainer
    MODEL_TYPE = IndexedParametricModel
    MODEL_FUNCTION_TYPE = IndexedModelFunction
    EXCEPTION_TYPE = IndexedFitException
    RESERVED_NODE_NAMES = {'data', 'model', 'cost',
                          'data_error', 'model_error', 'total_error',
                          'data_cov_mat', 'model_cov_mat', 'total_cov_mat',
                          'data_cor_mat', 'model_cor_mat', 'total_cor_mat'}

    def __init__(self,
                 data,
                 model_function,
                 cost_function=IndexedCostFunction_Chi2(
                    errors_to_use='covariance',
                    fallback_on_singular=True),
                 minimizer=None,
                 minimizer_kwargs=None):
        """
        Construct a fit of a model to a series of indexed measurements.

        :param data: the measurement values
        :type data: iterable of float
        :param model_function: the model function
        :type model_function: :py:class:`~kafe2.fit.indexed.IndexedModelFunction` or unwrapped native Python function
        :param cost_function: the cost function
        :type cost_function: :py:class:`~kafe2.fit._base.CostFunctionBase`-derived or unwrapped native Python function
        """
        # set/construct the model function object
        if isinstance(model_function, self.__class__.MODEL_FUNCTION_TYPE):
            self._model_function = model_function
        else:
            self._model_function = self.__class__.MODEL_FUNCTION_TYPE(model_function)

        # validate the model function for this fit
        self._validate_model_function_for_fit_raise()

        # set and validate the cost function
        if isinstance(cost_function, CostFunctionBase):
            self._cost_function = cost_function
        elif isinstance(cost_function, str):
            _cost_function_class = STRING_TO_COST_FUNCTION.get(cost_function, None)
            if _cost_function_class is None:
                raise IndexedFitException('Unknown cost function: %s' % cost_function)
            self._cost_function = _cost_function_class()
        else:
            self._cost_function = IndexedCostFunction_UserDefined(cost_function)
            # self._validate_cost_function_raise()
            # TODO: validate user-defined cost function? how?

        # declare cache
        self.__cache_total_error = None
        self.__cache_total_cov_mat = None
        self.__cache_total_cov_mat_inverse = None

        # initialize the Nexus
        self._init_nexus()

        # initialize the Fitter
        self._initialize_fitter(minimizer, minimizer_kwargs)

        self._fit_param_constraints = []
        self._loaded_result_dict = None

        # set the data after the cost_function has been set and nexus has been initialized
        self.data = data

    # -- private methods

    def _init_nexus(self):

        self._nexus = Nexus()

        for _type in ('data', 'model'):

            # add data and model for axis
            self._add_property_to_nexus(_type)
            # add errors for axis
            self._add_property_to_nexus('_'.join((_type, 'error')))

            # add cov mats for axis
            for _prop in ('cov_mat',):  # TODO: 'uncor_cov_mat'
                _node = self._add_property_to_nexus('_'.join((_type, _prop)))
                # add inverse
                self._nexus.add_function(
                    invert_matrix,
                    func_name='_'.join((_node.name, 'inverse')),
                    par_names=(_node.name,)
                )

        # 'total_error', i.e. data + model error in quadrature
        self._nexus.add_function(
            add_in_quadrature,
            func_name='total_error',
            par_names=(
                'data_error',
                'model_error'
            )
        )

        # 'total_cov_mat', i.e. data + model cov mats
        for _mat in ('cov_mat',):  # TODO: 'uncor_cov_mat'
            _node = (
                self._nexus.get('_'.join(('data', _mat))) +
                self._nexus.get('_'.join(('model', _mat)))
            )
            _node.name = '_'.join(('total', _mat))
            self._nexus.add(_node)

            # add inverse
            self._nexus.add_function(
                invert_matrix,
                func_name='_'.join((_node.name, 'inverse')),
                par_names=(_node.name,),
            )

        # get names and default values of all parameters
        _nexus_new_dict = self._get_default_values(
            model_function=self._model_function,
            x_name=None
        )

        # -- fit parameters

        self._fit_param_names = []  # names of all fit parameters (including nuisance parameters)
        self._poi_names = []  # names of the parameters of interest (i.e. the model parameters)
        for _par_name, _par_value in six.iteritems(_nexus_new_dict):
            # create nexus node for function parameter
            self._nexus.add(Parameter(_par_value, name=_par_name))

            # keep track of fit parameter names
            self._fit_param_names.append(_par_name)
            self._poi_names.append(_par_name)

        # -- nuisance parameters
        self._nuisance_names = []  # names of all nuisance parameters accounting for correlated errors

        self._nexus.add_function(lambda: self.poi_values, func_name='poi_values')
        self._nexus.add_function(lambda: self.parameter_values, func_name='parameter_values')
        self._nexus.add_function(lambda: self.parameter_constraints, func_name='parameter_constraints')

        # add the original function name as an alias to 'model'
        try:
            self._nexus.add_alias(self._model_function.name, alias_for='model')
        except NexusError:
            pass  # allow 'model' as function name for model

        self._nexus.add_function(
            collect,
            func_name="nuisance_vector"
        )

        # -- initialize nuisance parameters

        # TODO: implement nuisance parameters for indexed data

        # the cost function (the function to be minimized)
        _cost_node = self._nexus.add_function(
            self._cost_function.func,
            func_name=self._cost_function.name,
        )

        _cost_alias = self._nexus.add_alias('cost', alias_for=self._cost_function.name)

        self._nexus.add_dependency('poi_values', depends_on=self._poi_names)
        self._nexus.add_dependency('parameter_values', depends_on=self._fit_param_names)

        self._nexus.add_dependency(
            'model',
            depends_on=(
                'poi_values'
            )
        )

        self._nexus.add_dependency(
            'total_error',
            depends_on=(
                'data_error',
                'model_error',
            )
        )

        # TODO: add 'uncor_cov_mat'
        #for _side in ('data', 'model', 'total'):
        #    self._nexus.add_dependency(
        #        '{}_uncor_cov_mat'.format(_side),
        #        depends_on='{}_cov_mat'.format(_side)
        #    )

    def _invalidate_total_error_cache(self):
        self.__cache_total_error = None
        self.__cache_total_cov_mat = None
        self.__cache_total_cov_mat_inverse = None

    def _mark_errors_for_update(self):
        # TODO: implement a mass 'mark_for_update' routine in Nexus
        self._nexus.get('model').mark_for_update()
        self._nexus.get('data_error').mark_for_update()
        self._nexus.get('data_cov_mat').mark_for_update()
        self._nexus.get('data_cov_mat_inverse').mark_for_update()
        self._nexus.get('model_error').mark_for_update()
        self._nexus.get('model_cov_mat').mark_for_update()
        self._nexus.get('model_cov_mat_inverse').mark_for_update()
        self._nexus.get('total_error').mark_for_update()
        self._nexus.get('total_cov_mat').mark_for_update()
        self._nexus.get('total_cov_mat_inverse').mark_for_update()

    def _set_new_data(self, new_data):
        if isinstance(new_data, self.CONTAINER_TYPE):
            self._data_container = deepcopy(new_data)
        elif isinstance(new_data, DataContainerBase):
            raise IndexedFitException("Incompatible container type '%s' (expected '%s')"
                                      % (type(new_data), self.CONTAINER_TYPE))
        else:
            self._data_container = self._new_data_container(new_data, dtype=float)
        self._nexus.get('data').mark_for_update()

    def _set_new_parametric_model(self):
        self._param_model = self._new_parametric_model(self._model_function, self.parameter_values,
                                                       shape_like=self.data)
        self._mark_errors_for_update_invalidate_total_error_cache()

    # -- public properties

    @FitBase.data.getter
    def data(self):
        """array of measurement values"""
        return self._data_container.data

    @property
    def data_error(self):
        """array of pointwise data uncertainties"""
        return self._data_container.err

    @property
    def data_cov_mat(self):
        """the data covariance matrix"""
        return self._data_container.cov_mat

    @property
    def data_cov_mat_inverse(self):
        """inverse of the data covariance matrix (or ``None`` if singular)"""
        return self._data_container.cov_mat_inverse

    @property
    def data_cor_mat(self):
        """the data correlation matrix"""
        return self._data_container.cor_mat

    @property
    def model(self):
        """array of model predictions for the data points"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        return self._param_model.data

    @property
    def model_error(self):
        """array of pointwise model uncertainties"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        return self._param_model.err

    @property
    def model_cov_mat(self):
        """the model covariance matrix"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        return self._param_model.cov_mat

    @property
    def model_cov_mat_inverse(self):
        """inverse of the model covariance matrix (or ``None`` if singular)"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        return self._param_model.cov_mat_inverse

    @property
    def model_cor_mat(self):
        """the model correlation matrix"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        return self._param_model.cor_mat

    @property
    def total_error(self):
        """array of pointwise total uncertainties"""
        if self.__cache_total_error is None:
            _tmp = self.data_error**2
            _tmp += self.model_error**2
            self.__cache_total_error = np.sqrt(_tmp)
        return self.__cache_total_error

    @property
    def total_cov_mat(self):
        """the total covariance matrix"""
        if self.__cache_total_cov_mat is None:
            _tmp = self.data_cov_mat
            _tmp += self.model_cov_mat
            self.__cache_total_cov_mat = _tmp
        return self.__cache_total_cov_mat

    @property
    def total_cov_mat_inverse(self):
        """inverse of the total covariance matrix (or ``None`` if singular)"""
        if self.__cache_total_cov_mat_inverse is None:
            _tmp = self.total_cov_mat
            try:
                _tmp = np.linalg.inv(_tmp)
                self.__cache_total_cov_mat_inverse = _tmp
            except np.linalg.LinAlgError:
                pass
        return self.__cache_total_cov_mat_inverse

    # -- public methods

    ## add_error... methods inherited from FitBase ##

    def generate_plot(self):
        from kafe2.fit.indexed import IndexedPlot
        return IndexedPlot(self)

    def report(self, output_stream=sys.stdout,
               show_data=True,
               show_model=True,
               asymmetric_parameter_errors=False):
        """
        Print a summary of the fit state and/or results.

        :param output_stream: the output stream to which the report should be printed
        :type output_stream: TextIOBase
        :param show_data: if ``True``, print out information about the data
        :type show_data: bool
        :param show_model: if ``True``, print out information about the parametric model
        :type show_model: bool
        :param asymmetric_parameter_errors: if ``True``, use two different parameter errors for up/down directions
        :type asymmetric_parameter_errors: bool
        """
        _indent = ' ' * 4

        if show_data:
            output_stream.write(textwrap.dedent("""
                ########
                # Data #
                ########

            """))

            _data_table_dict = OrderedDict()
            _data_table_dict['Index'] = range(self.data_size)
            _data_table_dict['Data'] = self.data
            if self.has_data_errors:
                _data_table_dict['Data Error'] = self.data_error
                #_data_table_dict['Data Total Covariance Matrix'] = self.data_cov_mat
                _data_table_dict['Data Total Correlation Matrix'] = self.data_cor_mat

            print_dict_as_table(_data_table_dict, output_stream=output_stream, indent_level=1)

        if show_model:
            output_stream.write(textwrap.dedent("""
                #########
                # Model #
                #########

            """))

            #output_stream.write(_indent)
            output_stream.write(_indent + "Model Function\n")
            output_stream.write(_indent + "==============\n\n")
            output_stream.write(_indent * 2)
            output_stream.write(
                self._model_function.formatter.get_formatted(
                    with_par_values=False,
                    n_significant_digits=2,
                    format_as_latex=False,
                    with_expression=True
                )
            )
            output_stream.write('\n\n\n')

            _data_table_dict = OrderedDict()
            _data_table_dict['Index'] = range(self.data_size)
            _data_table_dict['Model'] = self.model
            if self.has_model_errors:
                _data_table_dict['Model Error'] = self.model_error
                #_data_table_dict['Model Total Covariance Matrix'] = self.model_cov_mat
                _data_table_dict['Model Total Correlation Matrix'] = self.model_cor_mat

            print_dict_as_table(_data_table_dict, output_stream=output_stream, indent_level=1)

        super(IndexedFit, self).report(output_stream=output_stream,
                                       asymmetric_parameter_errors=asymmetric_parameter_errors)
