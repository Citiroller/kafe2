from .._base import DReprError
from kafe.fit.representation._yaml_base import  YamlWriterMixin, YamlReaderMixin
from ._base import FitDReprBase
from .. import _AVAILABLE_REPRESENTATIONS
from kafe.fit.histogram.fit import HistFit
from kafe.fit.representation.container.yaml_drepr import DataContainerYamlReader,\
    DataContainerYamlWriter
from kafe.fit.indexed.fit import IndexedFit
from kafe.fit.xy.fit import XYFit
from kafe.fit.representation.model.yaml_drepr import ParametricModelYamlReader,\
    ParametricModelYamlWriter
from kafe.fit.xy_multi.fit import XYMultiFit

__all__ = ['FitYamlWriter', 'FitYamlReader']

class FitYamlWriter(YamlWriterMixin, FitDReprBase):

    def __init__(self, fit, output_io_handle):
        super(FitYamlWriter, self).__init__(
            output_io_handle=output_io_handle,
            fit=fit)
    
    @classmethod
    def _make_representation(cls, fit):
        _yaml_doc = dict()

        # -- determine model function type
        _type = cls._CLASS_TO_OBJECT_TYPE_NAME.get(fit.__class__, None)
        if _type is None:
            raise DReprError("Fit type unknown or not supported: %s" % fit.__class__)
        _yaml_doc['type'] = _type
        
        _yaml_doc['dataset'] = DataContainerYamlWriter._make_representation(fit._data_container)
        _yaml_doc['parametric_model'] = ParametricModelYamlWriter._make_representation(fit._param_model)
        
        #TODO cost function
        
        _yaml_doc['minimizer'] = fit._minimizer
        _yaml_doc['minimizer_kwargs'] = fit._minimizer_kwargs
        
        return _yaml_doc
    
class FitYamlReader(YamlReaderMixin, FitDReprBase):
    
    def __init__(self, input_io_handle):
        super(FitYamlReader, self).__init__(
            input_io_handle=input_io_handle,
            fit=None)

    @classmethod
    def _get_required_keywords(cls, yaml_doc, fit_class):
        _required_keywords = []
        if 'dataset' not in yaml_doc:
            _required_keywords += DataContainerYamlReader._get_required_keywords(
                yaml_doc,
                DataContainerYamlReader._OBJECT_TYPE_NAME_TO_CLASS.get(
                    cls._CLASS_TO_OBJECT_TYPE_NAME.get(fit_class)
                )
            )
        if 'parametric_model' not in yaml_doc:
            _required_keywords += ParametricModelYamlReader._get_required_keywords(
                yaml_doc,
                ParametricModelYamlReader._OBJECT_TYPE_NAME_TO_CLASS.get(
                    cls._CLASS_TO_OBJECT_TYPE_NAME.get(fit_class)
                )
            )
        return _required_keywords
    
    @classmethod
    def _convert_yaml_doc_to_object(cls, yaml_doc):
        # -- determine model function class from type
        _fit_type = yaml_doc.pop('type')
        _class = cls._OBJECT_TYPE_NAME_TO_CLASS.get(_fit_type, None)
        
        _data = DataContainerYamlReader._make_object(yaml_doc.pop('dataset'))
        _parametric_model = ParametricModelYamlReader._make_object(yaml_doc.pop('parametric_model'))
        #TODO cost function
        _minimizer = yaml_doc.pop('minimizer', None)
        _minimizer_kwargs = yaml_doc.pop('minimizer_kwargs', None)
        if _class is HistFit:
            _fit_object = HistFit(
                data=_data,
                model_density_function=_parametric_model._model_function_object,
                model_density_antiderivative=None,
                minimizer=_minimizer,
                minimizer_kwargs=_minimizer_kwargs
            )
        elif _class is IndexedFit:
            _fit_object = IndexedFit(
                data=_data,
                model_function=_parametric_model._model_function_object,
                minimizer=_minimizer,
                minimizer_kwargs=_minimizer_kwargs                
            )
        elif _class is XYFit:
            _fit_object = XYFit(
                xy_data=_data,
                model_function=_parametric_model._model_function_object,
                minimizer=_minimizer,
                minimizer_kwargs=_minimizer_kwargs
            )
        elif _class is XYMultiFit:
            _fit_object = XYMultiFit(
                xy_data=_data,
                model_function=_parametric_model._model_function_object,
                minimizer=_minimizer,
                minimizer_kwargs=_minimizer_kwargs
            )
        _fit_object._param_model = _parametric_model
        return _fit_object, yaml_doc
    
# register the above classes in the module-level dictionary
FitYamlReader._register_class(_AVAILABLE_REPRESENTATIONS)
FitYamlWriter._register_class(_AVAILABLE_REPRESENTATIONS)
