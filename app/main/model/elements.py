import re
from pymodm import MongoModel, fields, EmbeddedMongoModel
from pymodm.errors import DoesNotExist
from bson.objectid import ObjectId


class GlobalSettings:
   class Meta:
       # Set true --> cls variable will not be stored in the db
       final = True
       # Set true --> fields that are not defined in the model are ignored and deleted in the db when calling save()
       ignore_unknown_fields = True


class QCUser(MongoModel):
    class Meta:
        connection_alias = 'photon_wizard'

    dn = fields.CharField()
    username = fields.CharField()
    data = fields.DictField()
    is_authenticated = fields.BooleanField(default=True)
    is_active = fields.BooleanField(default=True)
    is_anonymous = fields.BooleanField(default=False)

    def get_id(self):
        return self.dn


class BaseElement(MongoModel):

    class Meta:
        connection_alias = 'photon_wizard'

    name = fields.CharField(blank=True)
    short_description = fields.CharField(blank=True)
    long_description = fields.CharField(blank=True)
    imports = fields.CharField(blank=True)
    syntax = fields.CharField(blank=True)
    tags = fields.ListField(blank=True)
    has_hyperparameters = fields.BooleanField(default=False)

    # def __init__(self):
    #     self._mongometa.final = True

    @property
    def system_name(self):
        pattern = re.compile('[\W_]+')
        cleaned_string = pattern.sub('_', self.name)
        # Todo lower?
        return cleaned_string

    @property
    def hyperparameter_objects(self):
        if self.has_hyperparameters:
            obj_list = []
            for i in self.hyperparameters:
                # try:
                query = Hyperparameter.objects.raw({'_id': i})
                nr_of_objects = query.count()
                if nr_of_objects > 0:
                    loaded_object = query.first()
                    obj_list.append(loaded_object)
            return obj_list
        else:
            return []

    def get_filtered_hyperparameter_objects(self, constraint_list):
        if self.has_hyperparameters:
            obj_list = []
            for i in self.hyperparameters:
                # try:
                if len(constraint_list) > 0:
                    query = Hyperparameter.objects.raw({'$and': [{'_id': i}, {'tags': {'$all': constraint_list}}]})
                    # query = Hyperparameter.objects.raw({'$and': [{'_id': i}, {'tags': constraint_list}]})
                else:
                    query = Hyperparameter.objects.raw({'_id': i})
                nr_of_objects = query.count()
                if nr_of_objects > 0:
                    loaded_object = query.first()
                    obj_list.append(loaded_object)
            return obj_list
        else:
            return []


class DataType(BaseElement):
    pass


class DataQuantity(BaseElement):
    lower_thres = fields.IntegerField(blank=True)
    upper_thres = fields.IntegerField(blank=True)


class AnalysisType(BaseElement):
    pass


class BrainAtlas(BaseElement):
    hyperparameters = fields.ListField(blank=True)
    has_hyperparameters = fields.BooleanField(default=True)


class NeuroTransformer(BaseElement):
    hyperparameters = fields.ListField(blank=True)
    has_hyperparameters = fields.BooleanField(default=True)


class CV(BaseElement):
    hyperparameters = fields.ListField(blank=True)
    arguments = fields.DictField(blank=True)
    has_hyperparameters = fields.BooleanField(default=True)


class Metric(BaseElement):
    allow_for = fields.ReferenceField(AnalysisType, blank=True)


class ImbStrategy(BaseElement):
    hyperparameters = fields.ListField(blank=True)
    arguments = fields.DictField(blank=True)


class Optimizer(BaseElement):
    hyperparameters = fields.ListField(blank=True)
    arguments = fields.DictField(blank=True)
    has_hyperparameters = fields.BooleanField(default=True)


class Transformer(BaseElement):
    hyperparameters = fields.ListField(blank=True)
    test_disabled = fields.BooleanField(blank=True)
    has_hyperparameters = fields.BooleanField(default=True)
    pre_processing = fields.BooleanField(blank=True, default=False)


class Estimator(BaseElement):
    hyperparameters = fields.ListField(blank=True)
    estimator_type = fields.ReferenceField(AnalysisType, blank=True)
    has_hyperparameters = fields.BooleanField(default=True)


class Hyperparameter(BaseElement):
    default_values = fields.ListField(blank=True)
    possible_values = fields.ListField(blank=True)
    value_type = fields.CharField(choices=['Categorical', 'Boolean', 'FloatRange', 'IntRange', 'Float', 'Int',
                                           'String', 'List'], blank=True)
    multi_select = fields.BooleanField(default=True)

    @property
    def default_objects(self):
        obj_list = []
        for i in self.default_values:
            # try:
            loaded_object = Defaultparameter.objects.raw({'_id': i}).first()
            obj_list.append(loaded_object)
        return obj_list


class Defaultparameter(BaseElement):
    values = fields.ListField(blank=True)


class PermutationTestInfos(EmbeddedMongoModel):

    class Meta:
        connection_alias = 'photon_wizard'

    n_perms = fields.IntegerField(blank=True)
    perm_id = fields.CharField(blank=True)
    permutation_file_path = fields.CharField(blank=True)
    duration_per_permutation = fields.FloatField(blank=True)
    permutation_celery_id = fields.CharField(blank=True)
    permutation_celery_status = fields.CharField(blank=True)


class ElementCombi(MongoModel):

    class Meta:
        connection_alias = 'photon_wizard'

    name = fields.CharField(blank=True)
    # referenced_element = fields.ReferenceField(BaseElement)
    referenced_element_id = fields.ObjectIdField(blank=True)
    # referenced_element_type = fields.CharField()
    hyperparameters = fields.DictField(blank=True)
    test_disabled = fields.BooleanField(default=False)

    def copy_me(self):
        return ElementCombi(name=self.name,
                            referenced_element_id=ObjectId(str(self.referenced_element_id)),
                            hyperparameters=dict(self.hyperparameters),
                            test_disabled=bool(self.test_disabled))


class Pipeline(MongoModel):

    class Meta:
        connection_alias = 'photon_wizard'

    id = fields.CharField(blank=True)

    name = fields.CharField(blank=True)
    creation_date = fields.DateTimeField(blank=True)
    project_name = fields.CharField(blank=True)
    photon_project_folder = fields.CharField(blank=True)
    user = fields.CharField(blank=True)
    status = fields.IntegerField(blank=True)
    description = fields.CharField(blank=True)

    # computation
    mongodb_connect_url = fields.CharField(blank=True,
                                           default='mongodb://localhost:27017/photon_results')
    collected_all_information = fields.BooleanField(default=False)
    celery_id = fields.CharField(blank=True)
    celery_status = fields.CharField(blank=True)
    # only used for online wizard
    photon_file_path = fields.CharField(blank=True)

    # Analysis Setup
    data_type = fields.ReferenceField(DataType, blank=True)
    data_quantity = fields.ReferenceField(DataQuantity, blank=True)
    analysis_type = fields.ReferenceField(AnalysisType, blank=True)
    permutation_test = fields.EmbeddedDocumentField(PermutationTestInfos, blank=True)

    # Data
    data_file = fields.CharField(blank=True)
    covariates = fields.CharField(blank=True)
    targets = fields.CharField(blank=True)
    features = fields.CharField(blank=True)
    groups = fields.CharField(blank=True)

    # Design and Pipeline Elements
    brainatlas = fields.ReferenceField(ElementCombi, blank=True)
    neuro_transformer = fields.ListField(blank=True)
    optimizer = fields.ReferenceField(ElementCombi, blank=True)
    inner_cv = fields.ReferenceField(ElementCombi, blank=True)
    outer_cv = fields.ReferenceField(ElementCombi, blank=True)
    metrics = fields.ListField(blank=True)
    best_config_metric = fields.ReferenceField(Metric, blank=True)
    transformer_elements = fields.DictField(blank=True)
    estimator_element_list = fields.ListField(blank=True)

    # will be filled during process to update list of items to choose from according to previous selections
    constraint_dict = fields.DictField(blank=True)

    @property
    def creation_time_str(self):
        return self.creation_date.strftime('%d.%m.%Y, %H:%M')

    @staticmethod
    def get_nr_elements_and_hyperparameters(t_dict):
        nr_of_transformer_elements = len(t_dict.keys())
        nr_of_hyperparameters = len([hyperparameter
                                     for name, hyperparameter_dict in t_dict.items()
                                     for hyperparameter, _ in hyperparameter_dict.items()])
        return nr_of_transformer_elements, nr_of_hyperparameters

    @staticmethod
    def element_combi_to_key_value_pair(element_combi_id, referenced_class):
        try:
            if isinstance(element_combi_id, str):
                element_combi_id = ObjectId(element_combi_id)
            element_combi = ElementCombi.objects.get({'_id': element_combi_id})
        except DoesNotExist as e:
            return "", {}
        actual_obj_name = Pipeline.name_for_id(element_combi.referenced_element_id, referenced_class)
        return actual_obj_name, element_combi.hyperparameters

    @staticmethod
    def name_for_id(obj_id, obj_class):
        actual_obj = obj_class.objects.get({'_id': obj_id})
        return actual_obj.name

    @staticmethod
    def key_value_pair_to_string(key_value_pair):
        name = key_value_pair[0]
        suffix = " |".join([value_key + ": " + str(value_value) for value_key, value_value in key_value_pair[1].items()])
        return dict([(name, suffix)])

    @property
    def transformer_dict(self):
        elements = dict()
        for _, transformer in self.transformer_elements.items():
            element = list(Transformer.objects.raw({'_id': transformer['ObjectId']}))[0]
            elements[element.name] = transformer['hyperparameters']
        return elements

    @property
    def transformer_summary(self):
        nr_of_transformer_elements, nr_of_hyperparameters = self.get_nr_elements_and_hyperparameters(
            self.transformer_dict)
        return "Your pipeline contains <b>" + str(
            nr_of_transformer_elements) + " transformer</b> elements specifying <b>" + str(
            nr_of_hyperparameters) + " hyperparameters.</b>"

    @property
    def estimator_dict(self):
        elements = dict()
        for estimator_id in self.estimator_element_list:
            elements.update([self.element_combi_to_key_value_pair(estimator_id, Estimator)])
        return elements

    @property
    def estimator_summary(self):
        estimator_dict = self.estimator_dict
        if len(estimator_dict) > 1:
            nr_of_transformer_elements, nr_of_hyperparameters = self.get_nr_elements_and_hyperparameters(
                self.estimator_dict)
            return "You choose between <b>" + str(nr_of_transformer_elements) \
                   + " learning algorithms</b> with a total of <b>" \
                   + str(nr_of_hyperparameters) + " hyperparameters.</b>"
        else:
            estimator = next(iter(estimator_dict.keys()))
            hyperparam_nr = len(estimator_dict[estimator])
            return "Your learning algorithm is <b>" + str(estimator) + "</b> with " \
                                                                       "" + str(hyperparam_nr) + " hyperparameters."

    @property
    def neuro_dict(self):
        neuro_dict = {}
        for neuro_id in self.neuro_transformer:
            neuro_dict.update([self.element_combi_to_key_value_pair(neuro_id, NeuroTransformer)])
        if self.brainatlas:
            neuro_dict.update([self.element_combi_to_key_value_pair(self.brainatlas._id, BrainAtlas)])
        return neuro_dict

    @property
    def neuro_summary(self):
        nr_of_transformer_elements, nr_of_hyperparameters = self.get_nr_elements_and_hyperparameters(self.neuro_dict)
        return "Your neuroimaging transformations consist of <b>" + str(nr_of_transformer_elements) + " elements</b> " \
                                                                                                      "with <b>" \
               + str(nr_of_hyperparameters) + " hyperparameters</b>."

    @property
    def meta_data_dict(self):
        meta_data_dict = dict()
        meta_data_dict["Optimizer"] = self.key_value_pair_to_string(self.element_combi_to_key_value_pair(self.optimizer._id, Optimizer))
        meta_data_dict["Inner Cross Validation Strategy"] = self.key_value_pair_to_string(self.element_combi_to_key_value_pair(self.inner_cv._id, CV))
        if self.outer_cv is not None:
            meta_data_dict["Outer Cross Validation Strategy"] = self.key_value_pair_to_string(self.element_combi_to_key_value_pair(self.outer_cv._id, CV))
        meta_data_dict["Metrics"] = dict([(self.name_for_id(metric_id, Metric), {}) for metric_id in self.metrics])
        if self.best_config_metric is not None:
            meta_data_dict["Best Config Metric"] = {self.best_config_metric.name: {}}
        else:
            meta_data_dict["Best Config Metric"] = {}
        return meta_data_dict

    @property
    def meta_data_summary(self):
        meta_data_dict = self.meta_data_dict
        best_metric = next(iter(meta_data_dict["Best Config Metric"]))
        optimizer = next(iter(meta_data_dict["Optimizer"].keys()))
        return "You optimize for <b>" + best_metric + "</b> using <b>" + str(optimizer) + "</b>."

    def copy_me(self):

        new_copy = Pipeline()

        list_of_native_types_to_copy = ['user', 'description', 'data_file', 'covariates', 'targets', 'features', 'groups',
                                        'neuro_transformer', 'transformer_elements', 'constraint_dict',
                                        'estimator_element_list', 'metrics']
        for attr_name in list_of_native_types_to_copy:
            attr_value = self.__getattribute__(attr_name)
            if attr_value is not None:
                class_name = attr_value.__class__
                value_copy = class_name(attr_value)
                new_copy.__setattr__(attr_name, value_copy)

        list_of_reference_fields = ['data_type', 'data_quantity', 'analysis_type', 'best_config_metric']
        for ref in list_of_reference_fields:
            old_ref = self.__getattribute__(ref)
            if old_ref is not None:
                new_obj_id = ObjectId(old_ref._id)
                new_copy.__setattr__(ref, new_obj_id)

        # copy element_combis
        list_of_element_combis=['brainatlas', 'optimizer', 'inner_cv', 'outer_cv']
        for ec in list_of_element_combis:
            old_ed = self.__getattribute__(ec)
            if old_ed is not None:
                new_ec = old_ed.copy_me()
                new_ec.save()
                new_copy.__setattr__(ec, new_ec)

        new_copy.save()
        return new_copy


class Rules(MongoModel):

    class Meta:
        connection_alias = 'photon_wizard'

    requirements = fields.ListField(blank=True)
    white_list = fields.ListField(blank=True)
    element_type = fields.CharField(blank=True)
    explanation = fields.CharField(blank=True)


class DefaultPipeline(MongoModel):

    class Meta:
        connection_alias = 'photon_wizard'

    name = fields.CharField()
    analysis_type = fields.ReferenceField(AnalysisType)
    data_type = fields.ReferenceField(DataType)
    method_to_call = fields.CharField(blank=True)
    property_dict = fields.DictField(blank=True)
    example_pipeline = fields.ReferenceField(Pipeline)
    complexity = fields.CharField(blank=True)
    display_order = fields.IntegerField(blank=True)

    @property
    def sorted_property_dict(self):
        return dict([(k, self.property_dict[k]) for k in sorted(self.property_dict)])
