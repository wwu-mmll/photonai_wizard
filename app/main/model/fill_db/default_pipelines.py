from app.main.model.elements import *
from app.main.model.tutorial_definition import create_boston_housing, create_neuro_classification, create_neuro_regression, create_breast_cancer
from app.main.model.code_creator import PhotonCodeCreator
# from app.main.wizard_config import WizardConfig
from pymodm import connect
import requests
import os
import time


def data_type_to_object_id(data_type):
    return DataType.objects.get({'name': data_type})._id


def analysis_type_to_object_id(analysis_type):
    return AnalysisType.objects.get({'name': analysis_type})


def generate_default_pipeline_items():

    list_of_available_tabular_pipelines = {0: 'basic', 2: 'estimator_switch', 1: 'feature_selection',
                                           3: 'estimator_switch_feature_selection'}
    header_dict = {'Tabular Data':
                       {'Classification': ["Standard Classifier", "Classifier Selection",
                                          "Feature Selection Classification", "Feature and Classifier Selection"],
                        'Regression': ["Standard Regressor", "Regressor Selection", "Feature Selection Regression",
                                       "Feature and Regressor Selection"]},

                   'Nifti Data':
                       {'Classification': ["Neuro Standard Classification", "Neuro Cortical Classification"],
                        'Regression': ["Neuro Standard Regression",  "Neuro Cortical Regression"]}}
    list_of_available_neuro_pipelines = {0: 'neuro_basic', 1: 'neuro_cortical'}
    setup_dictionary = {'Tabular Data': list_of_available_tabular_pipelines,
                        'Nifti Data': list_of_available_neuro_pipelines}

    for data_type, method_list in setup_dictionary.items():
        for analysis_type in ["Classification", "Regression"]:
            nr_of_pipelines = 0
            for display_nr, method_to_call in method_list.items():
                for hyperparameter_optimization in [False, True]:
                    nr_of_pipelines += 1
                    new_default_pipe = DefaultPipeline()
                    new_default_pipe.name = "Default Pipeline " + str(nr_of_pipelines)
                    new_default_pipe.analysis_type = analysis_type_to_object_id(analysis_type)
                    new_default_pipe.data_type = data_type_to_object_id(data_type)
                    new_default_pipe.display_order = display_nr
                    name = ''
                    if hyperparameter_optimization:
                        suffix = '_hyperparameter_optimization'
                    else:
                        suffix = ''

                    new_default_pipe.method_to_call = analysis_type.lower() + "_" + method_to_call + suffix

                    call_obj = getattr(DefaultPipelineRepo, new_default_pipe.method_to_call)
                    new_default_pipe.example_pipeline = call_obj(Pipeline())
                    new_default_pipe.example_pipeline.save()

                    new_default_pipe.property_dict = {}
                    if data_type == 'Tabular Data':
                        new_default_pipe.property_dict['Hyperparameter Optimization'] = hyperparameter_optimization
                        if "feature_selection" in method_to_call:
                            new_default_pipe.property_dict['Feature Selection'] = True
                        else:
                            new_default_pipe.property_dict['Feature Selection'] = False
                        if "estimator_switch" in method_to_call:
                            new_default_pipe.property_dict['Estimator Selection'] = True
                        else:
                            new_default_pipe.property_dict['Estimator Selection'] = False
                    else:
                         new_default_pipe.property_dict['Estimator Selection'] = False
                         new_default_pipe.property_dict['Feature Selection'] = False

                    pca_obj_id = Transformer.objects.get({'name': "PCA"})._id
                    if len([t for t, v in new_default_pipe.example_pipeline.transformer_elements.items()
                            if v["ObjectId"] == pca_obj_id]) > 0:
                        new_default_pipe.property_dict["Dimensionality Reduction"] = True
                    else:
                        new_default_pipe.property_dict["Dimensionality Reduction"] = False

                    if hyperparameter_optimization:
                        new_default_pipe.complexity = 'Optimized'
                    else:
                        new_default_pipe.complexity = 'Basic'

                    new_default_pipe.name = header_dict[data_type][analysis_type][display_nr]
                    new_default_pipe.save()


class DefaultPipelineBuilder:

    def __init__(self, pipeline: Pipeline, analysis_type: str, tune_hyperparameters: bool,
                 optimizer: str = 'random_grid_search', n_configurations: int = 30,
                 no_outer_cv: bool = False):
        self.pipeline = pipeline
        self.analysis_type = analysis_type
        self.pipeline.transformer_elements = dict()
        self.pipeline.estimator_element_list = list()
        if not optimizer:
            if tune_hyperparameters:
                self.define_optimization('random_grid_search', 30)
            else:
                self.define_optimization('grid_search', 0)
        else:
            self.define_optimization(optimizer, n_configurations)

        self.define_data_quantity()
        self.define_cross_validation(no_outer_cv)
        self.define_metrics()
        self.tune_hyperparameters = tune_hyperparameters

    def define_data_quantity(self):
        dq = DataQuantity.objects.get({'name': 'Between 101 and 500 samples'})
        self.pipeline.data_quantity = dq

    def define_metrics(self):
        if self.analysis_type == 'classification':
            self.pipeline.metrics = []
            for metric_name in ['accuracy', 'balanced_accuracy', 'specificity', 'sensitivity']:
                metric = Metric.objects.get({'name': metric_name})
                self.pipeline.metrics.append(metric._id)
            metric = Metric.objects.get({'name': 'balanced_accuracy'})
            self.pipeline.best_config_metric = metric._id
        else:
            self.pipeline.metrics = []
            for metric_name in ['mean_squared_error', 'mean_absolute_error', 'explained_variance']:
                metric = Metric.objects.get({'name': metric_name})
                self.pipeline.metrics.append(metric._id)
            metric = Metric.objects.get({'name': 'mean_absolute_error'})
            self.pipeline.best_config_metric = metric._id

    def define_optimization(self, optimizer_name, n_configurations):
        optimizer = list(Optimizer.objects.raw({'name': optimizer_name}))[0]
        optimizer_combi = ElementCombi()
        optimizer_combi.name = optimizer_name
        if not optimizer_name == 'grid_search':
            optimizer_combi.hyperparameters = {'n_configurations': "{}".format(n_configurations)}
        optimizer_combi.referenced_element_id = optimizer._id
        optimizer_combi.save()
        self.pipeline.optimizer = optimizer_combi

    def define_cross_validation(self, no_outer_cv=False):
        cv = CV.objects.get({'name': 'KFold'})
        inner_cv_combi = ElementCombi()
        inner_cv_combi.name = 'KFold'
        inner_cv_combi.referenced_element_id = cv._id
        inner_cv_combi.hyperparameters['n_splits'] = 3
        inner_cv_combi.hyperparameters['shuffle'] = True
        inner_cv_combi.save()
        self.pipeline.inner_cv = inner_cv_combi
        if not no_outer_cv:
            outer_cv_combi = ElementCombi()
            outer_cv_combi.name = 'KFold'
            outer_cv_combi.referenced_element_id = cv._id
            outer_cv_combi.hyperparameters['n_splits'] = 5
            outer_cv_combi.hyperparameters['shuffle'] = True
            outer_cv_combi.save()
            self.pipeline.outer_cv = outer_cv_combi

    def create_transformer_combi(self, transformer_name, hyperparameters, position):
        transformer_no = 'transformer_{}'.format(position)
        transformer = {}
        label_encoder = list(Transformer.objects.raw({'name': transformer_name}))[0]
        transformer['ObjectId'] = label_encoder._id
        transformer['position'] = position
        transformer['test_disabled'] = False
        transformer['is_active'] = True
        transformer['hyperparameters'] = hyperparameters
        self.pipeline.transformer_elements[transformer_no] = transformer

    def create_estimator_combi(self, estimator_name, hyperparameters):
        est = list(Estimator.objects.raw({'name': estimator_name}))[0]
        est_combi = ElementCombi()
        est_combi.referenced_element_id = est._id
        est_combi.hyperparameters = hyperparameters
        est_combi.save()
        self.pipeline.estimator_element_list.append(est_combi._id)

    def add_label_encoder(self, position: int):
        hyperparameters = {}
        self.create_transformer_combi('LabelEncoder', hyperparameters, position)

    def add_pca(self, position: int):
        if self.tune_hyperparameters:
            hyperparameters = {'n_components': "FloatRange(0.2, 0.99)"}
        else:
            hyperparameters = {'n_components': 0.8}
        self.create_transformer_combi('PCA', hyperparameters, position)

    def add_scaler(self, position: int):
        hyperparameters = {'with_mean': True, 'with_std': True}
        self.create_transformer_combi('StandardScaler', hyperparameters, position)

    def add_simple_imputer(self, position: int):
        hyperparameters = {'missing_values': 'np.nan', 'strategy': 'mean', 'fill_value': 0}
        self.create_transformer_combi('SimpleImputer', hyperparameters, position)

    def add_select_percentile(self, position: int):
        if self.analysis_type == 'classification':
            name = 'FClassifSelectPercentile'
        else:
            name = 'FRegressionSelectPercentile'
        if self.tune_hyperparameters:
            hyperparameters = {'percentile': [10, 25, 50, 75]}
        else:
            hyperparameters = {'percentile': 25}

        self.create_transformer_combi(name, hyperparameters, position)

    def add_random_forest(self):
        if self.analysis_type == 'classification':
            est_name = 'RandomForestClassifier'
            if self.tune_hyperparameters:
                hyperparameters = {'n_estimators': [25, 50, 75], 'criterion': 'gini', 'max_depth': None,
                                   'min_samples_split': 'IntegerRange(2, 10)', 'min_samples_leaf': 'IntegerRange(1, 10)'}
            else:
                hyperparameters = {'n_estimators': 50, 'criterion': 'gini', 'max_depth': None,
                                   'min_samples_split': 2, 'min_samples_leaf': 1}
        else:
            est_name = 'RandomForestRegressor'
            if self.tune_hyperparameters:
                hyperparameters = {'n_estimators': [25, 50, 75], 'criterion': 'mse', 'max_depth': None,
                                   'min_samples_split': 'IntegerRange(2, 10)', 'min_samples_leaf': 'IntegerRange(1, 10)'}
            else:
                hyperparameters = {'n_estimators': 50, 'criterion': 'mse', 'max_depth': None,
                                   'min_samples_split': 2, 'min_samples_leaf': 1}
        self.create_estimator_combi(est_name, hyperparameters)

    def add_svm(self):
        if self.analysis_type == 'classification':
            est_name = 'SVC'
            if self.tune_hyperparameters:
                hyperparameters = {'C': 'FloatRange(1e-7, 1e7, range_type="geomspace")', 'gamma': 'scale',
                                   'max_iter': 1e6, 'kernel': ["linear", "rbf"]}
            else:
                hyperparameters = {'C': 1, 'gamma': 'scale',
                                   'max_iter': 1e6, 'kernel': 'linear'}
        else:
            est_name = 'SVR'
            if self.tune_hyperparameters:
                hyperparameters = {'C': 'FloatRange(1e-7, 1e7, range_type="geomspace")', 'gamma': 'scale',
                                   'max_iter': 1e6, 'kernel': ["linear", "rbf"], 'epsilon': 0.1}
            else:
                hyperparameters = {'C': 1, 'gamma': 'scale',
                                   'max_iter': 1e6, 'kernel': 'linear', 'epsilon': 0.1}
        self.create_estimator_combi(est_name, hyperparameters)

    def add_ada_boost(self):
        if self.analysis_type == 'classification':
            est_name = 'AdaBoostClassifier'
            if self.tune_hyperparameters:
                hyperparameters = {'n_estimators': 'IntegerRange(30, 80)', 'learning_rate': 'FloatRange(0.1, 1)'}
            else:
                hyperparameters = {'n_estimators': 50, 'learning_rate': 1}
        else:
            est_name = 'AdaBoostRegressor'
            if self.tune_hyperparameters:
                hyperparameters = {'n_estimators': 'IntegerRange(30, 80)', 'learning_rate': 'FloatRange(0.1, 1)',
                                   'loss': 'linear'}
            else:
                hyperparameters = {'n_estimators': 50, 'learning_rate': 1, 'loss': 'linear'}

        self.create_estimator_combi(est_name, hyperparameters)

    def add_gaussian_process(self):
        if self.analysis_type == 'classification':
            est_name = 'GaussianProcessClassifier'
            if self.tune_hyperparameters:
                hyperparameters = {'optimizer': 'fmin_l_bfgs_b', 'n_restarts_optimizer': 0}
            else:
                hyperparameters = {'optimizer': 'fmin_l_bfgs_b', 'n_restarts_optimizer': 0}
        else:
            est_name = 'GaussianProcessRegressor'
            if self.tune_hyperparameters:
                hyperparameters = {'optimizer': 'fmin_l_bfgs_b', 'n_restarts_optimizer': 0, 'alpha': 1e-10}
            else:
                hyperparameters = {'optimizer': 'fmin_l_bfgs_b', 'n_restarts_optimizer': 0, 'alpha': 1e-10}
        self.create_estimator_combi(est_name, hyperparameters)

    def add_brain_mask(self):
        mask = BrainAtlas.objects.get({'name': 'MNI_ICBM152_GrayMatter'})
        combi = ElementCombi()
        combi.referenced_element_id = mask._id
        combi.save()
        self.pipeline.brainatlas = combi._id

    def add_cortical_atlas(self):
        atlas = list(BrainAtlas.objects.raw({'name': 'HarvardOxford_Cortical_Threshold_25'}))[0]
        combi = ElementCombi()
        combi.referenced_element_id = atlas._id
        combi.hyperparameters = {'roi_names': ['all']}
        combi.save()
        self.pipeline.brainatlas = combi._id


# --------------------------
# Screening
# --------------------------
class DefaultPipelineRepo:

    @staticmethod
    def test_default_pipelines():

        tabular_data_variations = ["basic", "estimator_switch", "feature_selection", "estimator_switch_feature_selection"]
        neuro_variations = ["neuro_basic", "neuro_cortical"]

        DefaultPipelineRepo._apply_test_function(tabular_data_variations)
        DefaultPipelineRepo._apply_test_function(neuro_variations)

    @staticmethod
    def _apply_test_function(method_names):
        test_server = "http://0.0.0.0:8003/run/"
        username = "default_pipe_test_user"
        origins = ["classification", "regression"]
        suffix = ["", "_hyperparameter_optimization"]

        for analysis_type in origins:
            for sf in suffix:
                for excel_function in method_names:
                    method_name = analysis_type + "_" + excel_function + sf
                    method = getattr(DefaultPipelineRepo, method_name)

                    if "neuro" in method_name:
                        if analysis_type == "classification":
                            pipe = create_neuro_classification(username)
                        else:
                            pipe = create_neuro_regression(username)
                    else:
                        if analysis_type == "classification":
                            pipe = create_breast_cancer(username)
                        else:
                            pipe = create_boston_housing(username)

                    updated_pipe = method(pipe)

                    updated_pipe.photon_file_path = os.path.join(updated_pipe.photon_project_folder, "photon_code.py")
                    updated_pipe.save()
                    with open(updated_pipe.photon_file_path, "w") as text_file:
                        photon_code = PhotonCodeCreator(udo_mode=True).create_code(updated_pipe)
                        text_file.write(photon_code)
                        text_file.close()

                    res = requests.get(test_server + str(updated_pipe._id) + "/photon")
                    print(method_name + ": " + str(res))
                    time.sleep(5)


    @staticmethod
    def classification_basic(pipeline):
        builder = DefaultPipelineBuilder(pipeline, analysis_type='classification',
                                         tune_hyperparameters=False, no_outer_cv=True)
        builder.add_label_encoder(0)
        builder.add_simple_imputer(1)
        builder.add_pca(2)
        builder.add_random_forest()
        return builder.pipeline

    @staticmethod
    def classification_basic_hyperparameter_optimization(pipeline):
        builder = DefaultPipelineBuilder(pipeline, analysis_type='classification',
                                         tune_hyperparameters=True)
        builder.add_label_encoder(0)
        builder.add_simple_imputer(1)
        builder.add_pca(2)
        builder.add_random_forest()
        return builder.pipeline

    @staticmethod
    def regression_basic(pipeline):
        builder = DefaultPipelineBuilder(pipeline, analysis_type='regression',
                                         tune_hyperparameters=False, no_outer_cv=True)

        builder.add_simple_imputer(0)
        builder.add_pca(1)
        builder.add_random_forest()
        return builder.pipeline

    @staticmethod
    def regression_basic_hyperparameter_optimization(pipeline):
        builder = DefaultPipelineBuilder(pipeline, analysis_type='regression',
                                         tune_hyperparameters=True)

        builder.add_simple_imputer(0)
        builder.add_pca(1)
        builder.add_random_forest()
        return builder.pipeline

    # --------------------------
    # Multiple Estimator
    # --------------------------
    @staticmethod
    def classification_estimator_switch(pipeline):
        builder = DefaultPipelineBuilder(pipeline, analysis_type='classification',
                                         tune_hyperparameters=False)

        builder.add_label_encoder(0)
        builder.add_simple_imputer(1)
        builder.add_pca(2)
        builder.add_random_forest()
        builder.add_gaussian_process()
        builder.add_ada_boost()
        builder.add_svm()
        return builder.pipeline

    @staticmethod
    def classification_estimator_switch_hyperparameter_optimization(pipeline):
        builder = DefaultPipelineBuilder(pipeline, analysis_type='classification',
                                         tune_hyperparameters=True)
        builder.add_label_encoder(0)
        builder.add_simple_imputer(1)
        builder.add_pca(2)
        builder.add_random_forest()
        builder.add_gaussian_process()
        builder.add_ada_boost()
        builder.add_svm()
        return builder.pipeline

    @staticmethod
    def regression_estimator_switch(pipeline):
        builder = DefaultPipelineBuilder(pipeline, analysis_type='regression',
                                         tune_hyperparameters=False)
        builder.add_simple_imputer(0)
        builder.add_pca(1)
        builder.add_random_forest()
        builder.add_gaussian_process()
        builder.add_ada_boost()
        builder.add_svm()
        return builder.pipeline

    @staticmethod
    def regression_estimator_switch_hyperparameter_optimization(pipeline):
        builder = DefaultPipelineBuilder(pipeline, analysis_type='regression',
                                         tune_hyperparameters=True)
        builder.add_simple_imputer(0)
        builder.add_pca(1)
        builder.add_random_forest()
        builder.add_gaussian_process()
        builder.add_ada_boost()
        builder.add_svm()
        return builder.pipeline

    # --------------------------
    # Feature Selection
    # --------------------------
    @staticmethod
    def classification_feature_selection(pipeline):
        builder = DefaultPipelineBuilder(pipeline, analysis_type='classification',
                                         tune_hyperparameters=False)
        builder.add_label_encoder(0)
        builder.add_simple_imputer(1)
        builder.add_select_percentile(2)
        builder.add_svm()
        return builder.pipeline

    @staticmethod
    def classification_feature_selection_hyperparameter_optimization(pipeline):
        builder = DefaultPipelineBuilder(pipeline, analysis_type='classification',
                                         tune_hyperparameters=True)
        builder.add_label_encoder(0)
        builder.add_simple_imputer(1)
        builder.add_select_percentile(2)
        builder.add_svm()
        return builder.pipeline

    @staticmethod
    def regression_feature_selection(pipeline):
        builder = DefaultPipelineBuilder(pipeline, analysis_type='regression',
                                         tune_hyperparameters=False)
        builder.add_simple_imputer(0)
        builder.add_select_percentile(1)
        builder.add_svm()
        return builder.pipeline

    @staticmethod
    def regression_feature_selection_hyperparameter_optimization(pipeline):
        builder = DefaultPipelineBuilder(pipeline, analysis_type='regression',
                                         tune_hyperparameters=True)
        builder.add_simple_imputer(0)
        builder.add_select_percentile(1)
        builder.add_svm()
        return builder.pipeline

    # ---------------------------------------
    # Feature Selection + Multiple Estimator
    # ---------------------------------------
    @staticmethod
    def classification_estimator_switch_feature_selection(pipeline):
        builder = DefaultPipelineBuilder(pipeline, analysis_type='classification',
                                         tune_hyperparameters=False)
        builder.add_label_encoder(0)
        builder.add_simple_imputer(1)
        builder.add_select_percentile(2)
        builder.add_svm()
        builder.add_random_forest()
        builder.add_ada_boost()
        builder.add_gaussian_process()
        return builder.pipeline

    @staticmethod
    def classification_estimator_switch_feature_selection_hyperparameter_optimization(pipeline):
        builder = DefaultPipelineBuilder(pipeline, analysis_type='classification',
                                         tune_hyperparameters=True)
        builder.add_label_encoder(0)
        builder.add_simple_imputer(1)
        builder.add_select_percentile(2)
        builder.add_svm()
        builder.add_random_forest()
        builder.add_ada_boost()
        builder.add_gaussian_process()
        return builder.pipeline

    @staticmethod
    def regression_estimator_switch_feature_selection(pipeline):
        builder = DefaultPipelineBuilder(pipeline, analysis_type='regression',
                                         tune_hyperparameters=False)
        builder.add_simple_imputer(0)
        builder.add_select_percentile(1)
        builder.add_svm()
        builder.add_random_forest()
        builder.add_ada_boost()
        builder.add_gaussian_process()
        return builder.pipeline

    @staticmethod
    def regression_estimator_switch_feature_selection_hyperparameter_optimization(pipeline):
        builder = DefaultPipelineBuilder(pipeline, analysis_type='regression',
                                         tune_hyperparameters=True)
        builder.add_simple_imputer(0)
        builder.add_select_percentile(1)
        builder.add_svm()
        builder.add_random_forest()
        builder.add_ada_boost()
        builder.add_gaussian_process()
        return builder.pipeline

    # ---------------------------------------
    # Neuro Screening
    # ---------------------------------------
    @staticmethod
    def classification_neuro_basic(pipeline):
        builder = DefaultPipelineBuilder(pipeline, analysis_type='classification',
                                         tune_hyperparameters=False)
        builder.add_brain_mask()
        builder.add_label_encoder(0)
        builder.add_pca(1)
        builder.add_svm()
        return builder.pipeline

    @staticmethod
    def classification_neuro_basic_hyperparameter_optimization(pipeline):
        builder = DefaultPipelineBuilder(pipeline, analysis_type='classification',
                                         tune_hyperparameters=True)
        builder.add_brain_mask()
        builder.add_label_encoder(0)
        builder.add_pca(1)
        builder.add_svm()
        return builder.pipeline

    @staticmethod
    def regression_neuro_basic(pipeline):
        builder = DefaultPipelineBuilder(pipeline, analysis_type='regression',
                                         tune_hyperparameters=False)
        builder.add_brain_mask()
        builder.add_pca(0)
        builder.add_svm()
        return builder.pipeline

    @staticmethod
    def regression_neuro_basic_hyperparameter_optimization(pipeline):
        builder = DefaultPipelineBuilder(pipeline, analysis_type='regression',
                                         tune_hyperparameters=True)
        builder.add_brain_mask()
        builder.add_pca(0)
        builder.add_svm()
        return builder.pipeline

    # ---------------------------------------
    # Neuro: Cortical
    # ---------------------------------------
    @staticmethod
    def classification_neuro_cortical(pipeline):
        builder = DefaultPipelineBuilder(pipeline, analysis_type='classification',
                                         tune_hyperparameters=False)
        builder.add_cortical_atlas()
        builder.add_label_encoder(0)
        builder.add_pca(1)
        builder.add_svm()
        return builder.pipeline

    @staticmethod
    def classification_neuro_cortical_hyperparameter_optimization(pipeline):
        builder = DefaultPipelineBuilder(pipeline, analysis_type='classification',
                                         tune_hyperparameters=True)
        builder.add_cortical_atlas()
        builder.add_label_encoder(0)
        builder.add_pca(1)
        builder.add_svm()
        return builder.pipeline

    @staticmethod
    def regression_neuro_cortical(pipeline):
        builder = DefaultPipelineBuilder(pipeline, analysis_type='regression',
                                         tune_hyperparameters=False)
        builder.add_cortical_atlas()
        builder.add_pca(0)
        builder.add_svm()
        return builder.pipeline

    @staticmethod
    def regression_neuro_cortical_hyperparameter_optimization(pipeline):
        builder = DefaultPipelineBuilder(pipeline, analysis_type='regression',
                                         tune_hyperparameters=True)
        builder.add_cortical_atlas()
        builder.add_pca(0)
        builder.add_svm()
        return builder.pipeline


if __name__ == "__main__":
    # DefaultPipelineRepo.test_default_pipelines()
    connect("mongodb://trap-umbriel:27017/photon-wizard2", connect=False, alias="photon_wizard")
    generate_default_pipeline_items()
