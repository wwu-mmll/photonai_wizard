import ast
import os
from ..model.elements import *
from bson.objectid import ObjectId
import numpy as np
import subprocess
import shlex
import builtins


class PhotonCodeCreator:

    def __init__(self, udo_mode=True):
        self.pipe_infos = None
        self.udo_mode = udo_mode

    def create_code(self, pipe_infos, final=False):
        self.pipe_infos = pipe_infos
        code = list()
        if self.pipe_infos is not None:
            if final:
                code.append(self._create_file_information())
                code.append(self._create_imports())
            # if with_warnings:
            #     code.append(self.create_warnings())

            code.append(self._create_persist_options())
            code.append(self._create_hyperpipe())
            code.append(self._create_neuro())
            code.append(self._create_transformers())
            code.append(self._create_estimator())
            code.append(self._create_data_loading())
            code.append(self._create_fit())

            merged_code = list()
            for block in code:
                if block is not None:
                    merged_code.extend(block)
            merged_code = ''.join(merged_code)
        return merged_code

    def create_code_for_perm_test(self, pipe_infos):

        self.pipe_infos = pipe_infos

        if self.pipe_infos is not None:

            def merge(code):
                merged_code = list()
                for block in code:
                    if block is not None:
                        merged_code.extend(block)
                return merged_code

            code_part_1 = list()
            code_part_1.append(self._create_file_information())
            code_part_1.append(self._create_imports(permutation_test=True, base_imports=True, further_imports=False))
            code_part_1.append(self.create_permutation_hyperpipe_function())

            code_part_1_final = ''.join(merge(code_part_1))

            code_part_2 = list()
            code_part_2.append(self._create_imports(permutation_test=True, base_imports=False, further_imports=True))
            code_part_2.append(self._create_persist_options())
            code_part_2.append(self._create_hyperpipe())
            code_part_2.append(self._create_neuro())
            code_part_2.append(self._create_transformers())
            code_part_2.append(self._create_estimator())
            code_part_2.append(self._create_fit(permutation_test=True))

            code_part_2_final = ''.join(merge(code_part_2))
            code_part_2_final = code_part_2_final.replace('\n', '\n    ')

            code_part_3 = list()
            code_part_3.append(self._create_data_loading())
            code_part_3.append(self.create_permutation_test_setup())
            code_part_3_final = ''.join(merge(code_part_3))

            merged_code = ''.join([code_part_1_final, code_part_2_final, code_part_3_final])
        return merged_code

    def _create_file_information(self):
        code = list()
        # code.append("\n# YOUR PROJECT'S LOCATION ")
        # code.append("\n# Project Folder: ")
        # code.append("\n# " + self.pipe_infos.photon_project_folder)
        code.append("\n# -------------------- GENERATED WITH PHOTON WIZARD (beta) ------------------------------")
        if self.pipe_infos.photon_project_folder:
            code.append("\n# PHOTON Project Folder: " + self.pipe_infos.photon_project_folder)
        code.append("\n")
        return code

    def _create_imports(self, permutation_test=False, base_imports=True, further_imports=True):
        code = list()

        if base_imports:
            code.append("""
import pandas as pd
import numpy as np""")
            if permutation_test:
                code.append("""                
from photonai.processing.permutation_test import PermutationTest""")

        if further_imports:
            code.append("""
from photonai.base import Hyperpipe, PipelineElement, OutputSettings, Switch, Preprocessing
from photonai.optimization import Categorical, IntegerRange, FloatRange""")
            if self.pipe_infos.brainatlas is not None:
                code.append("""
from photonai_neuro import NeuroBranch""")

            if permutation_test:
                code.append("\nimport numpy as np")

            cv_imports = ""
            if self.pipe_infos.outer_cv:
                imports_statement = "\n" + self._read_imports(self.pipe_infos.outer_cv).replace("\n", "")
                cv_imports += imports_statement
            if self.pipe_infos.inner_cv:
                imports_statement = "\n" + self._read_imports(self.pipe_infos.inner_cv).replace("\n", "")
                if imports_statement not in cv_imports:
                    cv_imports += imports_statement
            code.append("""
{}""".format(cv_imports.strip()))

            code.append("""
             """)

            # for key in self.pipe_infos._data._members:
            #     item = getattr(self.pipe_infos, key)
            #     if hasattr(item, 'imports'):
            #         if item.imports is not None and not isinstance(item, DataType):
            #             code.append(item.imports + '\n')
        return code

    @staticmethod
    def _read_imports(element_combi_id):
        if isinstance(element_combi_id, str):
            element_combi = ElementCombi.objects.get({'_id': element_combi_id})
        else:
            element_combi = element_combi_id
        referenced_obj = BaseElement.objects.get({'_id': element_combi.referenced_element_id})
        return referenced_obj.imports

    def _create_data_loading(self):
        code = list()
        if self.pipe_infos.data_file is not None:
            code.append("\n# Load data")
            code.append("""
df = pd.read_excel('{}')
X = np.asarray(df.iloc[:, {}])
y = np.asarray(df.iloc[:, {}])""".format(self.pipe_infos.data_file, self.pipe_infos.features, self.pipe_infos.targets))
            if self.pipe_infos.covariates and len(self.pipe_infos.covariates) > 0:
                code.append("""
covariates_var = np.asarray(df.iloc[:, {}])""".format(self.pipe_infos.covariates))
            if self.pipe_infos.groups and len(self.pipe_infos.groups) > 0:
                code.append("""
group_var = np.asarray(df.iloc[:, {}])""".format(self.pipe_infos.groups))
            code.append("\n")
        return code

    def _create_persist_options(self):
        # create logfile name
        code = list()
        code.append("\n# Specify how results are going to be saved")
        if self.udo_mode:
            code.append("""
output_settings = OutputSettings(mongodb_connect_url="{}",                                                                  
                             user_id="{}",
                             wizard_object_id="{}",
                             wizard_project_name="{}")
                    """.format(self.pipe_infos.mongodb_connect_url,
                               self.pipe_infos.user,
                               self.pipe_infos._id,
                               self.pipe_infos.project_name))
        return code

    def create_permutation_hyperpipe_function(self):
        return"""
        
def create_hyperpipe():        
"""

    def create_permutation_test_setup(self):
        code = """        
perm_tester = PermutationTest(create_hyperpipe, n_perms={}, n_processes=4, random_state=42,
                              permutation_id='{}')
perm_tester.fit(X, y""".format(self.pipe_infos.permutation_test.n_perms, self.pipe_infos.permutation_test.perm_id)
        code = self._append_kwargs(code, covariates=self.pipe_infos.covariates, groups=self.pipe_infos.groups)
        return code + ")"

    def _append_kwargs(self, code, covariates, groups):
        if groups and len(groups) > 0:
            code += ", groups=group_var"
        if covariates and len(covariates) > 0:
            code += ", confounder=covariates_var"
        return code

    def _create_hyperpipe(self):

        hyperpipe_string = """
hyperpipe = Hyperpipe('{}',""".format(self.pipe_infos.project_name)

        if self.pipe_infos.photon_project_folder:
            hyperpipe_string += """
                      project_folder = '{}',""".format(self.pipe_infos.photon_project_folder)
        else:
            hyperpipe_string += """
                      project_folder = './results',"""

        # define optimizer
        if self.pipe_infos.optimizer is not None:
            hyperpipe_string += """
                      optimizer="{}",""".format(self.pipe_infos.optimizer.name,)
            if self.pipe_infos.optimizer.hyperparameters is not None:
                hyperpipe_string += """
                      optimizer_params={},""".format(self._format_dict(self.pipe_infos.optimizer.hyperparameters,
                                                                           no_brackets=True))
        # get metrics
        metrics = list()
        if self.pipe_infos.metrics is not None:
            for obj_id in self.pipe_infos.metrics:
                metric = Metric.objects.raw({'_id': ObjectId(obj_id)}).first()
                metrics.append(metric.name)
        if len(metrics):
            hyperpipe_string += """
                      metrics={},""".format(metrics)
        best_config_metric = ''
        if self.pipe_infos.best_config_metric is not None:
            hyperpipe_string += """
                      best_config_metric="{}",""".format(self.pipe_infos.best_config_metric.name)

        if self.pipe_infos.outer_cv is not None:
            cv_args = list()
            for key, value in self.pipe_infos.outer_cv.hyperparameters.items():
                cv_args.append('{}={}'.format(key, value))
            merged_cv_args = ','.join(cv_args)
            hyperpipe_string += """
                      outer_cv = {}({}),""".format(self.pipe_infos.outer_cv.name, merged_cv_args)

        else:
            hyperpipe_string += """
                      use_test_set=False,"""

        # define inner cross-validation
        if self.pipe_infos.inner_cv is not None:
            cv_args = list()
            for key, value in self.pipe_infos.inner_cv.hyperparameters.items():
                cv_args.append('{}={}'.format(key, value))
            merged_cv_args = ', '.join(cv_args)
            hyperpipe_string += """
                      inner_cv = {}({}),""".format(self.pipe_infos.inner_cv.name, merged_cv_args)

        if self.pipe_infos.data_type:
            if self.pipe_infos.data_type.name == "Nifti Data":
                hyperpipe_string += """
                      verbosity=2,"""

                if self.pipe_infos.photon_project_folder is not None:
                    cache_folder = os.path.join(self.pipe_infos.photon_project_folder, "cache")
                    cache = '"' + str(cache_folder) + '"'
                    hyperpipe_string += """
                      cache_folder={},""".format(cache)
        if self.udo_mode:
            hyperpipe_string += """
                      output_settings=output_settings"""

        if hyperpipe_string[-1] == ",":
            hyperpipe_string = hyperpipe_string[:-1]
        hyperpipe_string += """)
        """

        code = list()
        code.append("\n# Define hyperpipe")
        code.append(hyperpipe_string)
        return code

    def _create_transformers(self):
        code = list()
        pre_processing_handled = False
        if hasattr(self.pipe_infos, 'transformer_elements'):
            transformer_dicts = self.pipe_infos.transformer_elements
            if transformer_dicts:
                code.append("\n# Add transformer elements")
                # put transformers in correct order
                transformer_index = [int(j["position"]) for i, j in transformer_dicts.items()]
                transformer_list = np.array([j for i, j in transformer_dicts.items()])
                order_index = np.argsort(transformer_index)
                sorted_transformers = transformer_list[order_index]

                for trans_item in sorted_transformers:
                    # 1. load transformer
                    transformer = Transformer.objects.raw({'_id': trans_item["ObjectId"]}).first()
                    name = transformer.name
                    test_disabled = trans_item["test_disabled"]
                    hyperparams, default_params = self.split_default_params_and_hyperparameters(trans_item["hyperparameters"])
                    hyperparam_str = self._format_dict(hyperparams)
                    default_param_str = self.format_default_params(default_params)
                    if not transformer.pre_processing:
                        code.append("""
hyperpipe += PipelineElement("{}", hyperparameters={}, 
                             test_disabled={}{})""".format(name, hyperparam_str, test_disabled, default_param_str))
                    else:
                        if not pre_processing_handled:
                            code.append("""
preprocessing_pipe = Preprocessing()
hyperpipe += preprocessing_pipe""")
                        pre_processing_handled = True
                        code.append("""
preprocessing_pipe += PipelineElement("{}") 
                        """.format(name))
        return code

    def add_pipeline_elements_to_code(self, obj_list, element_to_add_to, additional_defaults: str = ""):
        code = list()
        for obj_id in obj_list:
            element_combi = ElementCombi.objects.get({'_id': ObjectId(obj_id)})
            base_element_ref = BaseElement.objects.get({'_id': element_combi.referenced_element_id})
            name = base_element_ref.name
            hyperparams, default_params = self.split_default_params_and_hyperparameters(element_combi.hyperparameters)
            hyperparam_str = self._format_dict(hyperparams)
            default_param_str = self.format_default_params(default_params)
            code.append("""
{} += PipelineElement("{}", hyperparameters={}{}{})""".format(element_to_add_to, name, hyperparam_str, default_param_str,
                                                              additional_defaults))
        return ''.join(code)

    def _create_estimator(self):
        code = list()
        if self.pipe_infos.estimator_element_list is not None:
            if len(self.pipe_infos.estimator_element_list) > 1:
                code.append("\n# Add estimator")
                code.append("""
estimator_switch = Switch('EstimatorSwitch')""")
                code.append(self.add_pipeline_elements_to_code(self.pipe_infos.estimator_element_list, 'estimator_switch'))
                code.append("""
hyperpipe += estimator_switch                
""")
            else:
                code.append(self.add_pipeline_elements_to_code(self.pipe_infos.estimator_element_list, 'hyperpipe'))
                code.append("""
""")

        return code

    def _create_fit(self, permutation_test=False):
        code = list()
        if not permutation_test:
            if self.pipe_infos.data_file is not None:
                code.append("""
# Fit hyperpipe
hyperpipe.fit(X, y""")
                code = self._append_kwargs(code, covariates=self.pipe_infos.covariates, groups=self.pipe_infos.groups)
                code += ")"
        else:
            code.append("""
return hyperpipe                
                        """)
        return code

    def _create_neuro(self):
        code = list()
        if self.pipe_infos.brainatlas is not None:
            code.append("\n# Add neuro elements")
            atlas = self.pipe_infos.brainatlas
            code.append("""
neuro_branch = NeuroBranch('Neuro', nr_of_processes=3)""")
            if self.pipe_infos.neuro_transformer is not None:
                code.append(self.add_pipeline_elements_to_code(self.pipe_infos.neuro_transformer, 'neuro_branch',
                                                               ", batch_size=50"))

            atlas_object = BrainAtlas.objects.get({'_id': ObjectId(self.pipe_infos.brainatlas.referenced_element_id)})
            if atlas_object.name == "CustomMask":
                if isinstance(self.pipe_infos.brainatlas.hyperparameters['mask_image'], list):
                    mask_image = self.pipe_infos.brainatlas.hyperparameters['mask_image'][0]
                else:
                    mask_image = self.pipe_infos.brainatlas.hyperparameters['mask_image']
                code.append("""
neuro_branch += PipelineElement({}, hyperparameters={{}}, mask_image='{}', extract_mode='vec', batch_size=50)""".format(
                    "'BrainMask'",
                    mask_image))
            elif atlas_object.short_description == 'Mask':
                code.append("""
neuro_branch += PipelineElement({}, hyperparameters={{}}, mask_image='{}',
                                extract_mode='vec', batch_size=50)""".format("'BrainMask'", atlas_object.name))
            else:
                code.append("""
neuro_branch += PipelineElement({}, hyperparameters={{}}, atlas_name='{}',
                                rois={}, extract_mode='vec', batch_size=50)""".format("'BrainAtlas'",
                atlas_object.name, self.pipe_infos.brainatlas.hyperparameters['roi_names']))

            # add everything to the neuro branch
            code.append("""
hyperpipe += neuro_branch
            """)
        return code

    def split_default_params_and_hyperparameters(self, dictionary):

        no_default_params = {}
        default_params = {}

        for name, value in dictionary.items():
            if isinstance(value, list):
                if len(value) < 2:
                    # we have only one parameter, so we do not need to optimize
                    default_params[name] = value[0]
                else:
                    no_default_params[name] = value
            elif isinstance(value, str) and (value.startswith("IntegerRange") or value.startswith("FloatRange")):
                no_default_params[name] = value
            else:
                default_params[name] = value

        return no_default_params, default_params

    def format_default_params(self, dictionary):
        output_str = ""
        for name, element in dictionary.items():
            output_str += ", "
            output_str += str(name) + "="
            if isinstance(element, str):
                white_list = {"np.nan": "np.nan"}
                if element in white_list:
                    output_str += element
                else:
                    output_str += "'" + element + "'"
            else:
                 output_str += str(element)
        return output_str

    def format_string_to_code(self, write_to_dict_value, no_brackets:bool =False):
        white_list = {"np.nan": "np.nan"}
        string_output = ""
        # Write IntegerRange and FloatRange directly
        if isinstance(write_to_dict_value, str) and (
                write_to_dict_value.startswith("IntegerRange") or write_to_dict_value.startswith("FloatRange")):
            string_output += write_to_dict_value
        else:
            # write non-native python e.g. np.nan directly to code without quotes ''
            if isinstance(write_to_dict_value, str) and write_to_dict_value in white_list.keys():
                if no_brackets:
                    string_output += white_list[write_to_dict_value]
                else:
                    string_output += '[' + white_list[write_to_dict_value] + ']'
            else:

                try:
                    # try to interpret string as native python code,
                    # e.g. to catch a list of integers that is written as a string '[0, 1, 2, 3, 4]'
                    if isinstance(write_to_dict_value, str):
                        if no_brackets:
                            string_output += str(ast.literal_eval(write_to_dict_value))
                        else:
                            string_output += str(ast.literal_eval([write_to_dict_value]))
                    else:
                        # if we have a non-string, then it must be a native python type such as list,
                        # integer or float or None, so we casted the correct type already:
                        # then we can simply output it as string
                        if no_brackets:
                            string_output += str(write_to_dict_value)
                        else:
                            # if we have a list we dont need to add brackets anyway
                            if not isinstance(write_to_dict_value, list):
                                string_output += "[" + str(write_to_dict_value) + "]"
                            else:
                                string_output += str(write_to_dict_value)
                except ValueError as e:
                    string_output += '"' + write_to_dict_value + '"'
        return string_output

    @staticmethod
    def create_warnings(file_path):
        # curr_path = os.path.abspath(os.path.dirname(__file__))
        # filepath = os.path.join(curr_path, 'test_code.py')
        # cmd = 'pycodestyle --show-source --show-pep8 --ignore=E501 ' + os.path.join(curr_path, 'test_code.py')
        # output = self.__run_command(cmd)
        output = ''
        if file_path is not None:
            cmd = 'python -m py_compile ' + file_path
            output = PhotonCodeCreator._check_for_exception(cmd)
        return output

    def _format_dict(self, old_dict, no_brackets:bool = False):

        numer_of_params = 0
        string_output = "{"

        for key, value in old_dict.items():
            try:
                if numer_of_params > 0:
                    string_output += ", "
                string_output += "'" + key + "': "
                # if isinstance(value, list):
                #     write_to_dict_value = ', '.join('"{0}"'.format(w) for w in value)
                # else:
                #     write_to_dict_value = value

                string_output += self.format_string_to_code(value, no_brackets)
                numer_of_params += 1
            except builtins.SyntaxError as e:
                string_output += str(e)

        string_output += "}"
        return string_output

    @staticmethod
    def _check_for_exception(shell_command):
        try:
            p = subprocess.Popen(shlex.split(shell_command), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, error = p.communicate()
            p.terminate()
            if p.returncode != 0:
                msg_error = error.decode("ascii")
                prettified_error = """
    ''' 
    ! -------------- SYNTAX ERROR -----------------!
    {}
    ! -------------- SYNTAX ERROR -----------------!
    '''
                                   """.format(msg_error)
                return prettified_error
            else:
                return ""
        except OSError as e:
            # 12 = cannot allocate memory
            # -> ignore because syntax check is not that important.
            if e.errno == 12:
                return ""
            else:
                return e

    def __run_command(self, shell_command):
        output_list = []
        process = subprocess.Popen(shlex.split(shell_command), stdout=subprocess.PIPE)
        while True:
            output = process.stdout.readline()
            if output == b'' and process.poll() is not None:
                break
            if output:
                output_list.append("""#  {} \n""".format(output.strip().decode("ascii")))
        # rc = process.poll()
        process.terminate()
        return output_list


