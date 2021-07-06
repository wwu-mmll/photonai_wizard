

from ..controller.pipeline_helper import *
from ..model.elements import *
from ..model.code_creator import PhotonCodeCreator
from ..main import application, excelfiles


def udo_mode():
    return application.config['UDO_MODE']


class BasicStep:

    def setup(self):
        pass

    @staticmethod
    def generate_code(form_input):
        current_pipe = load_current_pipe()
        p_code = PhotonCodeCreator(udo_mode()).create_code(current_pipe)
        changed_nums = list()
        if form_input:
            if "photon_code" in form_input.form:
                old_code = form_input.form["photon_code"]
                old_code = old_code.replace('\r', '').split('\n')
                p_code_splitted = p_code.split('\n')
                diff = set(p_code_splitted).difference(set(old_code))
                for i in diff:
                    index = p_code_splitted.index(i)
                    changed_nums.append(index + 1)
        return p_code, changed_nums

    def process_input(self, form_input, selection_items):
        get_form_infos(form_input.form, selection_items)


class DataStep(BasicStep):

    def setup(self):
        # load pipe
        current_pipe = load_current_pipe()
        default_path = os.path.join(current_pipe.photon_project_folder, "features.xlsx")

        selection_items = list()
        if not udo_mode():
            selection_items.append(InputBox('Data Sheet', 'Provide an Excel sheet containing all the data',
                                            'data', 'data_file', default_value=default_path, is_required=False))

        selection_items.append(InputBox('Targets', 'Specify the column that contains the targets (zero-based, '
                                                   'the first column is the 0th column, the second the 1st, and so on...)',
                                        'targets', 'targets', default_value='0'))

        if current_pipe.data_type:
            if current_pipe.data_type.name == 'Nifti Data':
                selection_items.append(InputBox('Nifti Images',
                                                'Specify the columns that contain the nifti image filenames (absolute paths)',
                                                'features', 'features', default_value='1'))
            else:
                selection_items.append(
                    InputBox('Features', 'Specify the columns that contain the features (zero-based)',
                             'features', 'features', default_value='4:27'))

        # if groups is not set don't show group cv options

        all_cv_items = CV.objects.all()
        cv_items_group = [i._id for i in all_cv_items if i.name in ["GroupKFold",
                                                                    "GroupShuffleSplit",
                                                                    "LeaveOneGroupOut"]]

        if (current_pipe.outer_cv is not None and current_pipe.outer_cv.referenced_element_id in cv_items_group) or \
                (current_pipe.inner_cv is not None and current_pipe.inner_cv.referenced_element_id in cv_items_group):
            selection_items.append(InputBox('Groups', 'Specify the column that contains the group label (zero-based)',
                                            'groups', 'groups', default_value='', is_required=True))
        else:
            current_pipe.groups = None
            current_pipe.save()

        try:
            confounder_removal_found = False
            confounder_removal = Transformer.objects.get({"name": "ConfounderRemoval"})
            # search for confounder removal transformer:
            for _, transformer_dict in current_pipe.transformer_elements.items():
                if transformer_dict["ObjectId"] == confounder_removal._id:
                    selection_items.append(
                        InputBox('Covariates', 'Specify the columns that contain the covariates (zero-based)',
                                 'covariates', 'covariates', default_value='', is_required=True))
                    confounder_removal_found = True
                    break
            if not confounder_removal_found:
                current_pipe.covariates = None
                current_pipe.save()
        except DoesNotExist as e:
            pass
        return selection_items

    def process_input(self, form_input, selection_items):
        get_form_infos(form_input.form, selection_items)
        current_pipe = load_current_pipe()
        if udo_mode():
            try:
                if 'excel_file' not in form_input.files:
                    if current_pipe.data_file is None:
                        raise FileNotFoundError('Could not find a filename')
                else:
                    if not form_input.files['excel_file'].filename:
                        if not current_pipe.data_file:
                            raise Exception
                    else:
                        data_file = os.path.join(current_pipe.photon_project_folder,
                                                 form_input.files['excel_file'].filename)
                        if os.path.exists(data_file):
                            os.remove(data_file)

                        excelfiles.save(form_input.files['excel_file'], name=data_file)
                        current_pipe = load_current_pipe()
                        current_pipe.data_file = data_file
                        current_pipe.save()

            except Exception:
                fail_message = "Please provide an Excel file."
                raise Exception(fail_message)


class ProjectStep(BasicStep):

    def setup(self):
        selection_items = [InputBox('Analysis Name', 'Specify a name for this analysis.',
                                    'analysis_name', "name", full_width=True, default_value=''),
                           InputBox('Project Description', 'Please provide a short description of your project.',
                                    'project_description', 'description',
                                    default_value='', full_width=True)]

        if not udo_mode():
            selection_items.append(InputBox('Project Folder', 'Please give a local directory for your photon analysis',
                                            'photon_project_folder', 'photon_project_folder',
                                            default_value='/home/user1/project2'))
        return selection_items

    def process_input(self, form_input, selection_items):
        super().process_input(form_input, selection_items)
        pipe_obj = load_current_pipe()
        # save project name
        project_name = re.sub('[^0-9a-zA-Z]+', '', pipe_obj.name)
        pipe_obj.project_name = project_name.lower()
        pipe_obj.save()


class EstimatorStep(BasicStep):

    def setup(self):
        current_pipe = load_current_pipe()
        estimators = Estimator.objects.raw({'tags': {'$in': current_pipe.constraint_dict["analysis_type"]}})

        selection_items = [SelectionBox('Estimator', "Choose your estimator", estimators, False, 'estimator',
                                        'estimator_element_list', Estimator,
                                        has_parameters=True, find_many=True, has_test_disabled=False)]
        return selection_items

    def process_input(self, form_input, selection_items):
        get_form_infos(form_input.form, selection_items, save_element_combi_as_list=True)

        current_pipe = load_current_pipe()
        # if we are here, all information should be there in order to run the script
        current_pipe.collected_all_information = True
        current_pipe.save()


class TransformerStep(BasicStep):

    def setup(self):
        return list()

    def process_input(self, form_input, selection_items):
        current_pipe = load_current_pipe()
        transformer_pipe = {}

        for item_key, item_value in form_input.form.items():
            if re.match(r"transformer_[0-9]$", item_key):
                # we found an transformer
                transformer_pipe[item_key] = {}
                transformer_pipe[item_key]["ObjectId"] = ObjectId(item_value)
                transformer_pipe[item_key]["hyperparameters"] = {}
                prefix_should_be = item_key + "_" + item_value + "_"

                if not item_key + "_" + item_value + "_" + "test_disabled" in form_input.form:
                    transformer_pipe[item_key]["test_disabled"] = False

                for param_key, param_value in form_input.form.items():
                    if not item_key == param_key:
                        if param_key == "position_" + item_key:
                            transformer_pipe[item_key]["position"] = param_value
                        if param_key == item_key + "_is_active":
                            transformer_pipe[item_key]["is_active"] = param_value
                        if param_key == item_key + "_" + item_value + "_test_disabled":
                            get_value = form_input.form.getlist(param_key)[0]
                            if get_value == "on":
                                transformer_pipe[item_key]["test_disabled"] = True
                            else:
                                transformer_pipe[item_key]["test_disabled"] = False
                        elif param_key.startswith(prefix_should_be):
                            stripped_key = param_key[14::]
                            stripped_obj_id = stripped_key[25::]
                            persist_value = get_info_from_string(param_value)

                            if persist_value == '':
                                loaded_object = BaseElement.objects.raw({'_id': ObjectId(item_value)}).first()
                                persist_value = get_default_parameter_for_empty_string(loaded_object,
                                                                                       [],
                                                                                       stripped_obj_id)
                            transformer_pipe[item_key]["hyperparameters"][stripped_obj_id] = persist_value
                debug = True

        current_pipe.transformer_elements = transformer_pipe
        current_pipe.save()


class NeuroTransformerStep(BasicStep):

    def setup(self):
        selection_items = [SelectionBox('Neuro Transformer', "You may resample or smooth your images",
                                        NeuroTransformer.objects.all(), False, 'neuro_transformer',
                                        'neuro_transformer', NeuroTransformer, has_parameters=True, is_required=False,
                                        find_many=True),
                           SelectionBox('Brain Atlas', 'Pick a brain atlas', BrainAtlas.objects.all(), True,
                                        'brainatlas',
                                        'brainatlas', BrainAtlas, has_parameters=True)]

        return selection_items

    def process_input(self, form_input, selection_items):
        get_form_infos(form_input.form, selection_items, save_element_combi_as_list=True)


class PerformanceStep(BasicStep):

    def setup(self):
        current_pipe = load_current_pipe()

        metrics = Metric.objects.raw({'tags': {'$in': current_pipe.constraint_dict["analysis_type"]}})

        selection_items = [SelectionBox('Metrics', "Choose all performance metrics you want to calculate.",
                                        metrics, False, 'pipe_metrics', 'metrics', Metric, find_many=True),
                           SelectionBox('Best Config Metric',
                                        "Choose the performance metric that is minimized or maximizied in order to choose the best config.",
                                        metrics, True, 'best_config_metric', 'best_config_metric', Metric)]
        return selection_items

class OptimizationStep(BasicStep):

    def setup(self):
        optimizers = Optimizer.objects.all()
        curr_pipe = load_current_pipe()
        cv_items = CV.objects.raw({'tags': {'$in': curr_pipe.constraint_dict["data_quantity"]}})
        # cv_items = CV.objects.raw({'tags': {'$in': ['#500']}})

        constraint_list = curr_pipe.constraint_dict["data_quantity"]
        constraint_list_inner = list(constraint_list)
        constraint_list_inner.append('#inner_cv')
        constraint_list_outer = list(constraint_list)
        constraint_list_outer.append('#outer_cv')

        selection_items = [
            SelectionBox('Optimizer',
                         "Please specify the hyperparameter optimizer that will use to find the best hyperparameter configuration.",
                         optimizers, True, 'optimizer', 'optimizer', Optimizer, has_parameters=True),
            SelectionBox('Inner Cross Validation', "Choose your strategy for testing configurations.",
                         cv_items, True, 'inner_cv', 'inner_cv', CV, has_parameters=True,
                         constraint_list=constraint_list_inner),
            SelectionBox('Outer Cross Validation', "Choose your strategy for running the hyperparameter search.",
                         cv_items, True, 'outer_cv', 'outer_cv', CV, has_parameters=True,
                         constraint_list=constraint_list_outer, is_required=False)]
        return selection_items


class SetupStep(BasicStep):

    def setup(self):
        dts = DataType.objects.all()
        dqs = DataQuantity.objects.all()
        ats = AnalysisType.objects.all()

        selection_items = [SelectionBox('Data Type', "Choose the type of data you want to analyze.",
                                        dts, True, 'data_type', 'data_type', DataType),
                           SelectionBox('Data Quantity', "Specify the number of samples available for this analysis.",
                                        dqs, True, 'data_quantity', 'data_quantity', DataQuantity),
                           SelectionBox('Analysis Type', "Specify the type of your analysis.",
                                        ats, True, 'analysis_type', 'analysis_type', AnalysisType)]

        return selection_items

    def process_input(self, form_input, selection_items):
        # load current pipe
        pipe_exists = check_if_pipe_exists()

        if not pipe_exists:
            new_pipeline_object()

        # write information from form to database
        get_form_infos(form_input.form, selection_items)

        # reload with information inserted
        pipe_obj = load_current_pipe()

        if pipe_obj.constraint_dict is None:
            pipe_obj.constraint_dict = {}
        pipe_obj.constraint_dict.update(set_basic_constraints(pipe_obj))

        if pipe_obj.data_type is not None:
            if pipe_obj.data_type.name != "Nifti Data":
                # Reset neuro transformers if we don't have nifti data anymore
                if pipe_obj.neuro_transformer:
                    pipe_obj.neuro_transformer = None
                if pipe_obj.brainatlas:
                    pipe_obj.brainatlas = None

        pipe_obj.save()


class StepManager:
    STEP_DICT = {'optimization': OptimizationStep,
                 'setup': SetupStep,
                 'performance': PerformanceStep,
                 'transformers': TransformerStep,
                 'neuro_transformers': NeuroTransformerStep,
                 'estimator': EstimatorStep,
                 'project': ProjectStep,
                 'data': DataStep}

    def __init__(self, name: str):
        self.name = str
        self.step_object = self.STEP_DICT[name]()

    def setup(self):
        return self.step_object.setup()

    def process_input(self, form_input, selection_items):
        self.step_object.process_input(form_input, selection_items)

    @staticmethod
    def generate_code(form_input=None):
        return BasicStep.generate_code(form_input)
