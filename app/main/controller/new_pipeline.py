import uuid
import json

from flask import render_template, request, redirect, url_for, jsonify
from flask_login import login_required
from ..main import excelfiles
from flask_wtf import FlaskForm
from wtforms import FileField
from flask_wtf.file import FileAllowed
from pymongo import ASCENDING

from ..model.code_creator import PhotonCodeCreator
from ..controller.pipeline_helper import *
from app.main.model.fill_db.default_pipelines import DefaultPipelineRepo
from ..model.step_manager import StepManager


def wizard_step_kwargs_generator(step_nr: int, selection_items: list = []):

    # exception for neuro_transformers which share the position with transformers
    if step_nr == 7:
        tmp_step = 3
    else:
        tmp_step = step_nr
    active_list = ["active" if i <= tmp_step + 1 else "" for i in range(8)]

    line_numbers = ""
    if check_if_pipe_exists():
        current_pipe = load_current_pipe()
        p_code, changed_nums = StepManager.generate_code(request)
        line_numbers = ",".join([str(n) for n in changed_nums])
        if current_pipe.name:
            pipe_name = current_pipe.name
        else:
            pipe_name = "Design your pipeline"
    else:
        p_code = ''
        pipe_name = "Design your pipeline"

    step_names = ["setup", "optimization", "performance", "transformers", "estimator", "project", "data",
                  "neuro_transformers"]

    return {'photon_code': p_code,
            'pipe_name': pipe_name,
            'active_steps': active_list,
            'selection_items': selection_items,
            'line_numbers': line_numbers,
            'step_name': step_names[step_nr]}


@application.route('/general_infos', methods=["GET", "POST"])
@login_required
def general_infos():

    step_man = StepManager('setup')
    selection_items = step_man.setup()

    if request.method == 'POST':

        step_man.process_input(request, selection_items)
        if request.form["action"] == "previous":
            return redirect(url_for('project_history'))

        if request.form["action"] == "next":
            return redirect(url_for('optimization'))
    else:
        # load current pipe
        pipe_exists = check_if_pipe_exists()
        if pipe_exists:
            load_pipe_presets(selection_items)

    return render_template("pipeline_steps/general_infos.html",
                           **wizard_step_kwargs_generator(0, selection_items))


@application.route('/default_pipes/', methods=["GET", "POST"])
@login_required
def default_pipes():
    if request.method == 'POST':
        if request.form["action"] == "previous":
            return redirect(url_for('general_infos'))

    current_pipe = load_current_pipe_from_db()

    basic_header = 'Basic Default Pipelines'
    advanced_header = 'Advanced Default Pipelines'
    header_dict = {'Basic': basic_header, 'Optimized': advanced_header}
    default_pipe_dict = {basic_header: [], advanced_header: []}
    list_of_pipes = DefaultPipeline.objects.raw({'data_type': current_pipe.data_type._id,
                                                 'analysis_type': current_pipe.analysis_type._id}).order_by([('display_order', ASCENDING)])
    for p in list_of_pipes:
        default_pipe_dict[header_dict[p.complexity]].append(p)

    return render_template('pipeline_steps/default_pipes.html', default_pipe_dict=default_pipe_dict,
                           **wizard_step_kwargs_generator(1))


@application.route('/choose_default_pipe/<new>/<pipe_id>')
@login_required
def choose_default_pipe(new, pipe_id):
    if bool(new):
        current_pipe = new_pipeline_object()
    else:
        if check_if_pipe_exists():
            current_pipe = load_current_pipe()
        else:
            current_pipe = new_pipeline_object()

    default_pipe = DefaultPipeline.objects.get({'_id': ObjectId(pipe_id)})
    current_pipe.data_type = default_pipe.data_type
    current_pipe.analysis_type = default_pipe.analysis_type
    if current_pipe.constraint_dict is None:
        current_pipe.constraint_dict = dict()
    call_obj = getattr(DefaultPipelineRepo, default_pipe.method_to_call)
    updated_pipe = call_obj(current_pipe)
    current_pipe.constraint_dict.update(set_basic_constraints(current_pipe))
    updated_pipe.collected_all_information = True
    updated_pipe.save()
    # write_code_to_file(updated_pipe)
    session['pipe_id'] = str(updated_pipe._id)
    return redirect(url_for('project'))


@application.route('/default_pipeline_list', methods=['GET', 'POST'])
@login_required
def default_pipeline_list():
    default_pipe = DefaultPipeline.objects.all()
    return render_template('default_pipelines_overview.html', default_pipe_list=default_pipe)


@application.route('/update_code/<step_name>', methods=["POST"])
@login_required
def update_code(step_name):

    step_manager = StepManager(step_name)
    selection_items = step_manager.setup()
    step_manager.process_input(request, selection_items)
    p_code, changed_nums = step_manager.generate_code(request)

    answer = jsonify({'line_numbers':  ",".join([str(n) for n in changed_nums]),
                      'photon_code': p_code})
    return answer


@application.route('/optimization', methods=["GET", "POST"])
@login_required
def optimization():

    step_man = StepManager('optimization')
    selection_items = step_man.setup()

    load_pipe_presets(selection_items)

    if request.method == 'POST':

        step_man.process_input(request, selection_items)

        if request.form["action"] == "next":
            return redirect(url_for('metrics'))
        if request.form["action"] == "previous":
            return redirect(url_for('general_infos'))

    return render_template("pipeline_steps/optimization.html",
                           **wizard_step_kwargs_generator(1, selection_items))


@application.route('/metrics', methods=["GET", "POST"])
@login_required
def metrics():
    step_man = StepManager('performance')
    selection_items = step_man.setup()

    if request.method == 'POST':
        step_man.process_input(request, selection_items)

        current_pipe = load_current_pipe()
        if request.form["action"] == "next":
            if current_pipe.data_type.name == 'Nifti Data':
                return redirect(url_for('neuro_elements'))
            else:
                return redirect('transformers')
        if request.form["action"] == "previous":
            return redirect(url_for('optimization'))
    else:
        load_pipe_presets(selection_items)

    return render_template("pipeline_steps/metrics.html",
                           **wizard_step_kwargs_generator(2, selection_items))


@application.route('/neuro_elements', methods=["GET", "POST"])
@login_required
def neuro_elements():

    step_man = StepManager('neuro_transformers')
    selection_items = step_man.setup()
    if request.method == 'POST':
        step_man.process_input(request, selection_items)
        if request.form["action"] == "previous":
            return redirect(url_for('metrics'))
        else:
            return redirect(url_for('transformers'))
    else:
        load_pipe_presets(selection_items)

    return render_template("load_data/neuro_data.html",
                           **wizard_step_kwargs_generator(7, selection_items))


@application.route('/transformers', methods=["GET", "POST"])
@login_required
def transformers():

    selection_items = []

    current_pipe = load_current_pipe()
    trnsfrms = Transformer.objects.raw({'tags': {'$in': current_pipe.constraint_dict["analysis_type"]}})

    if request.method == "GET":

        # load existing transformer elements
        if current_pipe.transformer_elements:
            nr_of_already_existing_transformers = len(current_pipe.transformer_elements)
            for transformer_key, transformer_value in current_pipe.transformer_elements.items():#

                selection_box = SelectionBox('Transformer', "Choose your transformer", list(trnsfrms), True, transformer_key,
                                             'transformer_elements_' + transformer_key[-1::], Transformer,
                                             has_parameters=True,
                                             is_required=False,
                                             default_value=transformer_value["ObjectId"],
                                             is_active=transformer_value["is_active"],
                                             position=transformer_value["position"],
                                             test_disabled=transformer_value["test_disabled"],
                                             has_test_disabled=True)

                # load hyperparameters
                selection_box.hyperparameter_dict = {}
                for hyp_item_key, hyp_item_value in transformer_value["hyperparameters"].items():
                    hyp_key = selection_box.fieldset_id + "_" + str(transformer_value["ObjectId"]) + "_" + str(hyp_item_key)
                    selection_box.hyperparameter_dict[hyp_key] = hyp_item_value

                selection_items.append(selection_box)
        else:
            nr_of_already_existing_transformers = 0

        # fill up transformers up to 10 elements
        for i in range(nr_of_already_existing_transformers, 10):
            active = 'false'
            # preselect one first transformer item
            if i == 0:
                active = 'true'
            selection_items.append(SelectionBox('Transformer', "Choose your transformer", list(trnsfrms), True, 'transformer_' + str(i),
                                                'transformer_elements_' + str(i), Transformer,
                                                has_parameters=True,
                                                is_required=False,
                                                is_active=active,
                                                has_test_disabled=True,
                                                position=i))

        # Todo: Preselect StandardScaler for the first item , default_value = ObjID of STandardScaler
        # Todo: Preselect PCA for the second item , default_value = ObjID of PCA

        return render_template("pipeline_steps/transformer.html",
                               **wizard_step_kwargs_generator(3, selection_items))

    if request.method == 'POST':

        step_man = StepManager('transformers')
        step_man.process_input(request, None)

        if request.form["action"] == "next":
            return redirect(url_for('estimator'))
        if request.form["action"] == "previous":
            if current_pipe.data_type.name == 'Nifti Data':
                return redirect(url_for('neuro_elements'))
            else:
                return redirect(url_for('metrics'))


@application.route('/estimator', methods=["GET", "POST"])
@login_required
def estimator():

    step_man = StepManager('estimator')
    selection_items = step_man.setup()

    if request.method == 'POST':
        step_man.process_input(request, selection_items)

        if request.form["action"] == "next":
            return redirect(url_for('project'))
        if request.form["action"] == "previous":
            return redirect(url_for('transformers'))
    else:
        load_pipe_presets(selection_items)

    return render_template("pipeline_steps/estimator.html",
                           **wizard_step_kwargs_generator(4, selection_items))


def write_code_to_file(current_pipe):
    if udo_mode():
        filename = "photon_code.py"
        current_pipe.photon_file_path = os.path.join(current_pipe.photon_project_folder, filename)
        current_pipe.save()
        text_file = open(current_pipe.photon_file_path, "w")
    else:
        text_file_name = os.path.join(application.config['TMP_PHOTON_SCRIPT_FOLDER'], str(uuid.uuid4()) + ".py")
        current_pipe.photon_file_path = text_file_name
        current_pipe.save()
        text_file = open(text_file_name, 'w')

    # save code
    photon_code = PhotonCodeCreator(udo_mode=udo_mode()).create_code(current_pipe, final=True)
    text_file.write(photon_code)
    text_file.close()


@application.route('/project', methods=['GET', 'POST'])
@login_required
def project():

    step_man = StepManager('project')
    selection_items = step_man.setup()
    success_message = ""

    if request.method == "GET":
        load_pipe_presets(selection_items)
    else:
        step_man.process_input(request, selection_items)

        if udo_mode():
            pipe_obj = load_current_pipe()

            project_name = pipe_obj.project_name
            # check if project name already exists for this user
            pipes_with_the_same_name = Pipeline.objects.raw({'user': current_user.username,
                                                             'project_name': project_name}).count()

            # if project name is already set for user, add datetime stamp
            if pipes_with_the_same_name > 1:
                # another pipe already exists with this name
                new_project_name = pipe_obj.name + " " + str(datetime.datetime.now())[0:-7]
                pipe_obj.name = new_project_name
                pipe_obj.save()
                success_message = "<b>You have an already existing photon project with the same name.</b> <br>" \
                                  "In order to prevent data loss your project ist renamed to " + new_project_name

            user_folder = os.path.join("/spm-data/Scratch/photon_wizard", current_user.username)
            project_name = re.sub('[^0-9a-zA-Z]+', '', pipe_obj.name)
            project_name = project_name.lower()

            project_folder = os.path.join(user_folder, project_name)
            if not os.path.isdir(project_folder):
                oldmask = os.umask(000)
                os.makedirs(project_folder, 0o777)
                os.umask(oldmask)
            pipe_obj.photon_project_folder = project_folder

            pipe_obj.save()

        if success_message == "":
            if request.form["action"] == "previous":
                return redirect(url_for('estimator'))
            else:
                return redirect(url_for('load_data'))

    pipe_obj = load_current_pipe()

    return render_template("pipeline_steps/pipe_overview.html",
                           success=success_message,
                           pipe=pipe_obj,
                           **wizard_step_kwargs_generator(5, selection_items))


@application.route('/syntax_check', methods=['GET', 'POST'])
@login_required
def syntax_check():
    current_pipe = load_current_pipe()

    warnings = PhotonCodeCreator.create_warnings(current_pipe.photon_file_path)
    text_file = open(current_pipe.photon_file_path, "r")
    photon_code = text_file.read()
    if warnings:
        photon_code = ''.join([warnings, photon_code])

    if request.method == 'POST':
        if "action" in request.form:
            if request.form["action"] == "previous":
                return redirect(url_for('estimator'))
        else:
            # save new photon syntax
            new_photon_code = request.form["photon_syntax"]
            text_file = open(current_pipe.photon_file_path, "w")
            text_file.write(new_photon_code)
            text_file.close()
            warnings = PhotonCodeCreator.create_warnings(current_pipe.photon_file_path)
            if warnings:
                return redirect(url_for("syntax_check"))
            if udo_mode():
                return redirect(url_for("run_photon", pipe_id=current_pipe._id))
            else:
                return redirect(url_for('download_script', pipe_id=current_pipe._id))

    return render_template("pipeline_steps/final_syntax.html",
                           pipe_id=current_pipe._id,
                           photon_code=photon_code)


# !------------------- DATA -----------------------------------------------------------------------------------
class DataUploadForm(FlaskForm):
    excel_file = FileField("Your Excel file*", validators=[FileAllowed(excelfiles, 'Only Excel Allowed!')])


@application.route('/load_data', methods=["GET", "POST"])
@login_required
def load_data():
    form = DataUploadForm()

    step_man = StepManager('data')
    selection_items = step_man.setup()

    fail_message = ""
    if request.method == 'POST':
        try:
            step_man.process_input(request, selection_items)
        except Exception as e:
            fail_message = str(e)
        write_code_to_file(load_current_pipe())

        if fail_message == "":
            if request.form["action"] == "previous":
                return redirect(url_for('project'))
            else:
                if udo_mode():
                    return redirect(url_for('validate_data'))
                else:
                    return redirect(url_for('syntax_check'))
    else:
        current_pipe = load_current_pipe()
        load_pipe_presets(selection_items)
        form.excel_file.label = current_pipe.data_file
    return render_template("pipeline_steps/data.html",
                           form=form,
                           udo_mode=udo_mode(),
                           enable_multipart_form_data=True,
                           success=fail_message,
                           **wizard_step_kwargs_generator(6, selection_items))


@application.route('/load_data_for_validation', methods=['GET'])
def load_data_for_validation():
    current_pipe = load_current_pipe()
    html_data_array = load_excel(current_pipe)
    return_dict = {"targets": html_data_array[0], "features": html_data_array[2], "covariates": html_data_array[1],
                   "groups": html_data_array[3]}
    return json.dumps(return_dict)


@application.route('/validate_data', methods=['GET', 'POST'])
@login_required
def validate_data():
    if request.method == 'POST':
        if request.form["action"] == "previous":
            return redirect(url_for('load_data'))
        else:
            return redirect(url_for('syntax_check'))

    # load pipe
    current_pipe = load_current_pipe()
    pipe_name = current_pipe.name

    return render_template("load_data/validate_data.html",
                           photon_code=PhotonCodeCreator(udo_mode=udo_mode()).create_code(current_pipe),
                           active_steps=["active", "active", "active", "active",
                                         "active", "active", "active", "active"],
                           feature_header="Features",
                           pipe_name=pipe_name)

# ! ---------- END DATA ----------------------------------------------------------------------------------------
