import tailer
import requests
import uuid
import os
import shutil
import json
import datetime
import numpy as np
import matplotlib.pylab as plt
import plotly
import plotly.graph_objs as go

from typing import Union
from bson import ObjectId
from flask import render_template, session, redirect, url_for, send_from_directory, abort, request, jsonify
from flask_login import login_required

from pymongo import MongoClient
from pymongo.errors import AutoReconnect

from ..main import application
from ..model.elements import Pipeline, PermutationTestInfos
from ..model.code_creator import PhotonCodeCreator
from ..controller.pipeline_helper import load_pipe_from_db, get_current_log_file, load_current_pipe
from ..controller.new_pipeline import get_current_result_folder

from photonai.processing.permutation_test import PermutationTest


client = MongoClient(host=application.config["MONGODB_CS"], connect=False)
celery_db = client['photon-wizard-jobs']['celery_taskmeta']
photon_db = client['photon_results']['mdb_hyperpipe']


@application.before_request
def before_request():
    pass


@application.route('/', methods=['GET', 'POST'])
def index():
    if application.config['UDO_MODE']:
        return redirect(url_for('login'))  # Send them home
    else:
        return render_template('index_web.html')


@application.route('/load_pipeline/<pipe_id>')
@login_required
def load_pipeline(pipe_id):
    session['pipe_id'] = pipe_id
    return redirect(url_for('general_infos'))


@application.route('/new_pipeline')
@login_required
def new_pipeline():
    session['pipe_id'] = ''
    return redirect(url_for('general_infos'))


@application.route('/wizard_error')
@application.errorhandler(500)
@application.errorhandler(404)
@application.errorhandler(502)
@application.errorhandler(Exception)
def wizard_error(e):
    success = "Ooopsi, an error occured"
    if hasattr(e, 'code'):
        if e.code == 404:
            success = "...could not find that page. It may be cursed."
        elif e.code == 502:
            success = "Probably MongoDB has some problems. Give it a moment."
        elif e.code == 504:
            success = "Computing the output took too long. Maybe the working memory of the server is full. "
    elif "wizard_error" in session:
        success = session["wizard_error"]
    if "wizard_error_header" in session and session["wizard_error_header"]:
        header = session["wizard_error_header"]
    else:
        header = None

    success += "  " + str(e)

    session["wizard_error"] = ""
    session["wwizard_error_header"] = ""

    return render_template('error.html', error_msg=success, header=header)


@application.route('/hidden_quiz')
def hidden_quiz():
    return render_template('hidden_quiz.html')


@application.route('/project_history', methods=["GET", "POST"])
@login_required
def project_history():
    user = session['user']
    pipes = list(Pipeline.objects.raw({'$or': [{'user': user}]}))
    update_status(pipes)
    return render_template("project_history.html",  pipes=pipes)

@application.route('/FAQ', methods=["GET", "POST"])
def faq():
    return render_template("FAQ.html")


@application.route('/edit_pipe/<pipe_id>')
@login_required
def edit_pipe(pipe_id):
    session['pipe_id'] = pipe_id
    return redirect(url_for('syntax_check'))


@application.route('/copy_pipe/<pipe_id>')
@login_required
def copy_pipe(pipe_id):
    pipe_to_copy = load_pipe_from_db(pipe_id)
    new_pipe = pipe_to_copy.copy_me()
    session['pipe_id'] = str(new_pipe._id)
    return redirect(url_for('general_infos'))


@application.route('/delete_pipeline/<pipe_id>')
@login_required
def delete_pipeline(pipe_id):

    current_pipe = load_pipe_from_db(pipe_id)

    # cancel any occurring computations
    if current_pipe.celery_id:
        _cancel_computation(current_pipe.celery_id)
    if current_pipe.permutation_test and current_pipe.permutation_test.permutation_celery_id:
        _cancel_computation(current_pipe.permutation_test.permutation_celery_id)

    # delete in photon_results mongo gb
    # Todo change str to ObjectId with new photon version
    try:
        find_results = list(photon_db.find({'wizard_object_id': str(current_pipe._id)}))
        if len(find_results) > 0:
            delete_result = photon_db.delete_one({'wizard_object_id': current_pipe._id})
    except AutoReconnect as ar:
        print("Could not delete photon-db result tree for wizard pipeline")
    except Exception as e:
        print(e)
    finally:
        client.close()

    # delete folder
    try:
        if current_pipe.photon_project_folder is not None:
            shutil.rmtree(current_pipe.photon_project_folder)
    except FileNotFoundError:
        # then its already gone - we don't have to delete the folder
        pass

    current_pipe.delete()
    return redirect(url_for('project_history'))


@application.route('/cancel_computation/<pipe_id>')
@login_required
def cancel_computation(pipe_id):
    pipe = load_pipe_from_db(pipe_id)
    _cancel_computation(pipe.celery_id)
    # reset celery_id
    pipe.celery_id = ''
    pipe.celery_status = 'REVOKED'
    pipe.save()
    return redirect(url_for('project_history'))


@application.route('/cancel_permutation/<pipe_id>')
@login_required
def cancel_permutation(pipe_id):
    pipe = load_pipe_from_db(pipe_id)
    if hasattr(pipe, "permutation_test"):
        _cancel_computation(pipe.permutation_test.permutation_celery_id)
    # reset celery_id
    pipe.permutation_test = None
    pipe.save()
    return redirect(url_for('project_history'))


def _cancel_computation(celery_id):
    try:
        if celery_id is not None and celery_id != '':
            cancel_status = requests.get(application.config["TASK_CANCEL_URL"] + celery_id)
    except requests.exceptions.ConnectionError as e:
        print("Could not cancel pipeline:" + str(e))
        session["wizard_error"] = "Could not cancel pipeline due to connection problems." \
                                  " Please try to stop the analysis again " \
                                  "in some minutes. "
        abort(500)

@application.route('/titania_queue')
@login_required
def titania_queue():
    try:
        queue_info = requests.get(application.config["QUEUE_INFO_URL"])
        try:
            response_dict_all = json.loads(queue_info.content.decode())
        except Exception as e:
            abort(e)
    except ConnectionError as ce:
        session["wizard_error"] = "Could not retrieve worker status. Try again in some moments."
        abort(500)


    new_dict = dict()
    iterable_dicts = ["active", "reserved", "scheduled"]
    for status in iterable_dicts:
        response_dict = response_dict_all[status]
        for worker_name, worker_task_list in response_dict.items():
            if not worker_name in new_dict:
                new_dict[worker_name] = list()
            for worker_item in worker_task_list:
                if 'id' in worker_item:
                    celery_id_to_find = worker_item['id']
                    pipeline_obj = Pipeline.objects.raw({'$or': [{'celery_id': celery_id_to_find},
                                                                 {'permutation_test.permutation_celery_id': celery_id_to_find}]})
                    if pipeline_obj.count() > 0:
                       pipe = pipeline_obj.first()
                       new_dict[worker_name].append({'status': status, 'user': pipe.user, 'name': pipe.name})

    return render_template('titania_queue.html', worker_dict=new_dict)


@application.route('/currently_running/<pipe_id>')
@login_required
def currently_running(pipe_id):
    current_pipe = load_pipe_from_db(pipe_id)
    if current_pipe is None:
        session["wizard_error"] = "Could not find pipeline object. Is it deleted?"
        abort(500)
    current_path = current_pipe.photon_project_folder
    return render_template("pipeline_steps/currently_running.html",
                           pipe=current_pipe, current_path=current_path)


@application.route('/download_summary/<pipe_id>')
@login_required
def download_summary(pipe_id):
    pipe = load_pipe_from_db(pipe_id)
    current_folder = get_current_result_folder(pipe)

    if current_folder != "":
        filename = "photon_summary.txt"
        return send_from_directory(directory=current_folder, filename=filename)
    else:
        return redirect(url_for('currently_running'))


@application.route('/download_json/<pipe_id>')
@login_required
def download_json(pipe_id):
    pipe = load_pipe_from_db(pipe_id)
    current_folder = get_current_result_folder(pipe)

    if current_folder != "":
        filename = "photon_result_file.json"
        return send_from_directory(directory=current_folder, filename=filename, as_attachment=True)
    else:
        return redirect(url_for('currently_running'))


@application.route('/download_script/<pipe_id>')
@login_required
def download_script(pipe_id):
    pipe = load_pipe_from_db(pipe_id)
    filename = os.path.basename(pipe.photon_file_path)
    if application.config["UDO_MODE"]:
        return send_from_directory(directory=pipe.photon_project_folder, filename=filename)
    else:
        return send_from_directory(directory=application.config["TMP_PHOTON_SCRIPT_FOLDER"], filename=filename)


@application.route("/rerun_photon/<pipe_id>")
@login_required
def rerun_photon(pipe_id):
    session['pipe_id'] = pipe_id
    return redirect(url_for('run_photon'))


@application.route('/run_photon/<pipe_id>')
@login_required
def run_photon(pipe_id):
    current_pipe = load_pipe_from_db(pipe_id)
    photon_script = current_pipe.photon_file_path

    try:
        # cancel any computations or permutation tests and reset
        if current_pipe.celery_id:
            _cancel_computation(current_pipe.celery_id)
            current_pipe.celery_id = ''
            current_pipe.celery_status = ''
        if current_pipe.permutation_test and current_pipe.permutation_test.permutation_celery_id:
            _cancel_computation(current_pipe.permutation_test.permutation_celery_id)
        current_pipe.permutation_test = None
        current_pipe.save()

        if not os.path.isfile(photon_script):
            session["wizard_error"] = "Could not find photon source code file in your project folder! " \
                                      "Expected file to be at " + current_pipe.photon_file_path
            abort(500)

        res = requests.get(os.path.join(application.config["TASK_RUN_URL"], str(current_pipe._id) + "/photon"))
        try:
            response_dict = json.loads(res.content.decode())
            if 'celery_id' not in response_dict:
                raise requests.exceptions.ConnectionError("Request to server was not successful")
            current_pipe.celery_id = response_dict['celery_id']
            current_pipe.save()
        except:
            raise requests.exceptions.ConnectionError("Could not find task_id in response. Assuming error")
    except requests.exceptions.ConnectionError:
        session["wizard_error"] = "Could not connect to computing server. Please try to start the analysis again " \
                                  "in some minutes. "
        abort(500)

    return redirect(url_for('currently_running', pipe_id=current_pipe._id))


@application.route('/fetch_pipe_progress', methods=["POST"])
@login_required
def fetch_pipe_progress():

    if "pipe_id" in request.form:
        pipe_id = request.form["pipe_id"]
        if pipe_id != "":
            try:
                pipe_status = _get_pipe_status(pipe_id)
                new_log = get_current_log_file(current_pipe=load_pipe_from_db(pipe_id))
                if new_log == "":
                    text = ""
                else:
                    file = open(new_log, "rt")
                    text = tailer.tail(file, 50)
                    file.close()
            except OSError as e:
                # file not found...
                print(e)
                text = ""
    else:
        text = ""
        pipe_status = "UNPROCESSED"
    if text == "":
        lines_of_file = "The computation job is still pending. Waiting in queue... "
    else:
        lines_of_file = ''
        lines_of_file += '<br>'.join(text)

    response = {"log_text": lines_of_file,
                "status": pipe_status}
    return jsonify(response)


@application.route('/restart_permutation_test/<pipe_id>')
def restart_permutation_test(pipe_id):
    pipe_to_permute = load_pipe_from_db(pipe_id)
    sent_permutation_request(pipe_to_permute)
    return redirect(url_for('permutation', pipe_id=pipe_id, permutation_id=pipe_to_permute.permutation_test.perm_id))


def sent_permutation_request(pipe_to_permute):
    try:
        res = requests.get(os.path.join(application.config["TASK_RUN_URL"], str(pipe_to_permute._id) + "/perm"))
        try:
            response_dict = json.loads(res.content.decode())
            if 'celery_id' not in response_dict:
                raise requests.exceptions.ConnectionError("Request to server was not successful")
            else:
                pipe_to_permute.permutation_test.permutation_celery_id = response_dict["celery_id"]
                pipe_to_permute.save()
                return pipe_to_permute.permutation_test.permutation_celery_id
        except:
            raise requests.exceptions.ConnectionError("Could not find task_id in response. Assuming error")

    except requests.exceptions.ConnectionError:
        session["wizard_error"] = "Could not connect to computing server. Please try to start the analysis again " \
                                  "in some minutes. "
        abort(500)


@application.route('/permutation_test/<pipe_id>', methods=["GET", "POST"])
@login_required
def permutation_test(pipe_id):
    if request.method == 'POST':
        n_perms = int(request.form["n_perms"])

        pipe_to_permute = load_pipe_from_db(pipe_id)
        pipe_to_permute.permutation_test = PermutationTestInfos()
        pipe_to_permute.permutation_test.perm_id = uuid.uuid4()
        pipe_to_permute.permutation_test.n_perms = n_perms
        pipe_to_permute.permutation_test.permutation_file_path = os.path.join(pipe_to_permute.photon_project_folder,
                                                             'photon_permutation_test.py')
        pipe_to_permute.save()

        # set original computation as reference for calculating permutation test p values

        try:
            preparation_result = PermutationTest.prepare_for_wizard(pipe_to_permute.permutation_test.perm_id,
                                                                    wizard_id=ObjectId(pipe_id),
                                                                    mongo_db_connect_url=application.config["MONGODB_CS"]
                                                                                         + "/photon_results")
        except Exception as e:
            raise Exception("Could not prepare permutation test, failed communication with PHOTONAI package. "
                            + str(e))

        # check if pipe is better than dummy, otherwise doing a permutation test is not sensible
        if not preparation_result["usability"]:
            pipe_to_permute.permutation_test.perm_id = None
            pipe_to_permute.save()
            session["wizard_error_header"] = "Aborting permutation test."
            session["wizard_error"] = "Wizard applied some sanity check rules. Usefulness of a permutation test " \
                                      "for your analysis is doubted because your performance is either" \
                                      " not better than the dummy estimator or not established yet."
            abort(500)

            # get duration of original pipe

        duration = preparation_result["estimated_duration"]
        pipe_to_permute.permutation_test.duration_per_permutation = duration.total_seconds()
        pipe_to_permute.save()

        with open(pipe_to_permute.permutation_test.permutation_file_path, 'w') as text_file:
            photon_code = PhotonCodeCreator(udo_mode=application.config["UDO_MODE"]).create_code_for_perm_test(pipe_to_permute)
            text_file.write(photon_code)
            text_file.close()

        perm_id = sent_permutation_request(pipe_to_permute)

        return redirect(url_for('permutation',
                                pipe_id=pipe_id,
                                permutation_id=perm_id))
    else:
        return redirect(url_for('project_history'))


@application.route('/fetch_permutation_status/<pipe_id>/<permutation_id>')
def fetch_permutation_status(pipe_id, permutation_id):

    pipe = load_pipe_from_db(pipe_id)
    result = dict()

    result["pipe_name"] = pipe.name

    celery_status = _get_pipe_status(pipe_id, True)
    result["status"] = celery_status

    if celery_status not in ["REVOKED", "PENDING", "FAILURE"]:
        try:
            perm_result = PermutationTest._calculate_results(permutation_id,
                                                             mongodb_path=application.config["MONGODB_CS"]
                                                                          + "/photon_results")
        except Exception as e:
            raise e
        if perm_result is None:
            session["wizard_error"] = "Could not find permutation test results, please restart permutation test!"
            try:
                _cancel_computation(pipe.permutation_test.permutation_celery_id)
            except:
                pass
            pipe.permutation_test = None
            abort(500)

        plot_list = []
        if perm_result is not None:
            for metric, values in perm_result.perm_performances.items():
                p = perm_result.p_values[metric]
                true_value = perm_result.true_performances[metric]
                if len(values) > 0:
                    plot_list.append(plotly_perm_hist(true_value, values, metric, p))
        result["plot_html"] = '\n'.join(plot_list)

        left_over_perms = perm_result.n_perms - perm_result.n_perms_done
        if left_over_perms > 0:

            time_left = datetime.timedelta(
                seconds=(pipe.permutation_test.duration_per_permutation * left_over_perms) * 2)
            time_left_str = str(time_left)[:-7]
        else:
            time_left_str = '0'

        if perm_result.n_perms_done == 1:
            perm_result_done = 0
        else:
            perm_result_done = perm_result.n_perms_done

        result["pipe_n_todo"] = pipe.permutation_test.n_perms
        result["pipe_n_done"] = perm_result_done
        result["pipe_time_left"] = time_left_str
        result["pipe_p"] = str(perm_result.p_values)
    else:
        result["plot_html"] = ''
        result["pipe_n_todo"] = ''
        result["pipe_n_done"] = ''
        result["pipe_time_left"] = ''
        result["pipe_p"] = ''

    return jsonify(result)


@application.route('/permutation/<pipe_id>/<permutation_id>')
def permutation(pipe_id, permutation_id):
    return render_template('general/permutation_status.html', pipe_id=pipe_id,
                            permutation_id=permutation_id)


def plotly_perm_hist(performance: Union[int, float], perm_performances: np.array, metric: str,
                     p: Union[int, float] = None):
    import time
    y_lim = np.max(plt.hist(perm_performances, bins=50)[0])

    start = min(perm_performances)
    stop = max(perm_performances)
    size = (stop - start) / 50
    hist = go.Histogram(x=perm_performances, opacity=0.75, name='Permutation Performances',
                        xbins=dict(
                            start=start,
                            end=stop,
                            size=size
                        ))
    true_vline = go.Scatter(x=[performance, performance], y=[0, y_lim], name='True Performance',
                            line=dict(color=('rgb(205, 12, 24)'), width=5))
    data = [hist, true_vline]
    layout = go.Layout(title="", paper_bgcolor='rgba(0,0,0,0)', showlegend=False,
                       plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(title=metric.upper()),
                       annotations=[dict(x=performance, y=y_lim*1.1, xref='x', yref='y', text='p = {}'.format(p),
                                         showarrow=False)])
    plot_html = plotly.offline.plot({"data": data, "layout": layout},
                               auto_open=False, output_type='div')
    return """
    <div class="uk-width-1-2@s uk-width-1-3@m uk-width-1-4@l">{}</div>
    """.format(plot_html)


def update_status(pipes):
    for pipe in pipes:
        celery_db = client['photon-wizard-jobs']['celery_taskmeta']
        for celery_result in celery_db.find({'_id': pipe.celery_id}):
            pipe.celery_status = celery_result['status']
        client.close()
    return pipes


def _get_pipe_status(pipe_id, permutation_status=False):

    if pipe_id == "":
        return "PENDING"
    else:
        current_pipe = load_pipe_from_db(pipe_id)
        if not permutation_status:
            celery_id = current_pipe.celery_id
        else:
            celery_id =current_pipe.permutation_test.permutation_celery_id
        celery_result = list(celery_db.find({'_id': celery_id}))

        if celery_result:
            status = celery_result[0]['status']
            if not permutation_status:
                current_pipe.celery_status = status
            else:
                current_pipe.permutation_test.permutation_celery_status = status
            current_pipe.save()
            return status
        else:
            return "PENDING"
