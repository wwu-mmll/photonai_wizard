
import sys
import os
import time
import pandas as pd
import ast
import builtins
import datetime

from glob import glob
from flask import session, abort
from flask_login import current_user
from ..main import application
from ..model.elements import *
from ..model.input_model import *


pd.set_option('display.max_colwidth', 200)


def udo_mode():
    return application.config['UDO_MODE']


def get_form_infos(form: dict, selection_item_list: list, save_element_combi_as_list=False):
    hyperpipe_obj = load_current_pipe_from_db()
    if hyperpipe_obj is not None:
        # get all values out of the form
        for item in selection_item_list:
            val = None
            if item.find_object:
                try:
                    if not item.find_many:
                        if item.fieldset_id in form:
                            obj_id = form[item.fieldset_id]
                            val, combi_element = save_combi_element(item, form, obj_id)
                            if item.has_parameters:
                                val = combi_element
                        else:
                            if item.is_required:
                                print("Could not find value for " + item.fieldset_id + " in formular items." )
                                continue
                    else:
                        val = []
                        for obj_id in form.getlist(item.fieldset_id):
                            tmp_val, combi_element = save_combi_element(item, form, obj_id)
                            if item.has_parameters:
                                val.append(combi_element)
                            else:
                                val.append(tmp_val._id)
                except DoesNotExist:
                    print("Could not set hyperpipe attribute!", item.property_name)
                    pass
            else:
                val = form[item.fieldset_id]

            # check if element is already set
            if item.find_object and item.has_parameters:
                old_combi = getattr(hyperpipe_obj, item.property_name)
                if old_combi is not None:
                    if isinstance(old_combi, ElementCombi):
                        old_combi.delete()
                    elif isinstance(old_combi, list):
                        old_combi = list()

            if not save_element_combi_as_list:
                # and set value into hyperpipe object
                setattr(hyperpipe_obj, item.property_name, val)
            else:
                list_of_ids = list()
                if isinstance(val, ElementCombi):
                    setattr(hyperpipe_obj, item.property_name, val)
                else:
                    for element_combi in val:
                        element_combi.save()
                        list_of_ids.append(str(element_combi._id))
                    setattr(hyperpipe_obj, item.property_name, list_of_ids)

        hyperpipe_obj.save()
        return hyperpipe_obj


def save_combi_element(item, form, obj_id):
    val = item.object_type.objects.raw({'_id': ObjectId(obj_id)}).first()
    if item.has_parameters:
        combi_element = ElementCombi()
        combi_element.referenced_element_id = val._id
        combi_element.hyperparameters = {}
        combi_element.name = val.name
        # find all parameters
        parameters = {}
        for param_name, param_value in form.items():
            if param_name.startswith(item.fieldset_id):
                stripped_param_name = param_name[len(item.fieldset_id)+1::]
                if stripped_param_name[0:len(obj_id)] == obj_id:
                    if stripped_param_name.endswith('test_disabled'):
                        if param_value == 'on':
                            combi_element.test_disabled = True
                        else:
                            combi_element.test_disabled = False
                    else:
                        single_param_name = stripped_param_name[len(obj_id) + 1::]
                        value_to_persist = get_info_from_string(param_value)

                        if value_to_persist == '':
                            # if we get an empty string, try to load default parameter:
                            value_to_persist = get_default_parameter_for_empty_string(val, item.constraint_list,
                                                                                      single_param_name)

                        parameters[single_param_name] = value_to_persist
        combi_element.hyperparameters = parameters
        combi_element.save()
        return val, combi_element
    return val, None


def get_default_parameter_for_empty_string(val: object, constraint_list: list, single_param_name: str):
    try:
        tmp_hyperparameters = val.get_filtered_hyperparameter_objects(constraint_list)
        if tmp_hyperparameters is not None and len(tmp_hyperparameters) > 0:
            for hyp in tmp_hyperparameters:
                if hyp.name == single_param_name:
                    if hyp.default_objects is not None and len(hyp.default_objects) > 0:
                        default_object = hyp.default_objects[0]
                        value_to_persist = default_object.values
                        return value_to_persist
    except Exception as e:
        # do nothing because its just a try to safe the user
        print(e)
        return ''


def get_info_from_string(value_to_persist: str):
    if not value_to_persist.startswith("IntegerRange") and not value_to_persist.startswith("FloatRange"):
        # check if input values are native datatypes (int, doubles, etc.) or custom strings
        if len(value_to_persist) > 0:
            try:
                value_to_persist = ast.literal_eval(value_to_persist)
            except builtins.ValueError as e:
                if "," in value_to_persist:
                    value_to_persist = value_to_persist.split(",")
                else:
                    value_to_persist = [value_to_persist]
            except SyntaxError:
                value_to_persist = [value_to_persist]
        else:
            value_to_persist = ""
    return value_to_persist


def check_if_pipe_exists():
    pipe_id = session['pipe_id']
    if pipe_id is None or pipe_id == '':
        return False
    try:
        curr_pipe = Pipeline.objects.raw({'_id': ObjectId(pipe_id)}).first()
        return True
    except DoesNotExist:
        session["pipe_id"] = ""
        return False


def load_pipe_from_db(pipe_id):
    try:
        curr_pipe = Pipeline.objects.raw({'_id': ObjectId(pipe_id)}).first()
        return curr_pipe
    except DoesNotExist:
        return None


def load_current_pipe_from_db():
    pipe_id = session['pipe_id']
    if pipe_id is None or pipe_id == '':
        return None
    try:
        curr_pipe = Pipeline.objects.raw({'_id': ObjectId(pipe_id)}).first()
        return curr_pipe
    except DoesNotExist:
        return None

def load_current_pipe():
    pipe = load_current_pipe_from_db()
    if pipe is None:
        session['wizard_error'] = "We could not find your pipeline object in the database and " \
                                  "so we could not reload your photon project, so yes, actually everything is lost and " \
                                  "you have to do it all again... <br><b>Please don't cry</b>"
        abort(500)
    else:
        return pipe


def load_hyperparameters(fieldset_id, value):
    hyperparameter_dict = dict()
    for hyp_item_key, hyp_item_value in value.hyperparameters.items():
        hyp_key = fieldset_id + "_" + str(value.referenced_element_id) + "_" + str(hyp_item_key)
        hyperparameter_dict[hyp_key] = hyp_item_value
    return hyperparameter_dict


def load_pipe_presets(selection_item_list):

    curr_pipe = load_current_pipe_from_db()
    for item in selection_item_list:
        value = getattr(curr_pipe, item.property_name)
        if value is not None:
            if isinstance(item, InputBox):
                if value != "None":
                    item.default_value = value
            elif isinstance(item, SelectionBox):
                if not item.find_many and not item.has_parameters:
                    item.default_value = value._id
                if not item.find_many and item.has_parameters:
                    # if isinstance(item, ObjectId):
                    #     item.default_value = value
                    # else:
                    item.default_value = value.referenced_element_id
                    item.hyperparameter_dict = load_hyperparameters(item.fieldset_id, value)
                if item.find_many and not item.has_parameters:
                    item.default_value = value
                if item.find_many and item.has_parameters:
                    item.default_value = list()
                    item.hyperparameter_dict = dict()
                    if len(value) == 0:
                        item.default_value = None
                    else:
                        for obj_id in value:
                            loaded_element_combi = ElementCombi.objects.get({'_id': ObjectId(obj_id)})
                            item.default_value.append(loaded_element_combi.referenced_element_id)
                            item.hyperparameter_dict.update(load_hyperparameters(item.fieldset_id, loaded_element_combi))


    # debug = True


def load_excel(pipe):
    try:
        # df = pd.read_excel(pipe.data_file)
        df = pd.read_excel(pipe.data_file, nrows=30)
        # df = pd.concat([chunk[chunk['field'] > 1] for chunk in iter_csv])
    except Exception as e:
        print(e)
        return ("Couldn't load excel sheet: {}".format(sys.exc_info()[0]), None, None)

    try:
        c = pipe.features.split(':')
        c1 = c[0]
        c2 = c[-1]
        if c1 == c2:
            features = df.iloc[:50, [int(c1)]].to_html(classes='pandas_dataframe')
        else:
            features = df.iloc[:50, int(c1):int(c2)].to_html(classes='pandas_dataframe')
    except:
        features = "Invalid column selection: Remember to select a single column by " \
                   "specifying an integer corresponding to one column in the dataframe."
    try:
        c1 = pipe.targets.split(':')[0]
        targets = df.iloc[:50, [int(c1)]].to_html(classes='pandas_dataframe')
    except:
        targets = "Invalid column selection: Remember to select a single column by " \
                   "specifying an integer corresponding to one column in the dataframe."

    if pipe.covariates:
        try:
            c1 = pipe.covariates.split(':')[0]
            c2 = pipe.covariates.split(':')[-1]
            if c1 == c2:
                covariates = df.iloc[:50, [int(c1)]].to_html(classes='pandas_dataframe')
            else:
                covariates = df.iloc[:50, int(c1):int(c2)].to_html(classes='pandas_dataframe')
        except:
            covariates = "Invalid column selection: Remember to select a single column by " \
                       "specifying an integer corresponding to one column in the dataframe."
    else:
        covariates = "Not selected"

    if pipe.groups:
        try:
            c1 = pipe.groups.split(':')[0]
            groups = df.iloc[:50, [int(c1)]].to_html(classes='pandas_dataframe')
        except:
            groups = "Invalid column selection: Remember to select a single column by " \
                       "specifying an integer corresponding to one column in the dataframe."
    else:
        groups = "Not selected"
    return targets, covariates, features, groups


def get_current_result_folder(current_pipe):
    search_name = str(current_pipe.project_name) + "_results_*"
    spss_files = glob(os.path.join(current_pipe.photon_project_folder, search_name))
    if len(spss_files) > 0:
        most_recent = 0
        for i, file in enumerate(spss_files):
            # file = file.replace('-', '')
            date_str = file[-19::]
            new_date = time.strptime(date_str, "%Y-%m-%d_%H-%M-%S")

            if i != 0:
                if new_date > date:
                    most_recent = i
                    date = new_date
            else:
                date = new_date
        current_folder = spss_files[most_recent]
    else:
        current_folder = ""
    return current_folder


def get_current_log_file(current_pipe):
    current_folder = get_current_result_folder(current_pipe)
    if current_folder != "":
        return os.path.join(current_folder, "photon_output.log")
    else:
        return ""


def set_basic_constraints(pipe_obj):
    constraint_dict = dict()
    # save analysis type for offering correct metrics and estimators later
    analysis_type = pipe_obj.analysis_type.system_name
    constraint_dict["analysis_type"] = ['#' + analysis_type.lower()]

    # save data quantity constraint for selecting best default parameters later
    if pipe_obj.data_quantity:
        data_quantity = pipe_obj.data_quantity.upper_thres
        constraint_dict["data_quantity"] = ['#' + str(data_quantity)]
    return constraint_dict


def new_pipeline_object():
    pipe_obj = Pipeline()
    if udo_mode():
        pipe_obj.mongodb_connect_url = application.config["MONGODB_CS"] + "/photon_results"
    if udo_mode():
        pipe_obj.user = current_user.username
    else:
        pipe_obj.user = 'anonymous_photon_web_user'
    pipe_obj.creation_date = datetime.datetime.now()
    pipe_obj.status = 0
    pipe_obj.save()
    session['pipe_id'] = str(pipe_obj._id)
    return pipe_obj
