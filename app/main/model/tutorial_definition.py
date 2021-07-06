from app.main.model.elements import *
from flask import abort, session
from shutil import copyfile
import datetime
import os


def set_pipe_path(pipe, user_name, src_file):
    project_name = re.sub('[^0-9a-zA-Z]+', '', pipe.name).lower()
    pipe.project_name = project_name

    pipe.photon_project_folder = "/spm-data/Scratch/photon_wizard/" + user_name + "/" + project_name
    pipe.photon_file_path = "/spm-data/Scratch/photon_wizard/" + user_name + "/" + project_name + "/photon_code.py"
    if not os.path.isdir(pipe.photon_project_folder):
        oldmask = os.umask(000)
        os.makedirs(pipe.photon_project_folder, 0o777)
        os.umask(oldmask)
    target_features = "/spm-data/Scratch/photon_wizard/" + user_name + "/" + project_name + "/" + os.path.basename(src_file)
    try:
        copyfile(src_file, target_features)
    except FileNotFoundError as f:
        print(f)
        session["wizard_error"] = "Could not load tutorial data"
        abort(500)

    return target_features


def create_boston_housing(user_name: str):
    ####################################################################################################################
    #           BOSTON HOUSING                                                                                         #
    ####################################################################################################################
    pipe = Pipeline()
    pipe.name = 'Boston Housing ' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    pipe.user = user_name
    pipe.description = 'The Boston housing dataset contains 506 observations on housing prices for Boston suburbs and has 15 features. The medv variable is the target variable.'

    data_type = list(DataType.objects.raw({'name': 'Tabular Data'}))[0]
    data_quantity = list(DataQuantity.objects.raw({'name': "Between 101 and 500 samples"}))[0]
    analysis_type = list(AnalysisType.objects.raw({'name': "Regression"}))[0]
    pipe.data_type = data_type
    pipe.data_quantity = data_quantity
    pipe.analysis_type = analysis_type

    target_features = set_pipe_path(pipe, user_name,
                                    "/spm-data/Scratch/photon_wizard/examples/boston_housing/boston_data.xlsx")

    pipe.data_file = target_features
    pipe.targets = '13'
    pipe.features = '0:13'

    pipe = set_constraint_dict(pipe)
    pipe.save()
    return pipe


def set_constraint_dict(pipe):
    analysis_type = pipe.analysis_type.system_name
    pipe.constraint_dict["analysis_type"] = ['#' + analysis_type.lower()]

    # save data quantity constraint for selecting best default parameters later
    data_quantity = pipe.data_quantity.upper_thres
    pipe.constraint_dict["data_quantity"] = ['#' + str(data_quantity)]
    return pipe


def create_breast_cancer(user_name: str):
    ####################################################################################################################
    #           BREAST CANCER                                                                                          #
    ####################################################################################################################

    pipe = Pipeline()
    pipe.name = 'Breast Cancer ' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    pipe.user = user_name
    pipe.description = 'Classification of malignant and benign tumors. Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. ' \
                       'They describe characteristics of the cell nuclei present in the image.'

    data_type = list(DataType.objects.raw({'name': 'Tabular Data'}))[0]
    data_quantity = list(DataQuantity.objects.raw({'name': "Between 501 and 5000 samples"}))[0]
    analysis_type = list(AnalysisType.objects.raw({'name': "Classification"}))[0]
    pipe.data_type = data_type
    pipe.data_quantity = data_quantity
    pipe.analysis_type = analysis_type

    source_features = "/spm-data/Scratch/photon_wizard/examples/breast_cancer_data.xlsx"
    target_features = set_pipe_path(pipe, user_name, source_features)
    pipe.data_file = target_features

    pipe.targets = '1'
    pipe.features = '2:31'

    pipe = set_constraint_dict(pipe)
    return pipe


def create_neuro_classification(user_name: str):
    ###################################################################################################################
    #          PHOTON Neuro Gender Classification                                                                     #
    ###################################################################################################################
    pipe = Pipeline()

    pipe.name = 'Neuro Gender ' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    pipe.user = user_name
    pipe.description = 'Classify the gender of 100 subjects of the FOR2107 dataset using VBM8 data and PHOTON Neuro.'

    data_type = DataType.objects.get({'name': 'Nifti Data'})
    data_quantity = DataQuantity.objects.get({'name': "Between 501 and 5000 samples"})
    analysis_type = AnalysisType.objects.get({'name': "Classification"})
    pipe.data_type = data_type
    pipe.data_quantity = data_quantity
    pipe.analysis_type = analysis_type

    source_features = "/spm-data/Scratch/photon_wizard/examples/neuro_age/PAC2019_data_test.xlsx"
    target_features = set_pipe_path(pipe, user_name, source_features)

    pipe.data_file = target_features
    pipe.targets = '2'
    pipe.features = '4'

    pipe.save()
    return pipe


def create_neuro_regression(user_name: str):

    ####################################################################################################################
    #          PHOTON Neuro Age Regression                                                                             #
    ####################################################################################################################

    pipe = Pipeline()

    pipe.name = 'Neuro Age ' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    pipe.user = user_name
    pipe.description = 'Predict the age of 100 subjects of the FOR2107 dataset using VBM8 data and PHOTON Neuro.'

    data_type = DataType.objects.get({'name': 'Nifti Data'})
    data_quantity = DataQuantity.objects.get({'name': "Between 501 and 5000 samples"})
    analysis_type = AnalysisType.objects.get({'name': "Regression"})
    pipe.data_type = data_type
    pipe.data_quantity = data_quantity
    pipe.analysis_type = analysis_type

    source_features = "/spm-data/Scratch/photon_wizard/examples/neuro_age/PAC2019_data_test.xlsx"
    target_features = set_pipe_path(pipe, user_name, source_features)

    pipe.data_file = target_features
    pipe.targets = '1'
    pipe.features = '4'

    pipe.save()
    return pipe

