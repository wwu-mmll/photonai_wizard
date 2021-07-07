from app.main.model.elements import *
from photonai_neuro.brain_atlas import AtlasLibrary
from pymodm import connect
from faker import Faker
import pandas as pd


def fill_basic_elements():
    fake = Faker()
    # connect("mongodb://trap-umbriel:27017/photon-wizard2", connect=False, alias="photon_wizard")

    #-------------------------------------------#
    #           Analysis Type                   #
    #-------------------------------------------#

    # Tabular Data
    inst = DataType()
    inst.name = 'Tabular Data'

    inst.short_description = 'For simple data tables'
    inst.long_description = """
    Your data should be gathered in a single Excel table."""
    inst.save()

    # MRI data
    inst = DataType()
    inst.name = "Nifti Data"
    inst.short_description = 'Includes path to Nifti images'
    inst.long_description = """With this analysis you should provide Nifti images for every subject. Additionally,
    you can specify a brain mask to do a ROI-based analysis. The Nifti images will be loaded with the PHOTON Neuro
    module."""
    inst.save()


    #-------------------------------------------#
    #           Data Quantity                   #
    #-------------------------------------------#

    dqs = [((0, 100), "Up to 100 samples", '0 to 100'),
           ((101, 500), "Between 101 and 500 samples", '101 to 500'),
           ((501, 5000), "Between 501 and 5000 samples", '501 to 5000'),
           ((5001, 10000), "Between 5001 and 10000", '5001 to 10000'),
           ((10001, 999999), "More than 10000 examples", 'Lucky bastard')]

    for dq in dqs:
        inst = DataQuantity()
        inst.name = dq[1]
        inst.lower_thres = dq[0][0]
        inst.upper_thres = dq[0][1]
        inst.short_description = dq[2]
        inst.long_description = 'This will affect the recommended cross-validation strategy.'
        inst.save()


    #-------------------------------------------#
    #           Persist Options                 #
    #-------------------------------------------#


    #-------------------------------------------#
    #           Neuro Data                      #
    #-------------------------------------------#
    atlas_library = AtlasLibrary()

    atlases = [('AAL', {'roi_names': None}, 'Brain Atlas', 'An automated anatomical parcellation of the spatially normalized single-subject high-resolution T1 volume provided by the Montreal Neurological Institute (MNI).'),
               ('HarvardOxford_Cortical_Threshold_25', {'roi_names': None}, 'Brain Atlas', 'The Harvard Oxford Atlas provides access to cortical and subcortical regions from the Harvard-Oxford probabilistic atlas at either a 25% or 50% probability threshold. The atlas is in MNI152 space with 2mm isotropic voxels'),
               ('HarvardOxford_Subcortical_Threshold_25', {'roi_names': None}, 'Brain Atlas', 'The Harvard Oxford Atlas provides access to cortical and subcortical regions from the Harvard-Oxford probabilistic atlas at either a 25% or 50% probability threshold. The atlas is in MNI152 space with 2mm isotropic voxels'),
               ('HarvardOxford_Cortical_Threshold_50', {'roi_names': None}, 'Brain Atlas', 'The Harvard Oxford Atlas provides access to cortical and subcortical regions from the Harvard-Oxford probabilistic atlas at either a 25% or 50% probability threshold. The atlas is in MNI152 space with 2mm isotropic voxels'),
               ('HarvardOxford_Subcortical_Threshold_50', {'roi_names': None}, 'Brain Atlas', 'The Harvard Oxford Atlas provides access to cortical and subcortical regions from the Harvard-Oxford probabilistic atlas at either a 25% or 50% probability threshold. The atlas is in MNI152 space with 2mm isotropic voxels'),
               ('Yeo_7', {'roi_names': None}, 'Brain Atlas', 'Data from 1000 young, healthy adults were registered using surface-based alignment. All data were acquired on Siemens 3T scanners using the same functional and structural sequences. A clustering approach was employed to identify and replicate networks of functionally coupled regions across the cerebral cortex. The results revealed local networks confined to sensory and motor cortices as well as distributed networks of association regions that form interdigitated circuits. Within the sensory and motor cortices, functional connectivity followed topographic representations across adjacent areas.'),
               ('Yeo_7_Liberal', {'roi_names': None}, 'Brain Atlas', 'Data from 1000 young, healthy adults were registered using surface-based alignment. All data were acquired on Siemens 3T scanners using the same functional and structural sequences. A clustering approach was employed to identify and replicate networks of functionally coupled regions across the cerebral cortex. The results revealed local networks confined to sensory and motor cortices as well as distributed networks of association regions that form interdigitated circuits. Within the sensory and motor cortices, functional connectivity followed topographic representations across adjacent areas.'),
               ('Yeo_17', {'roi_names': None}, 'Brain Atlas', 'Data from 1000 young, healthy adults were registered using surface-based alignment. All data were acquired on Siemens 3T scanners using the same functional and structural sequences. A clustering approach was employed to identify and replicate networks of functionally coupled regions across the cerebral cortex. The results revealed local networks confined to sensory and motor cortices as well as distributed networks of association regions that form interdigitated circuits. Within the sensory and motor cortices, functional connectivity followed topographic representations across adjacent areas.'),
               ('Yeo_17_Liberal', {'roi_names': None}, 'Brain Atlas', 'Data from 1000 young, healthy adults were registered using surface-based alignment. All data were acquired on Siemens 3T scanners using the same functional and structural sequences. A clustering approach was employed to identify and replicate networks of functionally coupled regions across the cerebral cortex. The results revealed local networks confined to sensory and motor cortices as well as distributed networks of association regions that form interdigitated circuits. Within the sensory and motor cortices, functional connectivity followed topographic representations across adjacent areas.'),
               ('MNI_ICBM152_GrayMatter', None, 'Mask', 'A number of unbiased non-linear averages of the MNI152 database have been generated that combines the attractions of both high-spatial resolution and signal-to-noise while not being subject to the vagaries of any single brain (Fonov et al., 2011). The procedure involved multiple iterations of a process where, at each iteration, individual native MRIs were non-linearly fitted to the average template from the previous iteration, beginning with the MNI152 linear template. We present an unbiased standard magnetic resonance imaging template brain volume for normal population. These volumes were created using data from ICBM project.'),
               ('MNI_ICBM152_WhiteMatter', None, 'Mask', 'A number of unbiased non-linear averages of the MNI152 database have been generated that combines the attractions of both high-spatial resolution and signal-to-noise while not being subject to the vagaries of any single brain (Fonov et al., 2011). The procedure involved multiple iterations of a process where, at each iteration, individual native MRIs were non-linearly fitted to the average template from the previous iteration, beginning with the MNI152 linear template. We present an unbiased standard magnetic resonance imaging template brain volume for normal population. These volumes were created using data from ICBM project.' ),
               ('MNI_ICBM152_WholeBrain', None, 'Mask', 'A number of unbiased non-linear averages of the MNI152 database have been generated that combines the attractions of both high-spatial resolution and signal-to-noise while not being subject to the vagaries of any single brain (Fonov et al., 2011). The procedure involved multiple iterations of a process where, at each iteration, individual native MRIs were non-linearly fitted to the average template from the previous iteration, beginning with the MNI152 linear template. We present an unbiased standard magnetic resonance imaging template brain volume for normal population. These volumes were created using data from ICBM project.' ),
               ]
    for atlas in atlases:
        inst = BrainAtlas()
        inst.name = atlas[0]
        inst.short_description = atlas[2]
        inst.long_description = atlas[3]
        inst.imports = "from photonai_neuro import NeuroBranch"
        param_ids = list()

        if atlas[1] is not None:
            for key, val in atlas[1].items():
                param = Hyperparameter()
                param.name = key
                param.short_description = ""
                param.long_description = """"""

                default = Defaultparameter()
                default.save()
                atlas_object = atlas_library.get_atlas(atlas[0])
                possible_labels = [roi.label for roi in atlas_object.roi_list if roi.index != 0]
                param.possible_values = possible_labels

                param.default_values = [default._id]
                param.value_type = 'Categorical'
                param.multi_select = True
                param.save()
                param_ids.append(param._id)
            inst.hyperparameters = param_ids
        else:
            inst.has_hyperparameters = False
        inst.save()

    # Custom Mask
    inst = BrainAtlas()
    inst.name = 'CustomMask'
    inst.short_description = "Mask"
    inst.long_description = "Specify a path to your custom mask (nifti). The file path should be Linux compatible: i.e. '/spm-data/Scratch/.../'."
    param = Hyperparameter()
    param.name = "mask_image"
    param.short_description = "string"
    param.long_description = "Specify a filename including absolute path. The file path should be Linux compatible: i.e. '/spm-data/Scratch/.../'."
    default = Defaultparameter()
    default.values = "/spm-data/some/path"
    default.save()
    param.default_values = [default._id]
    param.multi_select = False
    param.save()
    inst.hyperparameters = [param._id]
    inst.imports = "from photonai_neuro import NeuroBranch"
    inst.save()

    # Resampling and Smoothing
    resampling = NeuroTransformer()
    resampling.name = "ResampleImages"
    resampling.short_description = "Neuro Transformation"
    resampling.long_description = "Resample your nifti images to your desired voxel size"
    voxel_size = Hyperparameter()
    voxel_size.name = "voxel_size"
    voxel_size.short_description = "Categorical"
    voxel_size.long_description = "Define the voxel size of the resulting images. Specify a single integer like 3 or 4 or " \
                                  "a list of multiple voxel sizes like [3, 4]. "
    default = Defaultparameter()
    default.values = "3"
    default.save()
    voxel_size.default_values = [default._id]
    voxel_size.save()
    resampling.hyperparameters = [voxel_size._id]
    resampling.imports = "from photonai_neuro import NeuroBranch"
    resampling.save()

    smoothing = NeuroTransformer()
    smoothing.name = "SmoothImages"
    smoothing.short_description = "Neuro Transformation"
    smoothing.long_description = "Smooth your nifti images"
    fwhm = Hyperparameter()
    fwhm.name = "fwhm"
    fwhm.short_description = "Categorical"
    fwhm.long_description = "Smooth nifti images with a Full Width Half Maximum Gauss kernel. To optimize this parameter, " \
                                 "use a list of possible values like [6, 8]."
    default = Defaultparameter()
    default.values = "8"
    default.save()
    fwhm.default_values = [default._id]
    fwhm.save()
    smoothing.hyperparameters = [fwhm._id]
    smoothing.imports ="from photonai_neuro import NeuroBranch"
    smoothing.save()


    #-------------------------------------------#
    #           Cross Validation                #
    #-------------------------------------------#


    cv_infos = pd.read_excel('/app/main/assets/db_files/cross_validation.xlsx', sheet_name='Tabelle1',
                             header=0, index_col='cv_name', engine='openpyxl')
    cv_hyp_infos = pd.read_excel('/app/main/assets/db_files/cross_validation.xlsx', sheet_name='Tabelle2',
                                 header=0, engine='openpyxl')

    for cv_name, cv_row in cv_infos.iterrows():
        cv = CV()
        cv.name = cv_name
        cv.short_description = cv_row['short_description']
        cv.long_description = cv_row['long_description']
        cv.imports = """
    {}
        """.format(cv_row['imports'])
        cv.tags = str(cv_row['tags']).split(', ')

        hyp_list = list()
        for _, hyp in cv_hyp_infos.iterrows():
            if hyp['cv_name'] == cv.name:
                param = Hyperparameter()
                param.name = hyp['hyperparameter']
                param.short_description = hyp['short_description']
                param.long_description = hyp['long_description']
                param.value_type = hyp['value_type']
                param.possible_values = str(hyp['possible_values']).replace('‚', '"').replace('‘', '"')
                param.tags = hyp['tags'].split(', ')
                default = Defaultparameter()
                default.values = str(hyp['default_values']).replace('‚', '"').replace('‘', '"')
                default.save()
                param.default_values = [default._id]
                param.save()
                hyp_list.append(param._id)
        cv.hyperparameters = hyp_list
        cv.save()




    regr = AnalysisType()
    regr.name = "Regression"
    regr.short_description = "continuous targets"
    regr.long_description = "Predicting a continuous-valued attribute associated with an object."
    regr.save()


    classif = AnalysisType()
    classif.name = "Classification"
    classif.short_description = "categorical targets"
    classif.long_description = "Identifying to which category an object belongs to."
    classif.save()

    metrics = [('mean_squared_error', regr, '#regression', 'Mean squared error regression loss'),
               ('mean_absolute_error', regr, '#regression', 'Mean absolute error regression loss'),
               ('explained_variance', regr, '#regression', 'Explained variance regression score function'),
               ('pearson_correlation', regr, '#regression', 'Pearson correlation between true and predicted scores.'),
               ('r2', regr, '#regression', 'R^2 (coefficient of determination) regression score function. '
                                           'Best possible score is 1.0, lower values are worse.'),
               ('accuracy', classif, '#classification', 'Accuracy classification score'),
               ('precision', classif, '#classification', 'The precision is the ratio tp / (tp + fp) where tp is the number '
                                                         'of true positives and fp the number of false positives. The precision '
                                                         'is intuitively the ability of the classifier not to label as positive '
                                                         'a sample that is negative. The best value is 1 and the worst value is 0.'),
               ('recall', classif, '#classification', 'The recall is the ratio tp / (tp + fn) where tp is the number of true '
                                                      'positives and fn the number of false negatives. The recall is intuitively '
                                                      'the ability of the classifier to find all the positive samples. '
                                                      'The best value is 1 and the worst value is 0.'),
               ('balanced_accuracy', classif, '#classification', 'The balanced accuracy in binary and multiclass classification '
                                                                 'problems to deal with imbalanced datasets. It is defined '
                                                                 'as the average of recall obtained on each class. '
                                                                 'The best value is 1 and the worst value is 0 when adjusted=False.'),
               ('sensitivity', classif, '#classification', ''),
               ('specificity', classif, '#classification', ''),
               ('f1_score', classif, '#classification', 'Compute the F1 score, also known as balanced F-score or F-measure. '
                                                        'The F1 score can be interpreted as a weighted average of the '
                                                        'precision and recall, where an F1 score reaches its best value at '
                                                        '1 and worst score at 0. The relative contribution of precision and '
                                                        'recall to the F1 score are equal.'),
               ('hamming_loss', classif, '#classification', 'The Hamming loss is the fraction of labels that are '
                                                            'incorrectly predicted.'),
               ('log_loss', classif, '#classification', 'Log loss, aka logistic loss or cross-entropy loss. This is the loss '
                                                        'function used in (multinomial) logistic regression and extensions '
                                                        'of it such as neural networks, defined as the negative '
                                                        'log-likelihood of the true labels given a probabilistic '
                                                        'lassifier’s predictions. The log loss is only defined for two or '
                                                        'more labels.'),
               ('auc', classif, '#classification', 'Compute Area Under the Curve (AUC) using the trapezoidal rule')]

    for metric in metrics:
        inst = Metric()
        inst.name = metric[0]
        inst.short_description = ''
        inst.long_description = metric[3]
        inst.tags = [metric[2]]
        inst.allow_for = metric[1]
        inst.save()

    imb_strategies = [('RandomUndersampling', {}), ('RandomOversampling', {})]

    for imb in imb_strategies:
        inst = ImbStrategy()
        inst.name = imb[0]
        inst.short_description = fake.sentence()
        inst.long_description = fake.text()
        inst.arguments = imb[1]
        inst.save()

    opts = [('grid_search', {}, 'Exhaustive Grid Search', 'Exhaustively generates candidates from a grid of hyperparameter '
                                                          'values specified. All hyperparameter configurations will be '
                                                          'tested. Be careful, the number of possible combinations of '
                                                          'hyperparameters easily explodes and computation time can '
                                                          'become large.'),
            ('random_grid_search', {'n_configurations': (30, 'Number of hyperparameter configurations to be tested.')},
             'Randomized Hyperparameter Optimization',
             'While using a grid of parameter settings is currently the most widely used method for parameter optimization, '
             'other search methods have more favourable properties. Randomized grid search draws k possible hyperparameter '
             'configurations.'),
            ('sk_opt', {'n_configurations': (30, 'Number of hyperparameter configurations to be tested.')},
             'Scikit Optimize', 'Scikit-Optimize, or skopt, is a simple and efficient library to minimize (very) expensive '
                                'and noisy black-box functions. It implements several methods for sequential model-based '
                                'optimization. skopt is reusable in many contexts and accessible.')]

    for opt in opts:
        inst = Optimizer()
        inst.name = opt[0]
        inst.short_description = opt[2]
        inst.long_description = opt[3]
        inst.arguments = opt[1]
        param_ids = list()
        for key, val in opt[1].items():
            param = Hyperparameter()
            param.name = key
            param.short_description = "Optimizer Configuration"
            param.long_description = val[1]

            default = Defaultparameter()
            default.values = val[0]
            default.save()

            param.default_values = [default._id]
            param.value_type = 'Int'
            param.save()
            param_ids.append(param._id)
        inst.hyperparameters = param_ids
        inst.save()



    #-------------------------------------------#
    #           Transformer                      #
    #-------------------------------------------#
    trans_infos = pd.read_excel('/app/main/assets/db_files/transformer.xlsx', sheet_name='Tabelle1', header=0,
                                index_col='transformer_name', engine='openpyxl')
    hyp_infos = pd.read_excel('/app/main/assets/db_files/transformer.xlsx', sheet_name='Tabelle2',
                              header=0, engine='openpyxl')


    for trans_name, trans_row in trans_infos.iterrows():
        inst = Transformer()
        inst.name = trans_name
        inst.short_description = trans_row['short_description']
        inst.long_description = trans_row['long_description']
        inst.tags = str(trans_row["tags"]).strip(" ").split(",")
        inst.pre_processing = bool(trans_row["pre_processing"])

        hyp_list = list()
        for _, hyp in hyp_infos.iterrows():
            if hyp['transformer_name'] == inst.name:
                param = Hyperparameter()
                param.name = hyp['hyperparameter']
                param.short_description = hyp['short_description']
                param.long_description = hyp['long_description']
                param.value_type = hyp['value_type']
                param.possible_values = str(hyp['possible_values']).replace('‚', "'").replace('‘', "'").replace('"', "'")
                default = Defaultparameter()
                default.values = str(hyp['default_values']).replace('‚', "'").replace('‘', "'").replace('"', "'")
                default.save()
                param.default_values = [default._id]
                param.save()
                hyp_list.append(param._id)
        inst.hyperparameters = hyp_list
        inst.save()


    #-------------------------------------------#
    #           Estimator                       #
    #-------------------------------------------#

    est_infos = pd.read_excel('/app/main/assets/db_files/estimator.xlsx', sheet_name='Tabelle1',
                              index_col='estimator_name', engine='openpyxl')
    hyp_infos = pd.read_excel('/app/main/assets/db_files/estimator.xlsx', sheet_name='Tabelle2',
                              header=0, engine='openpyxl')

    for est_name, est_row in est_infos.iterrows():
        est = Estimator()
        est.name = est_name
        est.short_description = est_row['short_description']
        est.long_description = est_row['long_description']
        # print("============================={}".format(est_row["estimator_type"]))
        if est_row['estimator_type'] == 'regr':
            est.estimator_type = regr
            est.tags = ['#regression']
        elif est_row['estimator_type'] == 'classif':
            est.estimator_type = classif
            est.tags = ['#classification']
        # else:
        #     raise ValueError('Estimator type not set correctly.')

        hyp_list = list()
        for _, hyp in hyp_infos.iterrows():
            if hyp['estimator_name'] == est.name:
                param = Hyperparameter()
                param.name = hyp['hyperparameter']
                param.short_description = hyp['short_description']
                param.long_description = hyp['long_description']
                param.value_type = hyp['value_type']
                param.possible_values = str(hyp['possible_values']).replace('‚', "'").replace('‘', "'").replace('"', "'")
                default = Defaultparameter()
                default.values = str(hyp['default_values']).replace('‚', "'").replace('‘', "'").replace('"', "'")
                default.save()
                param.default_values = [default._id]
                param.save()
                hyp_list.append(param._id)
        est.hyperparameters = hyp_list
        est.save()

    # #-------------------------------------------#
    # #           Tutorials                       #
    # #-------------------------------------------#
    # from app.main.model.TutorialDefinition import define_tutorials
    # define_tutorials()

debug = True

