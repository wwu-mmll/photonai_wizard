
# -------------------- GENERATED WITH PHOTON WIZARD (beta) ------------------------------
# PHOTON Project Folder: /home/user1/project2

import pandas as pd
import numpy as np
from photonai.base import Hyperpipe, PipelineElement, OutputSettings, Switch, Preprocessing
from photonai.optimization import Categorical, IntegerRange, FloatRange

    from sklearn.model_selection import ShuffleSplit        
             
# Specify how results are going to be saved
# Define hyperpipe
hyperpipe = Hyperpipe('adsd',
                      project_folder = '/home/user1/project2',
                      optimizer="sk_opt",
                      optimizer_params={'n_configurations': 30},
                      metrics=['precision', 'specificity'],
                      best_config_metric="recall",
                      outer_cv = ShuffleSplit(n_splits=3,test_size=0.1),
                      inner_cv = ShuffleSplit(n_splits=3, test_size=0.1))
        
# Add transformer elements
hyperpipe += PipelineElement("FastICA", hyperparameters={}, 
                             test_disabled=False, n_components=None)
# Add estimator
estimator_switch = Switch('EstimatorSwitch')
estimator_switch += PipelineElement("SVC", hyperparameters={'C': FloatRange(0.5, 2), 'kernel': ['linear', 'rbf']}, gamma='scale', max_iter=1000000)
estimator_switch += PipelineElement("RandomForestClassifier", hyperparameters={'n_estimators': IntegerRange(5, 20), 'min_samples_split': IntegerRange(2,5), 'min_samples_leaf': IntegerRange(1,3)}, criterion='gini', max_depth=None)
hyperpipe += estimator_switch                

# Load data
df = pd.read_excel('/home/user1/project2/features.xlsx')
X = np.asarray(df.iloc[:, 4:27])
y = np.asarray(df.iloc[:, 0])

# Fit hyperpipe
hyperpipe.fit(X, y)