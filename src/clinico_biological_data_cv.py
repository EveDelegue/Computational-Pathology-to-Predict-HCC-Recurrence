import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import ElasticNet, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import RobustScaler,SplineTransformer
from tabpfn_client import TabPFNClassifier, set_access_token
from tabfm import tabfm_v1_0_0_pytorch,TabFMClassifier
import warnings
warnings.filterwarnings("ignore")


set_access_token("tabpfn_sk_ZrEQfq7IBtBdWM_DJCwy4BIC4vfOZKLZhtdM3x3NaDg")

import yaml

with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

clinical_data = os.path.join(config["paths"]["pth_to_tab"],"Récidive_CHC.xlsx")

# read file in a panda dataframe
df = pd.read_excel(clinical_data,sheet_name=None)


#input_cols = ['Âge','Genre masculin', 'Nombre de nodules',
#                 'Expansif multinodulaire', 'Taille (cm)', 'Valeur exacte AFP pré-opératoire','log AFP','log taille'] #0.64 accuracy

input_cols = ['log taille','Nombre de nodules']#['Âge','Genre masculin', 'Nombre de nodules',
             #    'Nodule satellite','log AFP','log taille']

output_col = 'Récidive avant 2 ans'

#######

X_train = df['PB'][input_cols]
y_train = df['PB'][output_col]

X_train.to_excel('X_train.xlsx')

#hist = X_train.hist()
#plt.show()

####

model_list = []
# parameter grid
'''
# SVM
model_1 = SVC()

param_grid = [
  {'C': [0.5,1, 5], 'kernel': ['linear']},
  {'C': [0.5,1, 5], 'gamma': ['scale','auto'], 'kernel': ['rbf','sigmoid']},
 ]

model_list.append({'model':model_1,'params':param_grid})

# catboost
model = CatBoostClassifier(verbose=False)

param_grid=[{'iterations':[10,100],'depth':[2,5,10],'loss_function':["Logloss","CrossEntropy"]}]

model_list.append({'model':model,'params':param_grid})

# random decision tree
model_1 = DecisionTreeClassifier()

param_grid=[{'criterion':["gini","entropy","log_loss"]}]

model_list.append({'model':model_1,'params':param_grid})

# random forest
model_1 = RandomForestClassifier()

param_grid=[{'n_estimators':[10,20,50,100]}]

model_list.append({'model':model_1,'params':param_grid})

# gradient boosting
model_1 = GradientBoostingClassifier()

param_grid=[{'n_estimators':[10,50,100],"learning_rate":[0.1,0.2,0.01],"subsample":[0.5,1]}]

model_list.append({'model':model_1,'params':param_grid})

# AdaBoost
model_1 = AdaBoostClassifier()

param_grid=[{'n_estimators':[10,50,100],"learning_rate":[0.1,0.2,0.01]}]

model_list.append({'model':model_1,'params':param_grid})

# xgb classifier
model_1 = XGBClassifier(verbosity=0)

param_grid=[{'learning_rate':[0.1,0.3],'subsample':[0.5,1]}]

model_list.append({'model':model_1,'params':param_grid})

# mlp
model_1 = MLPClassifier() 

param_grid=[{'hidden_layer_sizes':[(),(10,),(100,)],'activation':['identity','tanh','relu','logistic']}]

model_list.append({'model':model_1,'params':param_grid})

# ElasticNet

model_1 = ElasticNet() 

param_grid=[{'alpha':[0.1,1,10],'l1_ratio':[0.5,1]}]

model_list.append({'model':model_1,'params':param_grid})

# Ridge classifier

model_1 = RidgeClassifier() 

param_grid=[{'alpha':[0.001,0.1,1,10]}]

model_list.append({'model':model_1,'params':param_grid})

# tabPFN
model_1 = TabPFNClassifier()
param_grid=[{'softmax_temperature':[0.7,0.9,1.2]}]

model_list.append({'model':model_1,'params':param_grid})
'''
# tabFM
model = tabfm_v1_0_0_pytorch.load()
model_1 = TabFMClassifier(model=model)
param_grid=[{}]

model_list.append({'model':model_1,'params':param_grid})


for model in model_list:
    inner_cv = KFold(n_splits=5,shuffle=True)

    new_model = Pipeline([('scaler','passthrough'),('preprocess','passthrough'),( 'classifier',model["model"]) ])
    new_params = [{'classifier__'+k:v for k,v in parametres.items()}|{'scaler':[None,RobustScaler()]}|{'preprocess':[None,SplineTransformer()]} 
                  for parametres in model["params"]]

    grid_search = GridSearchCV(estimator=new_model,param_grid=new_params,cv=inner_cv,scoring='accuracy')
    grid_search.fit(X_train,y_train)

    results = grid_search.cv_results_
    best_model = np.argmin(results["rank_test_score"])
    print(f'model : {model} params : {results["params"][best_model]} \n accuracy : {results["mean_test_score"][best_model]}, std : {results["std_test_score"][best_model]}, score time : {results["mean_score_time"][best_model]}+- {results["std_score_time"][best_model]} ')
