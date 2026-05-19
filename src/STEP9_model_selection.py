import pandas as pd
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, GridSearchCV,cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import ElasticNet, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import PolynomialFeatures
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.preprocessing import RobustScaler,MinMaxScaler,StandardScaler,SplineTransformer,PolynomialFeatures
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
import argparse
from sklearn.calibration import calibration_curve

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups",type=list[int],default=[1,2,3,4])
    args = parser.parse_args()
    return args


# Fixing all random seeds
SEED = 2025
np.random.seed(SEED)
random.seed(SEED)


nb_classes = 2
labels = ["Pas de Récidive", "Récidive"]
    
##########################################

df = pd.read_excel("data/tabs/input_dataframe_prognosis.xlsx")
df = df.sort_values(by="patient")

########################

df_pb = df.loc[df["Hôpital"] =="PB"]
df_hm = df.loc[df["Hôpital"] =="HM"]
df_bj = df.loc[df["Hôpital"] =="BJ"]

###################################

args = parse_arguments()
groups = args.groups

FINAL_COLS = []

if 1 in groups:
    FINAL_COLS.extend(["log1p_taille","log1p_AFP",
                       "Expansif multinodulaire","Nombre de nodules"])
if 2 in groups:
    FINAL_COLS.extend(["%P",
    "%P_max","NP_CntArea_norm",
    "P_CntArea_norm",
    "P_CntArea_norm_max"])
if 3 in groups:
    FINAL_COLS.extend(["density",
    "mean nucleus area",
    "anisocaryose",
    "nucleocyto index"])
if 4 in groups:
    FINAL_COLS.extend(["intra-tumoral",
    "peri-tumoral"])


#######

X_train = df_pb[FINAL_COLS]
y_train = df_pb["Récidive Globale"]

####
model_list = []
# parameter grid

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


for model in model_list:
    inner_cv = KFold(n_splits=5,shuffle=True)

    new_model = Pipeline([('scaler','passthrough'),('preprocess','passthrough'),( 'classifier',model["model"]) ])
    new_params = [{'classifier__'+k:v for k,v in parametres.items()}|{'scaler':[None,RobustScaler()]}|{'preprocess':[None,SplineTransformer()]} 
                  for parametres in model["params"]]

    grid_search = GridSearchCV(estimator=new_model,param_grid=new_params,cv=inner_cv)
    grid_search.fit(X_train,y_train)

    print(grid_search.best_estimator_)
    print(grid_search.best_score_)