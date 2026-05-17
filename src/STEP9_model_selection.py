import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, GridSearchCV,cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve

# Fixing all random seeds
SEED = 2025
np.random.seed(SEED)
random.seed(SEED)


nb_classes = 2
labels = ["Pas de Récidive", "Récidive"]
ticksx = np.arange(nb_classes) + 0.5
ticksy = np.arange(nb_classes) + 0.5


##########################

def compute_metrics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")  # Use "weighted" for multiclass
    PPV = precision_score(y_test, y_pred, average="weighted")
    Sensitivity = recall_score(y_test, y_pred, average="weighted")
    return accuracy, balanced_accuracy, f1, PPV, Sensitivity


def getWrongPatients(yhat, ytrue, patients):
    if len(np.where(yhat != ytrue)) != 0:
        return patients[np.where(yhat != ytrue)]
    else:
        print("No Misclassified Patients.")
        return np.array([0])
    
##########################################

df = pd.read_excel("data/tabs/input_dataframe_prognosis.xlsx")
df = df.sort_values(by="patient").drop("Nbre de lames", axis=1)


########################

df_pb = df.loc[df["Hôpital"] =="KB"].drop("Hôpital", axis=1)
df_hm = df.loc[df["Hôpital"] =="HM"].drop("Hôpital", axis=1)
df_bj = df.loc[df["Hôpital"] =="BJ"].drop("Hôpital", axis=1)

df_pb.shape, df_hm.shape, df_bj.shape

###################################

FINAL_COLS = [
    "Pattern expansif multinodulaire",
    "log1p_taille",
    "log1p_AFP",
    "%P",
    "%P_max",
    "NP_CntArea_norm",
    "P_CntArea_norm",
    "P_CntArea_norm_max",
    "Intra-tumoral",
    "Peri-tumoral",
    "density",
    "mean nucleus area",
    "anisocaryose",
    "nucleocyto index",
]

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
model_1 = XGBClassifier()

param_grid=[{'learning_rate':[0.1,0.3],'subsample':[0.5,1]}]

model_list.append({'model':model_1,'params':param_grid})

# 
model_1 = MLPClassifier() 

param_grid=[{'hidden_layer_sizes':[(),(10,),(100,)],'activation':['identity','tanh','relu','logistic']}]

model_list.append({'model':model_1,'params':param_grid})




for model in model_list:
    inner_cv = KFold(n_splits=5,shuffle=True)

    grid_search = GridSearchCV(estimator=model['model'],param_grid=model['params'],cv=inner_cv)
    grid_search.fit(X_train,y_train)

    print(grid_search.best_estimator_)
    print(grid_search.best_score_)