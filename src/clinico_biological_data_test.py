import pandas as pd
import os
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
import shap

import yaml


with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

clinical_data = os.path.join(config["paths"]["pth_to_tab"],"Récidive_CHC.xlsx")

# read file in a panda dataframe
df = pd.read_excel(clinical_data,sheet_name=None)


#input_cols = ['Âge','Genre masculin', 'Nombre de nodules',
#                 'Expansif multinodulaire', 'Taille (cm)', 'Valeur exacte AFP pré-opératoire','log AFP','log taille'] #0.64 accuracy

input_cols = [ 'Nombre de nodules','log taille']#,                 'Expansif multinodulaire','Âge','Genre masculin','log AFP']

output_col = 'Récidive avant 2 ans'

#######

X_train = df['PB'][input_cols]
y_train = df['PB'][output_col]

X_train.to_excel('X_train.xlsx')

#hist = X_train.hist()
#plt.show()

####

X_test_HM = df['HMN'][input_cols]
X_test_BJ = df['BJN'][input_cols]
y_test_HM = df['HMN'][output_col]
y_test_BJ = df['BJN'][output_col]

final_model = AdaBoostClassifier(learning_rate=0.1,n_estimators=100)

final_model.fit(X_train,y_train)
#results_BJ = final_model.predict(X_test_BJ)
#results_HM = final_model.predict(X_test_HM)

#print(f'model : {final_model}, score HM : {final_model.score(X_test_HM,y_test_HM)}, score BJN : {final_model.score(X_test_BJ,y_test_BJ)}')


explainer = shap.KernelExplainer(final_model.predict,X_train)
shap_values = explainer(X_train)
shap.plots.beeswarm(shap_values)
