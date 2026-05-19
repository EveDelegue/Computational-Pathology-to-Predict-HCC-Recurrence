from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
import random
import argparse
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler,MinMaxScaler,StandardScaler,SplineTransformer
from sklearn.svm import SVC

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups",type=list[int],default=[1,2,3,4])
    args = parser.parse_args()
    return args



# Fixing all random seeds
SEED = 2025
np.random.seed(SEED)
random.seed(SEED)

# best model
model = Pipeline(steps=[('scaler', RobustScaler()), ('preprocess', None),
                ('classifier', SVC(kernel='rbf',C=1.5,probability=True))])

df = pd.read_excel("data/tabs/input_dataframe_prognosis.xlsx")
df = df.sort_values(by="patient").drop("Nbre de lames", axis=1)


########################

df_pb = df.loc[df["Hôpital"] =="PB"].drop("Hôpital", axis=1)
df_hm = df.loc[df["Hôpital"] =="HM"].drop("Hôpital", axis=1)
df_bj = df.loc[df["Hôpital"] =="BJ"].drop("Hôpital", axis=1)


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

X_test_bj = df_bj[FINAL_COLS]
X_test_hm = df_hm[FINAL_COLS]
X_test = pd.concat([X_test_bj,X_test_hm])

y_test_bj = df_bj["Récidive Globale"]
y_test_hm = df_hm["Récidive Globale"]
y_test = pd.concat([y_test_bj,y_test_hm])
##################
model.fit(X_train,y_train)

from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.calibration import calibration_curve, CalibrationDisplay

bj_pred = model.predict(X_test_bj)
bj_score = model.predict_proba(X_test_bj)[:,1]
hm_pred = model.predict(X_test_hm)
hm_score = model.predict_proba(X_test_hm)[:,1]
full_pred = np.concat([bj_pred,hm_pred])
full_score = np.concat([bj_score,hm_score])


prob_true, prob_pred = calibration_curve(y_test, full_score, n_bins=10)
disp = CalibrationDisplay(prob_true, prob_pred, full_score)
disp.plot()
plt.show()

####################
accuracy = accuracy_score(y_test_bj, bj_pred)
accuracy_hm = accuracy_score(y_test_hm, hm_pred)
print(f'Test Accuracy: BJ => {accuracy*100:.2f}% HM => {accuracy_hm*100:.2f}%')
# compute F1 score
from sklearn.metrics import f1_score
f1 = f1_score(y_test_bj, bj_pred)
f1_hm = f1_score(y_test_hm, hm_pred)
print(f'F1 Score: BJ => {f1:.2f} HM => {f1_hm:.2f}')
# compute confusion matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_bj, bj_pred, labels=[0,1])
print('Confusion Matrix BJ :')
print(cm)
# display confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=[0,1])
disp.plot()
plt.show()
cm_hm = confusion_matrix(y_test_hm, hm_pred, labels=[0,1])
print('Confusion Matrix HM :')
print(cm_hm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_hm,
                              display_labels=[0,1])
disp.plot()
plt.show()

# compute sensitivity and specificity
tn, fp, fn, tp = cm.ravel().tolist()
if (tp + fn) == 0:
    sensitivity = 0.0
else:
    sensitivity = tp / (tp + fn)
if (tn + fp) == 0:
    specificity = 0.0
else:
    specificity = tn / (tn + fp)
print(f'Sensitivity BJ : {sensitivity:.2f}')
print(f'Specificity BJ : {specificity:.2f}')

# compute ppv and npv
if (tp + fp) == 0:
    ppv = 0.0
else:
    ppv = tp / (tp + fp)
if (tn + fn) == 0:
    npv = 0.0
else:
    npv = tn / (tn + fn)
print(f'PPV BJ : {ppv:.2f}')
print(f'NPV BJ : {npv:.2f}')

# compute sensitivity and specificity
tn, fp, fn, tp = cm_hm.ravel().tolist()
if (tp + fn) == 0:
    sensitivity = 0.0
else:
    sensitivity = tp / (tp + fn)
if (tn + fp) == 0:
    specificity = 0.0
else:
    specificity = tn / (tn + fp)
print(f'Sensitivity HM : {sensitivity:.2f}')
print(f'Specificity HM : {specificity:.2f}')
# compute ppv and npv
if (tp + fp) == 0:
    ppv = 0.0
else:
    ppv = tp / (tp + fp)
if (tn + fn) == 0:
    npv = 0.0
else:
    npv = tn / (tn + fn)
print(f'PPV HM : {ppv:.2f}')
print(f'NPV HM : {npv:.2f}')
# show roc curve
from sklearn.metrics import roc_curve, auc
plt.figure()
plt.subplot(1,2,1)
# BJ
fpr, tpr, thresholds = roc_curve(y_test_bj, bj_score)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic BJ ')
plt.legend(loc="lower right")
plt.subplot(1,2,2)
# HM
fpr, tpr, thresholds = roc_curve(y_test_hm, hm_score)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic HM ')
plt.legend(loc="lower right")
plt.show()
# show precision-recall curve
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test_bj, bj_score)
plt.figure()
plt.subplot(1,2,1)
# BJ
plt.plot(recall, precision, color='blue', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve BJ')
plt.subplot(1,2,2)
# HM 
precision, recall, thresholds = precision_recall_curve(y_test_hm, hm_score)
plt.plot(recall, precision, color='blue', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve HM')
plt.show()
######
import shap

explainer = shap.Explainer(model.predict,X_test)
shap_values = explainer(X_test)

shap.plots.beeswarm(shap_values)

plt.show()


#######

# accuracy en fonction du nb de patients dans le test


