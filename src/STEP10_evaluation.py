from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# best model
model = GradientBoostingClassifier(n_estimators=10, subsample=0.5)

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

X_test_bj = df_bj[FINAL_COLS]
X_test_hm = df_hm[FINAL_COLS]
y_test_bj = df_bj["Récidive Globale"]
y_test_hm = df_hm["Récidive Globale"]

##################
model.fit(X_train,y_train)

bj_pred = model.predict(X_test_bj)
bj_score = model.predict_proba(X_test_bj)[:,1]
hm_pred = model.predict(X_test_hm)
hm_score = model.predict_proba(X_test_hm)[:,1]

####################
accuracy = accuracy_score(y_test_bj, bj_pred)
print(f'Test Accuracy: {accuracy*100:.2f}%')
# compute confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_bj, bj_pred, labels=[0,1])
print('Confusion Matrix:')
print(cm)
# display confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay
plt.subplot(1,2,1)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=[0,1])
disp.plot()
plt.subplot(1,2,2)
cm = confusion_matrix(y_test_hm, hm_pred, labels=[0,1])
print('Confusion Matrix:')
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=[0,1])
disp.plot()
plt.show()
# compute F1 score
from sklearn.metrics import f1_score
f1 = f1_score(y_test_bj, bj_pred)
print(f'F1 Score: {f1:.2f}')
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
print(f'Sensitivity: {sensitivity:.2f}')
print(f'Specificity: {specificity:.2f}')
# compute ppv and npv
if (tp + fp) == 0:
    ppv = 0.0
else:
    ppv = tp / (tp + fp)
if (tn + fn) == 0:
    npv = 0.0
else:
    npv = tn / (tn + fn)
print(f'PPV: {ppv:.2f}')
print(f'NPV: {npv:.2f}')
# show roc curve
from sklearn.metrics import roc_curve, auc
plt.subplot(1,2,1)
fpr, tpr, thresholds = roc_curve(y_test_bj, bj_score)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.subplot(1,2,2)
fpr, tpr, thresholds = roc_curve(y_test_hm, hm_score)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
# show precision-recall curve
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test_bj, bj_score)
plt.figure()
plt.plot(recall, precision, color='blue', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.show()


