import os
import yaml
import torch
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, StratifiedShuffleSplit, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

features_paths = config["paths"]["pth_to_features"]
labels_path = 'data/Label_slides.xlsx'
data = 'fm'

if data == 'fm':
    # load data and labels
    df0 = pd.read_excel(labels_path, sheet_name="PB").dropna(subset=["Patient"])
    features_list = []
    labels_list = []
    patient_id_list = []
    # select patients 
    list_train_patients = os.listdir(features_paths)
    # loop through patients' slides
    for feature_file in list_train_patients:
        if feature_file.endswith('_features.pt'):
            # find the slide
            slide_id = feature_file.split('_')[0]
            feature_path = os.path.join(features_paths, feature_file)
            print(f"Slide ID: {slide_id}, Feature Path: {feature_path}")
            patient_id = int(slide_id[:-1])
            # compare to the list of required patients
            patient_list = df0["Patient"].tolist()
            if patient_id in patient_list:
                # load features from pt file
                features = torch.load(feature_path)['last_layer_embed']
                # add to the list
                features_list.append(features.squeeze())
                # read label from excel file
                label = int(df0[df0["Patient"] == int(slide_id[:-1])]["Récidive avant 2 ans"].values[0]) 
                labels_list.append(label)
                # read patient id from excel file
                patient_id = int(df0[df0["Patient"] == int(slide_id[:-1])]["Patient"].values[0])
                patient_id_list.append(patient_id)

    # convert to np array
    features_array = torch.stack(features_list, dim=0).cpu().numpy()
    labels_array = np.array(labels_list)
    patient_id_array = np.array(patient_id_list)
    # split between train and test for the first experiment
    train_0_array = features_array[patient_id_array < 90]
    test_0_array = features_array[patient_id_array >= 90]
    train_0_labels_array = labels_array[patient_id_array < 90]
    test_0_labels_array = labels_array[patient_id_array >= 90]

    #same for HM external validation
    HM_test_features_list = []
    HM_test_labels_list = []
    # read the right sheet
    df1 = pd.read_excel(labels_path, sheet_name="HMN").dropna(subset=["Patient"])
    list_train_patients = os.listdir(features_paths)
    for feature_file in list_train_patients:
        if feature_file.endswith('_features.pt'):
            # get slide id
            slide_id = feature_file.split('_')[0]
            feature_path = os.path.join(features_paths, feature_file)
            print(f"Slide ID: {slide_id}, Feature Path: {feature_path}")
            patient_id = int(slide_id[:-1])
            # compare to the list of patients in excel
            patient_list = df1["Patient"].tolist()
            if patient_id in patient_list:
                # load features from pt file
                features = torch.load(feature_path)['last_layer_embed']
                HM_test_features_list.append(features.squeeze())
                # read label from excel file
                label = int(df1[df1["Patient"] == int(slide_id[:-1])]["Récidive avant 2 ans"].values[0]) 
                HM_test_labels_list.append(label)

                # no need for patient id here

    # convert to np array
    HM_test_features_array = torch.stack(HM_test_features_list, dim=0).cpu().numpy()
    HM_test_labels_array = np.array(HM_test_labels_list)

    #same for BJN external validation
    BJN_test_features_list = []
    BJN_test_labels_list = []
    df2 = pd.read_excel(labels_path, sheet_name="BJN").dropna(subset=["Patient"])
    list_train_patients = os.listdir(features_paths)
    for feature_file in list_train_patients:
        if feature_file.endswith('_features.pt'):
            slide_id = feature_file.split('_')[0]
            feature_path = os.path.join(features_paths, feature_file)
            print(f"Slide ID: {slide_id}, Feature Path: {feature_path}")
            patient_id = int(slide_id[:-1])
            patient_list = df2["Patient"].tolist()
            if patient_id in patient_list:
                # load features from pt file
                features = torch.load(feature_path)['last_layer_embed']
                BJN_test_features_list.append(features.squeeze())
                # read label from excel file
                label = int(df2[df2["Patient"] == int(slide_id[:-1])]["Récidive avant 2 ans"].values[0]) 
                BJN_test_labels_list.append(label)

    # convert to np array
    BJN_test_features_array = torch.stack(BJN_test_features_list, dim=0).cpu().numpy()
    BJN_test_labels_array = np.array(BJN_test_labels_list)

else:
    df = pd.read_excel("data/tabs/input_dataframe_prognosis.xlsx")
    df_kb = df.loc[
        df["patient"].between(1, 110)
        | df["patient"].between(213, 222)
        | df["patient"].between(253, 260)
    ]

    cols_to_scale = [
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



    # convert to np array
    features_array = df_kb[cols_to_scale]
    labels_array = df_kb["Récidive Globale"]
    train_0_array = features_array[df_kb["patient"] < 90]
    test_0_array = features_array[df_kb["patient"] >= 90]
    train_0_labels_array = labels_array[df_kb["patient"] < 90]
    test_0_labels_array = labels_array[df_kb["patient"] >= 90]

    #same for HM external validation
    df_hm = df.loc[df["patient"].between(111, 160)]
    # patients from BJ
    df_bj = df.loc[df["patient"].between(161, 212) | df["patient"].between(223, 252)]


    # convert to np array
    HM_test_features_array = df_hm[cols_to_scale]
    HM_test_labels_array = df_hm["Récidive Globale"]

    #same for BJN external validation

    # convert to np array
    BJN_test_features_array = df_bj[cols_to_scale]
    BJN_test_labels_array = df_bj["Récidive Globale"]

# create a list of models 
models = []
names = ['SVM', 'Random Forest', 'Gradient Boosting', 'AdaBoost', 'XGBoost', 'Decision Tree', 'MLP']
# try a gaussian svm
svm  = make_pipeline(RobustScaler(), SVC(kernel="rbf", probability=True, C=1.5, gamma="scale"))
models.append(svm)
# try a CatBoost
catboost = CatBoostClassifier()
#models.append(catboost)
# try a random forest
rf = make_pipeline(RobustScaler(), RandomForestClassifier()) 
models.append(rf)
# try a gradient boosting
gb = make_pipeline(RobustScaler(), GradientBoostingClassifier()) 
models.append(gb)
# try a adaboost
ada = make_pipeline(RobustScaler(), AdaBoostClassifier() )
models.append(ada)
# try a xgboost
xgb = make_pipeline(RobustScaler(), XGBClassifier())
models.append(xgb)
# try a decision tree
dt = make_pipeline(RobustScaler(), DecisionTreeClassifier() )
models.append(dt)
# try a mlp
mlp = make_pipeline(RobustScaler(),MLPClassifier(hidden_layer_sizes=(), activation='identity') )
models.append(mlp)


results_dict = {}
for i in range(len(models)):
    model = models[i]
    model_name = names[i]
    # cross validate the model
    cv = KFold(n_splits=5, shuffle=True, random_state=42) # this is not good for the foundation model, as different slides of the same patient can be in the train and the test

    F1_scores = cross_val_score(model, features_array, labels_array, cv=cv,scoring='f1')
    accuracies = cross_val_score(model, features_array, labels_array, cv=cv,scoring='accuracy')
    # compute mean and std of F1 scores and accuracies
    mean_f1 = np.mean(F1_scores)
    std_f1 = np.std(F1_scores) 
    mean_acc = np.mean(accuracies) 
    std_acc = np.std(accuracies) 
    
    print(f"Model: {model_name}, Mean F1 Score: {mean_f1:.4f} (+/- {std_f1:.4f}), Mean Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})")

    # 73 - 38 split train test
    model.fit(train_0_array, train_0_labels_array)
    test_0_acc = model.score(test_0_array, test_0_labels_array)
    test_0_pred = model.predict(test_0_array)
    test_0_f1 = f1_score(test_0_labels_array, test_0_pred)
    # test on HM ad BJN 
    test_0_HM_acc = model.score(HM_test_features_array, HM_test_labels_array)
    test_0_HM_pred = model.predict(HM_test_features_array)
    test_0_HM_f1 = f1_score(HM_test_labels_array, test_0_HM_pred)
    test_0_BJN_acc = model.score(BJN_test_features_array, BJN_test_labels_array)
    test_0_BJN_pred = model.predict(BJN_test_features_array)
    test_0_BJN_f1 = f1_score(BJN_test_labels_array, test_0_BJN_pred)
    torch.save(model, f'data/models/{model_name}_classifier_73_{data}.pth')
    # test on test set
    model.fit(features_array, labels_array) 
    HM_test_acc = model.score(HM_test_features_array, HM_test_labels_array) 
    HM_test_pred = model.predict(HM_test_features_array)
    HM_test_f1 = f1_score(HM_test_labels_array, HM_test_pred)
    BJN_test_acc = model.score(BJN_test_features_array, BJN_test_labels_array) 
    BJN_test_pred = model.predict(BJN_test_features_array)
    BJN_test_f1 = f1_score(BJN_test_labels_array, BJN_test_pred)

    # save the model
    os.makedirs('data/models', exist_ok=True)
    torch.save(model, f'data/models/{model_name}_classifier_{data}.pth')
    # add results to the dict 
    results_dict[model_name] = [test_0_f1,test_0_acc,test_0_HM_f1,test_0_HM_acc,test_0_BJN_f1,test_0_BJN_acc , mean_f1,std_f1,mean_acc, std_acc, HM_test_acc, HM_test_f1, BJN_test_acc, BJN_test_f1]


# save results to a csv file
results_df = pd.DataFrame.from_dict(results_dict, orient='index', columns=['first split F1','first split Accuracy', 'first split HM F1','first split HM Acc',  'first split BJN F1',  'first split BJN Acc',    'Mean F1 Score', 'Std F1 Score', 'Mean Accuracy', 'Std Accuracy', 'HM Test Accuracy', 'HM Test F1 Score', 'BJN Test Accuracy', 'BJN Test F1 Score']) 
os.makedirs('results/models', exist_ok=True)
results_df.to_csv(f'results/models/classifier_results_{data}.csv')


