import torch
import pandas as pd
import os
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
PCA_rate = 1
norm = False
pca_model_path = 'data/models/pca_model.pth'
pca = None
if PCA_rate < 1.0:
    pca = torch.load(pca_model_path,weights_only=False)

class TensorPatientDataset(torch.utils.data.Dataset):
    def __init__(self, data_tensor, target_tensor, patient_ids):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.patient_ids = patient_ids

    def __len__(self):
        return self.data_tensor.size(0)

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index], self.patient_ids[index]

class EmbedingDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings_path, labels_path,split ='train',pca=None):
        if split == 'train':
            df0 = pd.read_excel(labels_path, sheet_name="PB").dropna(subset=["Patient"])
            self.dict_labels = {int(df0.at[i, "Patient"]): int(df0.at[i,"Récidive avant 2 ans"]) for i in range(len(df0))}
        elif split == 'test_HM':
            df1 = pd.read_excel(labels_path, sheet_name="HMN").dropna(subset=["Patient"])
            self.dict_labels = {int(df1.at[i, "Patient"]): int(df1.at[i,"Récidive avant 2 ans"]) for i in range(len(df1))}
        elif split == 'test_BJN':
            df2 = pd.read_excel(labels_path, sheet_name="BJN").dropna(subset=["Patient"])
            self.dict_labels = {int(df2.at[i, "Patient"]): int(df2.at[i,"Récidive avant 2 ans"]) for i in range(len(df2))}
        self.pt_files = {f.split('_')[0] : os.path.join(embeddings_path, f) for f in os.listdir(embeddings_path) if f.endswith('_features.pt')}
        # filter pt_files to keep only those in dict_labels
        self.pt_files = {k: v for k, v in self.pt_files.items() if int(k[:-1]) in self.dict_labels}
        self.slide_ids = list(self.pt_files.keys())
        self.pca = pca
        
    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        slide_id = self.slide_ids[idx]
        embedding = torch.load(self.pt_files[slide_id])['last_layer_embed']
        label = self.dict_labels[int(slide_id[:-1])]
        if self.pca is not None:
            embedding = torch.tensor(pca.transform(embedding.cpu()), dtype=torch.float32).to(device)
        patient_id = slide_id[:-1]
        target = torch.tensor(label, dtype=torch.float32)
        return embedding.to(device), target.to(device),patient_id

# Load the pre-trained last layer model

last_layer = torch.nn.Sequential(torch.nn.Linear(in_features=768,out_features=1)).to(device)
model_path = 'data/models/last_layer_final_1.pth' 
last_layer.load_state_dict(torch.load(model_path))

testDataset = EmbedingDataset(embeddings_path='data/features', labels_path="data/Label_slides.xlsx", split='test_BJN', pca=pca)
if norm:
    trainDataset = EmbedingDataset(embeddings_path='data/features', labels_path="data/Label_slides.xlsx", split='train', pca=pca)
    # normalize the data using RobustScaler
    from sklearn.preprocessing import RobustScaler, StandardScaler
    robust_scaler = RobustScaler()
    train_data = torch.cat([emb for emb, _,_ in trainDataset], dim=0)
    train_data = robust_scaler.fit_transform(train_data.cpu())
    # update trainDataset with normalized data
    trainDataset = torch.utils.data.TensorDataset(torch.tensor(train_data, dtype=torch.float32).to(device), torch.tensor([label for _, label,_ in trainDataset], dtype=torch.float32).to(device))

    # normalize test data using the same scaler
    test_data = torch.cat([emb for emb, _,_ in testDataset], dim=0)
    test_data = robust_scaler.transform(test_data.cpu())
    # update testDataset with normalized data
    testDataset = TensorPatientDataset(torch.tensor(test_data, dtype=torch.float32).to(device), torch.tensor([label for _, label,_ in testDataset], dtype=torch.float32).to(device), [patient_id for _, _, patient_id in testDataset])

testdataloader = torch.utils.data.DataLoader(testDataset, batch_size=1, shuffle=True)
print(len(testDataset))

last_layer.eval()
outputs_list = []
labels_list = []
patient_ids = []
with torch.no_grad():
    for embeds,labels,patient_id in testdataloader:
        outputs = last_layer(embeds)
        outputs_list.append(outputs.cpu())
        labels_list.append(labels.cpu())
        patient_ids.append(patient_id[0])

# compute a mean of the results accross patients
final_outputs_list = []
for patient_id in set(patient_ids):
    indices = [i for i, pid in enumerate(patient_ids) if pid == patient_id]
    if len(indices) > 1:
        # average outputs and labels
        output_ind = torch.stack([outputs_list[i] for i in indices])
        avg_output = torch.median(output_ind, dim=0).values
        label = labels_list[indices[0]]
    else:
        avg_output = outputs_list[indices[0]]
        label = labels_list[indices[0]]
    final_outputs_list.append((avg_output, label))


y_true = []
y_pred = []
for output, label in final_outputs_list:
    y_true.append(label.item())
    y_pred.append(output.item()>0.5)
accuracy = accuracy_score(y_true, y_pred)
print(f'Test Accuracy: {accuracy*100:.2f}%')
# compute confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred, labels=[0,1])
print('Confusion Matrix:')
print(cm)
# display confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=[0,1])
disp.plot()
plt.show()
# compute F1 score
from sklearn.metrics import f1_score
f1 = f1_score(y_true, y_pred)
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

y_scores = []
for output, label in final_outputs_list:
    y_scores.append(output.item())
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)
plt.figure()
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
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
plt.figure()
plt.plot(recall, precision, color='blue', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.show()

