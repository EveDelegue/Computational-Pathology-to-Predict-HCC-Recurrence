import torch
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import Subset
from sklearn.metrics import accuracy_score,roc_curve, auc,confusion_matrix, f1_score, precision_recall_curve
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--l1" , type=float, default=1/76, help="Regularization strength for L1 regularization")
    parser.add_argument("--step", type=str, default='final_train', help="Step to execute: 'cross_val', '73-38_train' or 'final_train'")
    parser.add_argument('--normalize', type=bool, default=True, help="Whether to normalize the data using RobustScaler")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    return args

args = parse_arguments()
regularization_type = 'L1'  # or None
lambda_reg = args.l1 # regularization strength
PCA_rate = 1
step =args.step # 'cross_val' or 'final_train'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate_last_layer(val_dataloader,last_layer):
    last_layer.eval()
    outputs_list = []
    labels_list = []
    patient_ids = []
    with torch.no_grad():
        for embeds,labels,patient_id in val_dataloader:
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
            avg_output = torch.mean(output_ind, dim=0)
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
    
    # compute confusion matrix
    
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    # compute F1 score
    f1 = f1_score(y_true, y_pred)
    
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

    # compute ppv and npv
    if (tp + fp) == 0:
        ppv = 0.0
    else:
        ppv = tp / (tp + fp)
    if (tn + fn) == 0:
        npv = 0.0
    else:
        npv = tn / (tn + fn)
    # show roc curve
    

    y_scores = []
    for output, label in final_outputs_list:
        y_scores.append(output.item())
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # show precision-recall curve
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    return {'accuracy' : accuracy,'cm' : cm, 'f1' : f1,'sensitivity' : sensitivity, 'specificity' : specificity, 'ppv' : ppv, 'npv' : npv, 'roc_auc' : roc_auc}

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
    def __init__(self, embeddings_path, labels_path,split ='train',pid=False):
        if split == 'train':
            df0 = pd.read_excel(labels_path, sheet_name="PB").dropna(subset=["Patient"])
            self.dict_labels = {int(df0.at[i, "Patient"]): int(df0.at[i,"Récidive avant 2 ans"]) for i in range(len(df0)) if int(df0.at[i,"Patient"])<90}
        elif split == 'val':
            df0 = pd.read_excel(labels_path, sheet_name="PB").dropna(subset=["Patient"])
            self.dict_labels = {int(df0.at[i, "Patient"]): int(df0.at[i,"Récidive avant 2 ans"]) for i in range(len(df0)) if int(df0.at[i,"Patient"])>=90}
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
        
    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        slide_id = self.slide_ids[idx]
        embedding = torch.load(self.pt_files[slide_id])['last_layer_embed']
        label = self.dict_labels[int(slide_id[:-1])]
        target = torch.tensor(label, dtype=torch.float32)
        patient_id = slide_id[:-1]
        return embedding.to(device), target.to(device),patient_id

T = 200

loss_fn = torch.nn.MSELoss()

if step == 'cross_val':
    trainDataset = EmbedingDataset(embeddings_path='data/features', labels_path="data/Label_slides.xlsx", split='train')
    trainDataset_2 = EmbedingDataset(embeddings_path='data/features', labels_path="data/Label_slides.xlsx", split='val')
    trainDataset = torch.utils.data.ConcatDataset([trainDataset, trainDataset_2])
    print(len(trainDataset))
    # pca path 
    if PCA_rate < 1.0:
        train_data = torch.cat([emb for emb, _ in trainDataset], dim=0)
        # fit pca on train data
        pca = PCA(n_components= PCA_rate)
        train_data_pca = pca.fit_transform(train_data.cpu())
        # update dataloaders with pca data
        trainDataset = torch.utils.data.TensorDataset(torch.tensor(train_data_pca, dtype=torch.float32).to(device), torch.tensor([label for _, label in trainDataset], dtype=torch.float32).to(device))
        traindataloader = torch.utils.data.DataLoader(trainDataset, batch_size=6, shuffle=True)
        # update last layer input features
        last_layer = torch.nn.Sequential(torch.nn.Linear(in_features=train_data_pca.shape[1], out_features=1)).to(device)
        print(f"PCA applied: reduced from 768 to {train_data_pca.shape[1]} features")
        # save pca model
        os.makedirs('data/models', exist_ok=True)
        torch.save(pca, 'data/models/pca_model.pth')

    #swa_model = torch.optim.swa_utils.AveragedModel(last_layer,multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.99))
    T = 200

    loss_fn = torch.nn.MSELoss()



    #swav_losses = []

    # Training loop for the final layer
    # cross validation setup
    global_results = []
    kfold = StratifiedKFold(n_splits = 5,shuffle=True)#KFold(n_splits=5, shuffle=True)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(trainDataset,y=[0]*len(trainDataset))):
        print(f"Fold {fold+1}")
        train_subset = Subset(trainDataset, train_idx)
        val_subset = Subset(trainDataset, val_idx)
        if args.normalize:
            # normalize train and val data using sklearn robust scaler
            train_data = torch.cat([emb for emb, _ ,_ in train_subset], dim=0)
            robust_scaler = RobustScaler()
            train_data = robust_scaler.fit_transform(train_data.cpu())
            # update train_subset with normalized data
            train_subset = TensorPatientDataset(torch.tensor(train_data, dtype=torch.float32).to(device), torch.tensor([label for _, label,_ in train_subset], dtype=torch.float32).to(device),[patient_id for _, _, patient_id in train_subset])
            # normalize val data using the same scaler
            val_data = torch.cat([emb for emb, _,_ in val_subset], dim=0)
            val_data = robust_scaler.transform(val_data.cpu())
            # update val_subset with normalized data
            val_subset = TensorPatientDataset(torch.tensor(val_data, dtype=torch.float32).to(device), torch.tensor([label for _, label,_ in val_subset], dtype=torch.float32).to(device),[patient_id for _, _, patient_id in val_subset])


        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=6, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=1, shuffle=True)

        last_layer = torch.nn.Sequential(torch.nn.Linear(in_features=768, out_features=1)).to(device)
        optimizer = torch.optim.Adam(last_layer.parameters(),lr = 1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T)
        l_losses = []
        v_losses = []
        min_cumulative_loss = float('inf')
        # train
        for epoch in range(T):  # number of epochs
            print(f"Epoch {epoch+1}")
            cumulative_loss = 0.0
            
            # train
            last_layer.train()
            for embeds,labels,_ in train_loader:
                # forward pass
                outputs = last_layer(embeds)
                loss = loss_fn(outputs.squeeze(), labels.squeeze())
                optimizer.zero_grad()
                cumulative_loss += loss.item()
                # Apply L1 regularization
                if regularization_type == 'L1':
                    l1_norm = sum(p.abs().sum() for p in last_layer.parameters())
                    loss += lambda_reg * l1_norm
                # backward pass and optimization
                loss.backward()
                optimizer.step()
                #swa_model.update_parameters(last_layer)
            l_losses.append(cumulative_loss / len(train_loader))

            print(f"Loss: {cumulative_loss / len(train_loader)}")
            scheduler.step()
            if min_cumulative_loss> cumulative_loss:
                min_cumulative_loss = cumulative_loss
                os.makedirs('data/models', exist_ok=True)
                torch.save(last_layer.state_dict(), f'data/models/last_layer_best_train_{fold}.pth')
                print("Saved best model with loss:", min_cumulative_loss/len(train_loader))
            # validation
            #last_layer.eval()
            #cumulative_loss = 0.0
            #swa_cumulative_loss = 0.0
            #with torch.no_grad():
            #    for embeds,labels,_ in val_loader:
            #        outputs = last_layer(embeds)
            #        loss = loss_fn(outputs.squeeze(), labels.squeeze())
            #        cumulative_loss += loss.item()
            #        #swa_outputs = swa_model(embeds)
            #        #swa_loss = loss_fn(swa_outputs.squeeze(), labels.squeeze())
            #        #swa_cumulative_loss += swa_loss.item()
            #    if cumulative_loss < min_cumulative_loss:
            #        min_cumulative_loss = cumulative_loss
            #        os.makedirs('data/models', exist_ok=True)
            #        #torch.save(last_layer.state_dict(), f'data/models/last_layer_best_{fold}.pth')
            #        print("Saved best model with loss:", min_cumulative_loss/len(val_loader))
            #print(f"Validation Loss: {cumulative_loss / len(val_loader)}")
            #print(f"SWA Validation Loss: {swa_cumulative_loss / len(val_loader)}")
            #v_losses.append(cumulative_loss / len(val_loader))
            #swav_losses.append(swa_cumulative_loss / len(val_loader))
        '''plt.plot(l_losses, label='Train Loss')
        plt.plot(v_losses, label='Validation Loss')
        #plt.plot(swav_losses, label='SWA Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()'''
        # validate 
        last_layer_best = torch.nn.Sequential(torch.nn.Linear(in_features=768, out_features=1)).to(device)
        if PCA_rate < 1.0 and pca is not None:
            last_layer_best = torch.nn.Sequential(torch.nn.Linear(in_features=pca.n_components_, out_features=1)).to(device)
        last_layer_best.load_state_dict(torch.load(f'data/models/last_layer_best_train_{fold}.pth'))
        print(f"Evaluation of the best model on fold {fold+1}")
        results = evaluate_last_layer(val_loader,last_layer_best) 
        print(results)
        # add results to global results
        global_results.append(results)
    # save global results to csv
    results_df = pd.DataFrame(global_results)
    os.makedirs('results/models', exist_ok=True)
    results_df.to_csv('results/models/last_layer_results.csv', index=False)


elif step == 'final_train':
    torch.manual_seed(args.seed)
    trainDataset = EmbedingDataset(embeddings_path='data/features', labels_path="data/Label_slides.xlsx", split='train')
    trainDataset_2 = EmbedingDataset(embeddings_path='data/features', labels_path="data/Label_slides.xlsx", split='val')
    trainDataset = torch.utils.data.ConcatDataset([trainDataset, trainDataset_2])
    # pick 10% for validation
    val_size = int(0.1 * len(trainDataset))
    train_size = len(trainDataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(trainDataset, [train_size, val_size])
    if args.normalize:
        # normalize train and val data using sklearn robust scaler
        train_data = torch.cat([emb for emb, _,_ in train_subset], dim=0)
        robust_scaler = RobustScaler()
        train_data = robust_scaler.fit_transform(train_data.cpu())
        # update train_subset with normalized data
        train_subset = TensorPatientDataset(torch.tensor(train_data, dtype=torch.float32).to(device), torch.tensor([label for _, label,_ in train_subset], dtype=torch.float32).to(device),[patient_id for _, _, patient_id in train_subset])
        # normalize val data using the same scaler
        val_data = torch.cat([emb for emb, _,_ in val_subset], dim=0)
        val_data = robust_scaler.transform(val_data.cpu())
        # update val_subset with normalized data
        val_subset = TensorPatientDataset(torch.tensor(val_data, dtype=torch.float32).to(device), torch.tensor([label for _, label,_ in val_subset], dtype=torch.float32).to(device),[patient_id for _, _, patient_id in val_subset])

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=6, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=1, shuffle=True)
    #valDataset = EmbedingDataset(embeddings_path='/media/eve/My Passport/data_hcc/features', labels_path="/home/eve/Downloads/Tableau 1 pour Eve(1).xlsx", split='val')
    #val_loader = torch.utils.data.DataLoader(valDataset, batch_size=1, shuffle=True)

    last_layer = torch.nn.Sequential(torch.nn.Linear(in_features=768, out_features=1)).to(device)
    optimizer = torch.optim.Adam(last_layer.parameters(),lr = 1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T)
    l_losses = []
    v_losses = []
    min_cumulative_loss = float('inf')
    # train
    for epoch in range(T):  # number of epochs
        print(f"Epoch {epoch+1}")
        cumulative_loss = 0.0
        
        # train
        last_layer.train()
        for embeds,labels,_ in train_loader:
            # forward pass
            outputs = last_layer(embeds)
            loss = loss_fn(outputs.squeeze(), labels.squeeze())
            optimizer.zero_grad()
            cumulative_loss += loss.item()
            # Apply L1 regularization
            if regularization_type == 'L1':
                l1_norm = sum(p.abs().sum() for p in last_layer.parameters())
                loss += lambda_reg * l1_norm
            # backward pass and optimization
            loss.backward()
            optimizer.step()
            #swa_model.update_parameters(last_layer)
        l_losses.append(cumulative_loss / len(train_loader))

        print(f"Loss: {cumulative_loss / len(train_loader)}")
        scheduler.step()
        # validation
        last_layer.eval()
        cumulative_loss = 0.0
        #swa_cumulative_loss = 0.0
        with torch.no_grad():
            for embeds,labels,_ in val_loader:
                    outputs = last_layer(embeds)
                    loss = loss_fn(outputs.squeeze(), labels.squeeze())
                    cumulative_loss += loss.item()
                    #swa_outputs = swa_model(embeds)
                    #swa_loss = loss_fn(swa_outputs.squeeze(), labels.squeeze())
                    #swa_cumulative_loss += swa_loss.item()
            if cumulative_loss < min_cumulative_loss:
                    min_cumulative_loss = cumulative_loss
                    os.makedirs('data/models', exist_ok=True)
                    torch.save(last_layer.state_dict(), f'data/models/last_layer_final.pth')
                    print("Saved best model with loss:", min_cumulative_loss/len(val_loader))
            print(f"Validation Loss: {cumulative_loss / len(val_loader)}")
            #print(f"SWA Validation Loss: {swa_cumulative_loss / len(val_loader)}")
            v_losses.append(cumulative_loss / len(val_loader))
        # save final model
    #os.makedirs('data/models', exist_ok=True) 
    #torch.save(last_layer.state_dict(), f'data/models/last_layer_final.pth')

    plt.plot(l_losses, label='Train Loss')
    plt.plot(v_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss and Validation Loss')
    plt.legend()
    plt.show()

## final training on the 73-38 train dataset
elif step == '73-38_train':
    torch.manual_seed(args.seed)
    train_subset = EmbedingDataset(embeddings_path='data/features', labels_path="data/Label_slides.xlsx", split='train')
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=6, shuffle=True)
    val_subset = EmbedingDataset(embeddings_path='data/features', labels_path="data/Label_slides.xlsx", split='val')
    if args.normalize:
        # normalize train and val data using sklearn robust scaler
        train_data = torch.cat([emb for emb, _ ,_ in train_subset], dim=0)
        robust_scaler = RobustScaler()
        train_data = robust_scaler.fit_transform(train_data.cpu())
        # update train_subset with normalized data
        train_subset = TensorPatientDataset(torch.tensor(train_data, dtype=torch.float32).to(device), torch.tensor([label for _, label,_ in train_subset], dtype=torch.float32).to(device),[patient_id for _, _, patient_id in train_subset])
        # normalize val data using the same scaler
        val_data = torch.cat([emb for emb,_, _ in val_subset], dim=0)
        val_data = robust_scaler.transform(val_data.cpu())
        # update val_subset with normalized data
        val_subset = TensorPatientDataset(torch.tensor(val_data, dtype=torch.float32).to(device), torch.tensor([label for _, label,_ in val_subset], dtype=torch.float32).to(device),[patient_id for _, _, patient_id in val_subset])

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=6, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=1, shuffle=True)
    #valDataset = EmbedingDataset(embeddings_path='/media/eve/My Passport/data_hcc/features', labels_path="/home/eve/Downloads/Tableau 1 pour Eve(1).xlsx", split='val')
    #val_loader = torch.utils.data.DataLoader(valDataset, batch_size=1, shuffle=True)

    last_layer = torch.nn.Sequential(torch.nn.Linear(in_features=768, out_features=1)).to(device)
    optimizer = torch.optim.Adam(last_layer.parameters(),lr = 1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T)
    l_losses = []
    v_losses = []
    min_cumulative_loss = float('inf')
    # train
    for epoch in range(T):  # number of epochs
        print(f"Epoch {epoch+1}")
        cumulative_loss = 0.0
        
        # train
        last_layer.train()
        for embeds,labels,_ in train_loader:
            # forward pass
            outputs = last_layer(embeds)
            loss = loss_fn(outputs.squeeze(), labels.squeeze())
            optimizer.zero_grad()
            cumulative_loss += loss.item()
            # Apply L1 regularization
            if regularization_type == 'L1':
                l1_norm = sum(p.abs().sum() for p in last_layer.parameters())
                loss += lambda_reg * l1_norm
            # backward pass and optimization
            loss.backward()
            optimizer.step()
            #swa_model.update_parameters(last_layer)
        l_losses.append(cumulative_loss / len(train_loader))

        print(f"Loss: {cumulative_loss / len(train_loader)}")
        scheduler.step()
        # validation
        last_layer.eval()
        cumulative_loss = 0.0
        #swa_cumulative_loss = 0.0
        with torch.no_grad():
            for embeds,labels,_ in val_loader:
                    outputs = last_layer(embeds)
                    loss = loss_fn(outputs.squeeze(), labels.squeeze())
                    cumulative_loss += loss.item()
                    #swa_outputs = swa_model(embeds)
                    #swa_loss = loss_fn(swa_outputs.squeeze(), labels.squeeze())
                    #swa_cumulative_loss += swa_loss.item()
            if cumulative_loss < min_cumulative_loss:
                    min_cumulative_loss = cumulative_loss
                    os.makedirs('data/models', exist_ok=True)
                    torch.save(last_layer.state_dict(), f'data/models/last_layer_73-38.pth')
                    print("Saved best model with loss:", min_cumulative_loss/len(val_loader))
            print(f"Validation Loss: {cumulative_loss / len(val_loader)}")
            #print(f"SWA Validation Loss: {swa_cumulative_loss / len(val_loader)}")
            v_losses.append(cumulative_loss / len(val_loader))
        # save final model
    #os.makedirs('data/models', exist_ok=True) 
    #torch.save(last_layer.state_dict(), f'data/models/last_layer_final.pth')

    plt.plot(l_losses, label='Train Loss')
    plt.plot(v_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss and Validation Loss')
    plt.legend()
    plt.show()
    

