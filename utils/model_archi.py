import torch
import torch.nn as nn
import torchvision.models as models

class IndepResNetModel(nn.Module):
    def __init__(self):
        super(IndepResNetModel,self).__init__()
        self.resnet1=models.resnet34(pretrained=False)
        self.resnet2=models.resnet34(pretrained=False)
        self.resnet3=models.resnet34(pretrained=False)
        
        for resnet in [self.resnet1, self.resnet2, self.resnet3]:
            for param in resnet.parameters():
                param.requires_grad=True
        
        self.resnet1.fc=nn.Linear(512, 16)
        self.resnet2.fc=nn.Linear(512, 16)
        self.resnet3.fc=nn.Linear(512, 16)
        self.fc2=nn.Linear(16*3, 32)
        self.fc3=nn.Linear(32, 16)
        self.fc4=nn.Linear(16,3)
        
        self.bn11 = nn.BatchNorm1d(num_features=16)
        self.bn12 = nn.BatchNorm1d(num_features=16)
        self.bn13 = nn.BatchNorm1d(num_features=16)
        self.bn2 = nn.BatchNorm1d(num_features=32)
        self.bn3 = nn.BatchNorm1d(num_features=16)

        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(0.6)

    def forward(self,r1, r2, r3):
        x1=self.bn11(self.resnet1(r1))
        x2=self.bn12(self.resnet1(r2))
        x3=self.bn13(self.resnet1(r3))
        x=torch.cat((x1,x2,x3), dim=1)
        x=self.relu(self.bn2(self.fc2(x)))
        x=self.dropout(x)
        x=self.relu(self.bn3(self.fc3(x)))
        x=self.dropout(x)
        out=self.fc4(x)
        return out
