import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
import argparse
import torch.utils.data as Data
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from antiberty import AntiBERTyRunner
from torch.utils.data import random_split
from sklearn import metrics
from tqdm import tqdm 
import random
antiberty = AntiBERTyRunner()

class Input(Dataset):
    def __init__(self,data):
        self.data=data
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        H = antiberty.embed([self.data[idx][0]])[0][0,:]
        L = antiberty.embed([self.data[idx][1]])[0][0,:]
        H_L=torch.cat((H, L), dim=0).to(torch.device('cpu'))
        label=self.data[idx][2]
        if label==1:
            weight=1/len1
        if label==0:
            weight=1/len2
        sample = {'input': H_L, 'label': label,'weight':weight}
        return sample


def get_data():
    data=[]

    with open('/public2022/tanwenchong/OAS/thera.fasta','r') as f:
        label=1
        while True:
            line1 = f.readline().strip()  
            line2 = f.readline().strip()
            line3 = f.readline().strip()
            line4 = f.readline().strip()             
            if not line1 or not line2 or not line3 or not line4:
                break
            #print(line2,line4)
            data.append([line2,line4,label])
    len1=len(data)
    filt=[]
    label=0
    with open ('/public2022/tanwenchong/OAS/result2','r') as f:
        lines=f.readlines()
        for line in lines:
            if line.rstrip() != '' and line[0] !='>':
                filt.append(int(line.rstrip()))
    filt=tuple(filt)
    print(len(filt))
    #filt = random.sample(range(0, 1954078), 6500)
    antibody=[]
    with open('/public2022/tanwenchong/OAS/pair.fasta', "r") as fasta_file:
        lines=fasta_file.readlines()
        for i in filt:
            seq_id=lines[i*4][1:]
            seq_id,_=seq_id.split('_')
            antibody.append(lines[i*4+1][:-2])
            antibody.append(lines[i*4+3][:-2])
            
            antibody.append(label)
            data.append(antibody)
            antibody=[]    
    len2=len(data)-len1
    return data,len1,len2

data,len1,len2=get_data()

dataset=Input(data)
class MLP(nn.Module):
    """ a simple 4-layer MLP """

    def __init__(self, nin, nout, nh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nout),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x= self.net(x)
        x=self.sigmoid(x)
        return x.squeeze(1)

num_train = int(0.9 * len(dataset))
num_val = len(dataset) - num_train 
train_dataset, valid_dataset = random_split(dataset, [num_train, num_val])
print(train_dataset[0]['input'].shape,train_dataset[0]['weight'])
weight=[]
for i in tqdm(range(len(train_dataset))):
    weight.append(train_dataset[i]['weight'])
print(weight)
batch_size=16
model=MLP(1024,1,128)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sampler = WeightedRandomSampler(weight, num_samples=len(weight), replacement=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
valid_loader = DataLoader(valid_dataset,batch_size=batch_size,shuffle=False)
model.to(device)
criterion = torch.nn.BCELoss()
learning_rate=5e-5
epochs=20
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,weight_decay=1e-16)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
def train(model,optimizer,loader,criterion):
    total_loss=0
    model.train()
    for batch in tqdm(loader):
        pred=model(batch['input'].to(device))
        loss = criterion(pred,batch['label'].float().to(device))
        total_loss+=loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return (total_loss)

def valid(model,loader):
    total_loss=0
    model.eval()
    predictions = []
    labels=[]
    with torch.no_grad():
        for batch in tqdm(loader):
            pred=model(batch['input'].to(device))
            loss = criterion(pred,batch['label'].float().to(device))
            total_loss+=loss.item()
            predictions+=pred.cpu().numpy().tolist()
            labels+=batch['label'].cpu().numpy().tolist()
    return (total_loss,metrics.roc_auc_score(labels,predictions))

for epoch in range(epochs):
    loss_t=train(model,optimizer,train_loader,criterion)
    print('epoch:{},train_loss:{}'.format(epoch,loss_t))
    loss_v,roc=valid(model,valid_loader)
    print('epoch:{},val_loss:{},roc:{}'.format(epoch,loss_v,roc))