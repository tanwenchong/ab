import torch
from models import EGNN
#from egnn_clean import EGNN

from tqdm import tqdm 
from torch.nn import Module, Linear, Softmax, BCEWithLogitsLoss, ModuleList
from sklearn import metrics
from torch_geometric.data import Data, Batch
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader,ImbalancedSampler
from get_dataset import PDBDataset
from torch.nn.utils.rnn import pad_sequence,pack_sequence,pack_padded_sequence,pad_packed_sequence
from torch.utils.data import random_split,WeightedRandomSampler

import matplotlib.pyplot as plt
from utils import *
from tensorboardX import SummaryWriter
writer = SummaryWriter(log_dir='/public2022/tanwenchong/OAS/log/logs')

root='/public2022/tanwenchong/OAS/process/'
dataset = PDBDataset(root)

batch_size=16
n_feat = 512
x_dim = 3
epochs=20
model = EGNN(in_node_nf=n_feat, in_edge_nf=0,hidden_nf=128,attention=True)
learning_rate=5e-5
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,weight_decay=1e-16)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = torch.nn.BCELoss()
model_path='/public2022/tanwenchong/OAS/model_2.pth'

def plot_auc(labels,predictions):
    fpr, tpr, thresholds = metrics.roc_curve(labels,predictions)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig('/public2022/tanwenchong/OAS/auc.png')



def train(model,optimizer,loader,criterion):
    total_loss=0
    model.train()
    for batch in tqdm(loader):
        batch.pad_edge_mask=batch.pad_edge_mask.view(-1,1)
        batch.pad_node_mask=batch.pad_node_mask.view(-1,1)
        n_nodes, _ = batch.pose.size()
        pred=model(h0=batch.x.to(device), x=batch.pose.to(device), edges=batch.edge_index.to(device), edge_attr=None, node_mask=batch.pad_node_mask.to(device), edge_mask=batch.pad_edge_mask.to(device),
                    n_nodes=300)
        loss = criterion(pred,batch.y.float().to(device))
        total_loss+=loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return (total_loss)

def val(model,loader,):
    total_loss=0
    predictions = []
    labels=[]
    model.eval()
    for batch in tqdm(loader):
        with torch.no_grad():
            batch.pad_edge_mask=batch.pad_edge_mask.view(-1,1)
            batch.pad_node_mask=batch.pad_node_mask.view(-1,1)
            n_nodes, _ = batch.pose.size()
            pred=model(h0=batch.x.to(device), x=batch.pose.to(device), edges=batch.edge_index.to(device), edge_attr=None, node_mask=batch.pad_node_mask.to(device), edge_mask=batch.pad_edge_mask.to(device),
                    n_nodes=300)
            loss = criterion(pred,batch.y.float().to(device))
            total_loss+=loss.item()

            predictions+=pred.cpu().numpy().tolist()
            labels+=batch.y.cpu().numpy().tolist()
    #print(labels)   
    #plot_auc(labels,predictions)
    return (total_loss,metrics.roc_auc_score(labels,predictions))

def train_model():
    num_train = int(0.9 * len(dataset))
    num_val = len(dataset) - num_train 
    train_dataset, valid_dataset = random_split(dataset, [num_train, num_val])
    sampler = ImbalancedSampler(train_dataset)
    train_loader = DataLoader(train_dataset,batch_size=batch_size,drop_last=True,shuffle=False,sampler=sampler)
    valid_loader = DataLoader(valid_dataset,batch_size=batch_size,drop_last=True,shuffle=False)
    len_t=len(train_loader)/8
    len_v=len(valid_loader)/8
    ref_roc=0
    for epoch in range(epochs):
        loss_t=train(model,optimizer,train_loader,criterion)/len_t
        writer.add_scalar('Train/Loss', loss_t, epoch)
        #print('epoch:{},train_loss:{}'.format(epoch,loss_t))
        loss_v,roc=val(model,valid_loader)
        writer.add_scalar('Test/Loss', loss_v, epoch)
        #print('epoch:{},val_loss:{},roc:{}'.format(epoch,loss_v/len_v,roc))
        if roc>ref_roc:
            #torch.save(model.state_dict(),model_path)
            ref_roc=roc
        for name, param in model.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
    writer.close()

def test_model():
    num_train = int(0.9 * len(dataset))
    num_val = len(dataset) - num_train 
    train_dataset, valid_dataset = random_split(dataset, [num_train, num_val])
    sampler = ImbalancedSampler(train_dataset)

    model.load_state_dict(torch.load(model_path))
    test_loader = DataLoader(valid_dataset,batch_size=1,drop_last=True,shuffle=False)
    predictions = []
    labels=[]
    model.eval()
    for batch in tqdm(test_loader):
        with torch.no_grad():
            batch.pad_edge_mask=batch.pad_edge_mask.view(-1,1)
            batch.pad_node_mask=batch.pad_node_mask.view(-1,1)
            n_nodes, _ = batch.pose.size()
            pred=model(h0=batch.x.to(device), x=batch.pose.to(device), edges=batch.edge_index.to(device), edge_attr=None, node_mask=batch.pad_node_mask.to(device), edge_mask=batch.pad_edge_mask.to(device),
                    n_nodes=300)
            predictions+=pred.cpu().numpy().tolist()
            labels+=batch.y.cpu().numpy().tolist()

    return (metrics.roc_auc_score(labels,predictions))

#train_model()
predictions,labels=test_model()
