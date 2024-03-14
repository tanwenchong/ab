import subprocess as sp
from tqdm import tqdm 
import os
from Bio import PDB
from Bio.PDB.PDBParser import PDBParser
from Bio.SeqUtils import seq1
import dill
import networkx as nx
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader,Dataset
from torch_geometric.utils import from_networkx
from itertools import combinations
from multiprocessing import Pool
from antiberty import AntiBERTyRunner
from torch_geometric.utils import to_dense_adj
antiberty = AntiBERTyRunner()
graph = list()
def get_chain_sequence(chain):
    # 初始化链的序列
    chain_sequence = ''

    # 遍历链中的残基
    for residue in chain:
        # 获取残基的氨基酸序列
        if PDB.is_aa(residue):
            chain_sequence += PDB.Polypeptide.three_to_one(residue.get_resname())

    return chain_sequence
def coors(pdb_file):
    parser = PDBParser(PERMISSIVE=1)
    structure = parser.get_structure('antibody', pdb_file)
    model = structure[0]
    X = []
    n=0
    cdr_only=False
    for chain in model.get_list():
        for residue in chain.get_list():
            n+=1
            if  residue.has_id('CA') :
                ca = residue['CA'].get_coord().tolist()
                aa=seq1(residue.get_resname())+str(n)
                X.append([aa,ca])
    return X
def build_graph(pdb_data):
    G = nx.Graph()

    # 添加节点和边
    for residue_id, residue_info in pdb_data:
        #print(residue_id, residue_info)
        G.add_node(residue_id, pose=residue_info)
    #print(G)
    # 添加边，可以根据原子之间的距离或其他相互作用进行连接
    #print(G.nodes.data())
    
    for (node1, coord1), (node2, coord2) in combinations(G.nodes.data(), 2):
        #print(coord1, coord2)
        dist = distance_3d(coord1['pose'], coord2['pose'])
        # 如果距离小于阈值，则添加边
        if dist < 8 :
            G.add_edge(node1, node2, distance=dist)

    return G

def distance_3d(coord1, coord2):
    # 计算两个三维坐标之间的欧几里德距离
    return ((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2 + (coord1[2] - coord2[2])**2)**0.5


def pad(matrix):
    max_size = 300  # 自定义padding的大小
    pad_rows = max_size - matrix.size(0)
    padded_matrix = F.pad(matrix, (0, 0, 0, pad_rows), value=0)
    return padded_matrix

def get_graph(new_pdb):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('antibody', new_pdb)
    for model in structure:
        # 遍历模型中的链
        for chain in model:
            # 根据链的标识符判断是轻链还是重链
            if chain.id == 'L':
                L= get_chain_sequence(chain)
            elif chain.id == 'H':
                H= get_chain_sequence(chain)    

    info=coors(new_pdb)
    #print(antiberty.embed(H))
    H_embeddings = antiberty.embed([H])[0][1:-1,:]
    L_embeddings = antiberty.embed([L])[0][1:-1,:]
    H_L=torch.cat((H_embeddings, L_embeddings), dim=0)
    protein_graph = build_graph(info)
    data = from_networkx(protein_graph)
    node_features=H_L.to(torch.device('cpu'))

    data.x = node_features
    data.y=torch.LongTensor([1])
    data.edge_index=data.edge_index.view(-1,2)
    print(data)
    graph.append(data)

def batch_stack(props):

    if not torch.is_tensor(props[0]):
        return torch.tensor(props)
    elif props[0].dim() == 0:
        return torch.stack(props)
    else:
        return torch.nn.utils.rnn.pad_sequence(props, batch_first=True, padding_value=0)

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def edge_index_batch(edge_index, num_nodes):  
    # 构建一个大小为 (num_nodes, num_nodes) 的零矩阵  
    batch_size=5
    rows, cols = [], []
    for index in range(batch_size):
        for edge in edge_index[index]:  
            i,j=edge
            rows.append(i + n_nodes * index)
            cols.append(j + n_nodes * index)
            
    return [torch.cat(rows), torch.cat(cols)]



def collate_fn(batch):
    
    #matrix = Batch.from_data_list(batch)
    #print(matrix)
    batch = {prop: batch_stack([mol[prop] for mol in batch]) for prop in batch[0].keys()}
    acid_mask=torch.ones([batch['x'].size()[0],batch['x'].size()[1]])
    batch['node_mask'] = acid_mask
    edge_mask = acid_mask.unsqueeze(1) * acid_mask.unsqueeze(2)
    #print(acid_mask.size())
    batch_size, n_nodes = acid_mask.size()
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask

    #edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
    batch['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)
    #batch['edge_index'] = 
    return batch


def get_loader():
    count=0
    files = os.listdir('/public2022/tanwenchong/OAS/thera')
    for file in tqdm(files):
        count+=1
        get_graph(file)
        if count==10:
            break
    #batched_graph = Batch.from_data_list(graph)
    dataset = MyDataset(graph)
    print(dataset[0])
    dataloader = DataLoader(dataset, batch_size=5, collate_fn=collate_fn, shuffle=True)
    with open('/public2022/tanwenchong/GNN/data/dataloader_nocom.pkl','wb') as f:
        dill.dump(dataloader, f)

get_loader()