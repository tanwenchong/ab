import subprocess as sp
from tqdm import tqdm 
import os
from Bio import PDB
from Bio.PDB.PDBParser import PDBParser
from Bio.SeqUtils import seq1
import pickle
import torch_geometric.transforms as T
from torch_geometric.transforms import Pad
import networkx as nx
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch,Dataset,DataLoader
from torch_geometric.utils import from_networkx
from itertools import combinations
from multiprocessing import Pool
from antiberty import AntiBERTyRunner
import torch_geometric
from radius import myRadiusGraph
from adjacency import AdjacencyFeatures
antiberty = AntiBERTyRunner()
graph = list()

def get_chain_sequence(chain):
    chain_sequence = ''
    for residue in chain:
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
def distance_3d(coord1, coord2):
    # 计算两个三维坐标之间的欧几里德距离
    return ((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2 + (coord1[2] - coord2[2])**2)**0.5
def build_graph(pdb_data):
    G = nx.Graph()
    for residue_id, residue_info in pdb_data:
        G.add_node(residue_id, pose=residue_info)
    for (node1, coord1), (node2, coord2) in combinations(G.nodes.data(), 2):
        dist = distance_3d(coord1['pose'], coord2['pose'])
        if dist < 8 :
            G.add_edge(node1, node2, distance=dist)
    return G




def get_graph(new_pdb):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('antibody', new_pdb)
    for model in structure:
        for chain in model:
            if chain.id == 'L':
                L= get_chain_sequence(chain)
            elif chain.id == 'H':
                H= get_chain_sequence(chain)    
    batch =  None
    dis=8
    max_edge=32
    info=coors(new_pdb)
    H_embeddings = antiberty.embed([H])[0][1:-1,:]
    L_embeddings = antiberty.embed([L])[0][1:-1,:]
    H_L=torch.cat((H_embeddings, L_embeddings), dim=0)
    protein_graph = build_graph(info)
    data = from_networkx(protein_graph)                                     
    node_features=H_L.to(torch.device('cpu'))
    data.x = node_features
    data.y=torch.LongTensor([0])
    return data


class PDBDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.raw_file_list=os.listdir('/public2022/tanwenchong/OAS/pair_pdb')
        self.processed_file_list=os.listdir('/public2022/tanwenchong/GNN/process')
        self.data = self.processed_file_list
        super().__init__(self.root)


    @property
    def processed_file_names(self):
        return self.processed_file_list

    @property
    def processed_dir(self) -> str:
        return '/public2022/tanwenchong/GNN/process/'

    def len(self):
        return len(self.data)

    def process(self):
        rem_files = set(self.raw_file_list) - set(self.processed_file_list)
        for file in rem_files:
            protein=file[:-4]
            data=get_graph(file)
            pad_transform = Pad(300,90000,0,0,add_pad_mask=True)
            #_transforms = []
            #_transforms.append(myRadiusGraph)
            #_transforms.append(AdjacencyFeatures)
            #pre_transform = T.Compose(_transforms)
            #data = pre_transform(data)
            data=pad_transform(data)
            torch.save(data, os.path.join('/public2022/tanwenchong/GNN/process/' +  f'{protein}.pt'))
    
    def get(self, idx):
        rep = self.data[idx]
        return torch.load(os.path.join('/public2022/tanwenchong/GNN/process/', f'{rep}'))   

root='/public2022/tanwenchong/OAS/process/'
#dataset = PDBDataset(root)
#dataset.process()
#train_dataloader = DataLoader(dataset,batch_size=32,drop_last=True,shuffle=True)
#for batch in train_dataloader:
#    print(batch)
#    break