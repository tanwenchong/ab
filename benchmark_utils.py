import json
from Bio import PDB
from Bio.PDB import NeighborSearch
from biotite.sequence.io.fasta import FastaFile, get_sequences
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm

import esm
import esm.inverse_folding

def esm_score(pdb,hight_chain):
        
        model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()

        model = model.eval()
        structure = esm.inverse_folding.util.load_structure(pdb)
        coords, native_seqs = esm.inverse_folding.multichain_util.extract_coords_from_complex(structure)
        native_seq = native_seqs[hight_chain]

        multi_ll, _ = esm.inverse_folding.multichain_util.score_sequence_in_complex(
            model, alphabet, coords, hight_chain, native_seq)
        
        coords, native_seq = esm.inverse_folding.util.load_coords(pdb, hight_chain)
        ll, _ = esm.inverse_folding.util.score_sequence(
            model, alphabet, coords, native_seq) 
 
        return multi_ll,ll


def get_sequence_from_pdb(pdb_file_path, chain_id):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file_path)
    sequence = ''
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for residue in chain:
                    sequence += PDB.Polypeptide.three_to_one(residue.get_resname())
    return sequence

def calculate_average_distance(pdb,antigen,heavy,cdrh3):
    # 1. 从PDB文件中读取结构数据
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb)

    # 2. 确定抗体和抗原的残基
    antibody_residues = []
    antigen_atoms = []
    antigen_ca=[]
    for model in structure:
        for chain in model:
            if chain.id in antigen:
                for residue in chain:
                    try:
                        antigen_ca.append(residue['CA'])
                        antigen_atoms+=([atoms for atoms in residue.get_atoms()])
                    except:
                        continue

            if chain.id == heavy:
                for residue in chain:
                    if int(cdrh3[0]) <= int(residue.get_id()[1]) <= int(cdrh3[1]):
                        antibody_residues.append(residue)
                

    # 3. 创建 NeighborSearch 对象以寻找最近的残基
    ns = NeighborSearch(antigen_atoms)
    ns_ca = NeighborSearch(antigen_ca)

    # 4. 对于每个抗体残基，找到最近的抗原残基并计算距离
    distances = []
    ca_distances=[]
    for antibody_residue in antibody_residues:
        antibody_atoms=antibody_residue.get_atoms()
        res_dis=list()
        for atoms in antibody_atoms:
            nearest_antigen_atom = ns.search(atoms.coord, 100,level='A')[0]
            distance = atoms - nearest_antigen_atom
            res_dis.append(distance)
        distances.append(min(res_dis))


        nearest_antigen_residue = ns.search(antibody_residue['CA'].coord, 100,level='A')[0]
        ca_distance = antibody_residue['CA'] - nearest_antigen_residue   
        ca_distances.append(ca_distance)

    # 5. 计算平均距离
    average_distance = sum(distances) / len(distances)
    average_cadistance = sum(ca_distances) / len(ca_distances)
    return average_distance,average_cadistance

