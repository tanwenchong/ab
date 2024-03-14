import os
import sys
import json
import re
from typing import Dict, List, Tuple
import pandas as pd
import argparse
from Bio import SeqIO
import math
from api.design import design
from abnumber import Chain
from multiprocessing import Pool,Manager
import multiprocessing
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('-g','--gpu')
args = parser.parse_args()
gpus=args.gpu.split(',')

def fasta_database_parser():
    databases = ["IGHV", "IGHJ", "IGKV", "IGKJ", "IGLV", "IGLJ"]

    heavy_V_database_dict = {}
    heavy_J_database_dict = {}
    light_KV_database_dict = {}
    light_KJ_database_dict = {}
    light_LV_database_dict = {}
    light_LJ_database_dict = {}
    for database in databases:
        for record in SeqIO.parse(f"/public2022/tanwenchong/antibody/cumab/CUMAb-main/IMGT_databases/{database}.fasta", "fasta"):
            name = record.name.split("|")[1]
            seq = str(record.seq)
            desc = record.description
            if "partial" not in desc and "rev" not in desc and desc.split("|")[3] == "F":
                if "H" in database:
                    if "V" in database:
                        if name.split("*")[0] not in heavy_V_database_dict:
                            if len(re.findall("C", seq)) <= 2:
                                heavy_V_database_dict[name.split("*")[0]] = seq
                    elif "J" in database:
                        if name.split("*")[0] not in heavy_J_database_dict:
                            heavy_J_database_dict[name.split("*")[0]] = seq
                elif "L" in database:
                    if "V" in database:
                        if name.split("*")[0] not in light_LV_database_dict:
                            if len(re.findall("C", seq)) <= 2:
                                light_LV_database_dict[name.split("*")[0]] = seq
                    elif "J" in database:
                        if name.split("*")[0] not in light_LJ_database_dict:
                            light_LJ_database_dict[name.split("*")[0]] = seq
                elif "K" in database:
                    if "V" in database:
                        if name.split("*")[0] not in light_KV_database_dict:
                            if len(re.findall("C", seq)) <= 2:
                                light_KV_database_dict[name.split("*")[0]] = seq
                    elif "J" in database:
                        if name.split("*")[0] not in light_KJ_database_dict:
                            light_KJ_database_dict[name.split("*")[0]] = seq
    return [heavy_V_database_dict, heavy_J_database_dict, light_KV_database_dict, light_KJ_database_dict, light_LV_database_dict, light_LJ_database_dict]

def make_full_chain(V_dict: Dict, J_dict: Dict) -> Dict:
    return_dict = {}
    for V in V_dict:
        for J in J_dict:
            key = V + "-" + J
            seq = V_dict[V] + J_dict[J]
            if len(re.findall("C", seq)) == 2:
                if key not in return_dict:
                    return_dict[key] = seq
    return return_dict

def parse_IMGT_databases() -> List:
    database_dicts = fasta_database_parser()
    heavy_V_database_dict = database_dicts[0]
    heavy_J_database_dict = database_dicts[1]
    light_KV_database_dict = database_dicts[2]
    light_KJ_database_dict = database_dicts[3]
    light_LV_database_dict = database_dicts[4]
    light_LJ_database_dict = database_dicts[5]
    heavy_dict = make_full_chain(heavy_V_database_dict, heavy_J_database_dict)
    kappa_dict = make_full_chain(light_KV_database_dict, light_KJ_database_dict)
    lambda_dict = make_full_chain(light_LV_database_dict, light_LJ_database_dict)
    return [heavy_dict, kappa_dict, lambda_dict]    

sequence_dicts = parse_IMGT_databases()

def screen_motifs(light_seq:str, heavy_seq:str, screens=["NG", "N[^P][ST]"]) -> bool:
    value = False
    light_CDRs = find_CDRs(light_seq)
    heavy_CDRs = find_CDRs(heavy_seq)
    for screen in screens:
        r_string = r"{}".format(screen)
        light_count = 0
        heavy_count = 0
        for light_CDR in light_CDRs:
            light_count += len(re.findall(r_string, light_CDR))
        for heavy_CDR in heavy_CDRs:
            heavy_count += len(re.findall(r_string, heavy_CDR))
        if len(re.findall(r_string, light_seq)) > light_count:
            value = True
        if len(re.findall(r_string, heavy_seq)) > heavy_count:
            value = True
    return value

def find_CDRs(sequence:str) -> List:
    chain1 = Chain(sequence, scheme='imgt')
    return [chain1.cdr1_seq,chain1.cdr2_seq,chain1.cdr3_seq]





ckpt = '/public2022/tanwenchong/antibody/dyMEAN/dyMEAN-main/checkpoints/multi_cdr_design.ckpt'
root_dir = '/public2022/tanwenchong/antibody/dyMEAN/design_multi'


manager = multiprocessing.Manager()
identifiers=manager.list()
frameworks=manager.list()
def get_frameworks(H_name):
    for L_name in sequence_dicts[1].keys():
        #if not screen_motifs(sequence_dicts[0][H_name],sequence_dicts[1][L_name]):
        identifiers.append(H_name+'_'+L_name)
        frameworks.append(
                (
                    ('H',sequence_dicts[0][H_name]),
                    ('L',sequence_dicts[1][L_name])
                )
            )


with Pool(60) as pool:
    pool.map(get_frameworks, sequence_dicts[0].keys())

print(len(identifiers))

pdbs = ['/public2022/tanwenchong/antibody/data/6y6c_fv4_clean_aho.clean.pdb' for _ in range(len(identifiers))]
epitope_defs = ['/public2022/tanwenchong/antibody/dyMEAN/data/epitope.json' for _ in range(len(identifiers))]
remove_chains = [['H','L'] for _ in range(len(identifiers))]

split_len = math.ceil(len(identifiers) / len(gpus))
identifier_splits = [identifiers[i * split_len: (i+1) * split_len] for i in range(len(gpus))]
framework_splits = [frameworks[i * split_len: (i+1) * split_len] for i in range(len(gpus))]
pdb_splits = [pdbs[i * split_len: (i+1) * split_len] for i in range(len(gpus))]
epitope_def_splits = [epitope_defs[i * split_len: (i+1) * split_len] for i in range(len(gpus))]
remove_chain_splits = [remove_chains[i * split_len: (i+1) * split_len] for i in range(len(gpus))]


def multi_design(identifier_splits,framework_splits,pdb_splits,epitope_def_splits,remove_chain_splits,gpu):
    print()
    design(ckpt=ckpt,  # path to the checkpoint of the trained model
            gpu=gpu,      # the ID of the GPU to use
            pdbs=pdb_splits,  # paths to the PDB file of each antigen (here antigen is all TRPV1)
            epitope_defs=epitope_def_splits,  # paths to the epitope definitions
            frameworks=framework_splits,      # the given sequences of the framework regions
            out_dir=root_dir,           # output directory
            identifiers=identifier_splits,    # name of each output antibody
            remove_chains=remove_chain_splits,# remove the original ligand
            enable_openmm_relax=True,   # use openmm to relax the generated structure
            auto_detect_cdrs=True)  # manually use '-'  to represent CDR residues


            
processes = []
for i in range(len(gpus)):
    process = multiprocessing.Process(target=multi_design, args=(identifier_splits[i],framework_splits[i],pdb_splits[i],epitope_def_splits[i],remove_chain_splits[i],gpus[i]))
    process.start()
    processes.append(process)
    
for process in processes:
    process.join()