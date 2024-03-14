
#python /public2022/tanwenchong/antibody/benchmark.py -j /public2022/tanwenchong/antibody/dyMEAN/dyMEAN-main/all_data/rabd_all.json -s -f -x
#python /public2022/tanwenchong/antibody/benchmark.py -p /public2022/tanwenchong/antibody/dyMEAN/design_result -m real -x -f -s -d
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-j','--json',required=False)
parser.add_argument('-p','--path')
parser.add_argument('-t','--type',required=False)
parser.add_argument('-m','--mode')
parser.add_argument('-o','--out',required=False)
parser.add_argument('-r','--rosetta', action='store_true') #not use
parser.add_argument('-x','--foldx', action='store_true')
parser.add_argument('-f','--esm_fold', action='store_true')
parser.add_argument('-s','--sequence', action='store_true')
parser.add_argument('-d','--dis', action='store_true')


args = parser.parse_args()
import json
from tqdm import tqdm
import subprocess as sp
import os
import pandas as pd
from multiprocessing import Pool
import multiprocessing
from score_pyrosetta import *
from antiberty import AntiBERTyRunner
from benchmark_utils import *
from abnumber import Chain
from Bio import PDB
from Bio.PDB import NeighborSearch

FOLDXBIN='/public2022/tanwenchong/app/foldx/foldx_20241231'

 

def test_score(line):
    item = json.loads(line)
    pdb=item["pdb"]
    os.chdir(args.path)
    if args.type=='diffab':
        pdb=pdb+'_generated.pdb'
    else:
        pdb=pdb+'.pdb'
    cdrh3_seq=get_sequence_from_pdb(pdb,item["heavy_chain"])
    chain = Chain(cdrh3_seq, scheme='imgt')
    cdrh3=[str(list(chain.cdr3_dict.keys())[0])[1:],str(list(chain.cdr3_dict.keys())[-1])[1:]]

    if args.foldx==True:
 
        try:
            result=os.popen(f'{FOLDXBIN} --command=AnalyseComplex --pdb={pdb} --analyseComplexChains={item["heavy_chain"]}{item["light_chain"]},{"".join(item["antigen_chains"])}')
            aff = float(result.read().split('\n')[-8].split(' ')[-1])
        except:
            aff=''
        
    if args.rosetta==True:
        #not use
        pdb_aho=f'/public2022/tanwenchong/antibody/dyMEAN/dyMEAN-main/all_structures/all_structures/aho/{pdb}'
        r_score=rosetta_score(pdb,f'{item["heavy_chain"]}{item["light_chain"]}',"".join(item["antigen_chains"]))

    if args.esm_fold==True:
        try:
            multi_ll,ll=esm_score(pdb,item["heavy_chain"])
        except:
            
            multi_ll=''
            ll=''

    if args.sequence==True:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        antiberty = AntiBERTyRunner()
        try:
            seq_ll = antiberty.pseudo_log_likelihood([cdrh3_seq])[0].item()
        except:
            seq_ll=''

    if args.dis==True:
        try:
            dis,ca_dis=calculate_average_distance(item["pdb_data_path"],item["antigen_chains"],item["heavy_chain"],cdrh3)
        except:
            dis,ca_dis='',''    
        
    if relax!=True:
        return [pdb,aff,multi_ll,ll,seq_ll,dis,ca_dis]
    
def real_score(pdb):

    cdrh3_seq=get_sequence_from_pdb(pdb,'H')
    chain = Chain(cdrh3_seq, scheme='imgt')
    cdrh3=[str(list(chain.cdr3_dict.keys())[0])[1:],str(list(chain.cdr3_dict.keys())[-1])[1:]]

    if args.foldx==True:
        result=os.popen(f'{FOLDXBIN} --command=AnalyseComplex --pdb={pdb} --analyseComplexChains=H,A')
        aff = float(result.read().split('\n')[-8].split(' ')[-1])

        
    if args.esm_fold==True:
        multi_ll,ll=esm_score(pdb,'H')


    if args.sequence==True:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        antiberty = AntiBERTyRunner()
        seq_ll = antiberty.pseudo_log_likelihood([cdrh3_seq])[0].item()


    if args.dis==True:
        dis,ca_dis=calculate_average_distance(pdb,'A','H',cdrh3)
   
    return [pdb,aff,multi_ll,ll,seq_ll]

def run_testset():
    with open(args.json, 'r') as fin:
        lines = fin.read().strip().split('\n')
    threads=30
    with Pool(threads) as pool:
        all_score=pool.map(test_score, lines)
    return all_score

def run_real():
    pdbs=os.listdir(args.path)
    threads=30
    with Pool(threads) as pool:
        all_score=pool.map(real_score, pdbs)

def save(all_score):
    columns = ['pdb', 'foldx_energy','esm_structure_multi','esm_structure_single','berty_sequence']
    df = pd.DataFrame(all_score, columns=columns)
    df.to_csv(f'/public2022/tanwenchong/antibody/{args.out}.csv', index=False)

if args.mode=='test':
    all_score=run_testset()
    save(all_score)
if args.mode=='real':
    all_score=run_real()
    columns = ['pdb', 'foldx_energy','esm_structure_multi','esm_structure_single','berty_sequence','atom_distance','CA_distance']
    df = pd.DataFrame(all_score, columns=columns)
    df.to_csv('/public2022/tanwenchong/antibody/dyMEAN/design.csv', index=False)



def fx(line): 

    item = json.loads(line)
    pdb=item["pdb"]
    os.chdir(args.path)
    if args.type=='diffab':
        pdb=pdb+'_generated.pdb'
    else:
        pdb=pdb+'.pdb'

    foldx_pdb=pdb
    sp.run(f'{FOLDXBIN} --command=AnalyseComplex --pdb={foldx_pdb} --analyseComplexChains={item["heavy_chain"]}{item["light_chain"]},{"".join(item["antigen_chains"])}',shell=True)


if args.mode=='foldx':
    with open(args.json, 'r') as fin:
        lines = fin.read().strip().split('\n')
    threads=40
    with Pool(threads) as pool:
        pool.map(fx, lines)