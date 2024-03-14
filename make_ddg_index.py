

mode='foldx'
pdb="/public2022/tanwenchong/PPI/8g02.pdb"
chain_id='G'

from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1


pdb_parser = PDBParser()
structure = pdb_parser.get_structure("8g02", pdb)
model = structure[0]
chain = model[chain_id]

if mode=='rdg':

    with open ('/public2022/tanwenchong/antibody/ddg/RDE-PPI-main/8g02_mutation.yml','a') as f:
        for residue in chain:

            if residue.id[0] == ' ':
    
                residue_id = residue.id[1]
 
                residue_name = residue.get_resname()
                f.write(f'- {seq1(residue_name)}{chain_id}{residue_id}*\n')

if mode=='foldx':
    #command=PositionScan
    #pdb=PS.pdb
    #positions=GA5a,GA14p (residue, chain, number, mutation)
    #usage:foldx --config=
    with open ('/public2022/tanwenchong/PPI/config.cfg','w+') as f:
        f.write('command=PositionScan\n')
        f.write(f'pdb={pdb}\n')
        f.write('positions=')
        for residue in chain:

            if residue.id[0] == ' ':
                residue_id = residue.id[1]
                residue_name = residue.get_resname()
                f.write(f'{seq1(residue_name)}{chain_id}{residue_id}a,')
        

