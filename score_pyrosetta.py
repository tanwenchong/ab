from pyrosetta import *
from pyrosetta.rosetta import *
from pyrosetta.teaching import *
import os
from rosetta.protocols.rosetta_scripts import *
from rosetta.protocols.antibody import *
from rosetta.protocols.antibody.design import *
from rosetta.utility import *
import pyrosettacolabsetup; pyrosettacolabsetup.install_pyrosetta()
import pyrosetta; pyrosetta.init()
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover


def rosetta_score(pdb,antibody,antigen):
    pose  =  pose_from_pdb(pdb)
    original_pose = pose.clone()

    interface=f'{antibody}_{antigen}'  #抗体链_抗原链
    mover = InterfaceAnalyzerMover(interface) #计算相互作用能量 
    mover.set_pack_separated(True)
    mover.apply(pose)
    return pose.scores['dG_separated']



