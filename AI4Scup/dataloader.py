import os
from rdkit import Chem
from rdkit import RDPaths
import numpy as np
from dgllife.model import AttentiveFPPredictor
from dgllife.utils import *
from functools import partial
from collections import Counter
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect 
#导入torch
import torch
import dgl
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from torch.utils.data import DataLoader

class MoleculeDataset(DGLDataset):
    def __init__(self, data_file, transform=None):
        self.data_file = data_file
        self.data = self.load_data()
        self.transform = transform
        self.atoms = ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At']

    def load_data(self):
        # 打开data_file.txt
        with open(self.data_file, 'r') as f:
            data = f.readlines()
        # 读取数据
        data = [line.strip().split() for line in data]
        # 返回数据
        return data
    # @property:
    def __len__(self):
        # 返回数据集长度
        return len(self.data)

    def chirality(atom):
        try:
            return one_hot_encoding(atom.GetProp('_CIPCode'), ['R', 'S']) + \
                [atom.HasProp('_ChiralityPossible')]
        except:
            return [False, False] + [atom.HasProp('_ChiralityPossible')]
    #处理smiles
    def process(self,smiles):
        # atom_count = Counter()
        # atom_count.update(atoms)
        mol = Chem.MolFromSmiles(smiles)
        # fp = Chem.RDKFingerprint(mol)
        morgan_fp = GetMorganFingerprintAsBitVect(mol, 5, nBits=1024)  
        morgan_fp = np.array(morgan_fp)
        atoms = self.atoms
        #atoms
        atom_featurizer = BaseAtomFeaturizer(
                        {'hv': ConcatFeaturizer([
                        partial(atom_type_one_hot, allowable_set=atoms,
                            encode_unknown=True),
                        partial(atom_degree_one_hot, allowable_set=list(range(10))),
                        atom_formal_charge, atom_num_radical_electrons,
                        partial(atom_hybridization_one_hot, encode_unknown=True),
                        #lambda atom: [0] #, A placeholder for aromatic information,
                        # atom_total_num_H_one_hot, self.chirality
                        ],
                        )})
        # bond_featurizer = BaseBondFeaturizer({
        #     'he': lambda bond: [0 for _ in range(10)]
        #     })
        # adj_matrix = Chem.GetAdjacencyMatrix(mol) # 邻接矩阵
        mol_graph = mol_to_bigraph(mol,node_featurizer=atom_featurizer)
        # , edge_featurizer=bond_featurizer
        res = [morgan_fp,mol_graph]
        return res

    def __getitem__(self, idx):
        smiles, label = self.data[idx]
        label = int(label)
        res = self.process(smiles)
        res = res + [label]
        return res


def collate(samples):
    morgan_fps,graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    batched_morgan_fps = torch.tensor(morgan_fps)
    batched_labels = torch.tensor(labels)
    return [batched_graph,batched_morgan_fps,batched_labels]

if __name__ == "__main__":
    smiles_list = ["CCO", "CC(C)(C)O", "C1=CC=CS1"]
    data = MoleculeDataset(data_file= "/root/AI4S/classification/hiv/data_test.txt")

    # dataloder   
    dataloader = GraphDataLoader(data, batch_size=32, shuffle=True,collate_fn=collate)
    for res in dataloader:
        print(res)
