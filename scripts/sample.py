import time
import argparse
import torch

from tqdm import tqdm
from pathlib import Path
from torch_geometric.data import Data, DataLoader
from torch.utils.data import Dataset
from .eval_utils import load_model, lattices_to_params_shape, get_crystals_list

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter
from pyxtal.symmetry import Group
import chemparse
import numpy as np
from p_tqdm import p_map

import os

import yaml
import torch
from omegaconf import OmegaConf

from pathlib import Path
import argparse

chemical_symbols = [
    # 0
    'X',
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og']

def diffusion(loader, model, step_lr):

    frac_coords, num_atoms, atom_types, lattices = [], [], [], []
    for idx, batch in enumerate(loader):

        if torch.cuda.is_available():
            batch.cuda()
        outputs, traj = model.sample(batch, step_lr = step_lr)
        frac_coords.append(outputs['frac_coords'].detach().cpu())
        num_atoms.append(outputs['num_atoms'].detach().cpu())
        atom_types.append(outputs['atom_types'].detach().cpu())
        lattices.append(outputs['lattices'].detach().cpu())

    frac_coords = torch.cat(frac_coords, dim=0)
    num_atoms = torch.cat(num_atoms, dim=0)
    atom_types = torch.cat(atom_types, dim=0)
    lattices = torch.cat(lattices, dim=0)
    lengths, angles = lattices_to_params_shape(lattices)

    return (
        frac_coords, atom_types, lattices, lengths, angles, num_atoms
    )

class SampleDataset(Dataset):

    def __init__(self, formula, num_evals, lengths, angles):
        super().__init__()
        self.formula = formula
        self.num_evals = num_evals
        self.lengths = lengths
        self.angles = angles
        self.structure = self.get_structure()

    def get_structure(self):
        self.composition = chemparse.parse_formula(self.formula)
        chem_list = []
        for elem in self.composition:
            num_int = int(self.composition[elem])
            chem_list.extend([chemical_symbols.index(elem)] * num_int)
        self.chem_list = chem_list

        lattice = Lattice.from_parameters(*self.lengths, *self.angles) 
        structure = Structure(lattice, chem_list, [[0, 0, 0]] * len(chem_list))
        return structure

    def __len__(self) -> int:
        return self.num_evals

    def __getitem__(self, index):
        lengths = torch.tensor(self.structure.lattice.lengths).view(1, -1) 
        angles = torch.tensor(self.structure.lattice.angles).view(1, -1) 
        # print(f"Sampling with lengths: {lengths}, angles: {angles}")

        return Data(
            atom_types=torch.LongTensor(self.chem_list),
            num_atoms=len(self.chem_list),
            num_nodes=len(self.chem_list),
            lengths=lengths,
            angles=angles
        )

def get_pymatgen(crystal_array):
    frac_coords = crystal_array['frac_coords']
    atom_types = crystal_array['atom_types']
    lengths = crystal_array['lengths']
    angles = crystal_array['angles']
    # print(f"Creating pymatgen structure with lengths: {lengths}, angles: {angles}")
    try:
        structure = Structure(
            lattice=Lattice.from_parameters(
                *(lengths.tolist() + angles.tolist())),
            species=atom_types, coords=frac_coords, coords_are_cartesian=False)
        return structure
    except:
        return None


def main(model_path, save_path, formula, num_evals, batch_size, step_lr, lengths, angles):
    print("Loaded lattice parameters:", lengths, angles)

    model, _, cfg = load_model(model_path, load_data=False)
    model.keep_lattice = True 

    if torch.cuda.is_available():
        model.to('cuda')

    tar_dir = os.path.join(save_path, formula)
    os.makedirs(tar_dir, exist_ok=True)

    print('Evaluate the diffusion model.')
    test_set = SampleDataset(formula, num_evals, lengths=lengths, angles=angles)
    test_loader = DataLoader(test_set, batch_size=min(batch_size, num_evals))

    (frac_coords, atom_types, lattices, lengths, angles, num_atoms) = diffusion(test_loader, model, step_lr)
    crystal_list = get_crystals_list(frac_coords, atom_types, lengths, angles, num_atoms)
    structure_list = p_map(get_pymatgen, crystal_list)

    for i, structure in enumerate(structure_list):
        tar_file = os.path.join(tar_dir, f"{formula}_{i+1}.cif")
        if structure is not None:
            writer = CifWriter(structure)
            writer.write_file(tar_file)
        else:
            print(f"{i+1} Error Structure.")

if __name__ == '__main__':
    main(model_path, save_path, formula, num_evals, batch_size, step_lr, lengths, angles)
