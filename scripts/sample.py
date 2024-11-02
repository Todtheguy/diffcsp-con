import time
import argparse
import torch

from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data import Data, Batch, DataLoader
from torch.utils.data import Dataset
from eval_utils import load_model, lattices_to_params_shape, get_crystals_list

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

    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []
    input_data_list = []
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

    def __init__(self, formula, num_evals):
        super().__init__()
        self.formula = formula
        self.num_evals = num_evals
        self.structure = self.get_structure()  # Dynamically initialize `self.structure`

    def get_structure(self):
        # Parse the formula to create a pymatgen Structure dynamically
        self.composition = chemparse.parse_formula(self.formula)
        chem_list = []
        for elem in self.composition:
            num_int = int(self.composition[elem])
            chem_list.extend([chemical_symbols.index(elem)] * num_int)
        self.chem_list = chem_list

        # Dynamically set lattice parameters based on structure details
        lattice = Lattice.from_parameters(*[5.43, 5.43, 5.43], *[90, 90, 90])  # Placeholder, replace with actual values or config-based parameters
        structure = Structure(lattice, chem_list, [[0, 0, 0]] * len(chem_list))  # Placeholder for atomic positions
        return structure

    def __len__(self) -> int:
        return self.num_evals

    def __getitem__(self, index):
        # Dynamically fetch lengths and angles from `self.structure.lattice`
        lengths = torch.tensor(self.structure.lattice.lengths).view(1, -1)  # Convert to 2D
        angles = torch.tensor(self.structure.lattice.angles).view(1, -1)    # Convert to 2D

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
    try:
        structure = Structure(
            lattice=Lattice.from_parameters(
                *(lengths.tolist() + angles.tolist())),
            species=atom_types, coords=frac_coords, coords_are_cartesian=False)
        return structure
    except:
        return None

# def main(args):
#     # load_data if do reconstruction.
#     model_path = Path(args.model_path)
#     model, _, cfg = load_model(
#         model_path, load_data=False)

#     if torch.cuda.is_available():
#         model.to('cuda')

#     tar_dir = os.path.join(args.save_path, args.formula)
#     os.makedirs(tar_dir, exist_ok=True)

#     print('Evaluate the diffusion model.')

#     test_set = SampleDataset(args.formula, args.num_evals)
#     test_loader = DataLoader(test_set, batch_size = min(args.batch_size, args.num_evals))

#     start_time = time.time()
#     (frac_coords, atom_types, lattices, lengths, angles, num_atoms) = diffusion(test_loader, model, args.step_lr)

#     crystal_list = get_crystals_list(frac_coords, atom_types, lengths, angles, num_atoms)

#     strcuture_list = p_map(get_pymatgen, crystal_list)

#     for i,structure in enumerate(strcuture_list):
#         tar_file = os.path.join(tar_dir, f"{args.formula}_{i+1}.cif")
#         if structure is not None:
#             writer = CifWriter(structure)
#             writer.write_file(tar_file)
#         else:
#             print(f"{i+1} Error Structure.")

def main():
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
    else:
        raise FileNotFoundError("config.yaml not found. Please ensure it's in the script directory.")

    model_path = config["model"]["model_path"]
    save_path = config["model"]["save_path"]
    formula = config["model"]["formula"]
    num_evals = config["model"].get("num_evals", 1)
    batch_size = config["model"].get("batch_size", 500)
    step_lr = config["model"].get("step_lr", 1e-5)

    model, _, cfg = load_model(model_path, load_data=False)
    model.keep_lattice = True 

    if torch.cuda.is_available():
        model.to('cuda')

    tar_dir = os.path.join(save_path, formula)
    os.makedirs(tar_dir, exist_ok=True)

    print('Evaluate the diffusion model.')
    test_set = SampleDataset(formula, num_evals)
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
    main()