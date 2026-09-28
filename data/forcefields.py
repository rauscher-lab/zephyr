import pathlib
import re
from dataclasses import dataclass

import MDAnalysis as mda
import numpy as np

import config


@dataclass
class Forcefield:
    """Dataclass to store all properties related to a given force field"""
    name: str
    dir: str | pathlib.Path
    rtp_basename: str
    types_filepath: pathlib.Path = None
    atom_types: list = None
    residue_info: dict = None
    vsite_types: list = None
    backbone_atoms_name: list = None
    gmx_energy_names: dict[str, str] = None

    def __post_init__(self):
        if self.types_filepath is None:
            self.types_filepath = config.data_dir / f'{self.name}_types.npz'
        if self.vsite_types is None:
            self.vsite_types = []
        if self.backbone_atoms_name is None:
            raise ValueError(f"Backbone atoms name must be defined.")
        if isinstance(self.dir, str):
            self.dir = pathlib.Path(self.dir)
        if self.gmx_energy_names is None:
            if 'charmm' in self.name:
                self.gmx_energy_names = charmm_gmx_energy_names
            elif 'amber' in self.name:
                self.gmx_energy_names = amber_gmx_energy_names
            else:
                raise ValueError(f"energy names for forcefield={self.name!r} are undefined.")

        # Save types into .npz file
        if not self.types_filepath.exists():
            if 'charmm' in self.name:
                supported_residues = charmm_supported_residues
            elif 'amber' in self.name:
                supported_residues = amber_supported_residues
            else:
                raise ValueError(f"supported residue types for forcefield={self.name!r} are undefined.")

            atom_types, res_info = import_ff_types(self.dir, rtp_basename=self.rtp_basename,
                                                   residues=supported_residues)
            saved_data = dict(atom_types=atom_types, residue_info=res_info)
            np.savez(self.types_filepath, **saved_data)

        with np.load(self.types_filepath, allow_pickle=True) as f:
            ff_data = dict(f)
        self.atom_types = ff_data['atom_types'].tolist() + self.vsite_types
        self.residue_info = ff_data['residue_info'].item()

    @property
    def gmx_name(self):
        return self.dir.stem

    def __setstate__(self, state):
        # Overwrite gmx_energy_names to allow dynamical changes of energy names
        if 'charmm' in state['name']:
            state['gmx_energy_names'] = charmm_gmx_energy_names
        elif 'amber' in state['name']:
            state['gmx_energy_names'] = amber_gmx_energy_names
        else:
            raise ValueError(f"energy names for forcefield={state['name']} are undefined.")

        for k, v in state.items():
            setattr(self, k, v)


def parse_gmx_rtp(filepath: str | pathlib.Path):
    """
    Parses a GROMACS .rtp force-field file
    Args:
        filepath: filepath of the .rtp file

    Returns:
        dictionary of the defined residues
        res_dict[res_name] = res_info  where res_info = dict(atoms=dict(),bonds=list(),impropers=list())
    """
    section_patt = re.compile(r'(?<=\[).+(?=\])')
    atom_type = r'\S'
    atoms_line_patt = re.compile(rf"({atom_type}+)\s+({atom_type}+)\s+([0-9\.\-]+)\s+(\d+)")
    bonds_line_patt = re.compile(rf"({atom_type}+)\s+({atom_type}+)")
    impropers_line_patt = re.compile(rf"({atom_type}+)\s+({atom_type}+)\s+({atom_type}+)\s+({atom_type}+)")

    with open(filepath) as file:
        lines = file.readlines()

    res_dict = dict()
    for line in lines:
        line = line.lstrip()
        if not line or line[0] == '\n' or line[0] == ';':
            continue

        if line[0] == '[':
            section_name = section_patt.search(line).group().strip()
            if section_name not in ['atoms', 'bonds', 'impropers', 'bondedtypes']:
                res_name = section_name
                res_dict[res_name] = dict()
            if section_name in ['atoms', 'bonds', 'impropers'] and section_name not in res_dict[res_name]:
                # res_dict[res_name][section_name] = list()
                if section_name == 'atoms':
                    res_dict[res_name][section_name] = dict(name=[], type=[], charge=[], charge_group=[])
                else:
                    res_dict[res_name][section_name] = list()
            continue

        if section_name == 'atoms':
            # match = atoms_line_patt.findall(line)[0]
            match = atoms_line_patt.match(line).groups()
            dict_info = dict(name=match[0], type=match[1], charge=float(match[2]), charge_group=int(match[3]))
            # res_dict[res_name][section_name][match[0]] = dict_info
            for k, v in dict_info.items():
                res_dict[res_name][section_name][k].append(v)
        elif section_name == 'bonds':
            bond_pair = list(bonds_line_patt.match(line).groups())
            res_dict[res_name][section_name].append(bond_pair)
        elif section_name == 'impropers':
            improper_tetrad = list(impropers_line_patt.match(line).groups())
            res_dict[res_name][section_name].append(improper_tetrad)

    return res_dict


def parse_gmx_tdb(filepath: str | pathlib.Path):
    """
    Parses a GROMACS .tdb file (terminal database file)
    Args:
        filepath: filepath of the .rtp file

    Returns:
        dictionary of the defined residues
        res_dict[res_name] = res_info  where res_info = dict(atoms=list(),bonds=list(),impropers=list())
    """
    section_patt = re.compile(r'(?<=\[).+(?=\])')

    with open(filepath) as file:
        lines = file.readlines()

    res_dict = dict()
    for line in lines:
        line = line.lstrip()
        if not line or line[0] == '\n' or line[0] == ';':
            continue

        if line[0] == '[':
            section_name = section_patt.search(line).group().strip()
            if section_name not in ['replace', 'add', 'delete', 'impropers', 'None']:
                res_name = section_name
                res_dict[res_name] = dict()
            if section_name in ['replace', 'add', 'delete', 'impropers'] and section_name not in res_dict[res_name]:
                res_dict[res_name][section_name] = list()
            if section_name == 'add':
                n_add_lines = 0
            continue

        # Remove end of line comments
        if ';' in line: line = line[:line.index(';')]

        elements = line.split()
        if section_name == 'replace':
            elements[-2], elements[-1] = float(elements[-2]), float(elements[-1])
            res_dict[res_name][section_name].append(elements)
        elif section_name == 'add':
            if n_add_lines % 2 == 0:
                elements[0], elements[1] = int(elements[0]), int(elements[1])
                res_dict[res_name][section_name].append(elements)
            else:
                elements[-3], elements[-2], elements[-1] = [float(e) for e in elements[-3:]]
                res_dict[res_name][section_name][-1].extend(elements)  # extend the previous line
            n_add_lines += 1
        elif section_name in ['impropers', 'delete']:
            res_dict[res_name][section_name].append(elements)

    return res_dict


def import_ff_types(force_field_dir: str | pathlib.Path, rtp_basename: str, residues: list[str] = None):
    """
    Imports a subset of atom types and residue info defined by a given forcefield for a given set of residues
    Args:
        force_field_dir: GROMACS force field directory
        rtp_basename: basename of the .rtp files.
        residues: name of residues to consider when defining the atom types subset
    Returns:
        atom_types: list of atom types defined by the given force field in the canonical amino acids + SRP
    """
    if residues is None:
        residues = canonical_aa_codes_3

    if isinstance(force_field_dir, str):
        force_field_dir = pathlib.Path(force_field_dir)

    # Get all atom types from the types defined in the .rtp file
    rtp_filepath = force_field_dir / f'{rtp_basename}.rtp'
    residue_info = parse_gmx_rtp(rtp_filepath)
    residue_info = {k: v for k, v in residue_info.items() if k in residues}
    atom_types = [atom_types[:] for v in residue_info.values() for atom_types in v['atoms']['type']]

    # Get extra atom types defined in terminal residues
    c_tdb_filepath = force_field_dir / f'{rtp_basename}.c.tdb'
    c_term_types_dict = parse_gmx_tdb(c_tdb_filepath)
    replacement_atom_types = [rep[-3] for v in c_term_types_dict.values() if 'replace' in v for rep in v['replace']]
    added_atom_types = [rep[-4] for v in c_term_types_dict.values() if 'add' in v for rep in v['add']]
    atom_types += replacement_atom_types + added_atom_types

    n_tdb_filepath = force_field_dir / f'{rtp_basename}.n.tdb'
    n_term_types_dict = parse_gmx_tdb(n_tdb_filepath)
    replacement_atom_types = [rep[-3] for v in n_term_types_dict.values() if 'replace' in v for rep in v['replace']]
    added_atom_types = [rep[-4] for v in n_term_types_dict.values() if 'add' in v for rep in v['add']]
    atom_types += replacement_atom_types + added_atom_types

    atom_types = sorted(set(atom_types))
    return atom_types, residue_info


# Support residues
canonical_aa_codes_3 = list(mda.lib.util.canonical_inverse_aa_codes.keys())

# Force-field-specific atom names
pdb_terminal_OO_names = ['O', 'OXT']
charmm_C_terminal_O_names = ['OT1', 'OT2']
amber_C_terminal_O_names = ['OC1', 'OC2']
charmm_backbone_atoms_name = ['N', 'CA', 'C', 'O', 'HA', 'HN', 'H1', 'H2', 'H3', 'HN1', 'HN2', 'OT1', 'OT2', 'HT2']
amber_backbone_atoms_name = ['N', 'CA', 'C', 'O', 'HA', 'H', 'H1', 'H2', 'H3', 'OC1', 'OC2']

# GMX energy names
gmx_energy_names = {'V_bonds': 'Bond', 'V_angles': None, 'V_dih_prop': 'Proper Dih.', 'V_dih_improp': None,
                    'V_dih_prop+improp': None, 'V_dih_cmap': None, 'V_LJ_SR': 'LJ (SR)', 'V_LJ_14': 'LJ-14',
                    'V_coulomb_SR': 'Coulomb (SR)', 'V_coulomb_recip': 'Coul. recip.', 'V_coulomb_14': 'Coulomb-14',
                    'V_tot': 'Potential'}
gmx_derived_energy_names = ['V_dih_prop+improp']

# Define force-field specific names of energy values
charmm_gmx_energy_names = gmx_energy_names.copy()
charmm_gmx_energy_names['V_angles'] = 'U-B'
charmm_gmx_energy_names['V_dih_improp'] = 'Improper Dih.'
charmm_gmx_energy_names['V_dih_cmap'] = 'CMAP Dih.'
charmm_supported_residues = canonical_aa_codes_3

amber_gmx_energy_names = gmx_energy_names.copy()
amber_gmx_energy_names['V_angles'] = 'Angle'
amber_gmx_energy_names['V_dih_improp'] = 'Per. Imp. Dih.'
amber_gmx_energy_names.pop('V_dih_cmap')
amber_supported_residues = [p + r for p in ['', 'C', 'N'] for r in canonical_aa_codes_3]

# Initialize forcefields
charmm36m = Forcefield(name='charmm36m', dir=config.data_dir / 'charmm36-nov2018.ff',
                       rtp_basename='merged', gmx_energy_names=charmm_gmx_energy_names,
                       vsite_types=['MNH3', 'MNH2', 'MCH3', 'MCH3S'],
                       backbone_atoms_name=charmm_backbone_atoms_name)
charmm22star = Forcefield(name='charmm22star', dir=config.data_dir / 'charmm22star.ff',
                          rtp_basename='aminoacids', gmx_energy_names=charmm_gmx_energy_names,
                          vsite_types=['MNH3', 'MNH2', 'MCH3', 'MCH3S'],
                          backbone_atoms_name=charmm_backbone_atoms_name)
amber14sb_OL15 = Forcefield(name='amber14sb_OL15', dir=config.data_dir / 'amber14sb_OL15.ff',
                            rtp_basename='aminoacids', gmx_energy_names=amber_gmx_energy_names,
                            backbone_atoms_name=amber_backbone_atoms_name)
forcefields: dict[str, Forcefield] = {f.name: f for f in [charmm36m, charmm22star, amber14sb_OL15]}

# Combine all atom and residue types defined by the force fields
force_field_atom_types = dict()
force_fields_res_info = dict()
force_fields_heavy_atom_types = dict()
for ff in forcefields.values():
    force_field_atom_types[ff.name] = ff.atom_types
    force_fields_res_info[ff.name] = ff.residue_info
    force_fields_heavy_atom_types[ff.name] = [t for t in ff.atom_types if not t.startswith('H')]

if __name__ == '__main__':
    charmm36_atom_types, charmm36_atom_res_info = import_ff_types(charmm36m.dir, 'merged', charmm_supported_residues)
    print('CHARMM36m:', set(charmm36_atom_types) ^ set(charmm36m.atom_types))
    charmm22star_atom_types, _ = import_ff_types(charmm22star.dir, 'aminoacids', charmm_supported_residues)
    print('CHARMM22*:', set(charmm22star_atom_types) ^ set(charmm22star.atom_types))
    amber14sb_atom_types, amber14sb_res_info = import_ff_types(amber14sb_OL15.dir, 'aminoacids',
                                                               amber_supported_residues)
    print('AMBER14:', set(amber14sb_atom_types) ^ set(amber14sb_OL15.atom_types))
