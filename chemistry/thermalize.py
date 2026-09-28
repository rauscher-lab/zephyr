from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import tempfile
import functools
import multiprocessing

import MDAnalysis as mda
import numpy as np
import torch
import tqdm

import config
from utils import parse_gmx_header, gmx_pdb2gmx, gmx_editconf, gmx_solvate, gmx_grompp, gmx_genion, \
    gmx_top_modify_posres, gmx_top_modify_molecules, gmx_mdrun, parse_gmx_mdp, modify_gmx_mdp, format_gmx_cmd, run_cmd, \
    gmx_anonymize, gmx_info, delayed_delete, Archiver, Timer
from data.data_classes import all_atoms_class_name, heavy_atoms_class_name, Biomolecule, prot_sel_crit
from data.datasets import load_dataset, XTCDataset, GeneratedDataset, BiomoleculeDataset
from analysis.metrics import analyze_dataset
from analysis.comparisons import compare_datasets

# MDP parameters used for equilibration
mdp_dir = config.data_dir / 'mdp'
gmx_EM_params_filepath = mdp_dir / 'em.mdp'
gmx_ions_params_filepath = mdp_dir / 'ions.mdp'
gmx_EQ_NVT_params_filepath = mdp_dir / 'EQ_NVT.mdp'
gmx_therm_NPT_params_filepath = mdp_dir / 'thermalize_NPT.mdp'
gmx_therm_NVT_params_filepath = mdp_dir / 'thermalize_NVT.mdp'


class GMXThermalizer:
    def __init__(self, top_filepath: str | pathlib.Path, forcefield_dir: str | pathlib.Path,
                 EM_params_filepath=gmx_EM_params_filepath, EQ_params_filepath=gmx_therm_NVT_params_filepath,
                 working_dir=None, terminals_code=None):
        self.top_filepath = pathlib.Path(top_filepath)
        self.EQ_params_filepath = pathlib.Path(EQ_params_filepath)
        self.EM_params_filepath = pathlib.Path(EM_params_filepath)
        self.forcefield_dir = pathlib.Path(forcefield_dir)
        self.is_working_dir_temp = working_dir is None
        if self.is_working_dir_temp:
            working_dir = tempfile.mkdtemp(prefix=f'{self.__class__.__name__}_')
        if terminals_code is None:
            terminals_code = ['1', '1']
        self.terminals_code = terminals_code
        self.working_dir = pathlib.Path(working_dir)

    def thermalize(self, mol: Biomolecule, resume=True, verbose=False, box_gro_filepath=None, box_top_filepath=None,
                   EM_steps=50000, EM_stepsize: float = None, EQ_time=100, element_class_out=None):
        """
        Thermalizes a given molecule by running a short NVT MD simulation using GROMACS.
        Args:
            mol: molecule that will be thermalized
            resume: resume the estimation
            verbose: output gmx function results
            box_gro_filepath: GROMACS .gro filepath of the template box
            box_top_filepath: GROMACS .top filepath of the template box
            EM_steps: Maximum number of steps in the energy minimization
            EM_stepsize: Step size used during EM ('emstep' .mdp parameter)
            EQ_time: Equilibration time in ps
            element_class_out: Element class of the output molecule. Defaults to the element class of the given mol.
        Returns:
            Biomolecule
        """
        # Create a working directory where outputs will be saved
        output_basename = f"{mol.ID}"
        working_dir = self.working_dir / output_basename
        topology_filepath = self.top_filepath

        if not resume and working_dir.exists():
            shutil.rmtree(working_dir)
        working_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(working_dir)

        # Create a PDB file of the molecule.
        pdb_filepath = working_dir / f"{output_basename}.pdb"
        if not pdb_filepath.exists():
            mol.save_pdb(top_filepath=topology_filepath, pdb_filepath=pdb_filepath)

        # Prepare system with pdb2gmx
        pdb2gmx_kwargs = dict(lb=0.0, forcefield_dir=self.forcefield_dir, ter=None,
                              terminals_protonation=self.terminals_code, ignh=mol.element_class == all_atoms_class_name)

        # If a box .top file is given, use the same virtual sites and water option as the one used in the given .top file
        if box_top_filepath is not None:
            box_top_header_attrs = parse_gmx_header(filepath=box_top_filepath)
            box_pdb2gmx_args = box_top_header_attrs['cmd_args']
            for n in {'-vsite', '-water'} & set(box_pdb2gmx_args):
                pdb2gmx_kwargs[n[1:]] = box_pdb2gmx_args[n]

        gmx_pdb2gmx(pdb_filepath=pdb_filepath, output_basename=output_basename, working_dir=working_dir,
                    verbose=verbose, **pdb2gmx_kwargs)
        prot_gro_filename = f"{output_basename}.gro"
        prot_top_filename = f"{output_basename}.top"

        # Create the simulation box with solvent if not given
        EM_input_gro_filepath = working_dir / f"{output_basename}_full.gro"
        System_top_filepath = EM_input_gro_filepath.with_suffix('.top')
        box_init_gro_filepath = working_dir / f"{output_basename}_box_init.gro"
        box_init_top_filepath = box_init_gro_filepath.with_suffix('.top')
        solvate_output_gro_filename = box_init_gro_filepath.with_stem(box_init_gro_filepath.stem + '_solvated').name
        if not EM_input_gro_filepath.exists():
            if box_gro_filepath is None:
                gmx_editconf(gro_filename=prot_gro_filename, gro_output_filename=box_init_gro_filepath.name, d=1.0,
                             verbose=verbose)
                if not box_init_top_filepath.exists():
                    shutil.copy(prot_top_filename, box_init_top_filepath.name)

                # Add solvent
                genion_tpr_filename = f"{output_basename}_ions.tpr"
                gmx_solvate(gro_filename=box_init_gro_filepath.name, top_filename=box_init_top_filepath.name,
                            output_gro_filename=solvate_output_gro_filename, verbose=verbose)

                # Copy the solvated box .top file and use it as the system box for EM and EQ.
                if not System_top_filepath.exists():
                    shutil.copy(box_init_top_filepath, System_top_filepath)

                # Add ions
                gmx_grompp(params_filename=gmx_ions_params_filepath, gro_filename=solvate_output_gro_filename,
                           top_filename=System_top_filepath.name, output_tpr_filename=genion_tpr_filename,
                           verbose=verbose)

                gmx_genion(tpr_filename=genion_tpr_filename, output_gro_filename=EM_input_gro_filepath.name,
                           output_top_filename=System_top_filepath.name, verbose=verbose, conc=0.15)
            else:
                assert box_top_filepath is not None, ".top file of the input box must be given."
                assert box_top_filepath.exists(), ".top file of the input box is missing."

                # Copy the input .top file to the working directory
                if not box_init_top_filepath.exists():
                    shutil.copy(working_dir / prot_top_filename, box_init_top_filepath)

                    # Modify the name of the position restraint file
                    gmx_top_modify_posres(box_init_top_filepath, pos_res_filename=f"{output_basename}_posre.itp")

                # Create a box of pure water with the same dimension as the input box
                box_input_uni = mda.Universe(box_gro_filepath)
                n_pos_ions = len(box_input_uni.select_atoms('resname NA').residues)
                n_neg_ions = len(box_input_uni.select_atoms('resname CL').residues)
                N_solvent_ori = len(box_input_uni.select_atoms('not protein').residues)
                if not box_init_gro_filepath.exists():
                    box_input_uni.select_atoms('water').write(box_init_gro_filepath)
                box_input_uni.trajectory.close()

                # Fill empty space with water molecules
                gmx_solvate(gro_filename=box_init_gro_filepath.name, top_filename=box_init_top_filepath.name,
                            output_gro_filename=solvate_output_gro_filename, verbose=verbose)

                # Insert the protein in the box of pure water
                pure_water_uni = mda.Universe(working_dir / solvate_output_gro_filename)
                prot_uni = mda.Universe(working_dir / prot_gro_filename)
                box_uni = mda.Merge(prot_uni.select_atoms('all'), pure_water_uni.select_atoms('all'))
                box_uni.dimensions = pure_water_uni.dimensions

                # Translate protein to center
                prot_sel = box_uni.select_atoms('protein')
                dx = pure_water_uni.atoms.center_of_geometry() - prot_sel.atoms.center_of_geometry()
                prot_sel.atoms.translate(dx)

                # Remove water molecules that are closest to the protein to match the solvent content of the input box .gro.
                # There must be N_H2O + N_ions water molecules remaining where
                # N_H2O = # of water of molecules in input box .gro and N_ions = # of ion molecules
                water_sel = box_uni.select_atoms('not protein')
                clash_cutoff = 1.5  # Angstrom

                # Calculate the number of water molecules that needs to be removed
                N_removed_water = len(pure_water_uni.select_atoms('not protein').residues) - N_solvent_ori

                # Find the water molecules that are within the cutoff distance of any atoms of the protein

                # (Old method of removing water molecules to match solvent content)
                # # Remove water molecules that are closest to the protein
                # water_mols = water_sel.residues[np.argsort(water_closest_dist)][N_removed_water:]

                # # (Old Method 2)
                # water_closest_dist = np.zeros(len(water_sel.residues))
                # for i, water_mol in enumerate(water_sel.residues):
                #     water_prot_dx = water_mol.atoms.positions[None, :, :] - prot_sel.positions[:, None, :]
                #     water_prot_atom_dist = np.linalg.norm(water_prot_dx, axis=-1)
                #     water_closest_dist[i] = water_prot_atom_dist.min()

                # Use capped_distance() with box input to account for periodic boundary conditions
                clashes_atom_ind, clashes_dist = mda.lib.distances.capped_distance(reference=prot_sel.positions,
                                                                                   configuration=water_sel.positions,
                                                                                   max_cutoff=clash_cutoff,
                                                                                   box=box_uni.dimensions)
                water_sel_resid_to_resind = np.zeros(water_sel.resids.max() + 1, dtype=int)
                water_sel_resid_to_resind[water_sel.residues.resids] = np.arange(len(water_sel.residues))
                clashes_water_ind = water_sel_resid_to_resind[water_sel.resids[clashes_atom_ind[:, 1]]]
                water_closest_dist = np.full(len(water_sel.residues), np.inf)
                np.minimum.at(water_closest_dist, clashes_water_ind, clashes_dist)

                # First remove all water molecules that are within the cutoff distance of the protein.
                is_water_not_clashing = water_closest_dist >= clash_cutoff
                N_water_clashes = is_water_not_clashing.size - is_water_not_clashing.sum()
                if N_water_clashes < N_removed_water:
                    # First, remove all clashes
                    water_mols = water_sel.residues[is_water_not_clashing]

                    # For the rest, select water molecules randomly
                    np.random.seed(hash(mol.ID) % (10 ** 8))  # Set seed for reproducibility
                    kept_water_ind = np.random.permutation(len(water_mols))[:N_solvent_ori]
                    water_mols = water_mols[kept_water_ind]
                else:
                    # Remove water molecules that are closest to the protein
                    kept_water_ind = np.argsort(water_closest_dist)[N_removed_water:]
                    water_mols = water_sel.residues[kept_water_ind]

                prot_and_water_sel = prot_sel + water_mols.atoms
                box_and_prot_gro_filepath = working_dir / f"{output_basename}_and_water.gro"
                prot_and_water_sel.write(box_and_prot_gro_filepath)

                # Copy the water box .top file and adjust its solvent content to match those of the given box .gro file
                N_water_mols = len(prot_and_water_sel.select_atoms('water').residues)
                molecules_content = dict(Protein_chain_A=1, SOL=N_water_mols)
                if not System_top_filepath.exists():
                    shutil.copy(box_init_top_filepath, System_top_filepath)
                    gmx_top_modify_molecules(System_top_filepath, molecules_content)

                # Add ions to the box to match the number of ions in the input box
                box_and_prot_tpr_filename = box_and_prot_gro_filepath.with_suffix('.tpr').name

                gmx_grompp(params_filename=gmx_ions_params_filepath, gro_filename=box_and_prot_gro_filepath.name,
                           top_filename=System_top_filepath.name, output_tpr_filename=box_and_prot_tpr_filename,
                           verbose=verbose)

                gmx_genion(tpr_filename=box_and_prot_tpr_filename, output_gro_filename=EM_input_gro_filepath.name,
                           output_top_filename=System_top_filepath.name, nn=n_neg_ions, np=n_pos_ions,
                           verbose=verbose)

                # Close Universes to avoid too many open files
                pure_water_uni.trajectory.close()
                prot_uni.trajectory.close()
                box_uni.trajectory.close()

        # Copy the .mdp parameter file used for energy minimization
        EM_params_filepath = working_dir / self.EM_params_filepath.name
        if not EM_params_filepath.exists():
            shutil.copy(self.EM_params_filepath, EM_params_filepath)

        # Modify the EM stepsize if given
        if EM_stepsize is not None:
            modify_gmx_mdp(EM_params_filepath, emstep=EM_stepsize)

        # Copy the .mdp parameter file used for equilibration
        EQ_params_filepath = working_dir / self.EQ_params_filepath.name
        if not EQ_params_filepath.exists():
            shutil.copy(self.EQ_params_filepath, EQ_params_filepath)

        # Modify the nsteps parameter of the .mdp file if given
        if EQ_time is not None:
            EQ_params = parse_gmx_mdp(EQ_params_filepath)
            nsteps = int(round(EQ_time / EQ_params['dt']))  # EQ_time is in ps
            modify_gmx_mdp(EQ_params_filepath, nsteps=nsteps)

        # Check if GROMACS has been compiled with thread_mpi. If so, use 1 thread-mpi rank
        gmx_info_dict = gmx_info()
        mdrun_kwargs = dict(ntmpi=1) if gmx_info_dict['MPI library'] == 'thread_mpi' else dict()

        # Minimize energy
        EM_output_basename = f"{output_basename}_EM"
        gmx_mdrun(gro_filepath=EM_input_gro_filepath, top_filename=System_top_filepath,
                  params_filepath=EM_params_filepath, output_basename=EM_output_basename, verbose=verbose,
                  nsteps=EM_steps, **mdrun_kwargs)

        # Equilibrate
        mdrun_kwargs |= dict(nb='gpu', pme='gpu', bonded='gpu')
        EQ_input_gro_filepath = (working_dir / EM_output_basename).with_suffix('.gro')
        EQ_output_basename = f"{output_basename}_EQ"
        # # Use GPU if there are no virtual sites
        # vsites = mda.Universe(EQ_input_gro_filepath).select_atoms('name M* or name V* or name DUM* or name EP*')
        # if len(vsites) == 0:
        #     mdrun_kwargs |= dict(nb='gpu', pme='gpu', bonded='gpu', update='gpu')
        gmx_mdrun(gro_filepath=EQ_input_gro_filepath, top_filename=System_top_filepath,
                  params_filepath=EQ_params_filepath, output_basename=EQ_output_basename, verbose=verbose,
                  grompp_kwargs={'maxwarn': 1}, **mdrun_kwargs)

        # Create a protein only .gro file of the final frame
        prot_only_gro_filename = f"{EQ_output_basename}_prot.gro"
        trjconv_cmd = format_gmx_cmd(cmd='trjconv', prefix='printf "Protein\nProtein\n" | ',
                                     s=f"{EQ_output_basename}.tpr", f=f"{EQ_output_basename}.gro",
                                     o=prot_only_gro_filename,
                                     center=None, pbc='mol', ur='compact')
        run_cmd(trjconv_cmd, output_filepath=prot_only_gro_filename, print_output=verbose)

        # Load the last frame and save the protein structure into a Biomolecule
        if element_class_out is None:
            element_class_out = mol.element_class
        gro_out_filepath = working_dir / prot_only_gro_filename
        gro_uni = mda.Universe(gro_out_filepath, convert_units=False)
        gro_uni_prot = gro_uni.select_atoms(prot_sel_crit)
        mol_out = Biomolecule.from_topology(top_filepath=topology_filepath,
                                            elements_position=gro_uni_prot.atoms.positions,
                                            element_class=element_class_out, ID=mol.ID)
        gro_uni.trajectory.close()
        return mol_out


def thermalize_worker_func(thermalizer: GMXThermalizer, mol: Biomolecule, **kwargs):
    try:
        return thermalizer.thermalize(mol, **kwargs)
    except Exception as e:
        print(f'Thermalization failed on molecule {mol.ID!r} with error:\n{e}')
        print('Re-trying with smaller EM step')

        if 'EM_stepsize' in kwargs:
            kwargs['EM_stepsize'] /= 100
        else:
            kwargs['EM_stepsize'] = 0.0001

        working_dir = thermalizer.working_dir / mol.ID
        if working_dir.exists():
            shutil.rmtree(working_dir)

        try:
            return thermalizer.thermalize(mol, **kwargs)
        except Exception as e:
            pass
            print(f'Thermalization failed a second time on molecule {mol.ID!r} with error:\n{e}')

        # Rename the working dir to indicate error
        working_dir_err = working_dir.with_name(working_dir.name + '_err')
        if not working_dir_err.exists():
            print(f'Renaming {working_dir.name!r} to {working_dir_err.name!r}')
            working_dir.rename(working_dir_err)
        else:
            print(f'{working_dir.name!r} was not renamed.')
        return None


def define_thermalization_configs(config_ID: str):
    configs = dict(EQ_params_filepath=gmx_therm_NVT_params_filepath, EM_params_filepath=gmx_EM_params_filepath)
    if config_ID in ['T100', 'W100', 'U100']:
        configs |= dict(EQ_time=100, EM_steps=10000)
        if config_ID == 'U100':
            configs['ref_t'] = 340  # CLN025 melting temperature
    else:
        raise ValueError(f"Thermalization config {config_ID!r} is not defined.")

    return configs


def thermalize_dataset(dataset: str | BiomoleculeDataset, config_ID: str = 'T100', parallel=False, regen=False,
                       dataset_suffix: str = None, allow_failures=False):
    """
    Creates a thermalized version of a dataset by immersing each molecule in a water box and running short MD simulation
    Args:
        dataset: dataset to thermalize
        config_ID: ID of the thermalization configurations. See define_thermalization_configs().
        parallel: Use workers to run parallel MD trajectories
        regen: delete previous thermalized dataset and regenerate it
        dataset_suffix: suffix added to the directory of the thermalized dataset. Defaults to '_{config_ID}'
        allow_failures: For molecules that failed to thermalize, use the original un-thermalized version.

    Returns:
        thermalized dataset of the same class as self
    """

    if isinstance(dataset, str):
        dataset = load_dataset(dataset)

    if dataset_suffix is None:
        dataset_suffix = f"_{config_ID}"

    if parallel:
        os.environ['OMP_NUM_THREADS'] = '1'

    # When debugging, use a subset for shorter processing time
    if config.debug:
        subset_dir = pathlib.Path(dataset.dir.parent, f'{dataset.name}_TDebug')
        size = min(len(dataset), 50)
        if not subset_dir.exists():
            dataset = dataset.subset(ind=torch.arange(size), inplace=False, inmemory=False, directory=subset_dir)
        else:
            dataset = load_dataset(directory=subset_dir)

    # Define directories where thermalized dataset will be saved
    therm_dataset_name = f"{dataset.name}{dataset_suffix}"
    therm_dataset_dir = dataset.dir.parent / therm_dataset_name
    therm_dataset_temp_dir = dataset.dir.parent / (therm_dataset_name + '_temp')
    thermalization_subdir = therm_dataset_temp_dir / 'gmx_thermalization'

    # Remove previous estimates if regenerating
    if regen:
        if therm_dataset_dir.exists():
            delayed_delete(therm_dataset_dir, delay=5)
        if therm_dataset_temp_dir.exists():
            delayed_delete(therm_dataset_temp_dir, delay=5)

    # Return thermalized dataset if found
    if dataset.__class__.is_processed(therm_dataset_dir):
        return dataset.__class__(directory=therm_dataset_dir)

    # Find the MD box configuration files for the given dataset
    box_files_basename = None
    box_files_stem = ['AAQAA3', 'CLN025', 'nup98_12', 'nup98_24', 'RS']
    if isinstance(dataset, XTCDataset):
        sim_dataset_name = dataset.name
    elif isinstance(dataset, GeneratedDataset):
        sim_dataset_name = dataset.model_dataset
    else:
        raise NotImplementedError
    for stem in box_files_stem:
        if sim_dataset_name.startswith(stem):
            box_files_basename = pathlib.Path(config.data_dir, 'MD_box', stem)
            break
    if box_files_basename is None:
        raise ValueError(f"Thermalization box files were not found for dataset {dataset.name!r}.")
    box_gro_filepath = box_files_basename.with_suffix('.gro')
    box_top_filepath = box_files_basename.with_suffix('.top')

    # Define the thermalization configs from the config ID
    thermalize_configs = define_thermalization_configs(config_ID)
    EQ_params_filepath = thermalize_configs.pop('EQ_params_filepath')
    EM_params_filepath = thermalize_configs.pop('EM_params_filepath')

    # Create a temporary .mdp file to keep the original intact
    EQ_params_filepath_ref = EQ_params_filepath
    EQ_params_filepath = thermalization_subdir / EQ_params_filepath_ref.name
    thermalization_subdir.mkdir(parents=True, exist_ok=True)
    shutil.copy(EQ_params_filepath_ref, EQ_params_filepath)

    # Change the reference temperature of the equilibration if defined in configs
    ref_t = thermalize_configs.pop('ref_t', None)
    if ref_t:
        modify_gmx_mdp(EQ_params_filepath, ref_t=ref_t)

    # Modify the nsteps parameter of the .mdp file using the EQ_time value of the config
    EQ_time = thermalize_configs.pop('EQ_time', None)
    if EQ_time is not None:
        EQ_params = parse_gmx_mdp(EQ_params_filepath)
        nsteps = int(round(EQ_time / EQ_params['dt']))  # EQ_time is in ps
        modify_gmx_mdp(EQ_params_filepath, nsteps=nsteps)
        thermalize_configs['EQ_time'] = None  # Ensures GMXThermalizer.thermalize() does not rewrite the file

    # Initialization
    thermalizer = GMXThermalizer(working_dir=thermalization_subdir, top_filepath=dataset.top_filepath,
                                 forcefield_dir=dataset.forcefield.dir, terminals_code=dataset.terminals_gmx_code,
                                 EQ_params_filepath=EQ_params_filepath, EM_params_filepath=EM_params_filepath)
    archiver = Archiver(dir=thermalization_subdir / f"molecules_thermalized_arch")
    timer = Timer(checkpoint_filepath=thermalization_subdir / 'thermalization_time_stats', resume=True)

    # Iterate through each remaining molecule
    inds_to_do = [i for i, ID in enumerate(dataset.molecules_ID) if ID not in archiver.keys]
    if inds_to_do:
        mols_iter = (dataset[i] for i in inds_to_do)
        worker_func = functools.partial(thermalize_worker_func, thermalizer, box_gro_filepath=box_gro_filepath,
                                        box_top_filepath=box_top_filepath, **thermalize_configs)

        if parallel:
            pool = multiprocessing.Pool()
            tasks_iter = pool.imap(worker_func, mols_iter)
        else:
            tasks_iter = map(worker_func, mols_iter)

        tasks_iter = tqdm.tqdm(tasks_iter, total=len(inds_to_do), desc=f'Thermalizing {dataset.name!r}')
        with archiver, timer:
            for mol in tasks_iter:
                if mol is not None:
                    archiver.save_to_buffer(**{mol.ID: mol})
                timer.save_checkpoint()

        if parallel:
            pool.close()

        os.chdir(dataset.dir.parent)

    # Remove the 'OMP_NUM_THREADS' environment variable that was added
    if parallel:
        os.environ.pop('OMP_NUM_THREADS', None)

    # Check if all molecules have thermalized without errors
    mols_thermalized = archiver.load()
    molecules_ID_error = [ID for ID in dataset.molecules_ID if ID not in mols_thermalized]
    if allow_failures:
        for ID in molecules_ID_error:
            mol_ori = dataset[dataset.molecules_ID.index(ID)]
            assert mol_ori.ID == ID, f"Original molecule ID {mol_ori.ID!r} does not match erroneous ID {ID!r}"
            mols_thermalized[ID] = mol_ori
    elif molecules_ID_error:
        raise ValueError(f"The following molecules ID are missing:\n{molecules_ID_error}")

    # Create a new dataset that has the thermalized molecules.
    if not dataset.__class__.is_processed(therm_dataset_temp_dir):
        mols_thermalized_list = [mols_thermalized[k] for k in dataset.molecules_ID]
        dataset_thermalized = dataset.__class__.from_molecules(mols_thermalized_list,
                                                               name=therm_dataset_name,
                                                               directory=therm_dataset_temp_dir)
        dataset_thermalized.inherit_metadata(dataset)
    else:
        dataset_thermalized = dataset.__class__(name=therm_dataset_name, directory=therm_dataset_temp_dir)

    # Save a copy of the thermalized structures in case they are needed after hydrogenization.
    processed_copy_filepath = dataset_thermalized.processed_npz_filepath.with_stem('molecules_thermalized')
    if not processed_copy_filepath.exists():
        shutil.copy(dataset_thermalized.processed_npz_filepath, processed_copy_filepath)

    # Add thermalization time to sampling time of GeneratedDatasets
    if isinstance(dataset_thermalized, GeneratedDataset):
        if dataset_thermalized.sampling_time is None:
            dataset_thermalized.sampling_time = 0.0
        dataset_thermalized.sampling_time += timer.elapsed_time
        dataset_thermalized.save_metadata()
        timer.delete_checkpoint()

    # Add hydrogen atoms to the thermalized dataset
    dataset_thermalized.hydrogenize()

    # Cleanup temporary files
    # Delete all gmx temp directories except 1 to keep as example
    for ID in dataset.molecules_ID[1:]:
        sub_working_dir = thermalizer.working_dir / ID
        if sub_working_dir.exists():
            shutil.rmtree(sub_working_dir)
        sub_working_dir_err = thermalizer.working_dir / (ID + '_err')
        if sub_working_dir_err.exists():
            shutil.rmtree(sub_working_dir_err)

    EQ_params_filepath.unlink(missing_ok=True)

    # In the example working directory, remove .log files and anonymize .top files.
    example_work_dir = thermalizer.working_dir / dataset.molecules_ID[0]
    for log_file in example_work_dir.rglob('*.log'):
        log_file.unlink()
    anonymized_files = list(example_work_dir.rglob('*.top')) + list(example_work_dir.rglob('*.mdp'))
    for file in anonymized_files:
        gmx_anonymize(file, file)

    # Destroy archive
    archiver.destroy()

    # Rename temporary dataset directory to target directory
    therm_dataset_temp_dir.rename(therm_dataset_dir)

    return dataset.__class__(directory=therm_dataset_dir)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='T100', help='ID of the thermalization configs.')
    p.add_argument('--dataset', type=str, default=None, help='Name of dataset to thermalize.')
    p.add_argument('--parallel', action='store_true', help='Run thermalization in parallel.')
    p.add_argument('--debug', action='store_true', help='Activate debug mode.')
    p.add_argument('--allow_fail', action='store_true', help='Allows molecules that failed thermalization.')
    args = p.parse_args()
    config.debug = args.debug

    if args.dataset is not None:
        dataset = load_dataset(args.dataset)
        dataset_thermalized = thermalize_dataset(dataset=dataset, parallel=args.parallel, config_ID=args.config,
                                                 allow_failures=args.allow_fail)
        dataset_thermalized_pruned = dataset_thermalized.prune()

        # Analyze heavy-atoms and all-atoms structures of the pruned dataset
        for elem_class in [all_atoms_class_name, heavy_atoms_class_name]:
            for d in [dataset_thermalized_pruned]:
                d.element_class = elem_class
                analyze_dataset(d)

        # Compare the thermalized dataset with its non-thermalized version
        compared_datasets = [dataset.model_dataset, dataset.prune(), dataset_thermalized_pruned]
        compare_datasets(compared_datasets, element_class=heavy_atoms_class_name)

    # # Test GMX Thermalizer
    # dataset_subset_dir = pathlib.Path(dataset_tempdir, f'{dataset.name}_tmpx2c2jptn')
    # if not dataset_subset_dir.exists():
    #     dataset = dataset.subset(ind=torch.arange(10), inplace=False, inmemory=False, directory=dataset_subset_dir)
    # else:
    #     dataset = load_dataset(directory=dataset_subset_dir)
    # print(dataset.dir)
    # thermalize_dataset(dataset, config_ID='W100')
    # exit()
