eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

# Activate conda environment
conda activate myenv

# Train models on main datasets
#  python train.py --configs config_CS1.py --dataset nup98_12 --dataset_split_set_ID 1
#  python train.py --configs config_HA1.py --dataset nup98_12 --dataset_split_set_ID 1

# Generate datasets to assess effect of psi on heavy-atom structure generation
#  python -m models.generate --configs config_HA1.py --dataset nup98_12 --ID Psi0 --ref_dataset_split train --pruning_method gmx_energy --n_samples train --psi 0.0
#  python -m models.generate --configs config_HA1.py --dataset nup98_12 --ID Psi1 --ref_dataset_split train --pruning_method gmx_energy --n_samples train --psi 1.0
#  python -m models.generate --configs config_HA1.py --dataset nup98_12 --ID Psi2 --ref_dataset_split train --pruning_method gmx_energy --n_samples train --psi 2.0
#  python -m models.generate --configs config_HA1.py --dataset nup98_12 --ID Psi5 --ref_dataset_split train --pruning_method gmx_energy --n_samples train --psi 5.0
#  python -m models.generate --configs config_HA1.py --dataset nup98_12 --ID Psi10 --ref_dataset_split train --pruning_method gmx_energy --n_samples train --psi 10.0

# Generate coarse datasets to assess effect of psi on coarse-structure sampling
#  python -m models.generate --configs config_CS1.py --dataset nup98_12 --ID Psi0 --lambda0 1.0 --psi 0.0 --n_samples train
#  python -m models.generate --configs config_CS1.py --dataset nup98_12 --ID Psi1 --lambda0 1.0 --psi 1.0 --n_samples train
#  python -m models.generate --configs config_CS1.py --dataset nup98_12 --ID Psi2 --lambda0 1.0 --psi 2.0 --n_samples train
#  python -m models.generate --configs config_CS1.py --dataset nup98_12 --ID Psi5 --lambda0 1.0 --psi 5.0 --n_samples train
#  python -m models.generate --configs config_CS1.py --dataset nup98_12 --ID Psi10 --lambda0 1.0 --psi 10.0 --n_samples train

# Psi scans where only psi_coarse is scanned. Psi_heavy is fixed to psi=2.0.
#  python -m models.generate --ID_coarse Psi0 --ID_fine _2 --dataset nup98_12 --configs_coarse config_CS1.py --configs_fine config_HA1.py --pruning_method gmx_energy --n_samples train --psi 2.0
#  python -m models.generate --ID_coarse Psi1 --ID_fine _2 --dataset nup98_12 --configs_coarse config_CS1.py --configs_fine config_HA1.py --pruning_method gmx_energy --n_samples train --psi 2.0
#  python -m models.generate --ID_coarse Psi2 --ID_fine _2 --dataset nup98_12 --configs_coarse config_CS1.py --configs_fine config_HA1.py --pruning_method gmx_energy --n_samples train --psi 2.0
#  python -m models.generate --ID_coarse Psi5 --ID_fine _2 --dataset nup98_12 --configs_coarse config_CS1.py --configs_fine config_HA1.py --pruning_method gmx_energy --n_samples train --psi 2.0
#  python -m models.generate --ID_coarse Psi10 --ID_fine _2 --dataset nup98_12 --configs_coarse config_CS1.py --configs_fine config_HA1.py --pruning_method gmx_energy --n_samples train --psi 2.0

#  python train.py --configs config_CS1.py --dataset nup98_24 --dataset_split_set_ID 1
#  python train.py --configs config_HA1.py --dataset nup98_24 --dataset_split_set_ID 1

#  python train.py --configs config_CS1.py --dataset CLN025 --dataset_split_set_ID 1
#  python train.py --configs config_HA1.py --dataset CLN025 --dataset_split_set_ID 1

#  python train.py --configs config_CS1.py --dataset RS --dataset_split_set_ID 1
#  python train.py --configs config_HA1.py --dataset RS --dataset_split_set_ID 1

#  python train.py --configs config_CS1.py --dataset AAQAA3 --dataset_split_set_ID 1
#  python train.py --configs config_HA1.py --dataset AAQAA3 --dataset_split_set_ID 1

# Generate main datasets
#  python -m models.generate --ID_coarse C1 --ID_fine H1 --dataset CLN025 --configs_coarse config_CS1.py --configs_fine config_HA1.py --pruning_method gmx_energy --n_samples train
#  python -m models.generate --ID_coarse C1 --ID_fine H1 --dataset AAQAA3 --configs_coarse config_CS1.py --configs_fine config_HA1.py --pruning_method gmx_energy --n_samples train
#  python -m models.generate --ID_coarse C1 --ID_fine H1 --dataset RS --configs_coarse config_CS1.py --configs_fine config_HA1.py --pruning_method gmx_energy --n_samples train
#  python -m models.generate --ID_coarse C1 --ID_fine H1 --dataset nup98_12 --configs_coarse config_CS1.py --configs_fine config_HA1.py --pruning_method gmx_energy --n_samples train
#  python -m models.generate --ID_coarse C1 --ID_fine H1 --dataset nup98_24 --configs_coarse config_CS1.py --configs_fine config_HA1.py --pruning_method gmx_energy --n_samples train

# Generate thermalized version of main datasets
#  python -m chemistry.thermalize --parallel --config T100 --dataset Diff_CS1_HA1_nup98_12_1_C1H1
#  python -m chemistry.thermalize --parallel --config T100 --dataset Diff_CS1_HA1_nup98_24_1_C1H1
#  python -m chemistry.thermalize --parallel --config U100 --dataset Diff_CS1_HA1_CLN025_1_C1H1
#  python -m chemistry.thermalize --parallel --config T100 --dataset Diff_CS1_HA1_AAQAA3_1_C1H1
#  python -m chemistry.thermalize --parallel --config T100 --dataset Diff_CS1_HA1_RS_1_C1H1

# Generate examples of denoising trajectories
#  python -m models.generate --ID_coarse C1 --ID_fine H1 --dataset nup98_12 --dataset_split_set_ID 1 --configs_coarse config_CS1.py --configs_fine config_HA1.py --save_trajectory --pruning_method gmx_energy --n_samples 5
#  python -m models.generate --ID_coarse C1 --ID_fine H1 --dataset nup98_24 --dataset_split_set_ID 1 --configs_coarse config_CS1.py --configs_fine config_HA1.py --save_trajectory --pruning_method gmx_energy --n_samples 5
#  python -m models.generate --ID_coarse C1 --ID_fine H1 --dataset RS --dataset_split_set_ID 1 --configs_coarse config_CS1.py --configs_fine config_HA1.py --save_trajectory --pruning_method gmx_energy --n_samples 5
#  python -m models.generate --ID_coarse C1 --ID_fine H1 --dataset CLN025 --dataset_split_set_ID 1 --configs_coarse config_CS1.py --configs_fine config_HA1.py --save_trajectory --pruning_method gmx_energy --n_samples 5
#  python -m models.generate --ID_coarse C1 --ID_fine H1 --dataset AAQAA3 --dataset_split_set_ID 1 --configs_coarse config_CS1.py --configs_fine config_HA1.py --save_trajectory --pruning_method gmx_energy --n_samples 5

# Record conditioners energy timeseries for sampling nup98_12
#  python -m models.generate --configs config_HA1.py --dataset nup98_12 --dataset_split_set_ID 1 --ID CondU --ref_dataset_split train --save_cond_U --n_samples train

# Train minimal models
#  python train.py --configs config_CS1.py --dataset nup98_12 --dataset_split_set_ID S1
#  python train.py --configs config_HA1.py --dataset nup98_12 --dataset_split_set_ID S1
#
#  python train.py --configs config_CS1.py --dataset nup98_24 --dataset_split_set_ID S1
#  python train.py --configs config_HA1.py --dataset nup98_24 --dataset_split_set_ID S1
#
#  python train.py --configs config_CS1.py --dataset CLN025 --dataset_split_set_ID S1
#  python train.py --configs config_HA1.py --dataset CLN025 --dataset_split_set_ID S1
#
#  python train.py --configs config_CS1.py --dataset RS --dataset_split_set_ID S1
#  python train.py --configs config_HA1.py --dataset RS --dataset_split_set_ID S1
#
#  python train.py --configs config_CS1.py --dataset AAQAA3 --dataset_split_set_ID S1
#  python train.py --configs config_HA1.py --dataset AAQAA3 --dataset_split_set_ID S1

# Train models on capped systems
#  python train.py --configs config_CS1.py --dataset nup98C_12 --dataset_split_set_ID 1
#  python train.py --configs config_HA1.py --dataset nup98C_12 --dataset_split_set_ID 1
#
#  python train.py --configs config_CS1.py --dataset nup98AC_12 --dataset_split_set_ID 1
#  python train.py --configs config_HA1.py --dataset nup98AC_12 --dataset_split_set_ID 1

# Train models that don't have cross-product modules
#  python train.py --configs config_CS1_NoCP.py --dataset nup98_12 --dataset_split_set_ID 1
#  python train.py --configs config_HA1_NoCP.py --dataset nup98_12 --dataset_split_set_ID 1

# Generate minimal datasets
#  python -m models.generate --ID_coarse C1 --ID_fine H1 --dataset nup98_12 --dataset_split_set_ID S1 --configs_coarse config_CS1.py --configs_fine config_HA1.py --model_epoch 2000 --pruning_method gmx_energy --n_samples test
#  python -m models.generate --ID_coarse C1 --ID_fine H1 --dataset nup98_24 --dataset_split_set_ID S1 --configs_coarse config_CS1.py --configs_fine config_HA1.py --model_epoch 2000 --pruning_method gmx_energy --n_samples test
#  python -m models.generate --ID_coarse C1 --ID_fine H1 --dataset CLN025 --dataset_split_set_ID S1 --configs_coarse config_CS1.py --configs_fine config_HA1.py --model_epoch 2000 --pruning_method gmx_energy --n_samples test
#  python -m models.generate --ID_coarse C1 --ID_fine H1 --dataset RS --dataset_split_set_ID S1 --configs_coarse config_CS1.py --configs_fine config_HA1.py --model_epoch 2000 --pruning_method gmx_energy --n_samples test
#  python -m models.generate --ID_coarse C1 --ID_fine H1 --dataset AAQAA3 --dataset_split_set_ID S1 --configs_coarse config_CS1.py --configs_fine config_HA1.py --model_epoch 2000 --pruning_method gmx_energy --n_samples test

# Generate datasets of capped systems
#  python -m models.generate --ID_coarse C1 --ID_fine H1 --dataset nup98C_12 --configs_coarse config_CS1.py --configs_fine config_HA1.py --pruning_method gmx_energy --n_samples train
#  python -m models.generate --ID_coarse C1 --ID_fine H1 --dataset nup98AC_12 --configs_coarse config_CS1.py --configs_fine config_HA1.py --pruning_method gmx_energy --n_samples train

# Generate datasets with models that don't have cross-product modules
#  python -m models.generate --ID_coarse C1 --ID_fine H1 --dataset nup98_12 --configs_coarse config_CS1_NoCP.py --configs_fine config_HA1_NoCP.py --pruning_method gmx_energy --n_samples train

# Chignolin Experiments
#  python train.py --configs config_CS1.py --dataset CLN025 --dataset_split_set_ID FinTr2_5
#  python train.py --configs config_HA1.py --dataset CLN025 --dataset_split_set_ID FinTr2_5

#  python train.py --configs config_HA1.py --dataset CLN025 --dataset_split_set_ID FinTr3
#  python train.py --configs config_CS1.py --dataset CLN025 --dataset_split_set_ID FinTr3

#  python train.py --configs config_CS1.py --dataset CLN025 --dataset_split_set_ID UinTr4
#  python train.py --configs config_HA1.py --dataset CLN025 --dataset_split_set_ID UinTr4

#  python train.py --configs config_CS1.py --dataset CLN025 --dataset_split_set_ID UinTr5
#  python train.py --configs config_HA1.py --dataset CLN025 --dataset_split_set_ID UinTr5

#  python train.py --configs config_CS1.py --dataset CLN025 --dataset_split_set_ID UinTr6
#  python train.py --configs config_HA1.py --dataset CLN025 --dataset_split_set_ID UinTr6

#  python -m models.generate --configs config_CS1.py --dataset CLN025 --dataset_split_set_ID FinTr2_5 --ID C1 --n_samples 25000
#  python -m models.generate --ID_coarse C1 --ID_fine H1 --dataset CLN025 --dataset_split_set_ID FinTr2_5 --configs_coarse config_CS1.py --configs_fine config_HA1.py --pruning_method gmx_energy --n_samples 25000

#  python -m models.generate --configs config_CS1.py --dataset CLN025 --dataset_split_set_ID FinTr3 --ID C1 --n_samples 25000
#  python -m models.generate --ID_coarse C1 --ID_fine H1 --dataset CLN025 --dataset_split_set_ID FinTr3 --configs_coarse config_CS1.py --configs_fine config_HA1.py --pruning_method gmx_energy --n_samples 25000

#  python -m models.generate --configs config_CS1.py --dataset CLN025 --dataset_split_set_ID UinTr4 --ID C1 --n_samples 25000
#  python -m models.generate --ID_coarse C1 --ID_fine H1 --dataset CLN025 --dataset_split_set_ID UinTr4 --configs_coarse config_CS1.py --configs_fine config_HA1.py --pruning_method gmx_energy --n_samples 25000

#  python -m models.generate --configs config_CS1.py --dataset CLN025 --dataset_split_set_ID UinTr5 --ID C1 --n_samples 25000
#  python -m models.generate --ID_coarse C1 --ID_fine H1 --dataset CLN025 --dataset_split_set_ID UinTr5 --configs_coarse config_CS1.py --configs_fine config_HA1.py --pruning_method gmx_energy --n_samples 25000

#  python -m models.generate --configs config_CS1.py --dataset CLN025 --dataset_split_set_ID UinTr6 --ID C1 --n_samples 25000
#  python -m models.generate --ID_coarse C1 --ID_fine H1 --dataset CLN025 --dataset_split_set_ID UinTr6 --configs_coarse config_CS1.py --configs_fine config_HA1.py --pruning_method gmx_energy --n_samples 25000
