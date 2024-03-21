#!/bin/bash -x

#SBATCH -n 1                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -p gpu,tambe_gpu
#SBATCH -t 0-4:00:0         # Runtime in D-HH:MM:SS, minimum of 10 minutes
#SBATCH --mem 32G          # Memory pool for all cores (see also --mem-per-cpu) MBs
#SBATCH --cpus-per-task 24
#SBATCH --gres gpu:1
#SBATCH -o bash-outputs/%A-%a.out  # File to which STDOUT will be written, %A inserts jobid %a inserts array id
#SBATCH -e bash-errors/%A-%a.err  # File to which STDOUT will be written, %A inserts jobid %a inserts array id
set -x
date
source ~/.bashrc
conda activate ~/../../tambe_lab/Users/luciagordon/tree-mapping/tree-mapping-env
python train_models.py experiment_configs/${1} ${2} ${3} ${4}