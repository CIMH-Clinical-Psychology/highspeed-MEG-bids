#!/bin/bash
#SBATCH --job-name=meg_maxfilter
#SBATCH --output=/zi/home/simon.kern/slurmlogs/slurm_%x_%A-%a.out
#SBATCH --error=/zi/home/simon.kern/slurmlogs/slurm_%x_%A-%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=1-30

source ~/Nextcloud/ZI/misc/setup_slurm.sh

SUBJECT=$(printf "%02d" ${SLURM_ARRAY_TASK_ID})

python run_maxfilter.py --subject ${SUBJECT}
