#!/bin/bash
#SBATCH -J BBA-EVAL
#SBATCH --mem=900GB
#SBATCH --gpus=1
#SBATCH --mail-type=NONE
#SBATCH --mail-user=omid.orh@gmail.com
#SBATCH --partition=all
#SBATCH -o ./slurm_outputs/EVAL-%j.out

module load anaconda
eval "$(conda shell.bash hook)"
conda activate /home/o_heidar/bbaenv
bash start_eval.sh