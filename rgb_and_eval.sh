#!/bin/bash
#SBATCH -J BETR-ATTN
#SBATCH --mem=300GB
#SBATCH --gpus=1
#SBATCH --mail-type=NONE
#SBATCH --mail-user=omid.orh@gmail.com
#SBATCH --partition=all
#SBATCH -o ./slurm_outputs/RGB-%j.out

module load anaconda
eval "$(conda shell.bash hook)"
conda activate /home/o_heidar/bbaenv
bash start_rgb.sh
bash start_eval
