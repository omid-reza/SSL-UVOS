#!/bin/bash
#SBATCH -J BBA-Mask
#SBATCH --mem=80GB
#SBATCH --gpus=1
#SBATCH --mail-type=NONE
#SBATCH --mail-user=omid.orh@gmail.com
#SBATCH --partition=all
#SBATCH -o ./slurm_outputs/MaskVisualization-%j.out
#SBATCH -w virya5

module load anaconda
eval "$(conda shell.bash hook)"
conda activate /home/o_heidar/bbaenv
python MaskVisualization.py