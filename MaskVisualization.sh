#!/bin/bash
#SBATCH -J BBA-Mask
#SBATCH --mem=130GB
#SBATCH --gpus=1
#SBATCH --mail-type=NONE
#SBATCH --mail-user=omid.orh@gmail.com
#SBATCH --partition=phy
#SBATCH -o ./slurm_outputs/MaskVisualization-%j.out
#SBATCH -w virya2

module load anaconda
eval "$(conda shell.bash hook)"
conda activate /home/o_heidar/bbaenv
python MaskVisualization.py