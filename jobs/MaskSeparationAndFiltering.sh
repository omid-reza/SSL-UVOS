#!/bin/bash
#SBATCH -J BBAMask
#SBATCH --mem=200GB
#SBATCH --gpus=1
#SBATCH --mail-type=NONE
#SBATCH --mail-user=omid.orh@gmail.com
#SBATCH --partition=all
#SBATCH -o ./slurm_outputs/Mask-%j.out
#SBATCH --chdir=../
#SBATCH -w virya5

module load anaconda
eval "$(conda shell.bash hook)"
conda activate /home/o_heidar/bbaenv

python src_masks/MaskSeparation.py
python src_masks/MaskObjectSelection.py
rm -rf masks