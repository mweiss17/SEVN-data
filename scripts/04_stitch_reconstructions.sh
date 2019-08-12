#!/bin/sh

POSES_FOLDER='data/SEVN/raw/poses'


# Export conda functions in subshell.
source ~/miniconda3/etc/profile.d/conda.sh
# Activate conda env.
conda activate SEVN-data


# Extracts camera positions from ORBSLAM2 .txt files and stitch them together.
python scripts/04_stitch_reconstructions.py --input_dir=$POSES_FOLDER --output_dir=$POSES_FOLDER --do_plot
