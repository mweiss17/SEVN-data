#!/bin/sh

IN_POSES_FILE='data/SEVN/raw/poses/coords.csv'
OUT_POSES_FILE='data/SEVN/processed/pos_ang.npy'
IN_PANORAMAS_FOLDER='data/SEVN/raw/vuze/2019-06-10/panorama'
OUT_PANORAMAS_FOLDER='data/SEVN/raw/panos'


# Export conda functions in subshell.
source ~/miniconda3/etc/profile.d/conda.sh
# Activate conda env.
conda activate SEVN-data


# Filter some panorama coordinates leaving around 2 panoramas per meter.
python scripts/05_filter_panos.py --input_file=$IN_POSES_FILE --output_file=$OUT_POSES_FILE --input_dir=$IN_PANORAMAS_FOLDER --output_dir=$OUT_PANORAMAS_FOLDER
