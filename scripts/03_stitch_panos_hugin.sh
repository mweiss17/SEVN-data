#!/bin/sh

VUZE_FOLDER='data/SEVN/raw/vuze'


# Export conda functions in subshell.
source ~/miniconda3/etc/profile.d/conda.sh
# Activate conda env.
conda activate SEVN-data

for UNDISTORT_FOLDER in $VUZE_FOLDER/*/undistort; do
    PANO_FOLDER=`echo ${UNDISTORT_FOLDER/undistort/panorama}`
    # Creates panoramas from vuze frame views.
    python scripts/03_stitch_panos_hugin.py --input_dir=$UNDISTORT_FOLDER --output_dir=$PANO_FOLDER
done
