#!/bin/sh

VUZE_FOLDER='data/SEVN/raw/vuze'

# Get the .MP4 in the video sub-folder of each date of data acquisition.
for VIDEO in $VUZE_FOLDER/*/video/*.MP4; do
  echo 'video file: '$VIDEO
  F=`echo ${VIDEO/video/jpg} | cut -d'.' -f1`
  mkdir -p ${F%/*}'/track_1'
  mkdir -p ${F%/*}'/track_2'
  # Videos to image sequence (30 frames per seconde).
  # Video track 1: correspond to the front and right camera.
  ffmpeg -i $VIDEO -r 30 -map 0:0 ${F%/*}'/track_1/'${F##*/}'_%06d.jpg'
  # Video track 2: correspond to the back and left camera.
  ffmpeg -i $VIDEO -r 30 -map 0:1 ${F%/*}'/track_2/'${F##*/}'_%06d.jpg'
done

# Export conda functions in subshell.
source ~/miniconda3/etc/profile.d/conda.sh
# Activate conda env.
conda activate SEVN-data

for JPG_FOLDER in $VUZE_FOLDER/*/jpg/*; do
    CROP_FOLDER=`echo ${JPG_FOLDER%/*/*}/crop`
    mkdir -p $CROP_FOLDER
    # Crops and rotates Vuze frames.
    python scripts/crop.py --input_dir=$JPG_FOLDER --output_dir=$CROP_FOLDER
done

for CROP_FOLDER in $VUZE_FOLDER/*/crop; do
    UNDISTORT_FOLDER=`echo ${CROP_FOLDER%/*}/undistort`
    echo $UNDISTORT_FOLDER
    mkdir -p $UNDISTORT_FOLDER
    # Undistort Vuze frames.
    python scripts/undistort.py --input_dir=$CROP_FOLDER --output_dir=$UNDISTORT_FOLDER
done
