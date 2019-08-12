#!/bin/sh

VUZE_FOLDER='data/SEVN/raw/vuze'

# Get the .MP4 in the 360 degree stabilized video sub-folder of each date of data acquisition.
for VIDEO in $VUZE_FOLDER/*/stabilized_video/*.MP4; do
  F=`echo ${VIDEO/video/panorama} | cut -d'.' -f1`
  echo 'video file: '$F
  mkdir -p ${F%/*}
  # Videos to image sequence (30 frames per seconde).
  ffmpeg -i $VIDEO -r 30 -map 0:0 $F'_%06d.jpg'
done
