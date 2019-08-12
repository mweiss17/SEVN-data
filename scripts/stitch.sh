#!/bin/sh
# $1=input_dir
# $2=output_dir
# $3=frame_num

# Assembles a Hugin .pto project file using equirectangular projection.
pto_gen -o '$2/pano_$3.pto' -p 2 -f 120 '$1/camera_1/$3.jpg' '$1/camera_3/$3.jpg' '$1/camera_5/$3.jpg' '$1/camera_7/$3.jpg'
# Control point detector for hugin.
cpfind --multirow --celeste -o '$2/pano_$3.pto' '$2/pano_$3.pto'
# Remove all non-credible control points.
cpclean -o '$2/pano_$3.pto' '$2/pano_$3.pto'
# Find vertical lines and assigns vertical control points to them.
linefind -o '$2/pano_$3.pto' '$2/pano_$3.pto'
# Control point optimization.
autooptimiser -a -m -l -s -o '$2/pano_$3.pto' '$2/pano_$3.pto'
# Change some output options of the project file
pano_modify --canvas=AUTO --crop=AUTO -o '$2/pano_$3.pto' '$2/pano_$3.pto'
# Stitching panorama
hugin_executor --stitching --prefix='$2/pano_$3' '$2/pano_$3.pto'

# Convert .tif to .jpg
convert "$2/pano_$3.tif" "$2/pano_$3.jpg"

# Remove unwanted format (.tif and .pto)
rm "$2/pano_$3.tif"
rm "$2/pano_$3.pto"
