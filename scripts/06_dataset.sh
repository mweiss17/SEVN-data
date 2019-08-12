# Export conda functions in subshell.
source ~/miniconda3/etc/profile.d/conda.sh
# Activate conda env.
conda activate SEVN-data

# Pre-process and write the labels, spatial graph, and lower resolution images to disk.
python scripts/06_dataset.py --data_path='data/SEVN' --do_plot --is_mini
