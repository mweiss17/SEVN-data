import os

import argparse
import glob
import numpy as np
import pandas as pd
from shutil import copyfile
from tqdm import tqdm

from matplotlib import pyplot as plt

from scripts.utils import find_nearby_nodes


def parse_args():
    '''
    Parser for the arguments.

    Returns
    ----------
    args : obj
        The arguments.

    '''
    parser = argparse.ArgumentParser(description='''Filter some panorama
                                                    coordinates leaving around
                                                    2 panoramas per meter.''')
    parser.add_argument('--input_file', type=str,
                        help='''Path to the .csv file containing the non
                                filtered coordinates.''')
    parser.add_argument('--output_file', type=str,
                        help='''Path to the .npy file containing the
                                filtered coordinates.''')
    parser.add_argument('--input_dir', type=str,
                        help='''Input directory containing the non filtered
                                panoramas.''')
    parser.add_argument('--output_dir', type=str,
                        help='''Output directory to copy the panoramas to
                                keep.''')
    parser.add_argument('--plot',
                        action='store_false',
                        help='Plot poses.')
    args = parser.parse_args()
    return args


def filter_poses(poses):
    '''
    Filter poses.

    Parameters
    ----------
    poses: pandas.DataFrame
        Dataframe containing the poses to filter.

    Returns
    -------
    pandas.DataFrame
        Dataframe containing the filtered poses.

    '''
    # Find poses to remove.
    to_filter = set()
    for i, node1 in tqdm(poses.iterrows(), leave=False,
                         total=poses.shape[0], desc='filtering nodes'):
        for j, node2 in find_nearby_nodes(poses, node1, 0.5).iterrows():
            if i == j or j in to_filter or i in to_filter:
                continue
            to_filter.add(j)
    # Filter poses.
    return poses[~poses.index.isin(to_filter)]


if __name__ == '__main__':
    # Load the arguments.
    args = parse_args()

    input_file = args.input_file
    # Check if input file exist.
    assert os.path.isfile(input_file), '--input_file does not exist.'
    print('input_file: {}'.format(input_file))

    output_file = args.output_file
    print('output_file: {}'.format(output_file))

    input_dir = args.input_dir
    print('input_dir: {}'.format(input_dir))

    output_dir = args.output_dir
    print('output_dir: {}'.format(output_dir))
    # Make sure output directory exist.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load poses
    poses = pd.read_csv(input_file, delimiter=',')
    # Reset column names.
    poses.columns = ['index', 'timestamp', 'x', 'y', 'z', 'angle']

    # Filter the poses.
    filtered_poses = filter_poses(poses)

    # Save filtered poses numpy array in a .npy file.
    np.save(args.output_file, filtered_poses.values)
    print('num poses filtered: {}'.format(len(poses) - len(filtered_poses)))
    print('num poses remaining: {}'.format(len(filtered_poses)))

    if args.plot:
        # Plot poses.
        x_f = [pose.x for idx, pose in filtered_poses.iterrows()]
        y_f = [pose.y for idx, pose in filtered_poses.iterrows()]
        plt.scatter(x_f, y_f)
        # Save figure.
        figure_fname = output_file.split('.')[0] + '.png'
        print('figure_fname: {}'.format(figure_fname))
        plt.savefig(figure_fname)

    # Find file names of the panoramas.
    paths = glob.glob(args.input_dir+'/*.jpg')

    # Find file names of the selected panoramas.
    good_paths = []
    nums = [str(int(x.timestamp * 30)).zfill(6) for idx, x in
            filtered_poses.iterrows()]
    for path in paths:
        for num in nums:
            if num in path:
                good_paths.append(path)

    for path in tqdm(good_paths, total=len(good_paths)):
        fname = os.path.join(args.output_dir, path.split('/')[-1])
        # Copy the selected panoramas from the HD.
        copyfile(path, fname)
