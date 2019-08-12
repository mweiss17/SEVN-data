import os

import argparse
import glob
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scripts.utils import check_dir, read_cameras_text


def parse_args():
    '''
    Parser for the arguments.

    Returns
    ----------
    args : obj
        The arguments.

    '''
    parser = argparse.ArgumentParser('''Extracts camera positions from ORBSLAM2
                                    .txt files and stitch them together.''')
    parser.add_argument('--input_dir',
                        type=str,
                        help='Input directory containg the .txt files.')
    parser.add_argument('--output_dir',
                        type=str,
                        help='''Output directory containg the .csv file with
                                the stitched camera positions.''')
    parser.add_argument('--do_plot',
                        action='store_true',
                        help='Plot poses.')
    args = parser.parse_args()
    return args


class Pose():
    def __init__(self, timestamp, t, angle):
        '''
        Pose class.

        Parameters
        ----------
        timestamp : float
            Timestamp.
        t: numpy array
            Array of x, y, z coordinates.
        angle: float
            Angle.

        '''
        self.timestamp = timestamp
        self.t = t
        self.angle = angle


def rotate_poses(poses, angle):
    '''
    Rotate poses in a list.

    Parameters
    ----------
    poses: list
        List of poses.
    angle: float
        Angle.

    Returns
    -------
    poses: list
        List of rotated poses.

    '''
    rot_point = poses[0].t
    rot_mat = np.array([[np.cos(np.deg2rad(angle)),
                         -np.sin(np.deg2rad(angle)), 0],
                        [np.sin(np.deg2rad(angle)),
                         np.cos(np.deg2rad(angle)), 0],
                        [0, 0, 1]])
    for pose in poses:
        tvec = pose.t - rot_point
        tvec = np.dot(rot_mat, tvec)
        pose.t = rot_point + tvec
        pose.angle = (pose.angle + angle) % 360
    return poses


def scale_poses(poses, ref, scale_x, scale_y):
    '''
    Scale poses.

    Parameters
    ----------
    poses: list
        List of poses to scale.
    ref: list
        List of reference poses.
    scale_x: float
        Scale x.
    scale_y: float
        Scale y.

    Returns
    -------
    poses: list
        List of scaled poses.

    '''
    dist = []
    for idx in range(len(poses)-1):
        dist.append(np.linalg.norm(poses[idx+1].t - poses[idx].t) / (
            poses[idx+1].timestamp - poses[idx].timestamp))

    ref_dist = []
    for idx in range(len(ref)-1):
        ref_dist.append(np.linalg.norm(ref[idx+1].t - ref[idx].t) / (
            ref[idx+1].timestamp - ref[idx].timestamp))

    avg_dist = np.average(dist)
    avg_ref_dist = np.average(ref_dist)
    scale_factor = avg_ref_dist / avg_dist

    for pose in poses:
        pose.t[0] = pose.t[0] * scale_factor * scale_x
        pose.t[1] = pose.t[1] * scale_factor * scale_y
    return poses


def stitch(poses1, poses2, scale_x, scale_y, angle, x, y):
    '''
    Stitch together two list of poses.

    Parameters
    ----------
    poses1: list
        List of poses 1.
    poses2: list
        List of poses 2.
    scale_x: float
        Scale x.
    scale_y: float
        Scale y.
    angle: float
        Angle.
    x: float
        x.
    y: float
        y.

    Returns
    -------
    reconstruction: list
        List of poses 1 and 2 stitched.

    '''
    poses2 = rotate_poses(poses2, angle)
    poses2 = scale_poses(poses2, poses1, scale_x, scale_y)
    intersection = poses1[-1]
    delta = intersection.t - poses2[0].t
    reconstruction = poses1
    for pose in poses2:
        pose.t = pose.t + delta + np.array([x, y, 0])
        reconstruction.append(pose)
    return reconstruction


def plot_data(x, y, l, fname):
    '''
    Plot poses.

    Parameters
    ----------
    x: list
        List of x.
    y: list
        List of y.
    l: int
        # of poses.
    fname: str
        File name.

    '''
    plt.scatter(x[0:l], y[0:l], c='blue')
    plt.scatter(x[l:-1], y[l:-1], c='red')
    plt.axis('equal')
    plt.savefig(fname)


def plot_poses(poses, l, fname):
    '''
    Plot poses.

    Parameters
    ----------
    poses: list
        List of poses.
    l: int
        # of poses.
    fname: str
        File name.

    '''
    x = []
    y = []
    for pose in poses:
        x.append(pose.t[0])
        y.append(pose.t[1])
    plot_data(x, y, l, fname)


if __name__ == '__main__':
    # Load the arguments.
    args = parse_args()

    input_dir = args.input_dir
    print('input_dir: {}'.format(input_dir))
    # Check if input directory exist and is not empty.
    check_dir(input_dir)

    output_dir = args.output_dir
    print('output_dir: {}'.format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load ORBSLAM2 outputs.
    reconstructions = {}
    for file in glob.glob(input_dir+'/*.txt'):
        idx = file.split('.')[0].split('/')[-1]
        reconstructions[idx] = [Pose(*x) for x in read_cameras_text(file)]

    # Stitch ORBSLAM2 outputs.
    poses = reconstructions['01']
    poses = stitch(poses, reconstructions['02'], 1.0, 1.0, -5.0, 1.0, 0.5)
    poses = stitch(poses, reconstructions['03'], 1.0, 1.0, -96.0, 0.0, 0.0)
    poses = stitch(poses, reconstructions['04'], 1.1, 1.1, 182.0, -6.0, 0.0)
    poses = stitch(poses, reconstructions['05'], 1.05, 1.05, 87.5, 0.0, 3.0)
    poses = stitch(poses, reconstructions['06'], 1.09, 1.09, -3.0, -1.0, 0.0)
    poses = stitch(poses, reconstructions['07'], 1.0, 1.0, -95.0, 0.0, -1.0)
    poses = stitch(poses, reconstructions['08'], 1.15, 1.0, -7.0, 0.0, -1.0)
    poses = stitch(poses, reconstructions['09'], 1.06, 1.06, -85.5, 0.0, 4.0)
    # We don't use the reconstruction 10.
    # poses = stitch(poses, reconstructions['10'], 1.15, 1.15, 180, 0.0, 0.0)
    poses = stitch(poses, reconstructions['11'], 1.0, 1.0, 90, -10.0, 220.0)
    poses = stitch(poses, reconstructions['12'], 1.05, 1.1, 5.0, 5.0, 0.0)
    poses = stitch(poses, reconstructions['13'], 1.0, 1.0, 91.0, 5.0, -12.5)
    poses = stitch(poses, reconstructions['14'], 1.0, 1.0, 87.0, 0.0, 3.0)
    poses = stitch(poses, reconstructions['15'], 1.1, 1.0, -1.0, 0.0, 0.0)
    poses = stitch(poses, reconstructions['16'], 1.0, 1.0, 176.0, 0.0, -1.0)
    poses = stitch(poses, reconstructions['17'], 1.0, 1.0, 181.5, -3.0, -2.0)
    poses = stitch(poses, reconstructions['18'], 1.2, 0.95, 82.0, 0.0, 0.0)

    if args.do_plot:
        # Plot poses.
        figure_fname = os.path.join(output_dir, 'poses.png')
        print('figure_fname: {}'.format(figure_fname))
        plot_poses(poses, len(poses), figure_fname)

    # Constuct pandas dataframe with the stitched camera positions.
    res = []
    for pose in poses:
        res.append((pose.timestamp,
                    pose.t[0], pose.t[1], pose.t[2],
                    pose.angle))
    res = pd.DataFrame(res, columns=['timestamp', 'x', 'y', 'z', 'angle'])
    # Save output.
    print('output fname: {}'.format(os.path.join(output_dir, 'coords.csv')))
    res.to_csv(os.path.join(output_dir, 'coords.csv'))
