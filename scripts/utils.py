import os

import numpy as np
import quaternion


def check_dir(dir_name):
    '''
    Check if a Directory is empty and also check exceptional situations.

    Parameters
    ----------
    dir_name: str
        Name of the directory to check.

    '''
    if os.path.exists(dir_name) and os.path.isdir(dir_name):
        if not os.listdir(dir_name):
            print('Directory is empty')
        else:
            print('Directory is not empty')
    else:
        print('Given Directory don\'t exists')


def find_nearby_nodes(coords_df, coords, radius):
    '''
    Find nearby nodes.

    Parameters
    ----------
    coords_df: pandas.DataFrame
        Dataframe containing coordinates of all the nodes.
    coords: pandas.DataFram
        Dataframe containing the reference node coordinates.
    radius: float
        Radius for which we are looking for nearby nodes from the reference
        node.

    Returns
    -------
    pandas.DataFrame
        Dataframe containing the nearby nodes coordinates.

    '''
    return coords_df[(coords_df.x > coords.x - radius) &
                     (coords_df.x < coords.x + radius) &
                     (coords_df.y > coords.y - radius) &
                     (coords_df.y < coords.y + radius)]


def find_angle(ref_q, q):
    '''
    Find angle.

    Parameters
    ----------
    ref_q: list
        Quaternion of the reference pose.
    q: list
        Quaternion of the pose.

    Returns
    -------
    angle: float
        Angle.

    '''
    rqvec = np.quaternion(ref_q[0], ref_q[1], ref_q[2], ref_q[3])
    qvec = np.quaternion(q[0], q[1], q[2], q[3])
    new_q = rqvec.inverse()*qvec
    ang = np.dot(np.array([0, 1, 0]),
                 quaternion.as_rotation_vector(new_q))*180/np.pi
    angle = ang % 360
    return angle


def read_cameras_text(path):
    '''
    Read SLAM .txt file and output a numpy array containing the poses.

    Parameters
    ----------
    path: str
        Path to the SLAM .txt output file.

    Returns
    -------
    poses: numpy.array
        Numpy array containing the poses.

    '''
    poses = []
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                timestamp = float(elems[0])
                t = np.array([float(elems[1]), float(elems[2]),
                              float(elems[3])])
                rot_mat = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
                t = np.dot(rot_mat, t)
                t[2] = 0.0
                q = np.array([float(elems[4]), float(elems[5]),
                              float(elems[6]), float(elems[7])])
                poses.append([timestamp, t, q])
    ref_pose = poses[0]
    angles = []
    for pose in poses:
        angles.append(find_angle(ref_pose[-1], pose[-1]))
    poses = np.array(poses)
    poses[:, -1] = angles
    return poses
