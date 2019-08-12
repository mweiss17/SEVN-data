import os

import argparse
import cv2
import datetime
import glob
import multiprocessing as mp
import numpy as np
import yaml

from scripts.utils import check_dir


def parse_args():
    '''
    Parser for the arguments.

    Returns
    ----------
    args : obj
        The arguments.

    '''
    parser = argparse.ArgumentParser('Undistort the frame views.')
    parser.add_argument('--input_dir',
                        type=str,
                        help='''Input directory containing the subfolders
                                of the 8 frame views.''')
    parser.add_argument('--output_dir',
                        type=str,
                        help='''Output directory containing the subfolders
                                of the preprocessed 8 frame view.''')
    parser.add_argument('--w',
                        type=int,
                        default=int(3200/2),
                        help='Desired frames width.')
    parser.add_argument('--h',
                        type=int,
                        default=int(2176/2),
                        help='Desired frames height.')
    parser.add_argument('--num_procs',
                        type=int,
                        default=8,
                        help='Nb. of processes for data parallelism.')
    args = parser.parse_args()
    return args


def opencv_matrix(loader, node):
    '''
    Construct node for OpenCV constructor.
    Taken from:
    https://gist.github.com/autosquid/66acc22b3798b36aea0a

    '''
    mapping = loader.construct_mapping(node, deep=True)
    mat = np.array(mapping['data'])
    mat.resize(mapping['rows'], mapping['cols'])
    return mat


def undistort_img(img, K, D, new_K, w, h, out_fname):
    '''
    Undistort an image. Then save the transformed
    image.

    Parameters
    ----------
    img: numpy array
        Image to crop.
    K: opencv-matrix
        Camera matrix..
    D: numpy array.
        Vector of distortion coefficients.
    new_K: opencv-matrix
        New camera matrix.
    w: int
        Desired image width.
    h: int
        Desired image height.
    out_fname: str
        Output image filename.

    '''
    # Computes undistortion and rectification maps.
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
    # Undistrort the image.
    undistorted_img = cv2.remap(img, map1, map2,
                                interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)
    # Save image.
    cv2.imwrite(out_fname, undistorted_img)


def undistort_proc(q, output_dir, w, h, cam_calib):
    '''
    Undistort vuze extracted frame views from a queue for multiprocessing
    and save them in the output_dir/camera_{cam_num} corresponding directories.

    Parameters
    ----------
    q: obj
        Queue class for multiprocessing. Contain the frame file names.
    output_dir: str
        Output directory.
    w: int
        Desired images width.
    h: int
        Desired images height.
    cam_calib: dict
        Dict containing cemara calibration informations.

    '''
    while True:
        try:
            file = q.get(True, 1)
            # Get file name.
            fname = file.split('/')[-1]
            # Get camera number.
            cam_num = file.split('/')[-2].split('_')[-1]
            out_fname = os.path.join(output_dir,
                                     f'camera_{cam_num}/{fname}')
            # Continue if frame already processed
            if os.path.isfile(out_fname):
                continue
            # Load frame.
            img = cv2.imread(file)
            # Get camera of interest calibration.
            calib = cam[str(cam_num)]
            K = calib['K']  # Camera matrix.
            D = calib['D']  # Vector of distortion coefficients.
            new_K = calib['new_K']  # New camera matrix
            # Undisort frame. Save it.
            undistort_img(img, K, D, new_K, w, h, out_fname)
        except:
            return


if __name__ == '__main__':
    # Load the arguments.
    args = parse_args()

    input_dir = args.input_dir
    print('input_dir: {}'.format(input_dir))

    output_dir = args.output_dir
    print('output_dir: {}'.format(output_dir))

    # Make sure input and output directories (one per camera) exist.
    for cam_num in range(8):
        in_camera_dir = os.path.join(input_dir, f'camera_{cam_num}')
        # Check if input directory exist and is not empty.
        check_dir(input_dir)
        out_camera_dir = os.path.join(output_dir, f'camera_{cam_num}')
        if not os.path.exists(out_camera_dir):
            os.makedirs(out_camera_dir)

    # Start recording time.
    datetime = datetime.datetime
    start_time = datetime.now()

    # Specify a constructor for OpenCV data type.
    yaml.add_constructor(u"tag:yaml.org,2002:opencv-matrix", opencv_matrix)

    # Loading
    with open('vuze_config/VZP1186200216.yml') as fin:
        c = fin.read()
        # Some operator on raw content of c may be needed.
        c = '%YAML 1.1' + os.linesep + '---' + c[len('%YAML:1.0'):] if \
            c.startswith("%YAML:1.0") else c
        result = yaml.load(c, Loader=yaml.Loader)

    cam = {}
    cam['0'] = result['CamModel_V2_Set']['CAM_0']
    cam['1'] = result['CamModel_V2_Set']['CAM_1']
    cam['2'] = result['CamModel_V2_Set']['CAM_2']
    cam['3'] = result['CamModel_V2_Set']['CAM_3']
    cam['4'] = result['CamModel_V2_Set']['CAM_4']
    cam['5'] = result['CamModel_V2_Set']['CAM_5']
    cam['6'] = result['CamModel_V2_Set']['CAM_6']
    cam['7'] = result['CamModel_V2_Set']['CAM_7']

    w = args.w  # Desired images width.
    h = args.h  # Desired images height.

    for k, v in cam.items():
        # Camera matrix.
        K = v['K']
        # Vector of distortion coefficients.
        D = np.array(v['DistortionCoeffs'])
        v['D'] = D
        # Returns the new camera matrix based on the free scaling parameter.
        # We set alpha=1, i.e. all source pixels are undistorted. We lose data
        # from the periphery, but we don't have any padded pixels without
        # valid data.
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
        v['new_K'] = new_K

    fnames = glob.glob(input_dir+'/*/*.jpg')
    print('# total frames: {}'.format(len(fnames)))

    # Data parallelism.
    q = mp.Queue()
    num_procs = args.num_procs  # Nb. of processes for data parallelism.
    procs = []
    for i in range(num_procs):
        p = mp.Process(target=undistort_proc, args=(q, output_dir, w, h, cam))
        procs.append(p)
        p.start()

    while fnames:
        if q.empty():
            q.put(fnames.pop())

    for p in procs:
        p.join()

    print('execution time: {}'.format(datetime.now() - start_time))
