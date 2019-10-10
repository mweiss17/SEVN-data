import os

import argparse
import cv2
import datetime
import glob
import multiprocessing as mp

from scripts.utils import check_dir


def parse_args():
    '''
    Parser for the arguments.

    Returns
    ----------
    args : obj
        The arguments.

    '''
    parser = argparse.ArgumentParser('Crops and rotates Vuze images.')
    parser.add_argument('--input_dir',
                        type=str,
                        help='''Input directory containing the images to
                              preprocessed.''')
    parser.add_argument('--output_dir',
                        type=str,
                        help='''Output directory containing the preprocessed
                              images.''')
    parser.add_argument('--w',
                        type=int,
                        default=int(3200/2),
                        help='Desired images width.')
    parser.add_argument('--h',
                        type=int,
                        default=int(2176/2),
                        help='Desired images height.')
    parser.add_argument('--num_procs',
                        type=int,
                        default=8,
                        help='Nb. of processes for data parallelism.')
    args = parser.parse_args()
    return args


def crop_img(img, x, y, w, h, out_fname):
    '''
    Crop and rotate 270 degrees clockwise an image. Then save the transformed
    image.

    Parameters
    ----------
    img: numpy array
        Image to crop.
    x: int
        x coordinates of the upper left corner of the crop image.
    y: int
        y coordinates of the upper left corner of the crop image.
    w: int
        Desired image width.
    h: int
        Desired image height.
    out_fname: str
        Output image filename.

    '''
    # Crop the frame.
    crop_img = img[y:y+h, x:x+w]
    # Rotate 270 degrees clockwise the frame.
    out_img = cv2.rotate(crop_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # Save frame.
    cv2.imwrite(out_fname, out_img)


def crop_proc(q, output_dir, w, h):
    '''
    Crop vuze extracted frames into 4 images (1 per camera) from a queue for
    multiprocessing and save them in the output_dir/camera_{cam_num}
    corresponding directories.

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

    '''
    while True:
        try:
            # Get file name.
            file = q.get(True, 1)
            # Get frame name.
            fname = file.split('/')[-1]
            # Get track number.
            # 1 correspond to the front and right camera and 2 to the back
            # and left camera.
            track_num = int(file.split('/')[-2].split('_')[-1])
            # Load frame.
            img = cv2.imread(file)
            for cam_num, (x, y) in enumerate([(0, 0), (w, 0), (0, h), (w, h)]):
                # cam_num 0 & 1 correspond to front camera, left and right view
                # cam_num 2 & 3 to right camera, front and back view
                # cam_num 4 & 5 to back camera, left and right view
                # cam_num 6 & 7 to left camera, front and back view
                cam_num = cam_num if track_num == 1 else 4 + cam_num
                # Output filename
                out_fname = os.path.join(output_dir,
                                         f'camera_{cam_num}/{fname}')
                # Continue if file already processed
                if os.path.isfile(out_fname):
                    continue
                # Crop and rotate frame. Save it.
                crop_img(img, x, y, w, h, out_fname)
        except:
            return


if __name__ == '__main__':
    # Load the arguments.
    args = parse_args()

    input_dir = args.input_dir
    print('input_dir: {}'.format(input_dir))
    # Check if input directory exist and is not empty.
    check_dir(input_dir)

    output_dir = args.output_dir
    print('output_dir: {}'.format(output_dir))

    # Make sure output directories (one per camera) exist.
    for cam_num in range(8):
        camera_dir = os.path.join(output_dir, f'camera_{cam_num}')
        if not os.path.exists(camera_dir):
            os.makedirs(camera_dir)

    # Start recording time.
    datetime = datetime.datetime
    start_time = datetime.now()

    # Get the list of frame file names.
    fnames = glob.glob(input_dir+'/*.jpg')
    print('# total frames: {}'.format(len(fnames)))

    # Data parallelism.
    q = mp.Queue()
    num_procs = args.num_procs  # Nb. of processes for data parallelism.
    procs = []
    w = args.w  # Desired images width.
    h = args.h  # Desired images height.
    for i in range(num_procs):
        p = mp.Process(target=crop_proc,
                       args=(q, output_dir, w, h))
        procs.append(p)
        p.start()

    while fnames:
        if q.empty():
            q.put(fnames.pop())

    for p in procs:
        p.join()

    print('execution time: {}'.format(datetime.now() - start_time))
