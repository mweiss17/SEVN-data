import os

import argparse
import cv2
import glob
from tqdm import tqdm


def parse_args():
    '''
    Parser for the arguments.

    Returns
    ----------
    args : obj
        The arguments.

    '''
    parser = argparse.ArgumentParser('''Crops top and bottom margins from
                                        images.''')
    parser.add_argument('--input_dir', type=str,
                        help='Input directory containing the images.')
    parser.add_argument('--output_dir', type=str,
                        help='Output directory containing the images.')
    parser.add_argument('--new_h', type=int, default=1088,
                        help='Desired image height.')
    parser.add_argument('--w', type=int, default=1088,
                        help='Image width.')
    parser.add_argument('--h', type=int, default=1600,
                        help='Image height.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Load the arguments
    args = parse_args()

    input_dir = args.input_dir
    print('input_dir: {}'.format(input_dir))

    output_dir = args.output_dir
    print('output_dir: {}'.format(output_dir))
    # Make sure output directory exist.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    w = args.w
    h = args.h
    new_h = args.new_h

    # Find image file names.
    fnames = glob.glob(input_dir+'/*.jpg')

    for fname in tqdm(fnames):
        # Read image.
        img = cv2.imread(fname)
        # Crop image.
        crop_img = img[int(h/2 - new_h/2):int(h/2 + new_h/2), 0:w]
        # Save image
        name = fname.split('/')[-1]
        out_fname = os.path.join(output_dir, name)
        cv2.imwrite(out_fname, crop_img)
