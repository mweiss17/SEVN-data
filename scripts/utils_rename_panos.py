import os

import argparse
import glob


def parse_args():
    '''
    Parser for the arguments.

    Returns
    ----------
    args : obj
        The arguments.

    '''
    parser = argparse.ArgumentParser('Rename images by adding the offset.')
    parser.add_argument('--input_dir', type=str,
                        help='''Input directory containing the images to
                                rename.''')
    parser.add_argument('--offset', type=int,
                        help='Offset to add.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Load the arguments
    args = parse_args()
    input_dir = args.input_dir
    offset = args.offset

    # Find image file names.
    fnames = glob.glob(input_dir+'/*.jpg')
    for fname in fnames:
        # Rename images by adding the offset.
        frame_num = int(fname.split('/')[-1].split('.')[0].split('_')[-1])
        frame_num = frame_num + offset
        out_fname = 'pano_' + '{:06d}'.format(frame_num) + '.jpg'
        out_fname = os.path.join(input_dir, out_fname)
        os.rename(fname, out_fname)
