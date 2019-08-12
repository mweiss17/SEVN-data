'''Stitch panoramas'''

import os

import argparse
import glob
import subprocess

from scripts.utils import check_dir


def parse_args():
    '''
    Parser for the arguments.

    Returns
    ----------
    args : obj
        The arguments.

    '''
    parser = argparse.ArgumentParser('''Creates panoramas from vuze frame
                                        views.''')
    parser.add_argument('--input_dir',
                        type=str,
                        help='Input directory containing frame views.')
    parser.add_argument('--output_dir',
                        type=str,
                        help='Output directory of created panoramas.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Load the arguments
    args = parse_args()

    input_dir = args.input_dir
    print('input_dir: {}'.format(input_dir))
    # Check if input directory exist and is not empty.
    check_dir(input_dir)

    output_dir = args.output_dir
    print('output_dir: {}'.format(output_dir))
    # Make sure output directory exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate of video frames.
    for file in glob.glob(input_dir+'/camera_0/*.jpg'):
        frame_num = file.split('/')[-1].split('.')[0]
        out_fname = os.path.join(output_dir, frame_num + '.jpg')
        # Continue if file already exist
        if not os.path.isfile(out_fname):
            try:
                # Run './scripts/stitch.sh' for each video frame.
                cmd = ['./scripts/stitch.sh', input_dir, output_dir, frame_num]
                print(cmd)
                print(subprocess.check_output(cmd))
            except KeyboardInterrupt:
                break
            except:
                pass
