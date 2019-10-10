import argparse


def parse_args():
    '''
    Parser for the arguments.

    Returns
    ----------
    args : obj
        The arguments.

    '''
    parser = argparse.ArgumentParser('Writes timestamp file for ORBSLAM2.')
    parser.add_argument('--output_fname', type=str,
                        help='Output .txt file name.')
    parser.add_argument('--file_len', type=int,
                        help='Number of images.')
    parser.add_argument('--fps', type=float, default=30,
                        help='Frame per second.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Load the arguments
    args = parse_args()
    output_fname = args.output_fname
    fps = args.fps
    file_len = args.file_len

    # Write timestamp file.
    file = open(output_fname, 'a')
    for i in range(1, file_len+1):
        out_fname = '{:06f}'.format(i/fps)
        file.write(out_fname+'\n')
    file.close()
