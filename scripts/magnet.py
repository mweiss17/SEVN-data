import argparse
import math
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt


def parse_args():
    '''
    Parser for the arguments.

    Returns
    ----------
    args : obj
        The arguments.

    '''
    parser = argparse.ArgumentParser('Script to read compass data from vuze.')
    parser.add_argument('--file_path',
                        type=str,
                        default='~/navi/RUN.CSV',
                        help='File path.')
    args = parser.parse_args()
    return args


def moving_average(a, n=10):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


if __name__ == '__main__':
    # Load the arguments
    args = parse_args()

    file_path = args.input_dir
    print('file_path : {}'.format(file_path))

    magnet_df = pd.read_csv(file_path, sep=',', header=None)
    cols = ["xMagnet", "yMagnet", "zMagnet"]

    data = []
    i = 0
    for idx, row in magnet_df.iterrows():
        i += 1
        row_data = {}
        for idx2, element in row.iteritems():
            if "=" in str(element):
                k, v = element.split("=")
                if k in cols:
                    row_data[k] = float(v)
        if len(row_data) > 1:
            data.append(row_data)

    x = []
    y = []
    for d in data:
        x.append(d['xMagnet'])
        y.append(d['yMagnet'])

    x = np.array(x)
    y = np.array(y)
    x_center = x.mean()
    y_center = y.mean()
    x_center = -0.08342219227857683
    y_center = 0.2666537047691143

    print(x_center, y_center)
    angles = np.array([math.atan2(y[i] - y_center, x[i] - x_center) for i in
                       range(len(x))]) * 180 / math.pi

    plt.plot(x, y)
    plt.show()

    plt.plot(angles)
    plt.show()

    plt.plot(moving_average(angles, 30))
    plt.show()

    plt.plot(moving_average(angles, 60))
    plt.show()
