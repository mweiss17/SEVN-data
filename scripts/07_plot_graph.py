import os

import argparse
import collections
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np

def parse_args():
    '''
    Parser for the arguments.

    Returns
    ----------
    args : obj
        The arguments.

    '''
    parser = argparse.ArgumentParser('Creates panoramas from vuze pictures.')
    parser.add_argument('--input_dir',
                        type=str,
                        help='Input directory containing the .pkl graph.')
    parser.add_argument('--zones',
                        action='store_true',
                        help='Different color for streets and corners.')
    args = parser.parse_args()
    return args

def plot_graph(input_dir, zones=None, graph_fname=None):

    G = nx.read_gpickle(graph_fname)

    pos = {k: v.get('coords')[0:2] for k, v in G.nodes(data=True)}
    if zones:
        # Get labels.
        coords = np.load(os.path.join(input_dir, 'pos_ang.npy'))
        coord_df = pd.DataFrame({'x': coords[:, 2],
                                  'y': coords[:, 3],
                                  'z': coords[:, 4],
                                  'angle': coords[:, -1],
                                  'timestamp': coords[:, 1],
                                  'frame': [int(x) for x in coords[:, 1]*30]})

        # Filter the pano coordinates by spatial relation.
        coord_df = pd.read_hdf(os.path.join(input_dir, "coord.hdf5"), key='df', index=False)
        corners = [node for node in G.nodes if node in
                   coord_df[coord_df.type == 'intersection'].index]
        streets = [node for node in G.nodes if node in
                   coord_df[coord_df.type == 'street_segment'].index]
        box = (24, 76, -125, 10)
        node_blacklist = []
        node_blacklist.extend([x for x in range(877, 879)])
        node_blacklist.extend([x for x in range(52, 56)])
        node_blacklist.extend([x for x in range(31, 39)])
        node_blacklist.extend([x for x in range(2040, 2045)])
        node_blacklist.extend([x for x in range(2057, 2063)])
        node_blacklist.extend([x for x in range(3661, 3669)])
        node_blacklist.extend([x for x in range(780, 784)])

        coord_df = coord_df[((coord_df.x > box[0]) &
                               (coord_df.x < box[1]) &
                               (coord_df.y > box[2]) &
                               (coord_df.y < box[3]))]
        coord_df = coord_df[~coord_df.index.isin(node_blacklist)]
        test_seg = [node for node in G.nodes if node in coord_df[coord_df.type == 'street_segment'].index]
        test_int = [node for node in G.nodes if node in coord_df[coord_df.type == 'intersection'].index]

        nx.draw_networkx_nodes(G, pos,
                               nodelist=corners,
                               node_color='#440154',
                               node_size=1,
                               alpha=0.8,
                               with_label=True)

        nx.draw_networkx_nodes(G, pos,
                               nodelist=streets,
                               node_color='#79d151',
                               node_size=1,
                               alpha=0.8)

        nx.draw_networkx_nodes(G, pos,
                               nodelist=test_seg,
                               node_color='#fde724',
                               node_size=1,
                               alpha=0.8,
                               with_label=True)

        nx.draw_networkx_nodes(G, pos,
                               nodelist=test_int,
                               node_color='#29788e',
                               node_size=1,
                               alpha=0.8,
                               with_label=True)

        edges = nx.draw_networkx_edges(G, pos=pos)

    else:
        nx.draw(G, pos, node_color='r', node_size=1)

    plt.axis('equal')
    plot_fname = os.path.join(input_dir, 'colored_graph.png')
    print('hist_fname {}:'.format(plot_fname))
    plt.savefig(plot_fname, transparent=True, dpi=1000)

    print('num_edges: {}'.format(len(G.edges)))
    print('num_nodes: {}'.format(len(G.nodes)))

    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    # Degree histogram
    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color='b')

    plt.title('Degree Histogram')
    plt.ylabel('Count')
    plt.xlabel('Degree')
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)
    hist_fname = os.path.join(input_dir, 'degree_hist.png')
    print('hist_fname {}:'.format(hist_fname))
    plt.savefig(hist_fname)


if __name__ == '__main__':
    # Load the arguments
    args = parse_args()

    input_dir = args.input_dir
    print('input_dir : {}'.format(input_dir))

    zones = args.zones

    # Load graph
    graph_fname = os.path.join(input_dir, 'graph.pkl')

    plot_graph(input_dir, zones, graph_fname)
