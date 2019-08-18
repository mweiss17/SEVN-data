import os

import argparse
import collections
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


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
        meta_fname = os.path.join(input_dir, 'meta.hdf5')
        meta_df = pd.read_hdf(meta_fname, key='df', index=False)
        corners = [node for node in G.nodes if node in
                   meta_df[meta_df.type == 'intersection'].frame]
        streets = [node for node in G.nodes if node in
                   meta_df[meta_df.type == 'street_segment'].frame]
        other = [node for node in G.nodes if node not in
                 meta_df[meta_df.type == 'street_segment'].frame]
        other = [node for node in other if node not in
                 meta_df[meta_df.type == 'intersection'].frame]

        nx.draw_networkx_nodes(G, pos,
                               nodelist=corners,
                               node_color='#e23f3a',
                               node_size=1,
                               alpha=0.8,
                               with_label=True)

        nx.draw_networkx_nodes(G, pos,
                               nodelist=streets,
                               node_color='#000cda',
                               node_size=1,
                               alpha=0.8)

        nx.draw_networkx_nodes(G, pos,
                               nodelist=other,
                               node_color='g',
                               node_size=20,
                               alpha=0.8)

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
