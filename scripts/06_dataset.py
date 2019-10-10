import os
import h5py
import tables
import argparse
import cv2
import gzip
import math
import networkx as nx
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import xml.etree.ElementTree as et

from utils import check_dir

from matplotlib import pyplot as plt


def parse_args():
    '''
    Parser for the arguments.

    Returns
    ----------
    args : obj
        The arguments.

    '''
    parser = argparse.ArgumentParser(description='''Pre-process and write the
                                                    labels, spatial graph, and
                                                    lower resolution images to
                                                    disk.''')
    parser.add_argument('--data_path', type=str, default='data/SEVN',
                        help='Path to the data directory.')
    parser.add_argument('--do_images', action='store_true',
                        help='''If true, build the lower resolution images to disk.''')
    parser.add_argument('--do_images_high_res', action='store_true',
                        help='''If true, build the high resolution images to disk.''')
    parser.add_argument('--do_graph', action='store_true',
                        help='If true, build the graph.')
    parser.add_argument('--do_plot', action='store_true',
                        help='If true, plot the data.')
    parser.add_argument('--is_test', action='store_true',
                        help='If true, plot the data.')
    args = parser.parse_args()
    return args


def cleanup_graph(coord_df, is_test):
    '''
    Description.

    Parameters
    ----------
    coord_df: pandas.DataFrame
        Dataframe containing the frame coordinates to clean.

    Returns
    -------
    coord_df: pandas.DataFrame
        Dataframe containing the frame coordinates.
    node_blacklist: list
        List containing the nodes to remove.
    edge_blacklist: list
        List containing the edges to remove.
    add_edges: list
        List containing the eges to add.

    '''
    # Construct the node blacklist.
    node_blacklist = [0, 928, 929, 930, 931, 1138, 6038, 6039, 5721, 5722,
                      6091, 6090, 6082, 6197, 6088, 4809, 5964, 5504, 5505,
                      5467, 5514, 174, 188, 189, 190, 2390, 2391, 2392, 2393,
                      1862, 1863, 1512, 1821, 4227, 1874, 3894, 3895, 3896,
                      3897, 3898, 2887, 608, 3025, 3090, 3013, 3090, 780, 781,
                      162, 1822, 1725, 1726, 1513, 3875, 4842, 4907, #4870,
                      4717, 6214, 6215, 5965, 5966, 4715, 3362, 5358, 3419,
                      5457, 5458, 3418]
    node_blacklist.extend([x for x in range(1330, 1348)])
    node_blacklist.extend([x for x in range(3034, 3071)])
    node_blacklist.extend([x for x in range(879, 892)])
    node_blacklist.extend([x for x in range(2971, 2983)])
    node_blacklist.extend([x for x in range(2888, 2948)])
    node_blacklist.extend([x for x in range(2608, 2629)])
    node_blacklist.extend([x for x in range(3091, 3098)])
    node_blacklist.extend([x for x in range(704, 780)])
    node_blacklist.extend([x for x in range(118, 128)])
    node_blacklist.extend([x for x in range(5724, 5748)])
    node_blacklist.extend([x for x in range(6186, 6197)])
    node_blacklist.extend([x for x in range(4891, 4896)])
    node_blacklist.extend([x for x in range(6083, 6088)])
    node_blacklist.extend([x for x in range(5516, 5594)])
    node_blacklist.extend([x for x in range(5955, 5964)])
    node_blacklist.extend([x for x in range(5459, 5467)])
    node_blacklist.extend([x for x in range(3400, 3418)])
    node_blacklist.extend([x for x in range(4261, 4266)])
    node_blacklist.extend([x for x in range(5506, 5514)])
    node_blacklist.extend([x for x in range(3876, 3894)])
    node_blacklist.extend([x for x in range(5340, 5358)])
    node_blacklist.extend([x for x in range(1122, 1132)])
    node_blacklist.extend([x for x in range(652, 704)])
    node_blacklist.extend([x for x in range(3899, 4227)])
    node_blacklist.extend([x for x in range(2394, 2410)])
    node_blacklist.extend([x for x in range(585, 608)])
    node_blacklist.extend([x for x in range(2043, 2057)])
    node_blacklist.extend([x for x in range(2244, 2252)])
    node_blacklist.extend([x for x in range(2847, 2887)])
    node_blacklist.extend([x for x in range(3636, 3652)])
    node_blacklist.extend([x for x in range(2834, 2847)])
    node_blacklist.extend([x for x in range(3652, 3661)])
    node_blacklist.extend([x for x in range(4228, 4261)])
    node_blacklist.extend([x for x in range(3251, 3257)])
    node_blacklist.extend([x for x in range(3229, 3237)])
    node_blacklist.extend([x for x in range(1835, 1843)])
    node_blacklist.extend([x for x in range(5102, 5113)])
    node_blacklist.extend([x for x in range(2828, 2834)])
    node_blacklist.extend([x for x in range(2605, 2608)])
    node_blacklist.extend([x for x in range(3632, 3636)])
    node_blacklist.extend([x for x in range(3661, 3667)])
    node_blacklist.extend([x for x in range(4712, 4715)])
    node_blacklist.extend([x for x in range(5286, 5338)])
    node_blacklist.extend([x for x in range(6216, 6247)])
    node_blacklist.extend([x for x in range(4852, 4857)])
    node_blacklist.extend([x for x in range(4874, 4884)])
    node_blacklist.extend([x for x in range(6092, 6096)])
    node_blacklist.extend([x for x in range(5748, 5754)])
    node_blacklist.extend([x for x in range(5716, 5724)])

    # Contruct the edge blacklist.
    edge_blacklist = [(913, 915), (835, 925), (824, 826), (835, 837),
                      (900, 902), (901, 903), (1534, 1536), (1511, 1724),
                      (191, 137), (50, 172), (612, 782), (1135, 1714),
                      (109, 107), (1875, 1824), (1771, 1875), (4843, 4845),
                      (4849, 4847), (4860, 4862), (4861, 4863), (4909, 4813),
                      (4910, 4908), (4506, 4897), (4502, 4897), (5754, 5756),
                      (5758, 5756), (4909, 4810), (1132, 1134)]

    # Construct the edges addlist.
    add_edges = [(902, 1329), (893, 894), (894, 895), (651, 2970), (638, 2983),
                 (637, 2983), (2637, 3098), (2629, 3088), (2954, 3026),
                 (3077, 3078), (3109, 3110), (2948, 2607), (0, 1722),
                 (6161, 6162), (6147, 6146), (6172, 6173), (6174, 6175),
                 (4465, 4906), (6212, 6213), (4465, 4464), (4467, 4466),
                 (6037, 5594), (5458, 3418), (5129, 5128), (100, 101),
                 (98, 1348), (99, 1348), (3630, 3631), (3631, 3632),
                 (3632, 3633), (3634, 3635), (2375, 3661), (1834, 2234),
                 (1834, 2233), (3366, 3367), (2827, 2363), (379, 611),
                 (2948, 3110), (2604, 3110), (3076, 3077), (782, 379),
                 (782, 380), (380, 611), (1, 1722), (1722, 185), (1722, 1121),
                 (107, 1712), (2042, 51), (173, 2042), (2375, 3667),
                 (2374, 3667), (2364, 2827), (163, 2349), (3631, 2389),
                 (3631, 2304), (1727, 1513), (1727, 1724), (1514, 1724),
                 (1514, 1727), (1781, 1853), (1781, 3228), (2242, 3116),
                 (3387, 5113), (3377, 3874), (3177, 5358), (5284, 5285),
                 (5285, 4840), (5285, 4841), (4841, 4843), (4851, 4857),
                 (4842, 4843), (4845, 4846), (4852, 4853), (4872, 4873),
                 (4874, 4875), (4876, 5308), (4873, 4776), (4873, 4775),
                 (4811, 4764), (4812, 4764), (4718, 4463), (6213, 4504),
                 (6213, 4503), (6213, 4505), (6096, 4489), (6096, 4896),
                 (5754, 6171), (5754, 6170), (6170, 6171), (6179, 6059),
                 (5715, 6040), (6178, 6177), (6045, 6210), (6177, 6210),
                 (6177, 6044), (5954, 5480), (5359, 3176), (5359, 4711),
                 (5359, 3177), (5456, 3188), (5456, 3187), (3186, 3248),
                 (5456, 3186), (161, 163)]

    return coord_df, node_blacklist, edge_blacklist, add_edges


def clean_positions(G, coord_df):
    '''
    Clean the graph and coordinates dataframe "by hand" by moving around some
    nodes.

    Parameters
    ----------
    G: networkx.Graph
        Graph.
    coord_df: pandas.DataFrame
        Dataframe contraining the frame coordinates.

    Returns
    -------
    G: networkx.Graph
        Cleaned graph.
    coord_df: pandas.DataFrame
        Cleaned Dataframe.

    '''
    # List of nodes to move.
    nodes_to_move = [([n for n in range(5129, 5266)],
                      [x*0.085 for x in range(0, 266-129)],
                      [y*0.07 for y in range(0, 266-129)]),
                     ([n for n in range(5266, 5285)],
                      [x*0.04 + 11.6 for x in range(0, 266-170)],
                      [y*0.1 + 9.5 for y in range(0, 266-170)]),
                     ([5285],
                      [11.3],
                      [8.8]),
                     ([n for n in range(4843, 4852)],
                      [-0.2, -0.2, -0.2, -0.3, 0, 0.2, 0.2, 0.3, 0],
                      [-1, 0, 1, 1, 1.3, 1.8, 2.2, 3.2, 3.8]),
                     ([n for n in range(4857, 4874)],
                      [-x*0.7 for x in range(0, 74-57)],
                      [y*0.2 for y in range(0, 74-57)]),
                     ([4872, 4873, 4762, 4464, 4718, 6172, 6171, 6170],
                      [-0.2, -0.7, 0, 0, -0.4, 0.6, 0.6, 0],
                      [0.3, 0, -0.3, 0.2, 0, 0, 0, -0.5]),
                     ([n for n in range(6096, 6142)],
                      [(46-x)*0.05 for x in range(0, 46)],
                      [(46-y)*0.01 for y in range(0, 46)]),
                     ([n for n in range(5754, 5784)],
                      [-(30-x)*0.02 for x in range(0, 30)],
                      [(30-y)*0.07 for y in range(0, 46)]),
                     ([n for n in range(5700, 5716)],
                      [x*0.02 for x in range(0, 16)],
                      [y*0.15 for y in range(0, 16)]),
                     ([n for n in range(5594, 5601)],
                      [(7-x)*0.05 for x in range(0, 7)],
                      [0 for y in range(0, 7)]),
                     ([n for n in range(5920, 5955)],
                      [-x*0.02 for x in range(0, 35)],
                      [0 for y in range(0, 35)]),
                     ([3305, 609, 610, 128, 129, 5515],
                      [0, -0.3, -0.2, 0, 0, 1],
                      [-0.2, 0, 0, -0.4, -0.3, 0.3]),
                     ([n for n in range(5359, 5371)],
                      [(12-x)*0.1 for x in range(0, 12)],
                      [(12-y)*0.02 for y in range(0, 12)]),
                     ([n for n in range(5435, 5457)],
                      [x*0.1 for x in range(0, 22)],
                      [0 for y in range(0, 22)])]

    # Clean the graph by moving around some nodes.
    for nodes, x, y in nodes_to_move:
        for idx, node in enumerate(nodes):
            G.nodes[node]['coords'][0] += x[idx]
            G.nodes[node]['coords'][1] += y[idx]
            coord_df.at[node, 'x'] = coord_df.loc[node]['x'] + x[idx]
            coord_df.at[node, 'y'] = coord_df.loc[node]['y'] + y[idx]
    return G, coord_df


def get_angle_between_nodes(G, n1, n2):
    '''
    Get angle between nodes.

    Parameters
    ----------
    G: networkx.Graph
        Graph.
    n1: int
        Single node 1.
    n2: int
        Single node 2.

    Returns
    -------
    angle: float
        Angle.

    '''
    x = G.nodes[n2]['coords'][0] - G.nodes[n1]['coords'][0]
    y = G.nodes[n2]['coords'][1] - G.nodes[n1]['coords'][1]
    angle = (math.atan2(y, x) * 180 / np.pi)
    return angle


def smallest_angles(a1, a2):
    '''
    Find smallest angles in a list compared to a reference.

    Parameters
    ----------
    a1: float
        Reference angle.
    a2: list
        List of angles.

    Return
    ------
    angles: list
        List of smallest angles.

    '''
    angles = []
    for a in a2:
        angle = a - a1
        angles.append(np.abs((angle + 180) % 360 - 180))
    return angles


def edge_is_usable(G, n1, n2):
    '''
    Edge is usable.

    Parameters
    ----------
    G: networkx.Graph
        Graph.
    n1: int
        Single node 1.
    n2: int
        Single node 2.

    Return
    ------
    bool
        If true, edge is usuable. Otherwise, it's not.

    '''
    # Find neighbors nodes.
    neighbors = [edge[1] for edge in list(G.edges(n1))]
    neighbor_angles = []
    for neighbor in neighbors:
        neighbor_angles.append(get_angle_between_nodes(G, n1, neighbor))

    # Find usuable edges.
    usable_edges = []
    for direction in [x*22.5 for x in range(-8, 8)]:
        angles = smallest_angles(direction, neighbor_angles)
        min_angle_node = neighbors[angles.index(min(angles))]
        if min(angles) < 22.5:
            usable_edges.append((n1, min_angle_node))
    # Is edge usuable?
    return (n1, n2) in usable_edges


def clean_edges(G, coord_df):
    '''
    Find edges that will not be used by the agent and print them.

    Parameters
    ----------
    G: networkx.Graph
        Graph.
    coord_df: pandas.DataFrame
        Dataframe contraining the frame coordinates.

    '''
    edges = []
    for n1, v in G.nodes(data=True):
        for n2 in [edge[1] for edge in list(G.edges(n1))]:
            if not edge_is_usable(G, n1, n2):
                edges.append((n1, n2))

    if len(edges) > 0:
        print('------------\nSome edges will not be used by the agent')
        print(edges)
        print('------------')


def plot_graph(G, figure_fname):
    '''
    Plot graph.

    Parameters
    ----------
    G: networkx.Graph
        Graph.
    figure_fname: str
        Figure file name.

    '''
    pos = {k: v.get('coords')[0:2] for k, v in G.nodes(data=True)}
    nx.draw_networkx(G, pos,
                     nodelist=G.nodes,
                     node_color='r',
                     node_size=10,
                     alpha=0.8,
                     with_label=True)
    plt.axis('equal')
    plt.savefig(figure_fname)


def construct_spatial_graph(coord_df, is_test):
    '''
    Filter the pano coordinates by spatial relation and create the graph.

    Parameters
    ----------
    coord_df: pandas.DataFrame
        Dataframe contraining the frame coordinates.

    Returns
    -------
    G: networkx.Graph
        Processed graph.
    coord_df: pandas.DataFrame
        Processed dataframe containing the frame coordinates.

    '''
    # Cleanup graph by hand.
    coord_df, node_blacklist, edge_blacklist, add_edges = cleanup_graph(coord_df, is_test)

    coord_df = coord_df[~coord_df.index.isin(node_blacklist)]

    # Init graph.
    G = nx.Graph()
    # Add nodes to the graph.
    G.add_nodes_from(coord_df.index)

    nodes = G.nodes
    for node_1_idx in tqdm(nodes, desc='Adding edges to graph'):
        # Add node informations to the graph.
        meta = coord_df[coord_df.index == node_1_idx]
        coords = np.array([meta['x'].values[0],
                           meta['y'].values[0],
                           meta['z'].values[0]])
        G.nodes[node_1_idx]['coords'] = coords
        G.nodes[node_1_idx]['timestamp'] = meta.timestamp
        G.nodes[node_1_idx]['angle'] = meta.angle
        G.nodes[node_1_idx]['old_index'] = node_1_idx

        # Find nearby nodes.
        radius = 1.1
        nearby_nodes = coord_df[(coord_df.x > coords[0] - radius) &
                                 (coord_df.x < coords[0] + radius) &
                                 (coord_df.y > coords[1] - radius) &
                                 (coord_df.y < coords[1] + radius)]
        for node_2_idx in nearby_nodes.index:
            if node_1_idx == node_2_idx:
                continue
            meta2 = coord_df[coord_df.index == node_2_idx]
            coords2 = np.array([meta2['x'].values[0],
                                meta2['y'].values[0],
                                meta2['z'].values[0]])
            G.nodes[node_2_idx]['coords'] = coords2
            # Compute edge between node 1 and 2 and add it to the graph.
            node_distance = np.linalg.norm(coords - coords2)
            G.add_edge(node_1_idx, node_2_idx, weight=node_distance)

    # Remove unwanted edges.
    for n1, n2 in edge_blacklist:
        if n1 in G.nodes and n2 in G.nodes:
            G.remove_edge(n1, n2)

    # Add wanted edges.
    for n1, n2 in add_edges:
        if n1 in G.nodes and n2 in G.nodes:
            G.add_edge(n1, n2)

    # Clean the graph and coordinates dataframe "by hand" by moving around
    # some nodes.
    G, coord_df = clean_positions(G, coord_df)

    # Find edges that will not be used by the agent and print them.
    clean_edges(G, coord_df)

    # Remap nodes
    mapping = coord_df.loc[coord_df.index, 'frame'].to_dict()
    G = nx.relabel_nodes(G, mapping)
    return G, coord_df


def label_segments(coord_df):
    '''
    Label street segment and intersection.

    Parameters
    ----------
    coord_df: pandas.DataFrame
        Dataframe to label.

    Returns
    -------
    coord_df: pandas.DataFrame
        Labeled dataframe.

    '''
    coord_df['type'] = None
    coord_df['group'] = None
    street_segments = [(25.3, 30.5, -113, -1.8),
                       (33, 38, -113.25, -2.5),
                       (62, 66, -115.8, -3.39),
                       (68.7, 76, -116, -4),
                       (35, 63.2, -125, -121),
                       (37, 62.5, -116.8, -115.1),
                       (38.3, 62.2, -3.3, -0.8),
                       (35.75, 62.5, 4.8, 6.6),
                       (1.8, 30.2, 105, 112.3)]
    intersections = [(24.7, 35, -123.5, -112.7),
                     (61.6, 73.6, -124.5, -116),
                     (26.38, 38.23, -2.4, 6.8),
                     (62.0, 72.2, -3.5, 6.2)]
    street_segments.extend([(-2.9, 25.4, -122.5, -121),
                            (-2.7, 24.7, -114.9, -113.5),
                            (0.9, 28.4, -2.0, 0.88),
                            (0.59, 26.6, 5.1, 8.3),
                            (-11.5, -7, -113.5, -1),
                            (-3.8, 1.3, -113.8, -0.4),
                            (26.5, 30.6, -112.5, -1.9)])
    intersections.extend([(-11.5, -2.7, -123, -113.5),
                          (-7.6, 0.52, -1.25, 8.5)])
    street_segments.extend([(-7.7, -4.8, 7.7, 105),
                            (-1, 1.6, 7.9, 105.1),
                            (25.5, 31.2, 6.9, 105.1),
                            (33, 39.9, 6.6, 105.2),
                            (61.9, 69.6, 6.4, 137.6),
                            (71.5, 76, 4.8, 137.4),
                            (30.9, 33.4, 111.6, 135.4),
                            (38.4, 40.3, 111.4, 135),
                            (3.5, 30.9, 134.5, 137.7),
                            (3.2, 31.5, 142, 143.6),
                            (40.9, 67.6, 135.5, 138.6),
                            (40.8, 67, 142.7, 145.6)])
    intersections.extend([(-8.7, 1.3, 105.2, 112.1),
                          (30.1, 39.8, 105, 111.8),
                          (-5, 4, 134.8, 143.3),
                          (30.9, 41.3, 134.5, 144.6),
                          (67.2, 77.9, 137, 145.5)])
    street_segments.extend([(-6.5, -3.3, 142.8, 260.6),
                            (0, 3, 143, 260),
                            (31, 35.1, 32.1, 260.6),
                            (38.7, 43, 144.2, 261),
                            (67, 70.6, 145.3, 260.6),
                            (72, 89.5, 145.3, 258.9),
                            (2.5, 33.7, 260.4, 262.3),
                            (2.75, 33.9, 270.4, 272.1),
                            (42.7, 67.9, 260.6, 262.2),
                            (42.9, 67.2, 270, 273)])
    intersections.extend([(-7.7, 2.6, 260.2, 271.6),
                          (33.6, 42.75, 260.69, 272.9),
                          (67.5, 92.7, 258.9, 274)])
    for idx, box in enumerate(street_segments):
        segment = coord_df[((coord_df.x > box[0]) &
                             (coord_df.x < box[1]) &
                             (coord_df.y > box[2]) &
                             (coord_df.y < box[3]))]
        segment['type'] = 'street_segment'
        segment['group'] = idx
        coord_df.loc[segment.index] = segment

    for idx, box in enumerate(intersections):
        intersection = coord_df[((coord_df.x > box[0]) &
                                  (coord_df.x < box[1]) &
                                  (coord_df.y > box[2]) &
                                  (coord_df.y < box[3]))]
        intersection['type'] = 'intersection'
        intersection['group'] = idx
        coord_df.loc[intersection.index] = intersection
    return coord_df


def process_labels(paths, shape, w, h, crop_margin):
    '''
    This function processes the labels into a nice format for the simulator.

    Parameters
    ----------
    paths: list
        List of label file names.
    shape: tuple
      Panos original shape.
    w: int
      Targeted image width.
    h: int
      Targeted image height.
    crop_margin: int
      Crop margin.

    Returns
    -------
    label_df: pandas.DataFrame
        Dataframe of labels.

    '''
    labels = []
    failed_to_parse = []
    for p in paths:
        xtree = et.parse(p)
        xroot = xtree.getroot()
        for idx, node in enumerate(xroot):
            if node.tag != 'object':
                continue
            frame = int(p.split('_')[-1].split('.')[0])
            text_label = node.find('name').text
            house_number = None
            if text_label.split('-')[0] == 'street_sign':
                try:
                    obj_type, street_name = text_label.split('-')
                except Exception as e:
                    print('street_sign: {}'.format(e))
                    failed_to_parse.append(text_label)
                    continue
            elif text_label.split('-')[0] == 'house_number':
                try:
                    obj_type, house_number, street_name = text_label.split('-')
                except Exception as e:
                    print('house_number: {}'.format(e))
                    failed_to_parse.append(text_label)
                    continue
            elif text_label.split('-')[0] == 'door':
                try:
                    obj_type, house_number, street_name = text_label.split('-')
                except Exception as e:
                    print('door: {}'.format(e))
                    failed_to_parse.append(text_label)
                    continue
            x_min = int(w *
                        int(node.find('bndbox').find('xmin').text) / shape[0])
            x_max = int(w *
                        int(node.find('bndbox').find('xmax').text) / shape[0])
            y_min = int((h - 2 * crop_margin) *
                        int(node.find('bndbox').find('ymin').text) / shape[1])
            y_max = int((h - 2 * crop_margin) *
                        int(node.find('bndbox').find('ymax').text) / shape[1])
            area = (x_max - x_min) * (y_max - y_min)
            labels.append((frame, obj_type, house_number, street_name, False,
                           x_min, x_max, y_min, y_max, area))
    label_df = pd.DataFrame(labels,
                            columns=['frame', 'obj_type', 'house_number',
                                     'street_name', 'is_goal',
                                     'x_min', 'x_max', 'y_min', 'y_max',
                                     'area'])
    print('Num labels failed to parse: {}'.format((len(failed_to_parse))))

    # Find target panos -- they are the ones with the biggest bounding box
    # an the house number
    doors = label_df[label_df.obj_type == "door"]
    addresses = set([x.house_number + '-' + x.street_name for i, x in
                     doors[['house_number', 'street_name']].iterrows()])
    for address in addresses:
        house_number, street_name = address.split('-')
        matched_doors = label_df[(label_df.obj_type == 'door') &
                                 (label_df.house_number == house_number) &
                                 (label_df.street_name == street_name)]
        label_df.at[matched_doors.area.idxmax(), 'is_goal'] = True
    return label_df


def normalize_image(image):
    '''
    Values calculated for SEVN: mean=[0.45247, 0.45871, 0.47285],
    std=[0.25556, 0.26181, 0.27931].
    '''
    normed_image = image / 255.0
    normed_image[:, :, 0] = (normed_image[:, :, 0] - 0.45247) / 0.25556
    normed_image[:, :, 1] = (normed_image[:, :, 1] - 0.45871) / 0.26181
    normed_image[:, :, 2] = (normed_image[:, :, 2] - 0.47285) / 0.27931
    return normed_image


def process_images(image_fname, paths, w, h, crop_margin):
    '''
    Loads in the pano images from disk, crops them and resizes them.

    Parameters
    ----------
    paths: list
        List of image file names.
    w: int
        Image width.
    h: int
        Image height.
    crop_margin: int
        Crop margin.

    Returns
    -------
    images: dict
        Dictionary containing the processed images.

    '''

    # TODO: writeout the large-scale images, too
    hdf5_file = tables.open_file(image_fname, mode='w')
    img_dtype = tables.Atom.from_dtype(np.dtype(np.float))
    storage = hdf5_file.create_earray(hdf5_file.root, 'images', img_dtype, shape=(0, h - 2 * crop_margin, w, 3))
    frames = np.zeros(len(paths))
    # Get panos and crop them
    for idx, path in enumerate(tqdm(paths, desc='Loading images')):
        frame = int(path.split('_')[-1].split('.')[0])
        frames[idx] = frame
        path = path.replace('jpg', 'png')
        image = cv2.imread(path)
        image = normalize_image(image)
        image = cv2.resize(image, (w, h))[:, :, ::-1]
        image = image[crop_margin:h - crop_margin]
        storage.append(image[None])
    hdf5_file.create_array(hdf5_file.root, 'frames', frames)
    hdf5_file.close()
    return


def create_dataset(data_path='data/SEVN', do_images=True, do_images_high_res=True, do_graph=True,
                   do_plot=False):
    '''
    If do_graph: pre-processes the pose data associated with the image
    and calls the fonction to create the graph and to process the labels.
    If do images: loads in the pano images from disk, crops them,
    resizes them, and writes them to disk.
    Otherwise load the processed data.

    Parameters
    ----------
    data_path: str
        Path to the data directory.
    do_images: bool
        If true, build the lower resolution images to disk. Default = True.
    do_graph: bool
        If true, build the graph. Default = False.
    do_plot: bool
        If true, plot the data. Default = False.

    Returns
    -------
    coord_df: pandas.DataFrame
        Dataframe containing the coordinates for each pano.
    G: networkx.Graph
        Graph.
    img_paths: list
        List of image file names.

    '''
    # Set up output path.
    output_path = data_path

    # Print output file names.
    coord_df_fname = os.path.join(output_path, 'processed/coord.hdf5')
    print('coord_df_fname: {}'.format(coord_df_fname))
    label_df_fname = os.path.join(output_path, 'processed/label.hdf5')
    print('label_df_fname: {}'.format(label_df_fname))
    graph_fname = os.path.join(output_path, 'processed/graph.pkl')
    print('graph_fname: {}'. format(graph_fname))
    if do_plot:
        plot_fname = os.path.join(output_path, 'processed/graph.png')
        print('figure_fname: {}'.format(plot_fname))
    if do_images:
        image_fname = os.path.join(output_path, 'processed/images.hdf5')
        print('image_fname: {}'.format(image_fname))
    if do_images_high_res:
        high_res_image_fname = os.path.join(output_path, 'processed/images-high-res.hdf5')
        print('high_res_image_fname: {}'.format(high_res_image_fname))

    if do_graph:
        # Load frame coordinates.
        coords = np.load(os.path.join(data_path, 'processed/pos_ang.npy'))
        coord_df = pd.DataFrame({'x': coords[:, 2],
                                  'y': coords[:, 3],
                                  'z': coords[:, 4],
                                  'angle': coords[:, -1],
                                  'timestamp': coords[:, 1],
                                  'frame': [int(x) for x in coords[:, 1]*30]})

        # Filter the pano coordinates by spatial relation.
        G, coord_df = construct_spatial_graph(coord_df, args.is_test)
        if do_plot:
            # Plot the graph.
            plot_graph(G, plot_fname)
        # Save graph.
        nx.write_gpickle(G, graph_fname)

        # Get existing label paths.
        label_paths = [data_path + '/raw/labels/pano_' + str(frame).zfill(6) +
                       '.xml' for frame in coord_df['frame'].tolist()]
        label_paths = [path for path in label_paths if os.path.isfile(path)]

        # Process labels.
        label_df = process_labels(label_paths,
                                  shape=(3840, 2160),
                                  w=224, h=126, crop_margin=int(126*(1/6)))
        label_df['obj_type'] = label_df['obj_type'].fillna('')
        label_df['is_goal'] = label_df['is_goal'].fillna(False)
        label_df.index = label_df.frame
        label_df = label_df.drop(['frame'], axis=1)
        label_df.to_hdf(label_df_fname, key='df', format='table')

        # Label street segment and intersection.
        coord_df = label_segments(coord_df)
        coord_df.index = coord_df.frame
        coord_df = coord_df.drop(['frame'], axis=1)
        coord_df.to_hdf(coord_df_fname, key='df', format='table')
    else:
        # Load coordinates.
        coord_df = pd.read_hdf(coord_df_fname,
                              key='df', index=False)
        # Load labels.
        label_df = pd.read_hdf(label_df_fname,
                              key='df', index=False)
        # Load graph.
        G = nx.read_gpickle(graph_fname)
        if do_plot:
            # Plot graph.
            plot_graph(G, plot_fname)

    # Get png names.
    frame_num = set(coord_df.index.tolist())
    img_paths = [data_path + '/raw/panos/pano_' + str(frame).zfill(6) +
                 '.jpg' for frame in frame_num]

    if do_images:
        # Loads in the pano images from disk, crops, resizes, and saves.
        h = 126
        w = 224
        crop_margin = int(h * (1/6))
        process_images(image_fname, img_paths, w=w, h=h, crop_margin=crop_margin)
    if do_images_high_res:
        h = 1920
        w = 3840
        crop_margin = int(h * (1/6))
        process_images(high_res_image_fname, img_paths, w=w, h=h, crop_margin=crop_margin)

    return coord_df, G, img_paths


if __name__ == '__main__':
    # Load arguments.
    args = parse_args()

    data_path = args.data_path
    print('data_path: {}'.format(data_path))
    # Check if input directory exist and is not empty.
    check_dir(data_path)

    do_images = args.do_images
    print('do_images: {}'.format(do_images))
    do_images_high_res = args.do_images_high_res
    print('do_images_high_res: {}'.format(do_images_high_res))
    do_graph = args.do_graph
    print('do_graph: {}'.format(do_graph))
    do_plot = args.do_plot
    print('do_plot: {}'.format(do_plot))

    # Create dataset.
    coord_df, G, img_paths = create_dataset(data_path=data_path,
                                           do_images=do_images,
                                           do_images_high_res=do_images_high_res,
                                           do_graph=do_graph,
                                           do_plot=do_plot)
