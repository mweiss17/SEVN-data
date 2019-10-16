import networkx as nx
import pandas as pd
import math
import numpy as np
import itertools
import matplotlib.pyplot as plt


def plot_intersection_street_segment(G, plot_fname):
    # Plot intersection and street segments
    pos = {k: v.get('coords')[0:2] for k, v in G.nodes(data=True)}
    intersection = set(n for n, v in G.nodes(data=True)
                       if v.get('type') == 'intersection')
    street_segment = set(n for n, v in G.nodes(data=True)
                         if v.get('type') == 'street_segment')
    # Street segment
    nx.draw_networkx(G, pos, nodelist=street_segment, node_color='b',
                     node_size=10, alpha=0.8, labels={}, with_label=True)
    # Intersection
    nx.draw_networkx(G, pos, nodelist=intersection, node_color='r',
                     node_size=10, alpha=0.8, labels={}, with_label=True)
    plt.axis('equal')
    plt.savefig(plot_fname, transparent=True, dpi=1000)
    plt.close()


def plot_groups(G, plot_fname):
    pos = {k: v.get('coords')[0:2] for k, v in G.nodes(data=True)}
    nodelists = [set(n for n, v in G.nodes(data=True)
                     if v.get('group') == i and v.get('type') == 'street_segment')
                 for i in set(coord['group'].tolist())]

    cmap = plt.get_cmap('gist_rainbow')
    num_colors = len(nodelists)
    colors = [cmap(1.*i/num_colors) for i in range(num_colors)]

    for idx, nodes in enumerate(nodelists):
        nx.draw_networkx(G, pos, nodelist=nodes, node_color=colors[idx],
                         node_size=10, alpha=0.8, labels={}, with_label=True)

    plt.axis('equal')
    plt.savefig(plot_fname, transparent=True, dpi=1000)
    plt.close()


def plot_node_degrees(G, plot_fname, intersection=True, street_segment=True):
    pos = {k: v.get('coords')[0:2] for k, v in G.nodes(data=True)}

    if intersection and not street_segment:
        nodelists = [set(n for n, v in G.degree
                     if v == i and G.node[n]['type'] == 'intersection')
                     for i in set(v for n, v in G.degree
                     if G.node[n]['type'] == 'intersection')]
    elif street_segment and not intersection:
        nodelists = [set(n for n, v in G.degree
                     if v == i and G.node[n]['type'] == 'intersection')
                     for i in set(v for n, v in G.degree
                     if G.node[n]['type'] == 'intersection')]
    else:
        nodelists = [set(n for n, v in G.degree
                     if v == i) for i in set(v for n, v in G.degree)]

    cmap = plt.get_cmap('gist_rainbow')
    num_colors = len(nodelists)
    colors = [cmap(1.*i/num_colors) for i in range(num_colors)]

    for idx, nodes in enumerate(nodelists):
        nx.draw_networkx(G, pos, nodelist=nodes, node_color=colors[idx],
                         node_size=10, alpha=0.8, labels={}, with_label=True)

    l = set(v for n, v in G.degree)
    plt.legend([x for pair in zip(l, l) for x in pair])
    plt.axis('equal')
    plt.savefig(plot_fname, transparent=True, dpi=1000)
    plt.close()


def plot_goal(G, plot_fname):
    pos = {k: v.get('coords')[0:2] for k, v in G.nodes(data=True)}
    is_goal = [n for n, v in G.nodes(data=True) if True in v.get('is_goal')]
    nx.draw_networkx(G, pos, nodelist=is_goal, node_color='#79d151',
                     node_size=5, alpha=0.8, labels={}, with_label=True)
    plt.axis('equal')
    plt.savefig(plot_fname, transparent=True, dpi=1000)
    plt.close()


def plot_door(G, plot_fname):
    pos = {k: v.get('coords')[0:2] for k, v in G.nodes(data=True)}
    door = [n for n, v in G.nodes(data=True) if 'door' in v.get('obj_type')]
    nx.draw_networkx(G, pos, nodelist=door, node_color='b',
                     node_size=5, alpha=0.8, labels={}, with_label=True)
    plt.axis('equal')
    plt.savefig(plot_fname, transparent=True, dpi=1000)
    plt.close()


def plot_street_sign(G, plot_fname):
    pos = {k: v.get('coords')[0:2] for k, v in G.nodes(data=True)}
    street_sign = [n for n, v in G.nodes(data=True) if 'street_sign' in v.get('obj_type')]
    nx.draw_networkx(G, pos, nodelist=street_sign, node_color='b',
                     node_size=5, alpha=0.8, labels={}, with_label=True)
    plt.axis('equal')
    plt.savefig(plot_fname, transparent=True, dpi=1000)
    plt.close()


def plot_house_number(G, plot_fname):
    pos = {k: v.get('coords')[0:2] for k, v in G.nodes(data=True)}
    house_number = [n for n, v in G.nodes(data=True) if 'house_number' in v.get('obj_type')]
    nx.draw_networkx(G, pos, nodelist=house_number, node_color='b',
                     node_size=5, alpha=0.8, labels={}, with_label=True)
    plt.axis('equal')
    plt.savefig(plot_fname, transparent=True, dpi=1000)
    plt.close()


def plot_house_number_sup1(G, plot_fname):
    pos = {k: v.get('coords')[0:2] for k, v in G.nodes(data=True)}
    house_number = [n for n, v in G.nodes(data=True)
                    if 'house_number' in v.get('obj_type') and
                    len(v.get('house_number')) >= 1]
    nx.draw_networkx(G, pos, nodelist=house_number, node_color='b',
                     node_size=5, alpha=0.8, labels={}, with_label=True)
    plt.axis('equal')
    plt.savefig(plot_fname, transparent=True, dpi=1000)
    plt.close()


DATA_PATH = 'data/SEVN/'

G = nx.read_gpickle(DATA_PATH + 'graph.pkl')

label = pd.read_hdf(DATA_PATH + 'label.hdf5', key='df', mode='r')
coord = pd.read_hdf(DATA_PATH + 'coord.hdf5', key='df', mode='r')

print(f'# nodes in G: {G.__len__()}')

# Clean G nodes meta-data
for n, v in G.node(data=True):
    del G.node[n]['old_index']
    del G.node[n]['timestamp']  # we can calculate it from the frame number
    G.node[n]['angle'] = float(v['angle'])


# coords df
# x                   49.4089
# y                  -115.969
# z                         0
# angle               182.483  #  rotation of the camera when it took the photograph
# timestamp           4515.83  #  time when the photo was taken
# frame                135474
# type         street_segment / segment
# group                     5


# Clean G edges meta-data
for u, v, d in G.edges(data=True):
    x1, y1, z1 = G.node[u]['coords']
    x2, y2, z2 = G.node[v]['coords']
    weight = ((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)**(1/2)
    G[u][v]['weight'] = weight

# Add coord meta-data to the G nodes
for n, v in G.node(data=True):
    assert G.node[n]['angle'] == coord.loc[n, 'angle']
    assert G.node[n]['coords'][0] == coord.loc[n, 'x']
    assert G.node[n]['coords'][1] == coord.loc[n, 'y']
    assert G.node[n]['coords'][2] == coord.loc[n, 'z']
    G.node[n]['coords'] = (coord.loc[n, 'x'],
                           coord.loc[n, 'y'],
                           coord.loc[n, 'z'])
    G.node[n]['type'] = coord.loc[n, 'type']
    G.node[n]['group'] = coord.loc[n, 'group']


# Sanity check
print(f'type: {set(coord["type"].tolist())}')

intersection = set(n for n, v in G.nodes(data=True)
                   if v.get('type') == 'intersection')

street_segment = set(n for n, v in G.nodes(data=True)
                     if v.get('type') == 'street_segment')


assert bool(set(intersection) & set(street_segment)) is False
assert len(intersection) + len(street_segment) == G.__len__()


# Add label meta-data to the G nodes
# label_df
# ['frame', 'obj_type', 'house_number', 'street_name', 'is_goal', 'x_min',
# 'x_max', 'y_min', 'y_max', 'area']

# We already have frame
# We don't need area (area = (x_max-x_min)*(y_max-y_min))

# Object type
obj_type = ['door', 'house_number', 'street_sign']

# SEVN street
street_SEVN = ['saint_laurent', 'clark', 'saint_urbain',
               'saint_zotique', 'beaumont', 'mozart',
               'jean_talon', 'beaubien']


# Visible street sign within SEVN
street_sign = ['beaubien', 'beaumont', 'belanger', 'clark', 'dante',
               'jean_talon', 'mozart', 'saint_laurent', 'saint_urbain',
               'saint_zotique', 'shamrock']


for n, v in G.node(data=True):
    G.node[n]['obj_type'] = []
    G.node[n]['is_goal'] = []
    G.node[n]['house_number'] = []
    G.node[n]['street_name'] = []
    G.node[n]['bbox'] = []

    if n in label.index:
        if isinstance(label.loc[n, 'obj_type'], str):
            assert label.loc[n, 'obj_type'] in obj_type
            obj_type = [str(label.loc[n, 'obj_type'])]

            if isinstance(label.loc[n, 'house_number'], str) and \
               label.loc[n, 'house_number'] != 'nan':
                assert label.loc[n, 'obj_type'] != 'street_sign'
                house_number = [str(label.loc[n, 'house_number'])]
            else:
                assert label.loc[n, 'obj_type'] == 'street_sign'
                house_number = [np.nan]

            assert label.loc[n, 'street_name'] in street_sign
            street_name = [str(label.loc[n, 'street_name'])]

            if str(label.loc[n, 'obj_type']) == 'door':
                assert bool(label.loc[n, 'is_goal']) in [True, False]
            else:
                assert bool(label.loc[n, 'is_goal']) is False
            is_goal = [bool(label.loc[n, 'is_goal'])]

            x_min = float(label.loc[n, 'x_min'])
            x_max = float(label.loc[n, 'x_max'])
            y_min = float(label.loc[n, 'y_min'])
            y_max = float(label.loc[n, 'y_max'])
            assert math.isnan(x_min) is False
            assert math.isnan(x_max) is False
            assert math.isnan(y_min) is False
            assert math.isnan(y_max) is False
            bbox = [(x_min, x_max, y_min, y_max)]

        else:
            obj_type = []
            house_number = []
            street_name = []
            is_goal = []
            bbox = []
            for idx, i in enumerate(label.loc[n, 'obj_type'].tolist()):

                assert label.loc[n, 'obj_type'].tolist()[idx] in obj_type
                obj_type.append(str(label.loc[n, 'obj_type'].tolist()[idx]))

                if isinstance(label.loc[n, 'house_number'].tolist()[idx], str) and \
                   label.loc[n, 'house_number'].tolist()[idx] != 'nan':
                    assert label.loc[n, 'obj_type'].tolist()[idx] != 'street_sign'
                    house_number.append(str(label.loc[n, 'house_number'].tolist()[idx]))
                else:
                    assert label.loc[n, 'obj_type'].tolist()[idx] == 'street_sign'
                    house_number.append(np.nan)

                assert label.loc[n, 'street_name'].tolist()[idx] in street_sign
                street_name.append(str(label.loc[n, 'street_name'].tolist()[idx]))

                if str(label.loc[n, 'obj_type'].tolist()[idx]) == 'door':
                    assert bool(label.loc[n, 'is_goal'].tolist()[idx]) in [True, False]
                else:
                    assert bool(label.loc[n, 'is_goal'].tolist()[idx]) is False
                is_goal.append(bool(label.loc[n, 'is_goal'].tolist()[idx]))

                x_min = float(label.loc[n, 'x_min'].tolist()[idx])
                x_max = float(label.loc[n, 'x_max'].tolist()[idx])
                y_min = float(label.loc[n, 'y_min'].tolist()[idx])
                y_max = float(label.loc[n, 'y_max'].tolist()[idx])
                assert math.isnan(x_min) is False
                assert math.isnan(x_max) is False
                assert math.isnan(y_min) is False
                assert math.isnan(y_max) is False
                bbox.append((x_min, x_max, y_min, y_max))

        G.node[n]['obj_type'] = obj_type
        G.node[n]['house_number'] = house_number
        G.node[n]['street_name'] = street_name
        G.node[n]['is_goal'] = is_goal
        G.node[n]['bbox'] = bbox

    else:
        G.node[n]['obj_type'] = []
        G.node[n]['house_number'] = []
        G.node[n]['street_name'] = []
        G.node[n]['is_goal'] = []
        G.node[n]['bbox'] = []


# Clean G nodes degree

# Nodes degree before cleaning
possible_nodes_degree = set(v for n, v in G.degree if
                            G.node[n]['type'] == 'street_segment')
print(f'Possible nodes degree within a segment: {possible_nodes_degree}')

# Average node spacing before cleaning
avg_w = []
for u, v, d in G.edges(data=True):
    avg_w.append(d['weight'])
print(f'Average node spacing: {np.mean(avg_w)}')


# Remove unuseful edges within street segments
to_del = []

degree_0 = set(n for n, v in G.degree if v == 0)
degree_1 = set(n for n, v in G.degree if v == 1)
assert bool(degree_0) is False
assert bool(degree_1) is False

degree_3 = set(n for n, v in G.degree
               if v == 3 and G.node[n]['type'] == 'street_segment')

while degree_3:

    for n1 in degree_3:
        neighbors = [n2 for n2 in nx.all_neighbors(G, n1)]
        edges = []
        for u, v in itertools.combinations(neighbors, 2):
            x1, y1, z1 = G.node[u]['coords']
            x2, y2, z2 = G.node[v]['coords']
            weight = ((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)**(1/2)
            edges.append([u, v, weight])
        edges = np.array(edges)
        edges = np.sort(edges.view('i8,i8,i8'),
                        order=['f2'], axis=0).view(np.float)
        candidates = [(n1, edges[0, 0]), (n1, edges[0, 1])]
        edges = np.array([[n1, n2, G[n1][n2]['weight']]
                         for n2 in nx.all_neighbors(G, n1)])
        edges = np.sort(edges.view('i8,i8,i8'),
                        order=['f2'], axis=0).view(np.float)

        if tuple(edges[2, 0:2]) in candidates:
            to_del.append(tuple(edges[2, 0:2]))
        else:
            to_del.append(tuple(edges[1, 0:2]))

    G.remove_edges_from(set(to_del))
    degree_3 = set(n for n, v in G.degree
                   if v == 3 and G.node[n]['type'] == 'street_segment')

degree_0 = set(n for n, v in G.degree if v == 0)
degree_1 = set(n for n, v in G.degree if v == 1)
assert bool(degree_0) is False
assert bool(degree_1) is False

# Nodes degree after cleaning
possible_nodes_degree = set(v for n, v in G.degree if
                            G.node[n]['type'] == 'street_segment')
print(f'Possible nodes degree within a segment after cleaning: {possible_nodes_degree}')


# Average node spacing after cleaning
avg_w = []
for u, v, d in G.edges(data=True):
    avg_w.append(d['weight'])
print(f'Average node spacing after cleaning: {np.mean(avg_w)}')

# Save new graph
nx.write_gpickle(G, DATA_PATH + 'new_graph.pkl')


###############################################################################
###############################################################################
# Remove unuseful edges within intersections
# This needs to be verified with the panos
###############################################################################
###############################################################################

nodelist = []
for u, v, d in G.edges(data=True):
    if G.node[u]['type'] == 'street_segment' and  \
       G.node[v]['type'] == 'intersection':
        nodelist.append(u)

    if G.node[u]['type'] == 'intersection' and \
       G.node[v]['type'] == 'street_segment':
        nodelist.append(v)

# Record the nodes that are too close to street segment (we don't want to
# acidentally remove them, those will be cleaned by hand after)
dont_touch = []
for n1 in set(nodelist):
    source_path_lengths = nx.single_source_dijkstra_path_length(
        G, n1, weight=None)
    for (n2, w) in source_path_lengths.items():
        if w <= 4.:
            if G.node[n2]['type'] == 'intersection':
                dont_touch.append(n2)

possible_nodes_degree = set(v for n, v in G.degree if
                            G.node[n]['type'] == 'intersection' and
                            n not in dont_touch)
print(f'Possible nodes degree within a intersection: {possible_nodes_degree}')

to_del = []

degree_3 = set(n for n, v in G.degree
               if v == 3 and G.node[n]['type'] == 'intersection' and
               n not in dont_touch)

while degree_3:

    for n1 in degree_3:
        neighbors = [n2 for n2 in nx.all_neighbors(G, n1)]
        edges = []
        for u, v in itertools.combinations(neighbors, 2):
            x1, y1, z1 = G.node[u]['coords']
            x2, y2, z2 = G.node[v]['coords']
            weight = ((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)**(1/2)
            if u == 133104 or v == 133104:
                stop = True
            edges.append([u, v, weight])
        edges = np.array(edges)
        edges = np.sort(edges.view('i8,i8,i8'),
                        order=['f2'], axis=0).view(np.float)
        candidates = [(n1, edges[0, 0]), (n1, edges[0, 1])]

        edges = np.array([[n1, n2, G[n1][n2]['weight']]
                         for n2 in nx.all_neighbors(G, n1)])
        edges = np.sort(edges.view('i8,i8,i8'),
                        order=['f2'], axis=0).view(np.float)

        if tuple(edges[2, 0:2]) in candidates:
            to_del.append(tuple(edges[2, 0:2]))
        else:
            to_del.append(tuple(edges[1, 0:2]))

    G.remove_edges_from(set(to_del))
    degree_3 = set(n for n, v in G.degree
                   if v == 3 and G.node[n]['type'] == 'street_segment')

degree_0 = set(n for n, v in G.degree if v == 0)
assert bool(degree_0) is False


# By hand correction
G.remove_edge(116829, 133068)


def weight_uv(u, v, G):
    x1, y1, z1 = G.node[u]['coords']
    x2, y2, z2 = G.node[v]['coords']
    weight = ((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)**(1/2)
    return weight


weight = weight_uv(133104, 116829, G)
G.add_edge(133104, 116829, weight=weight)

degree_1 = set(n for n, v in G.degree if v == 1)
assert bool(degree_1) is False

to_del_unsure = [(46976, 41282)]
to_del_sure = [
               # intersection 1
               (47316, 47367), (48045, 48087), (48066, 48108),
               (41247, 41282),
               # intersection 2
               (29246, 29213), (29277, 29372), (29372, 29445),
               (29372, 29277), (29412, 29471), (16017, 16059),
               (30617, 30579), (299282, 29778), (29829, 29750),
               (29721, 29778), (29694, 29637), (29694, 29750),
               (16059, 16118), (29928, 29778), (29667, 29721),
               (29246, 29313), (29313, 42690), (42690, 29412),
               (14904, 40104), (14904, 29045), (40104, 29045),
               (16086, 16118), (29502, 29549),
               # intersection 3
               (129141, 129099), (117002, 116961), (116961, 116919),
               (116919, 116877), (116939, 116898), (134745, 133422),
               (134745, 133422), (134745, 134790), (134768, 133463),
               # intersection 4
               (8760, 8802), (8781, 8736), (8826, 8868), (8897, 8943),
               (8781, 8826), (55338, 75330), (8943, 75330), (8943, 75330),
               (56325, 75125), (3240, 3204), (56367, 75125),
               (74837, 74859), (74837, 3168),
               (8700, 2960), (2982, 67083), (2960, 67083),
               # intersection 5
               (1575, 1518), (1391, 1445), (1461, 1473), (5157, 5199),
               (7388, 5877), (7583, 1803), (91692, 7365), (5877, 91692),
               (104421, 5643), (5378, 102161),
               (5226, 20759), (1416, 1473), (1803, 7584), (1445, 1518),
               (1473, 1518), (1518, 4932), (5253, 5337), (5233, 20759),
               (5877, 104448), (5877, 5643), (5643, 104448), (5253, 20759),
               (5337, 102161), (4932, 10215), (10215, 4980),
               # initersection 6
               (103104, 103133), (103133, 106206), (106242, 111822),
               (106272, 111822), (106416, 106458), (106458, 106518),
               (106482, 106548), (106914, 106829), (106866, 106800),
               (106773, 106829), (107930, 107888), (103395, 103362),
               (103362, 107978), (103133, 106206),
               (106617, 106572), (106437, 106482), (107930, 107880),
               # intersection 7
               (71402, 85632), (85868, 70668), (70925, 70880),
               (70902, 70851), (70880, 70824), (70851, 70800),
               (70824, 70781), (70800, 81657), (70824, 81657),
               (70824, 85068), (70800, 70761),
               # intersection 8
               (85848, 70668), (85868,  85829), (80312, 83133),
               (83379, 95316), (83309, 83351), (80277, 87068),
               (80312, 831029), (83156, 83109), (80409, 80363),
               (80385, 80340), (80517, 80457), (147932, 83811),
               (80457, 83832), (80312, 80277), (80312, 80277),
               (95463, 95421), (144032, 95492), (147932, 83832),
               (83607, 144032), (80517, 147932), (80517, 83832),
               # intersection 9
               (243891, 243705), (243767, 243651), (243705, 243618),
               (243387, 240906), (244200, 244158), (244200, 244241),
               (244389, 241146), (244179, 244220), (258608, 243618),
               (258608, 243594), (258608, 243570), (240906, 240863),
               (243408, 259380), (243387, 259380), (240863, 259380),
               (241095, 241146), (241116, 241172), (241116, 244368),
               (244389, 241116), (244368, 241146),
               # intersection 10
               (146445, 145314), (242108, 146994), (146994, 152114),
               (147015, 152114), (146952, 242108), (146952, 152114),
               (146952, 152088), (239901, 146715), (146715, 200858),
               (239901, 200858), (145349, 187610), (145280, 156741),
               (145251, 156741), (187586, 156716), (150747, 150680),
               (151890, 150710), (145068, 157052), (145038, 157049),
               (145349, 145314), (145062, 157025), (157025, 154380),
               (157025, 145038), (154346, 154380), (154346, 157049),
               (154346, 157025), (154346, 145011), (150680, 150710),
               (154346, 150747), (150747, 151890), (157049, 154406),
               (157025, 154406),
               # intersection 11
               (158067, 164615), (155442, 164582), (159507, 159552),
               (160017, 159960), (159995, 159933), (159960, 159911),
               (155474, 155442), (159744, 174993), (159639, 174993),
               (238211, 155723), (155723, 155688), (238211, 160257),
               (225621, 159960), (225600, 159933), (225600, 159995),
               (155688, 160257),
               # intersection 12
               (263832, 263885), (273014, 273056), (254268, 269198),
               (268377, 272724), (268358, 272724), (263904, 263856),
               (269370, 263856), (269405, 263856), (269370, 273056),
               (273056, 269405), (273056, 263885), (269405, 263885),
               # intersection 13
               (196239, 196280), (196259, 196301), (196301, 218741),
               (196344, 218741), (192087, 219015),
               (192087, 208008), (193278, 219015), (193278, 208008),
               (194577, 266091), (194577, 193697), (194540, 266091),
               (194540, 266091), (194540, 193659), (266091, 193604),
               (266091, 193632), (196301, 196344),
               (196301, 274068), (274068, 196344), (194799, 218273),
               (194828, 194939), (194856, 218273), (193697, 193659),
               (193604, 210558), (210558, 193659), (194939, 194856),
               (267435, 218173), (267435, 218273),
               # intersection 14
               (220203, 220161), (220182, 220140), (209079, 209000),
               (209079, 209034), (209172, 212015), (209172, 212034),
               (220140, 212034), (220140, 212015), (220110, 212034),
               (220110, 212015), (220110, 209034), (209520, 216558)
               ]

l = [41247, 16118, 42690, 40104, 40104, 55338, 56367, 56388, 74837,
     20759, 10215, 10244, 1391, 1416, 1445, 1473, 4932, 104448, 104421,
     104393, 104372, 103395, 103104, 106206, 111822, 85068, 81657,
     147932, 80517, 95276, 80277, 80277, 87068, 240863, 242108, 187610,
     146994, 156741, 156716, 147038, 200858, 239901, 242108, 146994,
     147015, 147038, 200858, 239901, 187610, 156741, 156716, 150747,
     150710, 150680, 154380, 154346, 157049, 157025, 164582, 164615,
     155442, 238211, 155723, 225621, 225600, 254268, 263885, 263832,
     263856, 192087, 208008, 196344, 194856, 267435, 220182, 220161,
     209034, 209000, 220110, 220203, 209552, 209520, 220140, 14904]

for i in l:
    G.node[i]['type'] = 'street_segment'


G.remove_edges_from(set(to_del_sure))

weight = weight_uv(269405, 263904, G)
G.add_edge(269405, 263904, weight=weight)
