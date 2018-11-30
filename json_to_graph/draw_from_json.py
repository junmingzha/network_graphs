#!/usr/bin/python3

"""
sudo -H pip3 install networkx matplotlib
sudo -H pip3 install pydot
sudo apt-get install graphviz

test.json
{
    "area": "Test Network",
    "directed": false,
    "multigraph": false,
    "nodes": [
    {"id": "R1"},
    {"id": "R2"},
    {"id": "R3"},
    {"id": "R4"},
    {"id": "R5"}
    ],
    "links": [
    {"source": "R2", "target": "R1", "weight": 10},
    {"source": "R3", "target": "R1", "weight": 10},
    {"source": "R4", "target": "R1", "weight": 10},
    {"source": "R4", "target": "R3", "weight": 10},
    {"source": "R5", "target": "R1", "weight": 10}
    ]
}

reference: https://graphviz.gitlab.io/_pages/pdf/dotguide.pdf
"""


import sys
import argparse
import json
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import write_dot
from networkx.readwrite import json_graph
import os
import operator


def all_node_ids(topology_json):

    node_ids = []
    
    for i,v in enumerate(topology_json['nodes']):
        for k,v in v.items():
            node_ids.append(v)

    return node_ids


"""
Find all nodes with their weighted closest ABR.
"""
def find_closest_abr(abr_list, graph, topology_json):

    # all_pairs_dijkstra_path() and all_pairs_dijkstra_path_length both return
    # a generator
    all_paths = dict(nx.all_pairs_dijkstra_path(graph))
    all_lengths = dict(nx.all_pairs_dijkstra_path_length(graph))

    if len(all_paths) != len(all_lengths):
        print("Path count is not equal to path length count, "
              "maybe some links are missing a weight?")
        return 1
    
    nodes = {}
    for node in topology_json['nodes']:

        # Skip abr to abr paths
        if node['id'] in abr_list:
            continue

        # Built up the path lengths from the current node back to each abr
        abr_lengths = {}
        for abr in abr_list:

            # If there is a path between this abr and this node
            if abr in all_paths[node['id']]:
                abr_lengths[abr] = all_lengths[node['id']][abr]

            # Only check nodes with a path to an (any) abr
            if len(abr_lengths) > 0:

                # Sorted dict by val, lowest to highest
                abr_lengths_sorted = sorted(abr_lengths.items(),
                                            key=operator.itemgetter(1))

                nodes[node['id']] = {}

                # If this node has a path to multiple abrs
                if len(abr_lengths_sorted) > 1:

                    # Check the multi-abr-paths to check if there are multiple best paths
                    for i in range(0, len(abr_lengths_sorted)-1):
                        if abr_lengths_sorted[i][1] == abr_lengths_sorted[i+1][1]:

                            """
                            if abr_lengths_sorted[i][0] not in nodes[node['id']]:
                                print("{} has the same cost to abr {} and {}".
                                      format(node['id'], abr_lengths_sorted[i][0],
                                             abr_lengths_sorted[i+1][0]))
                            """

                            nodes[node['id']].update({
                                                      abr_lengths_sorted[i][0]:
                                                      abr_lengths_sorted[i][1]
                                                     })
                            nodes[node['id']].update({
                                                      abr_lengths_sorted[i+1][0]:
                                                      abr_lengths_sorted[i+1][1]
                                                     })

                        else:
                            nodes[node['id']].update({
                                                      abr_lengths_sorted[i][0]:
                                                      abr_lengths_sorted[i][1]
                                                     })
                            break

                else:                
                    nodes[node['id']] = {
                                         abr_lengths_sorted[0][0]:
                                         abr_lengths_sorted[0][1]
                                        }


    # Sorted dict by key (node name)
    nodes_sorted = sorted(nodes.items())

    # node is a tuple: ('NODE01', {'ABR01': 60})
    ##for node in nodes_sorted:
    ##    for abr, weight in node[1].items():
    ##        print("{}, ABR: {}, weight: {}".format(node[0], abr, weight))

    return nodes_sorted


"""
Find all ABRs with their weighted closest nodes
"""
def find_closest_abr_node(abr_list, graph, topology_json):

    # all_pairs_dijkstra_path() and all_pairs_dijkstra_path_length both return
    # a generator
    all_paths = dict(nx.all_pairs_dijkstra_path(graph))
    all_lengths = dict(nx.all_pairs_dijkstra_path_length(graph))

    if len(all_paths) != len(all_lengths):
        print("Path count is not equal to path length count, "
              "maybe some links are missing a weight?")
        return 1
    
    abrs = {}
    for abr in abr_list:

        abrs[abr] = {}
        node_lengths = {}

        for node in topology_json['nodes']:

            # Skip abr to abr paths
            if node['id'] in abr_list:
                continue

            if node['id'] in all_paths[abr]:
                node_lengths[node['id']] = all_lengths[abr][node['id']]

        # Skip isolated nodes with no path to an abr
        if len(node_lengths) > 0:
            abrs[abr] = node_lengths

    abr_nodes = {}
    eq = {}
    for abr in abrs:

        for sub_abr in abrs:

            if abr not in abr_nodes:
                abr_nodes[abr] = {}

            if sub_abr not in abr_nodes:
                abr_nodes[sub_abr] = {}

            # Skip abr to abr paths
            if abr != sub_abr:

                for node, distance in abrs[abr].items():

                    if node in abrs[abr] and node in abrs[sub_abr]:

                        if abrs[abr][node] == abrs[sub_abr][node]:
                            if node not in eq:
                                eq[node] = {}
                            if abr not in eq[node]:
                                eq[node].update({abr: abrs[abr][node]})
                            if sub_abr not in eq[node]:
                                eq[node].update({sub_abr: abrs[sub_abr][node]})

                        elif abrs[abr][node] > abrs[sub_abr][node]:
                            abr_nodes[sub_abr].update({node: abrs[sub_abr][node]})
                        elif abrs[abr][node] < abrs[sub_abr][node]:
                            abr_nodes[abr].update({node: abrs[abr][node]})
                    
                    # Nodes with only one parent arb
                    else:
                        if node in abrs[abr]:
                            abr_nodes[abr].update({node: abrs[abr][node]})
                        elif node in abrs[sub_abr]:
                            abr_nodes[sub_abr].update({node: abrs[sub_abr][node]})

    ##for abr, nodes in abr_nodes.items():
    ##    print("Nodes closest to {}:".format(abr))
    ##    print(sorted(nodes.items()))
    ##print("These nodes have equal weight to multiple ABRs:")
    ##print(eq)

    return abr_nodes,eq


"""
Finds the longest weighted path between each ABR and each node with a path
to that ABR. This is the longest non-repeating path (the worst possible
case without breaking connectivity between node and ABR).
"""
def find_longest_paths_per_abr(abr_list, graph, topology_json):

    results = []

    for abr in abr_list:

        for node in topology_json['nodes']:

            all_paths = list(nx.nx.all_simple_paths(graph, source=abr,
                                                    target=node['id']))

            # Skips nodes with no path between them
            if len(all_paths) < 1:
                continue

            maxl = 0
            for i in range(0, len(all_paths)):
                if len(all_paths[i]) > maxl:
                    maxl = i

            weight = 0
            for index in range(0, len(all_paths[maxl])-1):
                for link in topology_json['links']:
                    if (link['source'] == all_paths[maxl][index] and 
                        link['target'] == all_paths[maxl][index+1]):
                        weight += link['weight']
                    elif (link['target'] == all_paths[maxl][index] and 
                        link['source'] == all_paths[maxl][index+1]):
                        weight += link['weight']

            ##print("Longest path from {} to {}, weight: {}, path: {}".
            ##      format(abr, node['id'], weight, all_paths[maxl]))
            results.append([abr, node['id'], weight, all_paths[maxl]])

    return results


"""
Find the node with the longest weighted shortest path to each ABR.
"""
def find_longest_shortest_path_per_abr(abr_list, graph):

    results = []

    # all_pairs_dijkstra_path() and all_pairs_dijkstra_path_length both return
    # a generator
    all_paths = dict(nx.all_pairs_dijkstra_path(graph))
    all_lengths = dict(nx.all_pairs_dijkstra_path_length(graph))

    print("Computed paths for {} node(s), computed weights for {} node(s)".
          format(len(all_paths), len(all_lengths)))

    if len(all_paths) != len(all_lengths):
        print("Path count is not equal to path length count, "
              "maybe some links are missing a weight?")
        return 1

    for abr in abr_list:

        longest_path = {}
        longest_length = {}

        for node, lengths in all_lengths.items():

            # Skip nodes without a path between them
            if abr not in all_lengths[node]:
                continue

            # Skip paths between ABRs
            if node in abr_list:
                continue

            if abr not in longest_length:
                longest_length[abr] = all_lengths[node][abr]
                longest_path[abr] = all_paths[node][abr]
            else:
                if longest_length[abr] < all_lengths[node][abr]:
                    longest_length[abr] = all_lengths[node][abr]
                    longest_path[abr] = all_paths[node][abr]

        ##print("Longest path to {} is from {}, weight {}: path: {}\n".
        ##      format(abr, longest_path[abr][0], longest_length[abr],
        ##      longest_path[abr]))

        results.append([abr, longest_path[abr][0], longest_length[abr],
              longest_path[abr]])

    return results


"""
Find the shortest weighted path between each ABR and each node with a path
to that ABR.
"""
def find_shortest_paths_per_abr(abr_list, graph):

    results = []

    # all_pairs_dijkstra_path() and all_pairs_dijkstra_path_length both return
    # a generator
    all_paths = dict(nx.all_pairs_dijkstra_path(graph))
    all_lengths = dict(nx.all_pairs_dijkstra_path_length(graph))

    print("Computed shortest paths for {} node(s), computed weights for {} "
          "node(s)".format(len(all_paths), len(all_lengths)))
    
    if len(all_paths) != len(all_lengths):
        print("Path count is not equal to path length count, maybe some links "
              "are missing a weight?")
        return 1

    for abr in abr_list:

        abr_paths = all_paths[abr]
        abr_lengths = all_lengths[abr]

        # Sort dict by value, lowest to highest
        abr_lengths_sorted = sorted(abr_lengths.items(),
                                    key=operator.itemgetter(1))
        
        for i in range(0, len(abr_lengths_sorted)):
            node = abr_lengths_sorted[len(abr_lengths_sorted)-i-1]
            # Skip paths between two ABRs
            if node[0] in abr_list:
                continue
            
            ##print("Node: {}, weight: {}, path: {}".format(node[0], node[1],
            ##                                              abr_paths[node[0]]))
            results.append([node[0], node[1], abr_paths[node[0]]])
        
        ##print("\n")

    return results


def find_all_childs(abr, graph, last_node, topology_json):

    all_childs = []

    for node in topology_json['nodes']:
        node = node['id']

        for path in list(nx.nx.all_simple_paths(graph, source=node, target=abr)):

            if ( (last_node in path) and 
                 (path[0] != last_node) and
                 (node not in all_childs) ):
                all_childs.append(node)

    return all_childs[::-1]


"""
Find all strings which are non-resilient paths from a node to an ABR. This
also print sub-sections of a path that are a string, for example:

    R3--R5  R9--R11
    |   |   |   |
R1--R2  R7--R8--R13--R14
    |   |   |   |
    R4--R6  R10-R12

Assuming R1 is an ABR, R13-R14 is a non-resilient sub-string within this path,
R7-R8 is another non-resilient string, R1-R2 is another.
"""
def find_strings(abr_list, graph, topology_json):

    """
    This funcion is an absolute disaster. Far too much is happening inside one
    function. This should be split into multiple functions. As one would expect
    with such an oversized/over complex function, there is a problem with it
    and it's difficult to troubleshoot. I haven't got time to split this into 
    multiple functions and fix the issue properly so I have added a hack, the
    node_history list. When we had layer 2 networks this function worked fine.
    The introduction of layer 3 and thus multiple AGNs/ABRs per area causes 
    the fuction to get stuck in an infinite loop oscilating between paths
    from the same node to each ABRs with the same length/cost.
    """

    strings = []
    starts = []
    ends = []

    # Minimum number of nodes that must be affacted by a string failure
    # for this function to report that failure:
    min_len = 1

    # Loop over every "abr" and for each "node" in the toplogy,
    # get all of it's paths back to the "abr"
    for abr in abr_list:
        for node in topology_json['nodes']:
            node = node['id']

            all_paths = list(nx.nx.all_simple_paths(graph, source=abr, target=node))

            # If this "node" has exactly one path back to "abr" it is a non-
            # resilient string. Check if it the path length is above the 
            # threshold for impact (minimum number of nodes that would go down)
            # if the string lost connectivity
            if ((len(all_paths)) == 1) and len(all_paths[0]) >= min_len:

                if all_paths[0] not in strings:
                    # New single path string
                    strings.append(all_paths[0])
                    starts.append(abr)
                    ends.append(node)

                # We have walked a string upto the minimum threshold length but
                # this might not be the end of the string. Compile a list of
                # nodes that are connected to this node
                last_node = all_paths[0][len(all_paths[0])-1]
                # affected == list of affected nodes
                affected = find_all_childs(abr, graph, last_node, topology_json)
                if len(affected) > 0:
                    sub_list = all_paths[0].copy()
                    for i in range (0, len(affected)):
                        sub_list.append(affected[i])
                    if sub_list not in strings:
                        # Appending new sub-string to list of non-resilient strings
                        strings.append(sub_list.copy())
                        starts.append(abr)
                        ends.append(sub_list[-1]) #ends.append(node)

            # If this node has multiple paths back to an abr
            elif (len(all_paths)) > 1:

                node_history = []

                # Loop over each node in each path and find any node(s) in the
                # path where resiliency occurs
                for path in all_paths:

                    node_next = False
                    node_temp = node

                     # Walk backwards up the path from "node" towards "abr"
                    for hop in path[::-1]:

                        if node_next:
                            node_next = False
                            node_temp = hop

                        #### EXPLAIN THIS HACK!!!
                        if (node,abr) in node_history:
                            continue
                        node_history.append((node,abr))

                        # Check if this "hop" on the path is the hop which
                        # introduces multiple paths towards the "abr"
                        if ( (len(list(nx.all_neighbors(graph, hop))) > 2) and
                             (len(list(nx.nx.all_simple_paths(graph,
                                                              source=abr,
                                                              target=hop))) > 1) ):

                            # It is...

                            """
                            Check if there are multiple next-hops on the
                            multiple paths between this "hop" and the "abr",
                            if there are multiple next hops this node is a
                            point of resiliency in the path between "node"
                            and "abr"
                            """
                            sub_paths = list(nx.nx.all_simple_paths(graph,
                                                                   source=abr,
                                                                   target=hop))
                            first_hops = []
                            for sub_path in sub_paths:
                                if sub_path[len(sub_path)-2] not in first_hops:
                                    first_hops.append(sub_path[len(sub_path)-2])

                            # This "hop" has more than one next hop towards the
                            # "abr"
                            if len(first_hops) > 1:

                                sub_string_shortest_path = nx.dijkstra_path(graph,
                                                                            source=node_temp,
                                                                            target=hop)

                                sub_string_shortest_path = sub_string_shortest_path[::-1]

                                if len(sub_string_shortest_path) >= min_len:

                                    a_sub_paths = list(nx.nx.all_simple_paths(graph,
                                                                              source=abr,
                                                                              target=node_temp))

                                    ##print("Path: : {}\n\n".format(a_sub_paths))

                                    a_first_hops = []
                                    for a_sub_path in a_sub_paths:
                                        if a_sub_path[len(a_sub_path)-2] not in a_first_hops:
                                            a_first_hops.append(a_sub_path[len(a_sub_path)-2])
 
                                    if len(a_first_hops) == 1:

                                        if sub_string_shortest_path not in strings:
                                            # Appending new sub-string to list
                                            # of non-resilient strings
                                            strings.append(sub_string_shortest_path)
                                            starts.append(hop)
                                            ends.append(node)
                                        
                                        #else:
                                            # String already exists in list

                                        node_next = True

                                    #else:
                                        # Source node was resilient

                                else:
                                    """
                                    String length too short.
                                    Start looking for a new string from the
                                    next hop
                                    """
                                    node_next = True

    ##strings = sorted(strings)
    results = []
    for i in range(0, len(strings)-1):
        results.append([starts[i], ends[i], strings[i]])
        ##print("Start of string is {}, end of string is {}, path is {}".
        ##      format(starts[i], ends[i], strings[i]))

    return results


def generate_reports(abr_list, graph, output_dir, topology_json):

    if not os.path.isdir(output_dir):
        print("Report directory doesn't exist: {}".format(output_dir))
        try:
            os.makedirs(output_dir, exist_ok=True)
            print("Created directory: {}".format(output_dir))
        except Exception as e:
            print("Couldn't create directory: {}\n{}".format(output_dir, e))
            return False


    results = find_longest_paths_per_abr(abr_list, graph, topology_json)
    if len(results) > 0:
        output = ("Longest path between each node and ABR "
                  "(worst case routing):\n")
        for result in results:
            output += ("Longest path from "+result[0]+" to "+result[1]+", "
                       "weight: "+str(result[2])+", "
                       "path: "+str(result[3])+"\n")

        ##print(output+"\n")
        save_report_text(output_dir+"/longest_path_to_abr.txt", output)
        save_report_json(output_dir+"/longest_path_to_abr.json", results)


    results = find_longest_shortest_path_per_abr(abr_list, graph)
    if len(results) > 0:
        output = ("Longest shortest path to each ABR (furthest node during "
                  "normal operations):\n")
        for result in results:
            output += ("Longest path to "+result[0]+" is from "+result[1]+", "
                       "weight: "+str(result[2])+", "
                       "path: "+str(result[3])+"\n")
        
        ##print(output+"\n")
        save_report_text(output_dir+"/longest_best_path_to_abr.txt", output)
        save_report_json(output_dir+"/longest_best_path_to_abr.json", results)


    results = find_shortest_paths_per_abr(abr_list, graph)
    if len(results) > 0:
        output = "Shotest path between each ABR and node (normal routing):\n"
        for result in results:
            output += ("Node: "+result[0]+", weight: "+str(result[1])+", "
                       "path: "+str(result[2])+"\n")

        ##print(output+"\n")
        save_report_text(output_dir+"/shortest_path_to_abr.txt", output)
        save_report_json(output_dir+"/shortest_path_to_abr.json", results)


    results = find_closest_abr(abr_list, graph, topology_json)
    if len(results) > 0:
        output = "Closest ABR to each node:\n"
        # result is a tuple: ('NODE01', {'ABR01': 60})
        for result in results:
            for abr, weight in result[1].items():
                output += ("Node: "+result[0]+", ABR: "+abr+", "
                           "Weight: "+str(weight)+"\n")
    
        ##print(output+"\n")
        save_report_text(output_dir+"/closest_abr_to_node.txt", output)
        save_report_json(output_dir+"/closest_abr_to_node.json", results)


    abr_results,eq_results = find_closest_abr_node(abr_list, graph, topology_json)
    output = "Closest node to each ABR:\n"
    if len(abr_results) > 0:
        for abr, nodes in abr_results.items():
            output += "Nodes closest to "+abr+":\n"
            output += str(sorted(nodes.items()))+"\n"

    if len(eq_results) > 0:
        output += "These nodes have equal weight to multiple ABRs:\n"
        output += str(eq_results)

    if (len(abr_results) > 0) or (len(eq_results) > 0):
        ##print(output+"\n")
        save_report_text(output_dir+"/closest_node_to_abr.txt", output)
        save_report_json(output_dir+"/closest_node_to_abr.json", results)


    results = find_strings(abr_list, graph, topology_json)
    if len(results) > 0:
        output = "Non-resilient strings from each node to each ABR:\n"
        for i in range(0, len(results)-1):
            output += ("Start of string is "+results[i][0]+", "
                       "end of string is "+results[i][1]+", "
                       "path is "+str(results[i][2])+"\n")

        ##print(output+"\n")
        save_report_text(output_dir+"/non_resilient_strings.txt", output)
        save_report_json(output_dir+"/non_resilient_strings.json", results)


def generate_topology_diagram(filename, graph):

    cmd = "dot -Gsize=100,100 -Gdpi=100 -Gnodesep=equally -Granksep=equally "
    cmd += "-Nshape=circle -Nstyle=filled -Nwidth=2 -Nheight=2 "
    cmd += "-Tpng -o"+filename+" topology.dot"
    write_dot(graph,'topology.dot')
    os.system(cmd)  ### No error checking!

    #plt.figure(1, figsize=(15,15)) 
    #plt.figure(num=None, figsize=(100, 100), dpi=80)
    #plt.set_size_inches(18.5, 10.5)
    #plt.axis('off')
    #plt.title(topology_json['area'])
    #plt.savefig(args['diagram_file'])

    return True


def load_positions(filename):

    if os.path.isfile(filename):

        try:
            positions_file = open(filename, 'r')
        except Exception as e:
            print("Couldn't open input positional file {}: {}".
                  format(filename, e))
            return False

        try:
            positions_json = json.load(positions_file)
        except Exception as e:
            print("Couldn't load input positional file JSON: {}".format(e))
            return False

        positions_file.close()

        return positions_json

    else:
        return None


def load_topology(filename):

    try:
        topology_file = open(filename, 'r')
    except Exception:
        print("Couldn't open inventory file {}".format(filename))
        return False

    try:
        topology_json = json.load(topology_file)
    except Exception as e:
        print("Couldn't load input topology JSON file: {}".format(e))
        return False

    topology_file.close()

    return topology_json


def parse_cli_args():

    parser = argparse.ArgumentParser(
        description='Read topology data from a JSON file and produce a '
                    'topology diagram and routing reports.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '-d', '--diagram-file',
        help='PNG filename to save the topology diagram as.',
        type=str,
        default='topology.png',
    )
    parser.add_argument(
        '-o', '--output-dir',
        help='Output directory for report files.',
        type=str,
        default='./reports',
    )
    parser.add_argument(
        '-p', '--positions-file',
        help='Optional JSON formatted positional data to keep graphs topology '
             'consistent after first run.',
        type=str,
        default='positions.json',
    )
    parser.add_argument(
        '-t', '--topology-file',
        help='JSON topology data file.',
        type=str,
        default='topology.json',
    )

    return vars(parser.parse_args())


def save_positions(filename, positions_data):

    try:
        positions_file = open(filename, 'w')
    except Exception as e:
        print("Couldn't open output positions file {}: {}".format(filename, e))
        return False

    try:
        json.dump(positions_data, positions_file, ensure_ascii=False,
                 sort_keys=True, indent=4)
    except Exception as e:
        print("Couldn't write to positional JSON file: {}".format(e))
        return False

    positions_file.close()

    return True


def save_report_json(filename, report_data):

    try:
        report_file = open(filename, 'w')
    except Exception as e:
        print("Couldn't open output JSON file {}: {}".format(filename, e))
        return False

    try:
        json.dump(report_data, report_file, ensure_ascii=False,
                 sort_keys=True, indent=4)
    except Exception as e:
        print("Couldn't write to report JSON fil {}: {}".format(filename, e))
        return False

    report_file.close()

    return True 


def save_report_text(filename, report_data):

    try:
        report_file = open(filename, "w")
    except Exception as e:
        print("Couldn't open output text file {}: {}".format(filename, e))
        return False

    try:
        report_file.write(report_data)
    except Exception as e:
        print("Couldn't write output test file {}: {}".format(filename, e))
        return False

    report_file.close()

    return True


def topology_has_changed(topology_json, positions_json):

    ret = 0

    for i,v in enumerate(topology_json['nodes']):
        for k,v in v.items():
            if v not in positions_json:
                print("Missing position for {}".format(v))
                ret = 1

    return ret


def main():

    args = parse_cli_args()

    topology_json = load_topology(args['topology_file'])
    if not topology_json:
        sys.exit(1)

    positions_json = load_positions(args['positions_file'])
    if positions_json == False:
        sys.exit(1)


    abr_list = []
    ### EXPLAIN THIS HACK YOU FIEND!
    for node in topology_json['nodes']:
        if "-GW" in node['id']:
            node['color'] = "lightblue"
            abr_list.append(node['id'])
        elif "-AGN" in node['id']:
            node['color'] = "lightblue"
            abr_list.append(node['id'])
        elif "agn0" in node['id']:
            node['color'] = "lightblue"
            abr_list.append(node['id'])

    # Create the graph from the loaded JSON data:
    graph = json_graph.node_link_graph(topology_json)
    print("Graph details:\n{}\n".format(nx.info(graph)))

    ### EXPLAIN THIS HACK YOU FIEND!
    for node in topology_json['nodes']:
        if "-GW" in node['id']:
            node.pop('color')
        elif "-AGN" in node['id']:
            node.pop('color')
        elif "agn0" in node['id']:
            node.pop('color')


    # As well as providing the coordinates, a list must be built of every node
    # id, nodes in that list will be fixed to the coords supplied: 
    fixed = all_node_ids(topology_json)

    # If a previous positional data set has been loaded, use it:
    if positions_json is not False:
        positions_data = nx.spring_layout(graph, pos=positions_json, fixed=fixed, k=5, iterations=2)

    # Else generate a random graph layout:
    else:
        positions_data = nx.spring_layout(graph, k=0.9, iterations=20)

    # spring_layout() returns a dict with the graph node as the key and
    # the value a two-tuple coords array, converts the array to a list:
    for k, v in positions_data.items():
        positions_data[k] = v.tolist()

    # Save the positional data for future runs
    if not save_positions(args['positions_file'], positions_data):
        sys.exit(1)

    nx.draw_networkx(graph, pos=positions_data, with_labels=True, node_size=300, font_size=10)

    # graph.edges(data=True) # Returns graph edges with additional data, e.g. 'weigth' key
    edge_labels=dict([((u,v,),d['weight']) for u,v,d in graph.edges(data=True)])
    # Argument edge_labels (dictionary) = Edge labels in a dictionary,
    # keyed by edge two-tuple with the of text labels as the value.
    nx.draw_networkx_edge_labels(graph, positions_data, edge_labels=edge_labels)

    generate_topology_diagram(args['diagram_file'], graph)
    
    if len(abr_list) > 0:
        generate_reports(abr_list, graph, args['output_dir'], topology_json)
    else:
        print("No gateway nodes found!")


if __name__ == '__main__':
    sys.exit(main())
