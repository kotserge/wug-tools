import pickle
import os
import math
import random
import csv

import networkx as nx
import pymc3 as pm

import numpy as np
from numpy.random import binomial, geometric, poisson

from collections import Counter

import graph_tool
from graph_tool.inference.blockmodel import BlockState
from graph_tool.inference import minimize_blockmodel_dl
from graph_tool.draw import graph_draw


def gen_fitted_graphs(path_in: str, path_out: str, seed: int, acc_funct=np.nanmedian, normalization=lambda x, y, z: x) -> None:
    """ Generates fitted graphs, and saves them in the output
    For each graph, 3 objects are saved.
    The fitted graph, the BlockState and the best distribution with its parameters.
    It also creates a csv in the top directory, containing some information about all graphs

    Parameters:
        :param path_in: Path to graphs
        :param path_out: Path to save the output (Subdirectories will be created per graph)
        :param seed: Seed to use for RNG
        :param acc_funct: accumalation function for weights
        :param normalization: normalization function
    """
    graphs_to_fitt = []
    path_out_fitted = []
    for _, _, files in os.walk(path_in):
        for file in files:
            graphs_to_fitt.append(nx.read_gpickle('{}/{}'.format(path_in, file)))
            path_out_fitted.append('{}/{}'.format(path_out, file))

    for ii, graph in enumerate(graphs_to_fitt):
        print('Calculating: ', path_out_fitted[ii])

        try:
            os.makedirs(path_out_fitted[ii])
        except FileExistsError:
            print('File or Directory for this graph exists, will not calculate/write results')
            print('==========================================')
            print('Done with: {}%'.format((ii + 1) / len(graphs_to_fitt) * 100))
            continue

        _, graph_tool_graph, _ = _generate_graphtool_graph(graph, seed=seed, acc_funct=acc_funct, normalization=normalization)
        distribution, state, history = _find_best_distribution(graph_tool_graph)

        history = [path_out_fitted[ii].split('/')[-1], *history]

        assert state is not None
        infer_param_dict = _infer_param_dict(graph_tool_graph, state, distribution)

        _write_result(path_out_fitted[ii], graph_tool_graph, state, infer_param_dict)

        print('==========================================')
        print('Done with: {}%'.format((ii + 1) / len(graphs_to_fitt) * 100))

        if ii == 0:
            full_history = ['Name', 'discrete-binomial', 'discrete-geometric', 'discrete-poisson', 'Max']
            _write_history(path_out, full_history, False)

        _write_history(path_out, history, True)


def _write_result(path: str, graph: graph_tool.Graph, state: BlockState, infer_param_dict: dict):
    file_prefix = path.split('/')[-1]

    with open('{}/{}.graph'.format(path, file_prefix), 'wb') as file:
        pickle.dump(graph, file)
    file.close()

    with open('{}/{}.state'.format(path, file_prefix), 'wb') as file:
        pickle.dump(state, file)
    file.close()

    with open('{}/{}.distribution'.format(path, file_prefix), 'wb') as file:
        pickle.dump(infer_param_dict, file)
    file.close()

    draw_graph(graph, state, '{}/{}.png'.format(path, file_prefix))


def _write_history(path: str, history: list, append_mode: bool = False):
    mode = 'w+' if not append_mode else 'a+'
    with open('{}/history.csv'.format(path), mode, newline='') as file:
        writer = csv.writer(file)
        writer.writerows([history])


def _minimize(graph: graph_tool.Graph, distribution="discrete-binomial", deg_corr=False) -> BlockState:
    return minimize_blockmodel_dl(graph,
                                  state_args=dict(deg_corr=deg_corr, recs=[graph.ep.weight], rec_types=[distribution]),
                                  multilevel_mcmc_args=dict(B_min=1, B_max=30, niter=100, entropy_args=dict(adjacency=False, degree_dl=False)))


def _find_best_distribution(graph: graph_tool.Graph):
    distributions = ["discrete-binomial", "discrete-geometric", "discrete-poisson"]

    best_distribution = ''
    best_state = None
    best_entropy = np.inf
    history = []
    for distribution in distributions:
        state: BlockState = _minimize(graph, distribution=distribution, deg_corr=False)
        entropy = state.entropy(adjacency=False, degree_dl=False)
        if entropy < best_entropy:
            best_entropy = entropy
            best_distribution = distribution
            best_state = state
        history.append(entropy)

    history.append(best_distribution)

    return best_distribution, best_state, history


def _infer_param_dict(graph: graph_tool.Graph, state: BlockState, distribution="discrete-binomial"):
    nodes = state.get_N()
    edges = state.get_E()
    num_communities = state.get_nonempty_B()
    nodes_per_community = sorted([v for v in state.get_nr().get_array() if v > 0], reverse=True)
    pdf = _pdf_dict(num_communities, distribution)

    return dict(nodes=nodes, edges=edges, communities=num_communities, community_size=nodes_per_community, pdf=pdf, param=_calculate_sbm_parameter_matrix(graph, state, distribution))


def _pdf_dict(size, distribution='discrete-binomial'):
    if distribution == 'discrete-binomial':
        pdf = [[binomial] * size] * size
    elif distribution == 'discrete-geometric':
        pdf = [[geometric] * size] * size
    elif distribution == 'discrete-poisson':
        pdf = [[poisson] * size] * size
    return pdf


def _calculate_sbm_parameter_matrix(graph: graph_tool.Graph, state: BlockState, distribution="discrete-binomial"):
    b = graph_tool.perfect_prop_hash([state.get_blocks()])[0]
    edges = graph.get_edges([graph.ep.weight])
    vertices = graph.get_vertices()

    block_dict = dict(sorted(Counter(b.get_array()).items(), key=lambda x: x[1], reverse=True))

    pdf = []
    edges_analyzed = 0
    for ii, (block_one, _) in enumerate(block_dict.items()):
        row = []
        for jj, (block_two, _) in enumerate(block_dict.items()):
            if jj < ii:
                row.append(pdf[jj][ii].copy())
                continue

            # ik, this is dirty, but it works fine
            edges_weights = [item[2] - 1 for item in edges if b[vertices[int(item[0])]] == block_one and b[vertices[int(item[1])]] == block_two or b[vertices[int(item[0])]] == block_two and b[vertices[int(item[1])]] == block_one]
            edges_analyzed += len(edges_weights)

            row.append(_pdf_dict_entry(_infer_distribution_parameters_from_weights(edges_weights, distribution), distribution))

        pdf.append(row)
    assert edges_analyzed == len(edges)
    return pdf


def _infer_distribution_parameters_from_weights(weights: list, distribution="discrete-binomial"):
    alpha = 0.1
    beta = 0.1

    with pm.Model() as model:  # context management
        if distribution == "discrete-binomial":
            p = pm.Beta('p', alpha=alpha, beta=beta)
            y = pm.Binomial('y', n=3, p=p, observed=weights)
        elif distribution == "discrete-geometric":
            p = pm.Beta('p', alpha=alpha, beta=beta)
            y = pm.Geometric('y', p=p, observed=weights)
        elif distribution == "discrete-poisson":
            p = pm.Gamma('p', alpha=alpha, beta=beta)
            y = pm.Poisson('y', mu=p, observed=weights)

        trace = pm.sample(2000, tune=1000, cores=4, return_inferencedata=False)

        return trace['p'].mean()


def _pdf_dict_entry(p, distribution):
    if distribution == 'discrete-binomial':
        return dict(n=3, p=p)
    elif distribution == 'discrete-geometric':
        return dict(p=p)
    elif distribution == 'discrete-poisson':
        return dict(lam=p)


def _generate_graphtool_graph(graph: nx.Graph, seed: int, acc_funct=np.nanmedian, normalization=lambda x, y, z: x):
    # Possibly not needed anymore, due to weights already beeing correct in the data-set
    graph = _update_weights(graph, attributes='judgments', seed=seed, acc_funct=acc_funct, normalization=normalization)

    graph_positive_edges = graph.copy()
    edges_negative = [(i, j) for (i, j) in graph_positive_edges.edges() if graph_positive_edges[i][j]['weight'] < 2.5]
    graph_positive_edges.remove_edges_from(edges_negative)
    position_information = nx.nx_agraph.graphviz_layout(graph_positive_edges, prog='sfdp')

    graph_tool_graph = _sem_eval_2_graph_tool(graph, position_information)

    return graph, graph_tool_graph, position_information


def _update_weights(graph=nx.Graph, attributes='judgments', acc_funct=np.nanmedian, non_value=0.0, normalization=lambda x, y, z: x, seed: int = 1234) -> nx.Graph:
    """
    Update edge weights from annotation attributes.
    :param G: graph
    :param attributes: list of attributes to be summarized
    :param acc_funct: accumalation function for weights
    :param non_value: value of non-judgment
    :param normalization: normalization function
    :return G: updated graph
    """
    rng = np.random.default_rng(seed)

    for (i, j) in graph.edges():
        try:
            values = [v[0] for k, v in graph[i][j][attributes].items()]
        except KeyError:
            raise KeyError('Given attribute does not exist')

        assert len(values) > 0

        data = [v if v != non_value else np.nan for v in values]  # exclude non-values

        weight = normalization(acc_funct(data), data, rng)

        graph[i][j]['weight'] = weight
    return graph


def _sem_eval_2_graph_tool(graph: nx.Graph, position_dict: dict) -> graph_tool.Graph:
    graph_tool_graph = graph_tool.Graph(directed=False)

    vertex_id = dict()
    for i, node in enumerate(graph.nodes()):
        vertex_id[node] = i

    new_weights = []
    for i, j in graph.edges():
        current_weight = graph[i][j]['weight']
        if current_weight != 0 and not np.isnan(current_weight):
            graph_tool_graph.add_edge(vertex_id[i], vertex_id[j])
            new_weights.append(current_weight)

    original_edge_weights = graph_tool_graph.new_edge_property("double")
    original_edge_weights.a = new_weights
    graph_tool_graph.ep['weight'] = original_edge_weights

    new_vertex_id = graph_tool_graph.new_vertex_property("string")
    for k, v in vertex_id.items():
        new_vertex_id[v] = k
    graph_tool_graph.vp.id = new_vertex_id

    vertex_position = graph_tool_graph.new_vertex_property("vector<double>")
    for k, v in position_dict.items():
        vertex = [vertex for vertex in graph_tool_graph.get_vertices() if graph_tool_graph.vp.id[graph_tool_graph.vertex(vertex)] == k][0]
        vertex_position[graph_tool_graph.vertex(vertex)] = v
    graph_tool_graph.vp.pos = vertex_position

    return graph_tool_graph


def _norm_weight_to_int(weight: float, weights: list, rng) -> int:
    if np.isnan(weight) or int(weight) == weight:
        return weight

    mean_weight = np.nanmean(weights)

    if mean_weight < weight:
        return weight - 0.5
    elif mean_weight > weight:
        return weight + 0.5
    return weight + (rng.integers(0, 2) - 0.5)


def draw_graph(graph: graph_tool.Graph, state: BlockState, title: str):
    b = state.get_blocks()
    b = graph_tool.perfect_prop_hash([b])[0]

    gray = [0.5, 0.5, 0.5, 1.0]
    black = [0.1, 0.1, 0.1, 1.0]
    ecolor = graph.new_edge_property("vector<double>")
    epen = graph.new_edge_property("double")

    for e in graph.edges():
        if graph.ep.weight[e] == 4:
            ecolor[e] = black
            epen[e] = 2
        elif graph.ep.weight[e] == 3:
            ecolor[e] = black
            epen[e] = 0.9
        else:
            ecolor[e] = gray
            epen[e] = 0.5

    # Maybe use graphviz
    graph_draw(graph, pos=graph.vp.pos, vertex_size=12, vertex_fill_color=b, edge_color=ecolor, edge_pen_width=epen,
               fit_view=True, adjust_aspect=False, ink_scale=0.9, output_size=(640, 480), output=title, overlap=True)


if __name__ == '__main__':
    gen_fitted_graphs('data/wugs/dwug_de/graphs/full/', 'data/wugs/full_int_fit/dwug_de', 777, normalization=_norm_weight_to_int)
