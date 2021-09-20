import pickle
import os
import math
import random
import csv

import networkx as nx
import numpy as np
import pymc3 as pm

import graph_tool
from graph_tool.inference.blockmodel import BlockState
from graph_tool.inference import minimize_blockmodel_dl
from graph_tool.draw import graph_draw


def gen_fitted_graphs(path_in: str, path_out: str) -> None:
    """ Generates fitted graphs, and saves them in the output
    For each graph, 3 objects are saved.
    The fitted graph, the BlockState and the best distribution with its parameters.
    It also creates a csv in the top directory, containing some information about all graphs

    Parameters:
        :param path_in: Path to graphs
        :param path_out: Path to save the output (Subdirectories will be created per graph)
    """
    graphs_to_fitt = []
    path_out_fitted = []
    for _, _, files in os.walk(path_in):
        for file in files:
            graphs_to_fitt.append(nx.read_gpickle('{}/{}'.format(path_in, file)))
            path_out_fitted.append('{}/{}'.format(path_out, file))

    full_history = [['Name', 'discrete-binomial', 'discrete-geometric', 'discrete-poisson', 'Max']]
    for ii, graph in enumerate(graphs_to_fitt):
        print('Calculating: ', path_out_fitted[ii])
        _, graph_tool_graph, _ = _generate_graphtool_graph(graph)
        distribution, state, history = _find_best_distribution(graph_tool_graph)

        history = [path_out_fitted[ii].split('/')[-1], *history]

        assert state is not None
        infered_param_inside, infered_param_outside = _infer_distribution_parameters_from_graph(graph_tool_graph, state, distribution)

        try:
            os.makedirs(path_out_fitted[ii])
            _write_result(path_out_fitted[ii], graph_tool_graph, state, distribution, infered_param_inside, infered_param_outside)
            full_history.append(history)
        except FileExistsError:
            print('File or Directory for this graph exists, will not write results')

        print('==========================================')
        print('Done with: {}%'.format((ii + 1) / len(graphs_to_fitt) * 100))

    _write_history(path_out, full_history)


def _write_result(path: str, graph: graph_tool.Graph, state: BlockState, distribution: str, infered_param_inside: float, infered_param_outside: float):
    file_prefix = path.split('/')[-1]

    with open('{}/{}.graph'.format(path, file_prefix), 'wb') as file:
        pickle.dump(graph, file)
    file.close()

    with open('{}/{}.state'.format(path, file_prefix), 'wb') as file:
        pickle.dump(state, file)
    file.close()

    with open('{}/{}.distribution'.format(path, file_prefix), 'wb') as file:
        pickle.dump({'distribution': distribution, 'infered_param_inside': infered_param_inside, 'infered_param_outside': infered_param_outside}, file)
    file.close()

    draw_graph(graph, state, '{}/{}.png'.format(path, file_prefix))


def _write_history(path: str, history: list):
    with open('{}/history.csv'.format(path), 'w+', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(history)


def _minimize(graph: graph_tool.Graph, distribution="discrete-binomial", deg_corr=False) -> BlockState:
    return minimize_blockmodel_dl(graph,
                                  state_args=dict(deg_corr=deg_corr, recs=[graph.ep.original_weight], rec_types=[distribution]),
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


def _infer_distribution_parameters_from_graph(graph: graph_tool.Graph, state: BlockState, distribution="discrete-binomial"):
    inside_weights, outside_weights = _calculate_weigts_of_graph(graph, state)
    return _infer_distribution_parameters_from_weights(inside_weights, distribution), _infer_distribution_parameters_from_weights(outside_weights, distribution)


def _calculate_weigts_of_graph(graph: graph_tool.Graph, state: BlockState):
    b = graph_tool.perfect_prop_hash([state.get_blocks()])[0]
    edges = graph.get_edges([graph.ep.original_weight])
    vertices = graph.get_vertices()

    outside_edges = [item for item in edges if b[vertices[int(item[0])]] != b[vertices[int(item[1])]]]
    outside_weights = [item[2] - 1 for item in outside_edges]

    inside_edges = [item for item in edges if b[vertices[int(item[0])]] == b[vertices[int(item[1])]]]
    inside_weights = [item[2] - 1 for item in inside_edges]
    return inside_weights, outside_weights


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

        # print(pm.summary(trace).to_string()), pmcy3 does not have a summary function
        # pm.traceplot(trace)

        return trace['p'].mean()


def _generate_graphtool_graph(graph: nx.Graph):
    # Possibly not needed anymore, due to weights already beeing correct in the data-set
    graph = _update_weights(graph, attributes='judgments')

    graph_positive_edges = graph.copy()
    edges_negative = [(i, j) for (i, j) in graph_positive_edges.edges() if graph_positive_edges[i][j]['weight'] < 2.5]
    graph_positive_edges.remove_edges_from(edges_negative)

    position_information = nx.nx_agraph.graphviz_layout(graph_positive_edges, prog='sfdp')

    graph_tool_graph = _sem_eval_2_graph_tool(graph, position_information)

    return graph, graph_tool_graph, position_information


def _update_weights(graph=nx.Graph, attributes='judgments', test_statistic=np.median, non_value=0.0, normalization=lambda x: x) -> nx.Graph:
    """
    Update edge weights from annotation attributes.
    :param G: graph
    :param attributes: list of attributes to be summarized
    :param test_statistic: test statistic to summarize data
    :param non_value: value of non-judgment
    :param normalization: normalization function
    :return G: updated graph
    """
    for (i, j) in graph.edges():
        try:
            values = [v[0] for k, v in graph[i][j][attributes].items()]
        except KeyError:
            raise KeyError('Given attribute does not exist')

        assert len(values) > 0

        data = [v for v in values if not v == non_value]  # exclude non-values

        if data != []:
            weight = normalization(test_statistic(data))
        else:
            weight = float('nan')

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
    graph_tool_graph.ep['original_weight'] = original_edge_weights

    shifted_weight = graph_tool_graph.new_edge_property("int")
    shifted_weight.a = list(map(lambda x: int(x * 2 - 2), original_edge_weights.a))
    graph_tool_graph.ep['shifted_weight'] = shifted_weight

    edge_weight = graph_tool_graph.new_edge_property("double")
    edge_weight.a = list(map(lambda x: x - 2.5, original_edge_weights.a))
    graph_tool_graph.ep['weight'] = edge_weight

    new_vertex_id = graph_tool_graph.new_vertex_property("string")
    graph_tool_graph.vp.id = new_vertex_id

    for k, v in vertex_id.items():
        new_vertex_id[v] = k

    vertex_position = graph_tool_graph.new_vertex_property("vector<double>")
    graph_tool_graph.vp.pos = vertex_position

    for k, v in position_dict.items():
        vertex = [vertex for vertex in graph_tool_graph.get_vertices() if graph_tool_graph.vp.id[graph_tool_graph.vertex(vertex)] == k][0]
        vertex_position[graph_tool_graph.vertex(vertex)] = v

    return graph_tool_graph


def draw_graph(graph: graph_tool.Graph, state: BlockState, title: str):
    b = state.get_blocks()
    b = graph_tool.perfect_prop_hash([b])[0]

    gray = [0.5, 0.5, 0.5, 1.0]
    black = [0.1, 0.1, 0.1, 1.0]
    ecolor = graph.new_edge_property("vector<double>")
    epen = graph.new_edge_property("double")

    for e in graph.edges():
        if graph.ep.original_weight[e] == 4:
            ecolor[e] = black
            epen[e] = 2
        elif graph.ep.original_weight[e] == 3:
            ecolor[e] = black
            epen[e] = 0.9
        else:
            ecolor[e] = gray
            epen[e] = 0.5

    # Maybe use graphviz
    graph_draw(graph, pos=graph.vp.pos, vertex_size=12, vertex_fill_color=b, edge_color=ecolor, edge_pen_width=epen,
               fit_view=True, adjust_aspect=False, ink_scale=0.9, output_size=(640, 480), output=title, overlap=True)


if __name__ == '__main__':
    gen_fitted_graphs('data/wugs/dwug_en/graphs/semeval/', 'data/wugs/float_n3_fit/dwug_en')
