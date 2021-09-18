import os
import pickle
import graph_tool
from graph_tool.inference.blockmodel import BlockState


def all_fitted(path: str):
    for _, dirs, _ in os.walk(path):
        for i, dir in enumerate(dirs):
            print('======================')
            print('Working on: ', dir)
            fitted_wug2simulation_param(dir, '{}/{}'.format(path, dir))
            print('======================')
            print('Done with {}%'.format((i + 1) / len(dirs) * 100))


def fitted_wug2simulation_param(name: str, path: str):
    _, state, distribution = _load_data(name, path)

    nodes = state.get_N()
    edges = state.get_E()
    num_communities = state.get_nonempty_B()
    nodes_per_community = sorted([v for v in state.get_nr().get_array() if v > 0], reverse=True)

    sim_param_dict = dict(nodes=nodes, edges=edges, communities=num_communities, community_size=nodes_per_community)

    for k, v in distribution.items():
        if v == 'discrete-binomial':
            sim_param_dict['n'] = 4
        sim_param_dict[k] = v

    with open('{}/{}_param.dict'.format(path, name), 'wb') as file:
       pickle.dump(sim_param_dict, file)
    file.close()


def _load_data(name: str, path: str):
    with open('{}/{}.state'.format(path, name), 'rb') as file:
        state: BlockState = pickle.load(file)
    file.close()

    with open('{}/{}.distribution'.format(path, name), 'rb') as file:
        distribution: dict = pickle.load(file)
    file.close()

    with open('{}/{}.graph'.format(path, name), 'rb') as file:
        graph: graph_tool.Graph = pickle.load(file)
    file.close()

    return graph, state, distribution


if __name__ == '__main__':
    all_fitted('data/wugs/fitted3/dwug_de')
    # fitted_wug2simulation_param('Truppenteil', 'data/wugs/fitted3/dwug_de/Truppenteil')
