import graph_tool.all as gt
from graph_tool.inference.blockmodel import BlockState
from graph_tool import VertexPropertyMap
import pickle

# g_acc are just the accuracy scores, thus not interesting
with open('data/best_fit/g_accuracies', 'rb') as file:
    g_acc = pickle.load(file)
file.close()

print(g_acc)

# acctual information, (dist used,<class 'graph_tool.inference.blockmodel.BlockState'>)
with open('data/best_fit/g_dist_states', 'rb') as file:
    g_dist_states: dict = pickle.load(file)
file.close()

# for k, v in g_dist_states.items():
#   print('Key ', k, ' Distribution: ', v[0])
# exit()

for k, v in g_dist_states.items():
    print('============')
    print('Key Value: ', k)
    print('Distribution: ', v[0])
    state: BlockState = v[1]
    print('State: ', state)
    print('Nodes: ', state.get_N())
    print('Edges: ', state.get_E())
    print('Blocks: ', state.get_B())
    # pos, _ = state.draw()
    # state.draw(output='old/{}.png'.format(k))
    graph: gt.Graph = state.g
    print('Edge Properties: ', graph.list_properties())
    print('Parameters Edges', state.get_rec_params())
    print('============')
