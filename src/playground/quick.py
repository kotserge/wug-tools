import os
import numpy as np
from graph_tool.inference.blockmodel import BlockState
import pickle


with open('data/best_fit/g_dist_states', 'rb') as file:
    g_dist_states: dict = pickle.load(file)
file.close()

path = 'data/wugs/new_fitted/dwug_de/'
count = 0
false_count = 0
mean_diff = []
for _, dirs, _ in os.walk(path):
    for i, dir in enumerate(dirs):
        with open('{0}/{1}/{1}.state'.format(path, dir), 'rb') as file:
            state: BlockState = pickle.load(file)
            dir = '-'.join(dir.split('_'))
            count += 1
            v = g_dist_states.get(dir, None)
            if v is not None and v[1].get_B() != state.get_nonempty_B():
                print(dir, v[1].get_B(), state.get_nonempty_B(), v[1].get_B() - state.get_nonempty_B())
                mean_diff.append(v[1].get_B() - state.get_nonempty_B())
                false_count += 1

print('Median diff: ', np.median(mean_diff), 'Mean diff: ', np.nanmean(mean_diff))
print('False: {}%'.format(false_count / count * 100))
