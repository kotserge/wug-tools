from numpy import rot90
import pandas
import pickle
import seaborn
import matplotlib.pyplot as plt

dfs = []
for s in ['', '1', '2', '3']:
    path_de = 'data/wugs/fitted{}/dwug_de/history.csv'.format(s)
    path_en = 'data/wugs/fitted{}/dwug_en/history.csv'.format(s)

    df_de = pandas.read_csv(path_de)
    df_en = pandas.read_csv(path_en)

    dfs.append(pandas.concat([df_de, df_en]))

# acctual information, (dist used,<class 'graph_tool.inference.blockmodel.BlockState'>)
with open('data/best_fit/g_dist_states', 'rb') as file:
    g_dist_states: dict = pickle.load(file)
file.close()

count = {'discrete-binomial': 0, 'discrete-geometric': 0, 'discrete-poisson': 0}
for k, v in g_dist_states.items():
    count[v[0]] += 1

fig, axs = plt.subplots(ncols=5)
df_e = pandas.DataFrame(list(count.items()), columns=['Distribution', 'Count'])
p = seaborn.barplot(data=df_e, x='Distribution', y='Count', ax=axs[0])

p.set(title='Old fitted data')
for item in p.get_xticklabels():
    item.set_rotation(90)

for i in range(len(dfs)):
    p = seaborn.countplot(data=dfs[i], x='Max', ax=axs[i + 1])
    p.set(title='New, Round {}'.format(i + 1))
    for item in p.get_xticklabels():
        item.set_rotation(90)
plt.show()
