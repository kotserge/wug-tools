from numpy import rot90
import pandas
import pickle
import seaborn
import matplotlib.pyplot as plt

path_de = 'data/wugs/new_fitted/dwug_de/history.csv'
path_en = 'data/wugs/new_fitted/dwug_en/history.csv'

df_de = pandas.read_csv(path_de)
df_en = pandas.read_csv(path_en)

df = pandas.concat([df_de, df_en])

path_de_bad = 'data/wugs/bad_new_fitted/dwug_de/history.csv'
path_en_bad = 'data/wugs/bad_new_fitted/dwug_en/history.csv'

df_de = pandas.read_csv(path_de_bad)
df_en = pandas.read_csv(path_en_bad)

df_bad = pandas.concat([df_de, df_en])

path_de_float = 'data/wugs/float_new_fitted/dwug_de/history.csv'
path_en_float = 'data/wugs/float_new_fitted/dwug_en/history.csv'

df_de = pandas.read_csv(path_de_float)
df_en = pandas.read_csv(path_en_float)

df_float = pandas.concat([df_de, df_en])

# acctual information, (dist used,<class 'graph_tool.inference.blockmodel.BlockState'>)
with open('data/best_fit/g_dist_states', 'rb') as file:
    g_dist_states: dict = pickle.load(file)
file.close()

count = {'discrete-binomial': 0, 'discrete-geometric': 0, 'discrete-poisson': 0}
for k, v in g_dist_states.items():
    count[v[0]] += 1

fig, axs = plt.subplots(ncols=4)
df_e = pandas.DataFrame(list(count.items()), columns=['Distribution', 'Count'])
p = seaborn.barplot(data=df_e, x='Distribution', y='Count', ax=axs[0])

p.set(title='Old fitted data', ylim=[0, 100])
for item in p.get_xticklabels():
    item.set_rotation(90)

p = seaborn.countplot(data=df, x='Max', ax=axs[1])
p.set(title='New Fitted', ylim=[0, 100])
for item in p.get_xticklabels():
    item.set_rotation(90)

p = seaborn.countplot(data=df_bad, x='Max', ax=axs[2])
p.set(title='New Bad Fitted', ylim=[0, 100])
for item in p.get_xticklabels():
    item.set_rotation(90)

p = seaborn.countplot(data=df_float, x='Max', ax=axs[3])
p.set(title='New Float Fitted', ylim=[0, 100])
for item in p.get_xticklabels():
    item.set_rotation(90)

plt.show()
