import random
import networkx as nx

from clustering_interface import wsbm_clustering
from clustering_interface import louvain_clustering
from clustering_interface import chinese_whispers_clustering

# generate random graph
G = nx.gnm_random_graph(5, 5)
for (u, v) in G.edges():
    G.edges[u, v]['weight'] = random.randint(0, 10)

cw_cluster = chinese_whispers_clustering(G)
print(cw_cluster)

lm_cluster = louvain_clustering(G)
print(lm_cluster)

wsbm_clusters = wsbm_clustering(G)
print(wsbm_clusters)
