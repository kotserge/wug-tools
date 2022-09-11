import random
import networkx as nx

from clustering.clustering_interface import chinese_whispers_clustering
from manipulation.graph_manipulation import remove_edges

graph = nx.readwrite.gpickle.read_gpickle('data/test/graph/full/Rezeption')
graph = remove_edges(graph)

cluster = chinese_whispers_clustering(graph)

print(cluster)