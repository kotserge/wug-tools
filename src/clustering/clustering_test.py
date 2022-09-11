import random
import networkx as nx

from clustering_interface import correlation_clustering
# generate random graph
graph = nx.readwrite.gpickle.read_gpickle('data/test/graph/full/Rezeption')

cluster = correlation_clustering(graph)

print(cluster)