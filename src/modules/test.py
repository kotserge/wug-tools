import random
import networkx as nx

from clustering_interface import wsbm_clustering
# generate random graph
graph = nx.readwrite.gpickle.read_gpickle('data/Manschette')

cluster = wsbm_clustering(graph)

print(cluster)