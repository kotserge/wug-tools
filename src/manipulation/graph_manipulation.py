import networkx as nx
import numpy as np

def remove_edges(graph: nx.Graph, key=lambda x: np.isnan(x) ) -> nx.Graph:
    """
    Creates a copy where edges are removed based a key function which considers only the weight of the edge.

    Parameters
    ----------
    graph : nx.Graph
        The graph to be copied and modified.
    key : function
        The function which determines if an edge is removed or not. The function should take a single argument which is the weight of the edge.
    
    Returns
    -------
    nx.Graph
        The modified graph.
    """
    _graph = graph.copy()

    for edge in _graph.edges:
        if key(_graph.get_edge_data(*edge)['weight']):
            _graph.remove_edge(*edge)

    return _graph