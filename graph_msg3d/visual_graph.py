'''
# @ Author: S.W.
# @ Description: Implementation of matplotlib to draw a graph
'''
# @ Author: S.W.
# @ Description: Implementation of matplotlib to draw a graph

import matplotlib.pyplot as plt
import networkx as nx
from copy import deepcopy
import numpy as np


def normalize_adjacency_matrix(A):
    node_degrees = A.sum(-1)
    degs_inv_sqrt = np.power(node_degrees, -0.5)
    norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt
    return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)


def k_adjacency(A, k, with_self=False, self_factor=1):
    assert isinstance(A, np.ndarray)
    I = np.eye(len(A), dtype=A.dtype)
    if k == 0:
        return I
    Ak = np.minimum(np.linalg.matrix_power(A + I, k), 1) \
         - np.minimum(np.linalg.matrix_power(A + I, k - 1), 1)
    # Ak= np.minimum(np.linalg.matrix_power(A + I, k), 1) - np.minimum(np.mean(np.stack([np.linalg.matrix_power(A + I, q) for q in range(k)],0),0),1)
    if with_self:
        Ak += (self_factor * I)
    return Ak


def dicentered_k_adjacency(A, k, with_self=False, self_factor=1):
    assert isinstance(A, np.ndarray)
    I = np.eye(len(A), dtype=A.dtype)
    if k == 0:
        return I
    # else:
    #     Ak = np.minimum(np.linalg.matrix_power(A + I, k), 1) - np.minimum(np.linalg.matrix_power(A + I, k - 1) + np.mean(np.stack([np.linalg.matrix_power(A + I, p+1) - np.linalg.matrix_power(A + I, p) for p in range(k)], 0), 0) +np.mean(np.stack([np.linalg.matrix_power(A + I, p+1) for p in range(k)], 0), 0), 1)
    
    else:
        Ak = np.mean(np.stack([np.minimum(np.linalg.matrix_power(A + I, k), 1) - np.minimum(np.mean(np.stack([np.linalg.matrix_power(A + I, p+1) - np.linalg.matrix_power(A + I, p) for p in range(k)], 0), 0), 1),
                     np.minimum(np.linalg.matrix_power(A + I, k), 1) - np.minimum(np.linalg.matrix_power(A + I, k - 1), 1)],0),0)

    if with_self:
        Ak += (self_factor * I)
    return Ak


class AdjacencyMatrix(object):

    def __init__(self, nodes):
        self.nodes = nodes

    def plot_graph_1(self, graph, title='Grafo 1'):

        graph_layout = nx.Graph()
        adjacent_matrix = deepcopy(graph)

        for n in range(1, self.nodes + 1):
            graph_layout.add_node(n)

        for i in range(self.nodes):
            for j in range(self.nodes):

                if (adjacent_matrix[i][j] != 0):
                    graph_layout.add_edge(i + 1, j + 1)
                    adjacent_matrix[i][j] = 0
                    adjacent_matrix[j][i] = 0

        dictionary = nx.spring_layout(graph_layout)
        nx.draw_networkx_nodes(graph_layout, dictionary)
        nx.draw_networkx_labels(graph_layout, dictionary)
        nx.draw_networkx_edges(graph_layout, dictionary)
        plt.title(title)
        plt.show()


def main():
    qt_nodes = int(input(f"number of vertices:"))
    qt_arestas = int(input(f"number of edges:"))
    A_binary = np.zeros([qt_nodes, qt_nodes], dtype=int)
    for i in range(0, qt_arestas):
        aresta1 = int(input(f"the first edge {i + 1}: "))
        aresta2 = int(input(f"the second edge {i + 1}: "))
        A_binary[aresta1 - 1, aresta2 - 1] = 1
        A_binary[aresta2 - 1, aresta1 - 1] = 1
        print()
    print('-=' * 20)
    print('the complete adj matrix')
    print(A_binary)
    A_powers = [A_binary + np.eye(len(A_binary)) for k in range(8)]
    A_powers = [normalize_adjacency_matrix(g) for g in A_powers]
    A_powers = [np.linalg.matrix_power(g, k) for k, g in enumerate(A_powers)]
    A_powers = np.concatenate(A_powers)
    plt.matshow(A_powers, cmap=plt.cm.Blues)
    plt.show()

    disentangled_A_powers = [k_adjacency(A_binary, k, with_self=True) for k in range(8)]
    disentangled_A_powers = np.concatenate([normalize_adjacency_matrix(g) for g in disentangled_A_powers])
    plt.matshow(disentangled_A_powers, cmap=plt.cm.Blues)
    plt.show()

    dicentered_A_powers = [dicentered_k_adjacency(A_binary, k, with_self=True) for k in range(8)]
    dicentered_A_powers = (np.concatenate([normalize_adjacency_matrix(g) for g in dicentered_A_powers]))
    plt.matshow(dicentered_A_powers, cmap=plt.cm.Blues)
    plt.show()

    Object = AdjacencyMatrix(qt_nodes)
    Object.plot_graph_1(A_binary, 'Graph')


if __name__ == '__main__':
    main()
