# coding: utf-8
import networkx as nx
import numpy as np
from sklearn.utils import shuffle
from scipy.sparse import dok_matrix


class SignGraph:
    def __init__(self, filename, seq='\t', split_ratio=0.8, weight=False, directed=False):
        self.adj_matrix = None
        self.train_edges = None
        self.test_edges = None
        self.g = nx.DiGraph()
        self.weight = weight
        self.directed = directed
        edgefile = open(filename)
        edges = [map(int, line.strip().split(seq)) for line in edgefile]
        self.all_edges = edges

        source_node, targt_node, sign = zip(*edges)
        nodes = list(set(source_node) | set(targt_node))
        if len(nodes)-1 != max(nodes):
            print '======= len(nodes)-1 != max(nodes)! =========='
        # print len(nodes)
        for node in xrange(max(nodes)+1):
            self.g.add_node(node)
        edges = shuffle(edges)
        training_size = int(split_ratio * len(edges))
        self.train_edges = edges[:training_size]
        self.test_edges = edges[training_size:]
        if split_ratio == 1.0:
            self.test_edges = self.train_edges
        edgefile.close()
        self.getGraph(weight=self.weight, directed=self.directed)

    def getGraph(self, weight, directed):
        for line in self.train_edges:
            if directed:
                if not weight:
                    src, tgt, sign = line
                    self.g.add_edge(src, tgt)
                    self.g[src][tgt]['weight'] = 1.0
                    self.g[src][tgt]['sign'] = sign
                else:
                    src, tgt, sign, weight = line
                    self.g.add_edge(src, tgt)
                    self.g[src][tgt]['weight'] = float(weight)
                    self.g[src][tgt]['sign'] = sign
            else:
                if not weight:
                    src, tgt, sign = line
                    self.g.add_edge(src, tgt)
                    self.g.add_edge(tgt, src)
                    self.g[src][tgt]['weight'] = 1.0
                    self.g[tgt][src]['weight'] = 1.0
                    self.g[src][tgt]['sign'] = sign
                    self.g[tgt][src]['sign'] = sign
                else:
                    src, tgt, sign, weight = line
                    self.g.add_edge(src, tgt)
                    self.g.add_edge(tgt, src)
                    self.g[src][tgt]['weight'] = float(weight)
                    self.g[tgt][src]['weight'] = float(weight)
                    self.g[src][tgt]['sign'] = sign
                    self.g[tgt][src]['sign'] = sign
        print ('node: {}, edges: {}').format(len(self.g.nodes), len(self.g.edges))

    def to_adjmatrix(self):
        self.adj_matrix = dok_matrix((len(self.g.nodes()), len(self.g.nodes())), np.int_)
        for edge in self.g.edges():
            self.adj_matrix[edge[0], edge[1]] = self.g[edge[0]][edge[1]]['sign']
        self.adj_matrix = self.adj_matrix.tocsr()
        return self.adj_matrix
