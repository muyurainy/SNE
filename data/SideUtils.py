# coding: utf8
import numpy as np
from torch.utils import data
from .SignedGraph import SignGraph
import os


class SineData(data.Dataset):

    def __init__(self, filename, seq='\t', train=True, test=False, split_ratio=0.8):
        self.train = train
        self.test = test
        self.G = SignGraph(filename, seq=seq, split_ratio=split_ratio, directed=True)
        self.tuple_list = self.get_tuple(self.G.to_adjmatrix().todense())

    def get_tuple(self, adj, threshold=200):
        tuple_list = []
        for i in xrange(adj.shape[0]):
            poss = np.where(adj[i, :] == 1)[1]
            negs = np.where(adj[i, :] == -1)[1]
            tuples = []
            for pos in poss:
                for neg in negs:
                    tuples.append((pos+1, neg+1))
            if len(negs) == 0:
                for pos in poss:
                    tuples.append((pos+1, 0))

            if len(tuples) > threshold:
                rand = np.random.permutation(len(tuples))[0:threshold]
                newTuples = []
                for ind in rand:
                    newTuples.append(tuples[ind])
                tuples = newTuples
            for tup in tuples:
                tuple_list.append([i+1, tup[0], tup[1]])
        return tuple_list

    def __getitem__(self, index):
        return self.tuple_list[index]

    def __len__(self):
        return len(self.tuple_list)
