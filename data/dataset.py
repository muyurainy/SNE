# coding: utf8
import numpy as np
from math import pow
from torch.utils import data
from .SignedGraph import SignGraph


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


class SideData(object):

    def __init__(self, config, seq='\t', split_ratio=0.8):
        self.config = config
        self.G = SignGraph(config.filename, seq=seq, split_ratio=split_ratio,
                           directed=True)
        if self.config.weighted:
            print '============ no consider ! ==========='
        else:
            if self.config.directed:
                num_edges = len(self.G.g.in_edges())
                self.freq = {node: 1.0 * len(self.G.g.in_edges(node)) / num_edges
                             for node in self.G.g.nodes}
            else:
                num_edges = len(self.G.g.out_edges())
                self.freq = {node: 1.0 * len(self.G.g.out_edges(node)) / num_edges
                             for node in self.G.g.nodes}
        self.walks = self.walk(config.window_size, config.walk_length, config.num_walks)
        self.preprocess(config.window_size)

    def walk(self, window_size, walk_length, num_walks):
        nodes = np.array(self.G.g.nodes)
        walks = []
        v_str = {v: v for v in self.G.g.nodes}
        v_str['+'] = '+'
        v_str['-'] = '-'
        self.v_str = v_str
        num_pairs_required = window_size * (walk_length - (window_size + 1) / 2)
        if self.config.weighted:
            print '============ no consider ! ==========='
        else:
            print 'start random walk on Memory!'
            for cnt in range(num_walks):
                np.random.shuffle(nodes)
                for node in nodes:
                    if len(self.G.g[node].keys()) == 0:
                        continue
                    num_pairs = 0
                    while num_pairs < num_pairs_required:
                        walks.append(self.random_walk(
                            g=self.G.g, walk_length=walk_length,
                            start=node, rand=np.random.RandomState(),
                            subsample=self.config.subsample))
                        num_pairs += ((len(walks[-1]) + 1) * (len(walks[-1]) - 1) / 8
                                      if len(walks[-1]) < 2 * window_size - 1
                                      else window_size * (len(walks[-1]) - window_size) / 2)
        return walks

    def random_walk(self, g, walk_length, start, rand, subsample):
        """
        Generate single random walk for unweighted network
        Parameters:
            g: networkx object
            walk_length: int
                the length of each length
            start: int
                start node
            rand: numpy random object
            subsample: float
                subsample rate

        Returns:
            :list
                single random walk list
        """
        walk = [start]
        cur = start
        sign = 1
        while len(walk) < 2 * walk_length - 1:
            if len(g[cur].keys()) == 0:
                break
            nxt = rand.choice(list(g[cur].keys()))
            if len(g[nxt].keys()) == 0 or rand.rand() < np.sqrt(subsample / self.freq[nxt]):
                walk.append('+' if sign * g[cur][nxt]['sign'] > 0 else '-')
                walk.append(nxt)
                sign = 1
            elif g[cur][nxt] < 0:
                sign *= -1
            cur = nxt
        return [self.v_str[node] for node in walk]

    def preprocess(self, window_size):
        """
        Generate word freq, input tuples

        Renturns:
            :list
                input tuple list
        """
        # ====== Generate word_freq ======
        self.word_freq = {}
        for sentence in self.walks:
            for word in sentence:
                if word != '+' and word != '-':
                    if word in self.word_freq:
                        self.word_freq[word] += 1
                    else:
                        self.word_freq[word] = 1
        print self.word_freq
        # ====== Generate node pair ======
        node_pair = []
        for sentence in self.walks:
            example_pos = 0
            label_pos = 2
            num_pos = 0
            num_neg = 0
            while(True):
                if label_pos - example_pos > 2 * window_size \
                        or example_pos + 2 > len(sentence) \
                        or label_pos > len(sentence):
                    num_pos = 0
                    num_neg = 0
                    example_pos += 2
                    label_pos = example_pos
                    label_pos += 2
                    if example_pos + 2 > len(sentence):
                        break
                if sentence[label_pos - 1] == '-':
                    num_neg += 1
                else:
                    num_pos += 1
                node_pair.append((sentence[example_pos],
                                  sentence[label_pos], num_pos, num_neg))
                label_pos += 2
        # ===== Init Neg table ======
        table_size = 1e8
        NEG_SAMPLE_POWER = 0.75
        norm = sum([pow(self.freq[i], NEG_SAMPLE_POWER) for i in self.freq])
        neg_table = ['' for i in range(int(table_size))]
        p = 0
        i = 0
        for node in self.freq:
            p += 1.0 * pow(self.freq[node], NEG_SAMPLE_POWER) / norm
            while i < table_size and (1.0 * i) / table_size < p:
                neg_table[i] = node
                i += 1
        # ===== Negtive Sample ======
        self.data = []
        for pair in node_pair:
            if pair[3] % 2 == 1:
                sign = -1
            else:
                sign = 1
            neg_list = []
            for _ in range(self.config.neg_sample_size):
                neg_node = neg_table[np.random.randint(0, table_size)]
                while neg_node == pair[0] or neg_node == pair[1]:
                    neg_node = neg_table[np.random.randint(0, table_size)]
                neg_list.append(neg_node)
            self.data.append([pair[0], pair[1], sign, neg_list])
            __import__('pdb').set_trace()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
