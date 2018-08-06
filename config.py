# coding: utf8
import warnings


def parse(self, kwargs):
    for k, v in kwargs.iteritems():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribut %s" % k)
        setattr(self, k, v)

    print('user config:')
    for k, v in self.__class__.__dict__.iteritems():
        if not k.startswith('__'):
            print(k, getattr(self, k))


class SiNEConfig(object):
    filename = 'data/dataset/wiki-Vote.txt'
    model = 'SiNE'
    batch_size = 80
    CUDA = False
    num_workers = 4
    N = 0  # nodes length + virtual node
    epochs = 100
    lr = 0.5
    weight_decay = 1e-4
    _1st_out_features = 20
    _2st_out_features = 20
    _3st_out_features = 1
    dimension = 20
    split_ratio = 0.8


# DefaultConfig.parse = parse
SiNEConfig.parse = parse
SiNEconfig = SiNEConfig()
# opt.parse = parse


class SideConfig(object):
    filename = 'data/dataset/out.ucidata-gama'  # 'data/dataset/wiki-Vote.txt'
    directed = True
    weighted = False
    subsample = 1e-3
    num_walks = 10  # 80
    walk_length = 40
    window_size = 5
    dimension = 128
    neg_sample_size = 20
    weight_decay = 1e-2
    batch_size = 16
    lr = 0.025
    epochs = 100
    model = 'Side'
    CUDA = False
    num_workers = 4
    split_ratio = 0.8


SideConfig.parse = parse
Sideconfig = SideConfig()
