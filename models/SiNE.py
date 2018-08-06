# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .BasicModule import BasicModule


class SiNE(BasicModule):

    def __init__(self, config):
        super(SiNE, self).__init__()
        self.config = config
        self.model_name = 'SiNE'
        self._1st_layer = FirstLayer(config.dimension, config._1st_out_features)
        self._2nd_layer = nn.Linear(config._1st_out_features, config._2st_out_features)
        self._3rd_layer = nn.Linear(config._2st_out_features, config._3st_out_features)
        self.dropout = nn.Dropout(0.5)
        self.embedding = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(config.N, config.dimension)))

    def init_model_weight(self):
        nn.init.xavier_uniform(self._2nd_layer.weight)
        nn.init.xavier_uniform(self._3rd_layer.weight)

        nn.init.uniform(self._2nd_layer.bias, a=-0.5, b=0.5)
        nn.init.uniform(self._3rd_layer.bias, a=-0.5, b=0.5)

    def t_to_v(self, t):
        if self.config.CUDA:
            return Variable(t.cuda())
        else:
            return Variable(t)

    def forward(self, data):
        x1, x2, x3 = data
        h1 = F.tanh(self._1st_layer(self.embedding[self.t_to_v(x1)], self.embedding[self.t_to_v(x2)]))
        h2 = F.tanh(self._1st_layer(self.embedding[self.t_to_v(x1)], self.embedding[self.t_to_v(x3)]))
        h1 = F.tanh(self._2nd_layer(h1))
        h2 = F.tanh(self._2nd_layer(h2))
        self.dropout(h1)
        self.dropout(h2)
        h1 = F.tanh(self._3rd_layer(h1))
        h2 = F.tanh(self._3rd_layer(h2))
        self.tuple_loss = torch.sum(
            torch.max(
            torch.cat((self.t_to_v(torch.zeros(h1.shape))
                      , h2 + 1 - h1), dim=1), dim=1)[0])
        # print self.tuple_loss
        return self.tuple_loss

    def get_embedding(self):
        return self.embedding.cpu().data.numpy()[1:]


class FirstLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(FirstLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w1 = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(in_features, out_features)))
        self.w2 = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(in_features, out_features)))
        self.b = nn.Parameter(nn.init.uniform(torch.Tensor(out_features), a=-0.1, b=0.1))

    def forward(self, x1, x2):
        return torch.matmul(x1, self.w1) + torch.matmul(x2, self.w2) + self.b
