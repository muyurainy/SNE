#  coding: utf-8
from config import SiNEconfig as config
import models
from data import SineData
import torch
from torch.utils.data import DataLoader
from utils import Task
import os
import cPickle as pickle


def train(**kwargs):
    config.parse(kwargs)
    if os.path.exists(config.filename + '_' + str(config.split_ratio) + 'SineData.pkl'):
        train_data = pickle.load(file(config.filename + '_' + str(config.split_ratio) + 'SineData.pkl'))
        print 'exists SineData.pkl, load it!'
    else:
        train_data = SineData(config.filename, split_ratio=config.split_ratio)
        pickle.dump(train_data, file(config.filename + '_' + str(config.split_ratio) + 'SineData.pkl', 'w'))
    config.N = train_data.G.g.number_of_nodes() + 1
    model = getattr(models, config.model)(config)   # .eval()
    if torch.cuda.is_available():
        model.cuda()
        config.CUDA = True
    train_dataloader = DataLoader(train_data, config.batch_size, shuffle=True,
                                  num_workers=config.num_workers)
    #  optimizer = torch.optim.SGD(model.parameters(),lr = config.lr, weight_decay = config.weight_decay)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.95,
                                     weight_decay=config.weight_decay)
    task = Task(train_data.G)
    # model.train()
    for epoch in range(config.epochs):
        total_loss = 0.0
        for idx, data in enumerate(train_dataloader):
            #  if config.CUDA:
                #  data = map(lambda x: Variable(x.cuda()), data)
            #  else:
                #  data = map(lambda x: Variable(x), data)
            optimizer.zero_grad()
            loss = model(data)
            loss.backward()
            optimizer.step()
            if config.CUDA:
                total_loss += loss.cpu().data.numpy()
            else:
                total_loss += loss.data.numpy()
        print 'epoch {0}, loss: {1}'.format(epoch, total_loss)
        task.link_sign_prediction_split(model.get_embedding())
        #  if epoch % 20 == 0:
            #  for param_group in optimizer.param_groups:
                #  param_group['lr'] = param_group['lr'] * 0.95
    # model.eval()


def help():
    print('''
    usage : python file.py <function> [--args=value]
    <function> := train | help
    example:
            python {0} train --env='env0701' --lr=0.01
            python {0} help
    avaiable args:'''.format(__file__))

    from inspect import getsource
    source = (getsource(config.__class__))
    print(source)


if __name__ == '__main__':
    import fire
    fire.Fire()
