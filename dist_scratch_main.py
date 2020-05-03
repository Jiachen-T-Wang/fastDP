import os, sys
from random import Random

import numpy as np
import time

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist

from dpsgd import DPSGD, DPAdam, DPAdagrad, DPRMSprop
from mlp import Network
from utility import *


def init_process(rank, size, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '172.31.8.207'
    os.environ['MASTER_PORT'] = '23456'
    dist.init_process_group(backend, rank=rank, world_size=size)


""" Gradient averaging. """
def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


def DPtrain(model, device, data_file, optimizer, epoch_nb, target_acc):
  
  # TODO: partition data
  # x_train, y_train, x_test, y_test = read_data(data_file)

  batch_size = optimizer.batch_size
  minibatch_size = optimizer.minibatch_size

  train_partition, (x_test, y_test), batch_size = partition_dataset(batch_size, data_file)

  print('Rank:', dist.get_rank(), 'Train Set Size:', len(train_partition))

  model.train()

  lossFnc = nn.BCEWithLogitsLoss().cuda()

  batches_per_epoch = int(len(train_partition) / batch_size)
  minibatches_per_batch = int(batch_size / minibatch_size)

  for epoch in range(epoch_nb):
    epoch_loss = 0
    epoch_acc = 0

    end = time.time()

    for i in range(batches_per_epoch):
      idx = np.random.randint(0, len(train_partition), batch_size)
      x_train_batch, y_train_batch = train_partition[idx]

      optimizer.zero_grad()

      for _ in range(minibatches_per_batch):
        idx = np.random.randint(0, batch_size, minibatch_size)

        x_train_mb, y_train_mb = x_train_batch[idx].cuda(non_blocking=True), y_train_batch[idx].cuda(non_blocking=True)

        optimizer.zero_minibatch_grad()
        pred_mb = model(x_train_mb)
        loss = lossFnc(pred_mb, y_train_mb.unsqueeze(1))
        acc = binary_acc(pred_mb, y_train_mb.unsqueeze(1))
        loss.backward()

        average_gradients(model)

        optimizer.minibatch_step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
      optimizer.step()

      #print('Epoch:', epoch, 'Batch: ['+str(i)+'/'+str(batches_per_epoch)+']',
      #      'Loss:', batch_loss, 'Acc:', batch_acc)

    epoch_time = time.time()-end

    epoch_loss, epoch_acc = epoch_loss/len(train_partition), epoch_acc/len(train_partition)

    test_loss, test_acc = test(model, device, x_test, y_test)

    print('Rank ', dist.get_rank(), 'Epoch:', epoch, 
          'Train Loss:', epoch_loss, 'Test Loss:', test_loss,
          'Train Acc:', epoch_acc, 'Test Acc', test_acc,
          'Time:', epoch_time)


def test(model, device, X_data, Y_data):
    model.eval()
    lossFnc = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        X_data, Y_data = X_data.to(device), Y_data.to(device)
        Y_pred = model(X_data)
        loss = lossFnc(Y_pred, Y_data.unsqueeze(1))
        acc = binary_acc(Y_pred, Y_data.unsqueeze(1))

    return loss.item(), acc.item()



""" Dataset partitioning helper """
class Partition(object):

    def __init__(self, data, index):

        x_train, y_train = data

        self.x_train = x_train
        self.y_train = y_train
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        data_idx = self.index[idx]
        return self.x_train[data_idx], self.y_train[data_idx]


class DataPartitioner(object):

    def __init__(self, data, sizes=[0.5, 0.5], seed=1234):

        rng = Random()
        rng.seed(seed)

        x_train, y_train = data

        self.data = data
        self.partitions = []

        data_len = len(x_train)

        indexes = np.arange(data_len)
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


""" Partitioning PUMS """
def partition_dataset(batch_size, data_file):

    x_train, y_train, x_test, y_test = read_data(data_file)
    
    size = dist.get_world_size()
    # batch_size = int(batch_size / float(size))  # Replace 128 to overall batch_size
    partition_sizes = [1.0 / size for _ in range(size)]
    partitioner = DataPartitioner((x_train, y_train), partition_sizes)

    train_set = partitioner.use(dist.get_rank())

    return train_set, (x_test, y_test), batch_size


if __name__ == "__main__":

    DIR = 'data/'
    size = 2
    rank = int(sys.argv[1])
    torch.manual_seed(1234)

    print("Initialize Process Group...")

    init_process(rank, size, backend='gloo')
    local_rank = int(sys.argv[2])
    device = torch.device('cuda', local_rank)

    print("Initialize Model...")

    model = Network()
    model.to(device)

    l2_norm_clip = 3
    noise_multiplier = 0.9
    batch_size = 256
    minibatch_size = 3 

    optimizer = DPSGD(
        params = model.parameters(),
        l2_norm_clip = l2_norm_clip,
        noise_multiplier = noise_multiplier,
        batch_size = batch_size,
        minibatch_size = minibatch_size, 
        lr = 0.01,
    )

    print('Begin DP Training')

    DPtrain(model, device, DIR+'CaPUMS5full.csv', optimizer, epoch_nb=10, target_acc=0.65)





