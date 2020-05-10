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


"""
def init_process(master_ip, master_port, rank, size, backend='gloo'):
    os.environ['MASTER_ADDR'] = master_ip
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group(backend, rank=rank, world_size=size)
"""

""" 
    Gradient averaging. 
    Note: Parameters are never broadcast between processes. The module performs
    an all-reduce step on gradients and assumes that they will be modified
    by the optimizer in all processes in the same way.        
"""
def gradients_allreduce(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


def DPtrain(model, device, data_file, optimizer, epoch_nb):

  # Since we do not have a sampler that can automatically distribute data
  # we need to write our own version of data partitioner
  batch_size = optimizer.batch_size
  minibatch_size = optimizer.minibatch_size

  train_partition, (x_test, y_test), batch_size = partition_dataset(batch_size, data_file)

  print('Rank:', dist.get_rank(), 'Train Set Size:', len(train_partition))

  model.train()

  lossFnc = nn.BCEWithLogitsLoss()

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

        gradients_allreduce(model)

        optimizer.minibatch_step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
      optimizer.step()

    epoch_time = time.time()-end

    epoch_loss, epoch_acc = epoch_loss/(batches_per_epoch*batch_size), epoch_acc/(batches_per_epoch*batch_size)

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
    batch_size = int(batch_size / float(size)) 
    partition_sizes = [1.0 / size for _ in range(size)]
    partitioner = DataPartitioner((x_train, y_train), partition_sizes)

    train_set = partitioner.use(dist.get_rank())

    return train_set, (x_test, y_test), batch_size


if __name__ == "__main__":

    print('Collect Inputs...')

    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int)
    parser.add_argument("--master_ip", type=str)
    parser.add_argument("--master_port", type=str)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--dist_backend", type=str, default='nccl')
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--path", type=str, default='data/CaPUMS5full.csv')
    parser.add_argument("--l2_norm_clip", type=float, default=3)
    parser.add_argument("--noise_multiplier", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--minibatch_size", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()

    # Total Number of distributed processes
    size = args.size

    # The private IP address and port for master node
    master_ip, master_port = args.master_ip, args.master_port

    # Global rank of the current process
    rank = args.rank

    # Local rank of the current process
    local_rank = args.local_rank

    # Distributed backend type
    dist_backend = args.dist_backend

    # Number of epochs to train for
    num_epochs = args.num_epochs

    # Number of additional worker processes for dataloading
    workers = args.workers

    # Data Path
    path = args.path

    print("Initialize Process Group...")

    # Initialize Process Group
    init_process(master_ip, master_port, rank, size, backend=dist_backend)


    device = torch.device('cuda', local_rank)

    print("Initialize Model...")

    torch.manual_seed(1234)

    model = Network()
    model.to(device)

    # training parameters setup
    l2_norm_clip = args.l2_norm_clip
    noise_multiplier = args.noise_multiplier
    batch_size = args.batch_size
    minibatch_size = args.minibatch_size
    lr = args.lr

    optimizer = DPSGD(
        params = model.parameters(),
        l2_norm_clip = l2_norm_clip,
        noise_multiplier = noise_multiplier,
        batch_size = batch_size,
        minibatch_size = minibatch_size, 
        lr = lr,
    )

    print('Begin DP Training')

    DPtrain(model, device, path, optimizer, epoch_nb=10)

