import time
import os, sys
import numpy as np
import pandas as pd
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.parallel
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
from torch.multiprocessing import Pool, Process
from sklearn.model_selection import train_test_split

from dpsgd import DPSGD, DPAdam, DPAdagrad, DPRMSprop
from mlp import Network
from utility import *


def init_process(master_ip, master_port, rank, size, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = master_ip
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group(backend, rank=rank, world_size=size)


def DPtrain(model, device, train_loader, optimizer, epoch_nb):

    _, test_data = read_data_package('./CaPUMS5full.csv', max_line=10000)
    x_test, y_test = test_data[:, :-1], test_data[:, -1]

    print('Rank of Training Process:', dist.get_rank())

    model.train()

    lossFnc = nn.BCEWithLogitsLoss()

    batch_size = optimizer.batch_size
    minibatch_size = optimizer.minibatch_size

    minibatches_per_batch = int(batch_size / minibatch_size)


    for epoch in range(epoch_nb):

        # Set seed for DistributedSampler
        train_loader.sampler.set_epoch(epoch)

        epoch_loss = 0
        epoch_acc = 0
        n_batch = 0

        start = time.time()

        for i, train_data in enumerate(train_loader):
            x_train_batch, y_train_batch = train_data[:, :-1].to(device), train_data[:, -1].to(device)
            n_batch += 1

            optimizer.zero_grad()

            for _ in range(minibatches_per_batch):
                idx = np.random.randint(0, batch_size, minibatch_size)
                x_train_mb, y_train_mb = x_train_batch[idx], y_train_batch[idx]

                optimizer.zero_minibatch_grad()
                pred_mb = model(x_train_mb)
                loss = lossFnc(pred_mb, y_train_mb.unsqueeze(1))
                acc = binary_acc(pred_mb, y_train_mb.unsqueeze(1))
                loss.backward()
                optimizer.minibatch_step()

                epoch_loss += loss.item()
                epoch_acc += acc.item()
                
            optimizer.step()

        epoch_time = time.time()-start

        epoch_loss, epoch_acc = epoch_loss/(n_batch*minibatches_per_batch), epoch_acc/(n_batch*minibatches_per_batch)

        test_loss, test_acc = test(model, device, x_test, y_test)

        print('Rank ', dist.get_rank(), 'Epoch:', epoch, 
              'Train Loss:', epoch_loss, 'Test Loss:', test_loss,
              'Train Acc:', epoch_acc, 'Test Acc', test_acc,
              'Time:', epoch_time)


"""
def validate(model, device, val_loader, epoch):
    for _, test_data in enumerate(val_loader):
        x_test, y_test = test_data[:, :-1], test_data[:,-1]
        test_loss, test_acc = test(model, device, x_test, y_test)
        print('Epoch:', epoch, 'Test Loss:', test_loss, 'Test Acc:', test_acc)      
"""


def test(model, device, X_data, Y_data):
    model.eval()
    lossFnc = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        X_data, Y_data = X_data.to(device), Y_data.to(device)
        Y_pred = model(X_data)
        loss = lossFnc(Y_pred, Y_data.unsqueeze(1))
        acc = binary_acc(Y_pred, Y_data.unsqueeze(1))

    return loss.item(), acc.item()


if __name__=='__main__':
    
    # Number of epochs to train for
    num_epochs = 10

    # Total Number of distributed processes
    size = 2

    # Distributed backend type
    dist_backend = 'nccl'

    # The private IP address and port for master node
    master_ip, master_port = '172.31.37.213', '23456'

    # Rank of the current process
    rank = int(sys.argv[1])

    # Number of additional worker processes for dataloading
    workers = 2

    # Data Path
    path = './CaPUMS5full.csv'

    # Number of additional worker processes for dataloading
    workers = 2

    print("Initialize Process Group...")

    # Initialize Process Group
    init_process(master_ip, master_port, rank, size, backend=dist_backend)

    # Establish Local Rank and set device on this node
    local_rank = int(sys.argv[2])
    dp_device_ids = [local_rank]
    device = torch.device('cuda', local_rank)


    print("Initialize Model...")

    torch.manual_seed(1234) # missing even in the tutorial
    model = Network()
    model.to(device)

    # Use 'DistributedDataParallel' module
    # This is a blocking function
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=dp_device_ids, output_device=local_rank)

    # training parameters setup
    l2_norm_clip = 3
    noise_multiplier = 0.9
    batch_size = 256
    minibatch_size = 3 

    batch_size = int(batch_size / size)

    optimizer = DPSGD(
        params = model.parameters(),
        l2_norm_clip = l2_norm_clip,
        noise_multiplier = noise_multiplier,
        batch_size = batch_size,
        minibatch_size = minibatch_size, 
        lr = 0.01,
    )


    print("Initialize Dataloaders...")

    trainset, valset = read_data_package(path)

    # Use DistributedSampler module to handle distributing the dataset across nodes when training
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=(train_sampler is None), 
                                               num_workers=workers, pin_memory=False, sampler=train_sampler,
                                               drop_last=True)
    
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=False)

    DPtrain(model, device, train_loader, optimizer, num_epochs)

