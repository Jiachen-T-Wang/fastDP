import numpy as np
import pandas as pd
import csv
import time
import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import SGD, Adam, Adagrad, RMSprop

from sklearn.model_selection import train_test_split

from dpsgd import DPSGD, DPAdam, DPAdagrad, DPRMSprop
from mlp import Network
from utility import *


# Regular Training
def train(model, device, data_file, optimizer, epoch_nb, batch_size):

  if optimizer==None:
    optimizer = SGD(model.parameters(), lr=0.01)
  
  x_train, y_train, x_test, y_test = read_data(data_file)

  model.train()

  lossFnc = nn.BCEWithLogitsLoss()

  batches_per_epoch = int(x_train.shape[0] / batch_size)

  for epoch in range(epoch_nb):
    loss_epoch = 0
    acc_epoch = 0

    end = time.time()

    for _ in range(batches_per_epoch):
      idx = np.random.randint(0, x_train.shape[0], batch_size)
      x_train_batch, y_train_batch = x_train[idx].to(device), y_train[idx].to(device)

      optimizer.zero_grad()
      pred_batch = model(x_train_batch)
      loss = lossFnc(pred_batch, y_train_batch.unsqueeze(1))
      acc = binary_acc(pred_batch, y_train_batch.unsqueeze(1))

      loss_epoch += loss.item()
      acc_epoch += acc.item()

      loss.backward()
      optimizer.step()

    epoch_time = time.time() - end

    loss_epoch /= batches_per_epoch
    acc_epoch /= batches_per_epoch

    test_loss, test_acc = test(model, device, x_test, y_test)
    atk_acc = inversion_atk(model, device, x_train, y_train, target_col=8)

    print('Epoch:', epoch, 'Train Loss:', loss_epoch, 'Train Acc:', acc_epoch, 
          'Test Loss:', test_loss, 'Test Acc', test_acc, 
          'Atk Acc:', atk_acc, 'Time:', epoch_time)


def DPtrain(model, device, data_file, optimizer, epoch_nb):
  
  x_train, y_train, x_test, y_test = read_data(data_file)

  model.train()

  lossFnc = nn.BCEWithLogitsLoss()

  batch_size = optimizer.batch_size
  minibatch_size = optimizer.minibatch_size

  batches_per_epoch = int(x_train.shape[0] / batch_size)
  minibatches_per_batch = int(batch_size / minibatch_size)

  for epoch in range(epoch_nb):
    batch_loss = 0
    batch_acc = 0

    end = time.time()

    for i in range(batches_per_epoch):
      idx = np.random.randint(0, x_train.shape[0], batch_size)
      x_train_batch, y_train_batch = x_train[idx], y_train[idx]

      optimizer.zero_grad()

      for _ in range(minibatches_per_batch):
        idx = np.random.randint(0, batch_size, minibatch_size)
        x_train_mb, y_train_mb = x_train_batch[idx].to(device), y_train_batch[idx].to(device)

        optimizer.zero_minibatch_grad()
        pred_mb = model(x_train_mb)
        loss = lossFnc(pred_mb, y_train_mb.unsqueeze(1))
        acc = binary_acc(pred_mb, y_train_mb.unsqueeze(1))
        loss.backward()
        optimizer.minibatch_step()

        batch_loss += loss.item()
        batch_acc += acc.item()

      batch_loss /= minibatches_per_batch
      batch_acc /= minibatches_per_batch
        
      optimizer.step()

    epoch_time = time.time() - end

    train_loss, train_acc = test(model, device, x_train, y_train)
    test_loss, test_acc = test(model, device, x_test, y_test)
    atk_acc = inversion_atk(model, device, x_train, y_train, target_col=8)

    print('Epoch:', epoch, 'Train Loss:', train_loss, 'Test Loss:', test_loss,
          'Train Acc:', train_acc, 'Test Acc', test_acc, 'Atk Acc:', atk_acc, 
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


if __name__=='__main__':

    print('Collect Inputs...')

    parser = argparse.ArgumentParser()
    parser.add_argument("--settings_path", type=str)
    args = parser.parse_args()

    settings_json_fname = args.settings_path
    train_settings = json.load(open(settings_json_fname))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Number of epochs to train for
    num_epochs = train_settings['num_epoch']

    # Data Path
    path = train_settings['path']

    print('Initialize Model ...')

    model = Network()
    model.to(device)

    # training parameters setup
    l2_norm_clip = train_settings['l2_norm_clip']
    noise_multiplier = train_settings['noise_multiplier']
    batch_size = train_settings['batch_size']
    minibatch_size = train_settings['minibatch_size']
    lr = train_settings['lr']

    optimizer = DPSGD(
        params = model.parameters(),
        l2_norm_clip = l2_norm_clip,
        noise_multiplier = noise_multiplier,
        batch_size = batch_size,
        minibatch_size = minibatch_size, 
        lr = lr,
    )

    print('Begin DP Training')

    DPtrain(model, device, path, optimizer, epoch_nb=num_epochs)
