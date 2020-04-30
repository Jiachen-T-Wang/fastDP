import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import csv

from dpsgd import DPSGD, DPAdam, DPAdagrad, DPRMSprop
from mlp import Network


DIR = './drive/My Drive/Colab Notebooks/CS205/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    return acc


def read_data(file, max_line=10000):
    line = 0
    with open(file, 'rt') as f:
        next(f) # skip labels
        X, Y = [], []
        reader = csv.reader(f)
        for row in reader:

            if sum(np.array(row)=='NA'): # leave out all NAs
                continue

            x1, x2 = map(int, row[:5]), map(int, row[6:-2])

            x = list(x1) + list(x2)

            X.append(x)
            Y.append(int(row[5]))
            line += 1
            if line > max_line: break

    X, Y = np.array(X), np.array(Y)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=69)
    print(x_train.shape, y_train.shape)

    return torch.Tensor(x_train), torch.Tensor(y_train), torch.Tensor(x_test), torch.Tensor(y_test)


# Regular Training

def train(model, device, data_file, optimizer, epoch_nb, batch_size):
  
  x_train, y_train, x_test, y_test = read_data(data_file)

  model.train()

  lossFnc = nn.BCEWithLogitsLoss()

  batches_per_epoch = int(x_train.shape[0] / batch_size)

  for epoch in range(epoch_nb):
    loss_epoch = 0
    acc_epoch = 0
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

    loss_epoch /= batches_per_epoch
    acc_epoch /= batches_per_epoch

    print('Epoch:', epoch, 'Loss:', loss_epoch, 'Acc:', acc_epoch)


def DPtrain(model, device, data_file, optimizer, epoch_nb, target_acc):
  
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

      print('Epoch:', epoch, 'Batch: ['+str(i)+'/'+str(batches_per_epoch)+']',
            'Loss:', batch_loss, 'Acc:', batch_acc)

    train_loss, train_acc = test(model, device, x_train, y_train)
    test_loss, test_acc = test(model, device, x_test, y_test)

    print('Epoch:', epoch, 'Train Loss:', train_loss, 'Test Loss:', test_loss,
        'Train Acc:', train_acc, 'Test Acc', test_acc)

    if test_acc > target_acc:
        return


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

    DPtrain(model, device, DIR+'CaPUMS5full.csv', optimizer, epoch_nb=10, target_acc=0.65)
