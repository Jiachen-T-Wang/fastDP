import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist


from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import csv, os


MAX_LINE = 10000


def inversion_atk(model, device, x_train, y_train, target_col):
    
    model.eval()

    correct = 0.0

    if x_train.shape[0] > 10000:
        x_train = x_train[:10000]

    for i in range(x_train.shape[0]):
        row_x = np.stack([x_train[i] for _ in range(2)])
        row_x[0, target_col] = 0
        row_x[1, target_col] = 1
        row_y = np.repeat(y_train[i], 2)
        row_x = torch.Tensor(row_x).to(device)
        pred = torch.sigmoid(model(row_x)).cpu().data.numpy()
        row_y = row_y.cpu().data.numpy().reshape((2, 1))
        errors = np.abs(row_y - pred)
        guess = np.argmin(errors)

        if guess == x_train[i, target_col]:
            correct += 1

    return correct / x_train.shape[0]


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    return acc


def read_data(file, max_line = MAX_LINE):
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

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=69)
    print(x_train.shape, y_train.shape)

    return torch.Tensor(x_train), torch.Tensor(y_train), torch.Tensor(x_test), torch.Tensor(y_test)


# change (combined x and y)
def read_data_package(file, max_line = MAX_LINE):
    line = 0
    with open(file, 'rt') as f:
        next(f) # skip labels
        dataset = []
        reader = csv.reader(f)
        for row in reader:
            if sum(np.array(row)=='NA'): # leave out all NAs
                continue
            x1, x2 = map(int, row[:5]), map(int, row[6:-2])
            #x = list(x1) + list(x2)
            #X.append(x)
            #Y.append(int(row[5]))
            dataset.append(list(x1) + list(x2) + [int(row[5])])
            line += 1
            if line > max_line: 
                break
    dataset = np.array(dataset)
    train, val = train_test_split(dataset, test_size=0.2, random_state=69)
    return torch.Tensor(train), torch.Tensor(val)


def init_process(master_ip, master_port, rank, size, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = master_ip
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group(backend, rank=rank, world_size=size)



