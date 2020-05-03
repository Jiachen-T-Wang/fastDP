import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import csv


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

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=69)
    print(x_train.shape, y_train.shape)

    return torch.Tensor(x_train), torch.Tensor(y_train), torch.Tensor(x_test), torch.Tensor(y_test)



