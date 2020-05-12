import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


"""
Deep Learning Architecture for predicting unemployment rate
Input Dimension: 9, Output Dimension: 1
Feel free to change model architectures
"""

class Network(nn.Module):
  def __init__(self):
    super(Network, self).__init__()
    self.fc1 = nn.Linear(9, 50)
    self.fc2 = nn.Linear(50, 1)

  def forward(self, x):
    x = self.fc1(x)
    x = F.relu(x)
    output = self.fc2(x)
    return output