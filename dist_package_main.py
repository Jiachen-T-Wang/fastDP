import time
import sys
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

import numpy as np
import pandas as pd
import csv

from dpsgd import DPSGD, DPAdam, DPAdagrad, DPRMSprop
from mlp import Network

DIR = './'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def binary_acc(y_pred, y_test):
	y_pred_tag = torch.round(torch.sigmoid(y_pred))
	correct_results_sum = (y_pred_tag == y_test).sum().float()
	acc = correct_results_sum/y_test.shape[0]
	return acc

# change (combined x and y)
def read_data(file, max_line = 10000):
	line = 0
	with open(file, 'rt') as f:
		next(f) # skip labels
		dataset = []
		reader = csv.reader(f)
		for row in reader:
			if sum(np.array(row)=='NA'): # leave out all NAs
				continue
			x1, x2 = map(int, row[:5]), map(int, row[6:-2])
            # x = list(x1) + list(x2)
            # X.append(x)
            # Y.append(int(row[5]))
			dataset.append(list(x1) + list(x2) + [int(row[5])])
			line += 1
			if line > max_line: 
				break
	dataset = np.array(dataset)
	train, val = train_test_split(dataset, test_size=0.2, random_state=69)
	return torch.Tensor(train), torch.Tensor(val)


def DPtrain(model, device, train_loader, optimizer, epoch):

 	# x_train, y_train, x_test, y_test = read_data(data_file)
	model.train()

	lossFnc = nn.BCEWithLogitsLoss()

	batch_size = optimizer.batch_size
	minibatch_size = optimizer.minibatch_size

	minibatches_per_batch = int(batch_size / minibatch_size)

	# for epoch in range(epoch_nb):
	for _, train_data in enumerate(train_loader):
		x_train, y_train = train_data[:, :-1], train_data[:,-1]
		batches_per_epoch = int(x_train.shape[0] / batch_size)

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
		# test_loss, test_acc = test(model, device, x_test, y_test)
		print('Epoch:', epoch, 'Train Loss:', train_loss, 'Train Acc:', train_acc)

def validate(model, device, val_loader, epoch):
	for _, test_data in enumerate(val_loader):
		x_test, y_test = test_data[:, :-1], test_data[:,-1]
		test_loss, test_acc = test(model, device, x_test, y_test)
		print('Epoch:', epoch, 'Test Loss:', test_loss, 'Test Acc:', test_acc)		



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
	# Number of additional worker processes for dataloading
	workers = 2
	
	# Number of epochs to train for
	num_epochs = 10

	# Number of distributed processes
	world_size = 2

	# Distributed backend type
	dist_backend = 'nccl'

	# -  **dist\_url** - URL to specify the initialization method of the
	#    process group. This may contain the IP address and port of the rank0
	#    process or be a non-existant file on a shared file system. Here,
	#    since we do not have a shared file system this will incorporate the
	#    **node0-privateIP** and the port on node0 to use.

	# Url used to setup distributed training
	dist_url = "tcp://172.31.8.137:23456"

	print("Initialize Process Group...")

	# Initialize Process Group
	dist.init_process_group(backend=dist_backend, 
	                        init_method=dist_url, 
	                        rank=int(sys.argv[1]), 
	                        world_size=world_size)

	# Establish Local Rank and set device on this node
	local_rank = int(sys.argv[2])
	dp_device_ids = [local_rank]
	torch.cuda.set_device(local_rank)

	# Construct Model
	model = Network().cuda()

	# Next, we make the model ``DistributedDataParallel``, which handles the
	# distribution of the data to and from the model and is critical for
	# distributed training. The ``DistributedDataParallel`` module also
	# handles the averaging of gradients across the world, so we do not have
	# to explicitly average the gradients in the training step.
	# This is a blocking function
	model = torch.nn.parallel.DistributedDataParallel(model, device_ids=dp_device_ids, output_device=local_rank)

	# parameters setup
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

	print("Initialize Dataloaders...")

	trainset, valset = read_data("./CaPUMS5full.csv")

	# Create DistributedSampler to handle distributing the dataset across nodes when training
	# This can only be called after torch.distributed.init_process_group is called
	train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)

	# Create the Dataloaders to feed data to the training and validation steps
	train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=(train_sampler is None), 
	                                           num_workers=workers, pin_memory=False, sampler=train_sampler)
	
	val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=False)

	for epoch in range(num_epochs):
		# Set epoch count for DistributedSampler
		# In distributed mode, calling the set_epoch method 
		# is needed to make shuffling work; each process 
		# will use the same random seed otherwise.
		train_sampler.set_epoch(epoch)

		# train for one epoch
		print("\nBegin Training Epoch {}".format(epoch+1))
		DPtrain(model, device, train_loader, optimizer, epoch)
		
		# evaluate on validation set
		print("Begin Validation @ Epoch {}".format(epoch+1))
		validate(model, device, val_loader, epoch)

    # DPtrain(model, device, DIR+'CaPUMS5full.csv', optimizer, epoch_nb=10, target_acc=0.65)
