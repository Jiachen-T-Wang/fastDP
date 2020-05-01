import torch
import pandas
from torch.utils.data import Dataset


"""
Let’s create a dataset class for our face landmarks dataset. 
We will read the csv in __init__ but leave the reading of images to __getitem__. 
This is memory efficient because all the images are not stored in the memory at once but read as required.
"""


class PUMS_loader(Dataset):
	""" California PUMS Dataset """

	def __init__(self, csv_file, root_dir):
		self.df = pd.read_csv(csv_file)
		self.root_dir = root_dir

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
			
		return self.df[idx]

