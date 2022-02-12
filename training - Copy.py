import datetime
import os

import sys

sys.path.append('C:/Users/rohan/Dissertation/transformer-master - Copy/transformer-master')
sys.path.append('C:/Users/rohan/Dissertation/transformer-master - Copy/transformer-master/src')

import datetime

import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import seaborn as sns

from tst import Transformer
from tst.loss import OZELoss

from dataset import OzeDataset
from utils import compute_loss
from visualization import map_plot_function, plot_values_distribution, plot_error_distribution, plot_errors_threshold, plot_visual_sample


# Training parameters
DATASET_PATH = 'dataset/dataset.npz'
BATCH_SIZE = 8#64#8
NUM_WORKERS = 8
LR = 2e-4
EPOCHS = 30

# Model parameters
d_model = 64 # Lattent dim
q = 8 # Query size
v = 8 # Value size
h = 8 # Number of heads
N = 4 # Number of encoder and decoder to stack
attention_size = 100#12 # Attention window size
dropout = 0.2 # Dropout rate
pe = None # Positional encoding
chunk_mode = None

d_input = 37  # From dataset
d_output = 8  # From dataset

# Config
sns.set()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

if not os.path.exists("logs"):
    os.mkdir("logs")
if not os.path.exists("models"):
    os.mkdir("models")


torch.cuda.empty_cache()
import gc
gc.collect()

print(torch.cuda.memory_summary(device=None, abbreviated=False))



import json
from ABBA import ABBA as ABBA

trial_dataset = np.load(DATASET_PATH)
labels_path = "labels.json"

with open(labels_path, "r") as stream_json:
    labels = json.load(stream_json)

R = trial_dataset['R'].astype(np.float32)
X = trial_dataset['X'].astype(np.float32)
Z = trial_dataset['Z'].astype(np.float32)

print("R shape")
print(np.shape(R))
print("X shape")
print(np.shape(X))
print("Z shape")
print(np.shape(Z))

m = Z.shape[0]  # Number of training example
K = Z.shape[-1]  # Time serie length

Z = Z.transpose((0, 2, 1))
X = X.transpose((0, 2, 1))

R = np.tile(R[:, np.newaxis, :], (1, K, 1))

print("new R shape")
print(np.shape(R))
print("new X shape")
print(np.shape(X))
print("new Z shape")
print(np.shape(Z))

# Store R, Z and X as x and y
_x = np.concatenate([Z, R], axis=-1)
_y = X

print("_x shape")
print(np.shape(_x))

# print("np.shape(_x)")
# print(np.shape(_x))
# print("np.shape(_x)[2]")
# print(np.shape(_x)[2])


# Normalize
# if _normalize == "mean":
mean = np.mean(_x, axis=(0, 1))
std = np.std(_x, axis=(0, 1))
_x = (_x - mean) / (std + np.finfo(float).eps)

_mean = np.mean(_y, axis=(0, 1))
_std = np.std(_y, axis=(0, 1))
_y = (_y - _mean) / (_std + np.finfo(float).eps)

# Using ABBA

abba = abba=ABBA(max_len=672, verbose=1)

print(np.shape(_x))

dim1 = np.shape(_x)[0]
dim2 = np.shape(_x)[1]
dim3 = np.shape(_x)[2]

pieces = np.empty(shape=(dim1, dim2, dim3))
print(np.shape(pieces))

# for i in range(dim1):
for i in range(10):
    print("{}th sample _x[i][..][j]".format(i))
#     for j in range (dim3):
    for j in range (dim3):
        print("{}th metric _x[i][..][j]".format(j))
#         print(_x[i,:,j])
        print("iteration {} of {}".format((i+1)*(j+1),10*dim3))
        abbaPieces = abba.compress(_x[i,:,j])
        print("compression step done")
        pieces = np.append(pieces,abbaPieces)
        print("stacking step done")

print("compression done...")

# pieces = abba.compress(_x)
ABBA_string, centers = abba.digitize(pieces)
# Apply inverse transform.
ABBA_numeric = mean + np.dot(std, abba.inverse_transform(ABBA_string, centers, normalised_time_series[0]))

# One hot encode symbolic representation. Create list of all symbols
# in case symbol does not occur in symbolic representation. (example:
# flat line and insist k>1)
alphabet = sorted([chr(97+i) for i in range(len(centers))])
sequence = np.array([[0 if char != letter else 1 for char in alphabet] for letter in ABBA_string])

# Convert to float32

# _x = torch.half(_x)
_x = torch.half(sequence)
_y = torch.half(_y)