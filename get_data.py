# Initial test to see if we have already have the data
from os.path import exists

if exists('./data/test/pytorch_test_data_1_v4.pt'):
    
    raise Exception("Pytorch test data alreadyt exists")

########################################################
#
# If no  data, then continue with the script, and 
# create it
#
########################################################

import json
import numpy as np
import pickle as pkl
import sys
import tensorflow as tf
import torch

from copy import copy
from collections import defaultdict
from os import makedirs
from os.path import isdir
from local_paths import ingestion_program_path, input_dir_path

from sys import path
from tqdm import tqdm

# Append python path to import data manager
program_dir = ingestion_program_path 
assert isdir(program_dir)
path.append(program_dir)
from data_manager import DataManager 


# Setip data manager
basename='task1_v4'
input_dir=input_dir_path
data_manager = DataManager(basename, input_dir)

# Grab training data
training_data = data_manager.load_training_data()

###############################################################################################
# Get All Training data
###############################################################################################

# Create the tensorflow dataloader, we are using a batch size of 1 to get all of the data
data_batches = training_data.batch(1, drop_remainder=True)

# Make lists to store the pytorch data
p_xs = []
p_ys = []

for data in tqdm(data_batches):

    # Convert the batch to be compatible with pytorch
    # This will require a transpose to convert from channels first to channels last, 
    # for the feature data, but for the target data
    batch = data[0]
    y = data[1]
    batchn = np.transpose(data[0],(0,3,1,2)).copy().astype('float32')
    batcht = torch.tensor(batchn,dtype=torch.float32,requires_grad=True)
    batch_p_y = copy(torch.tensor(
                                    np.asarray(y).copy().astype('float32'),
                                  dtype=torch.float32,requires_grad=True)
                    )

    batch_p_x = copy(batcht)

    p_xs.append(batch_p_x)
    p_ys.append(batch_p_y)

# Concatenate the lists to create full torch arrays
X = torch.cat(p_xs, axis=0)
y = torch.cat(p_ys, axis=0)

# Create the directory to store the test data
makedirs('./data/test/', exist_ok=True)

# Put data into a torch tensor and then save
torch.save((X,y),'./data/test/pytorch_test_data_1_v4.pt')
