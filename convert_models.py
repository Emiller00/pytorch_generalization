from sys import path
from local_paths import ingestion_program_path, input_dir_path
from os import makedirs
from os.path import isdir

program_dir =ingestion_program_path

assert isdir(program_dir)
path.append(program_dir)

from data_manager import DataManager 

from scripts.ct_model import CTModel
from tqdm import tqdm

import torch

# Setip data manager

basename='task1_v4'

input_dir= input_dir_path

data_manager = DataManager(basename, input_dir)

# Grab training data

training_data = data_manager.load_training_data()

# Make sure you have the correct  folder
 
makedirs('pt_models', exist_ok=True)

for model_name in tqdm(data_manager.model_ids[:20]):
    
    model = data_manager.load_model(model_name)

    ct_model = CTModel(input_model=model,data_type='float32')

    ct_model.to('cpu')

    torch.save(ct_model,'./pt_models/{}.pth'.format(model_name))
