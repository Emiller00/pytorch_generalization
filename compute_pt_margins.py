from local_paths import ingestion_program_path, input_dir_path

from os import listdir, makedirs
from os.path import join
from sys import path

from time import time

import argparse
import pickle as pkl
import torch

from collections import defaultdict
from copy import copy

from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.ct_model import CTModel
from scripts.custom_dataset import CustomDatasetInMemory
from scripts.margin_dist_tools import get_margin_distribution, get_normalized_margin

# Get relative paths
data_path = "./data/test/pytorch_test_data_1_v4.pt"
pt_models_path = "./pt_models"
output_path = "./data/margins/pytorch"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_data_loader(data_path, device="cpu", batch_size=64, num_workers=0, pin_memory=False):
    '''
    This is a function to instantiate a dataloader. 

    Arguments:

        data_path <string>: The path to the pytorch tensor with the data we intend to load into the dataloader. 

        device <string>: The device where the data will be loaded to.

        batch_size <int>: The size of each batch, when the dataloader is run as an interator

        num_workers <int>: The number of workers, if the data is loaded directly to the GPU, this should be 0

        pin_memory <bool>: A flag to determine if we will pin the memory or not
    '''


    # Instantiate a simple custom dataset
    dataset = CustomDatasetInMemory(data_path, map_location=device)

    # Create a DataLoader to load the entire dataset
    # Input the parameters, while deferring to their defaults, if necessary
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return dataloader


def main():

    # Parse Arguments
    parser = argparse.ArgumentParser(description="Perform operations on a list of numbers.")

    parser.add_argument("--num", "-n", default=0)
    parser.add_argument("--batch_size", "-b", default=64)
    parser.add_argument("--workers", "-w", default=0, type=int)
    parser.add_argument("--pin_memory", action="store_true")

    args = parser.parse_args()

    # Setup initial variabls
    # These will most likely not change
    basename = "task1_v4"
    input_dir = input_dir_path
    batch_size = args.batch_size
    n = int(args.num)

    if args.pin_memory:
        pin_memory = True
    else:
        pin_memory = False

    # Get the list of available models
    # From this list, get the model we want to run
    model_list = [i.replace(".pth", "") for i in listdir(pt_models_path)]

    if n != -1:
        model_name = model_list[n]
    else:
        model_name = model_list[0]

    print("Model {} of {} is {} ".format(str(n), str(len(model_list)), model_name))

    model_path = "./pt_models/{}.pth".format(model_name)

    ct_model = torch.load(model_path)

    ct_model = ct_model.to(device)

    # Define total noramalized margin
    all_normalized_margins = defaultdict(list)

    # Set up the dataloader, and use it to compute the nornalized margin distribution
    print(" running model {} with {} workers".format(model_name, args.workers))

    data_loader = get_data_loader(
        data_path=data_path, device=device, batch_size=64, num_workers=args.workers, pin_memory=pin_memory
    )


    for n, (X, y) in tqdm(enumerate(data_loader)):

        # make sure every batch is the correct batch size
        if X.shape[0] == batch_size:
            X.requires_grad_()
            y.requires_grad_()

            X = X.to(device)
            y = y.to(device)

            margin_dist, int_out = get_margin_distribution(model=ct_model, X=X, y=y, margin_length=4)

            normalized_margin = {k: i for k, i in enumerate(get_normalized_margin(md=margin_dist, int_out=int_out))}

            for k, v in normalized_margin.items():
                all_normalized_margins[k].append(v)

    makedirs(output_path, exist_ok=True)
    pkl.dump(all_normalized_margins, open(join(output_path, "{}.pkl".format(model_name)), "wb"))

if __name__ == "__main__":
    main()
