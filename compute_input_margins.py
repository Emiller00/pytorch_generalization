from os import listdir, makedirs
from os.path import join
import argparse
import pickle as pkl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.custom_dataset import CustomDatasetInMemory
# Import routine for computing input margins following
# "Input margins can predict generalization too" by Mouton et al.
from scripts.input_margin_tools import get_input_margin

# Paths
data_path = "./data/test/pytorch_test_data_1_v4.pt"
pt_models_path = "./pt_models"
output_path = "./data/input_margins"


def get_data_loader(data_path, device="cpu", batch_size=64, num_workers=0, pin_memory=False):
    dataset = CustomDatasetInMemory(data_path, map_location=device)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return dataloader


def main():
    parser = argparse.ArgumentParser(description="Compute input margins for a model")
    parser.add_argument("--num", "-n", default=0)
    parser.add_argument("--batch_size", "-b", default=64)
    parser.add_argument("--workers", "-w", default=0, type=int)
    parser.add_argument("--pin_memory", action="store_true")
    args = parser.parse_args()

    batch_size = args.batch_size
    model_list = [i.replace(".pth", "") for i in listdir(pt_models_path)]
    n = int(args.num)

    if n != -1:
        model_name = model_list[n]
    else:
        model_name = model_list[0]

    model_path = join(pt_models_path, f"{model_name}.pth")
    ct_model = torch.load(model_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ct_model = ct_model.to(device)

    pin_memory = args.pin_memory

    data_loader = get_data_loader(
        data_path=data_path,
        device=device,
        batch_size=batch_size,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )

    all_margins = []
    for _, (X, y) in tqdm(enumerate(data_loader)):
        if X.shape[0] == batch_size:
            X = X.to(device).requires_grad_()
            y = y.to(device)
            margins = get_input_margin(model=ct_model, X=X, y=y)
            all_margins.extend(margins)

    makedirs(output_path, exist_ok=True)
    pkl.dump(all_margins, open(join(output_path, f"{model_name}.pkl"), "wb"))


if __name__ == "__main__":
    main()
