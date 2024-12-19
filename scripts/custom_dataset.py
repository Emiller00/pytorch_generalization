import torch
from torch.utils.data import Dataset

class CustomDatasetInMemory(Dataset):
    def __init__(self, data_path,map_location=None):
        self.data_path = data_path
        if map_location:
            self.X, self.y = torch.load(data_path,map_location=map_location)
            print ('Loading data to {}'.format(map_location))
        else:
            self.X, self.y = torch.load(data_path)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        #return self.X[idx].requires_grad_(), self.y[idx]
        return self.X[idx], self.y[idx]
