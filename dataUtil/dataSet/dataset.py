import torch
from torch.utils import data


class TSCDataset(data.Dataset):
    def __init__(self, data_instances):
        self.data = data_instances

    def __getitem__(self, index):
        X = []
        Y = []
        for seq, label in self.data:
            X.append(seq)
            Y.append(label)
        return torch.tensor(X[index]), torch.tensor(Y[index])

    def __len__(self):
        return len(self.data)

    def dim(self):
        return self.data[0][0].shape[1]
