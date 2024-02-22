import torch
from torch.utils import data


class MShapeDataset(data.Dataset):
    def __init__(self, train_pairs):
        self.train_pairs = train_pairs

    def __getitem__(self, index):
        anchor = torch.tensor(self.train_pairs[index][0])
        negative = torch.tensor(self.train_pairs[index][1], dtype=torch.float32)
        positive = torch.tensor(self.train_pairs[index][2], dtype=torch.float32)
        return anchor, negative, positive

    def __len__(self):
        return len(self.train_pairs)
