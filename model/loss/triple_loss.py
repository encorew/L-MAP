# define triplet loss function
from torch import nn
import torch.nn.functional as F


class TripleLoss(nn.Module):
    def __init__(self):
        super(TripleLoss, self).__init__()
        self.margin = 0.2

    def forward(self, anchor, positive, negative):
        pos_dist = (anchor - positive).pow(2).sum(1)
        neg_dist = (anchor - negative).pow(2).sum(1)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()
        # we can also use #torch.nn.functional.pairwise_distance(anchor,positive, keep_dims=True),
        # which computes the euclidean distance.
