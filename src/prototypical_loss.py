import torch.nn as nn
import torch.nn.functional as F
import torch

class PrototypicalLoss(nn.Module):
    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support=n_support

    def forward(self, input, target):
        return prototypical_loss(input, target, self.n_support)

def euclidean_dist(x,y):
    '''
    Compute euclidean distance between two tensors
    '''
    n = x.size(dim=0)
    m=y.size(dim=0)
    d=x.size(dim=1)
    if d!= y.size(1):
        raise Exception
    
    x=x.unsqueeze(1).expand(n,m,d) # ?
    y=y.unsqueeze(0).expand(n,m,d)

    return torch.pow(x-y,2).sum(2)

def prototypical_loss(input, target, n_support):