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
    Compute euclidean distance square between two tensors
    Params:
    - x
        Tensor x.shape = N x D, contains N vectors of length D
    - y
        Tensor y.shape = M x D, contains M vectors of length D
    Returns:
    - dist
        Tensor dist.shape = N x M. Each element of 'dist', i.e. dist[i][j], equals to
        the euclidiean distance square of vector x_input[i] and vector y_input[j]
    '''

    n = x.size(dim=0)
    m=y.size(dim=0)
    d=x.size(dim=1)
    if d!= y.size(1):
        raise Exception
    
    x=x.unsqueeze(1).expand(n,m,d)
    y=y.unsqueeze(0).expand(n,m,d)

    dist = torch.pow(x-y,2).sum(2)
    return dist


def prototypical_loss(input, target, n_support):
    # input=model_output
    # target=y
    target_cpu=target.to('cpu')
    input_cpu=input.to('cpu')

    def supp_idxs(c):
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(dim=1)
    
    classes = torch.unique(target_cpu)
    n_classes=len(classes)

    # assuming n_query, n_target constants
    n_query=target_cpu.eq(classes[0].item()).sum().item() - n_support

    support_idxs=list(map(supp_idxs, classes))

    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)

    query_samples = input.to('cpu')[query_idxs]
    dists=euclidean_dist(query_samples, prototypes)

    log_p_y=F.log_softmax(-dists,dim=1).view(n_classes,n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat=log_p_y.max(2)
    acc_val=y_hat.eq(target_inds.squeeze(2)).float().mean()

    return loss_val, acc_val