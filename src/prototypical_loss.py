import torch.nn as nn
import torch.nn.functional as F
import torch

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
    
    # target is made up of 'n_classes x (n_support+n_query)' labels
    classes = torch.unique(target)
    n_classes=len(classes)

    # 'task_idxs' is a matrix of shape 'n_classes x (n_support+n_query)'
    # row i of 'task_idxs' correspond to all the coordinates of samples of class i
    task_idxs = torch.stack(list(map(lambda c: torch.argwhere(target == c).squeeze(),classes)), dim=0)
    n_query = task_idxs.shape[1] - n_support

    # support_idxs[i] are all the coordinates of all support labels(of class[i]) in 'target' tensor
    # so as query_idxs[i]
    support_idxs, query_idxs = task_idxs.split([n_support, n_query], dim=1)
    support_samples = input[support_idxs]
    query_samples = input[query_idxs]

    # prototypes[i] are the average vector across all support samples' latent features(of class[i])
    prototypes = support_samples.mean(dim=1)
    
    # dists[i][j][k] means the euclidean distance between
    # the jth query sample of class[i] and the prototype of class[k]
    dists=euclidean_dist(query_samples.view(n_classes*n_query,-1), prototypes).view(n_classes, n_query, n_classes)

    # code below euqals to the following pseudocode:
    # for c in all_classes:
    #   for q in query_set_of_class_c:
    #       computes softmax of q over all_prototypes
    log_p_y=F.log_softmax(-dists,dim=2)

    # according to the paper, loss can be computed as:
    # for c in all_classes:
    #   for q in query_set_of_class_c:
    #       loss += softmax probability of dist(q, prototype_of_class_c)
    #       which equals to log_p_y[c][q][c]
    # loss /= (n_classes+n_query)
    target_inds = torch.arange(0, n_classes).to(target.device)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(dim=2, index=target_inds).squeeze().view(-1).mean()
    
    # y_hat has the same shape of query_samples
    _, y_hat=log_p_y.max(2)
    acc_val=y_hat.eq(target_inds.squeeze(2)).float().mean()

    return loss_val, acc_val
