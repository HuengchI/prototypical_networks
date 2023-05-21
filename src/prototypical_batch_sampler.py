import numpy as np
import torch

class PrototypicalBatchSampler(object):
    '''
    To yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples'.
    In fact at every iteration the batch indexes will refer to 'num_support' + 'num_query' samples for 'classes_per_it' random classes.

    __len__ returns the number of episodes per epoch(same as 'self.iterations')
    '''
    def __init__(self, labels, classes_per_it, num_samples, iterations):
        '''
        Args:
        - labels: an iterable containing all the labels for the current dataset samples indexes will be infered from this iterable.
        - iterations: number of episodes per epoch
        - num_samples: number of samples for each episodes for each class (support + query)
        '''
        super(PrototypicalBatchSampler, self).__init__()
        self.labels=labels
        self.classes_per_it=classes_per_it
        self.sample_per_class=num_samples
        self.iterations=iterations

        self.classes, self.counts=np.unique(self.labels, return_counts=True)
        self.classes = torch.tensor(self.classes, dtype=torch.int64)

        # create a matrix, indexes, of dim: classes X max(elements per class)
        # fill it with nans
        # for every class c, fill the relative row with the indices samples belonging to c
        # in numel_per_class we store the number of samples for each class/row

        # for omniglot dataset, there're 4122 classes and each class contains 20 samples
        # so the matrix size is 4122 X 20

        # creating such a matrix makes it easy to randomly sample batches with Pytorch API such as 'randperm'
        self.idxs=range(len(self.labels))
        self.indexes=np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.indexes = torch.tensor(self.indexes)
        self.numel_per_class = torch.zeros_like(self.classes)
        for idx, label in enumerate(self.labels):
            label_idx = np.argwhere(self.classes == label).item()
            self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx
            self.numel_per_class[label_idx] += 1
        self.indexes.to(dtype=torch.int)

    def __iter__(self):
        '''
        yeild a batch(episode) of indexes
        '''
        spc = self.sample_per_class
        cpi=self.classes_per_it

        for it in range(self.iterations):
            batch_size = spc*cpi
            batch=torch.empty(batch_size,dtype=torch.int64)
            c_idxs = torch.randperm(len(self.classes))[:cpi]
            for i, c in enumerate(self.classes[c_idxs]): # iterating over the permutated classes list
                s=slice(i*spc,(i+1)*spc) # the covered length of s is ONE 'sample_per_class'

                label_idx=torch.arange(len(self.classes)).long()[self.classes==c].item() # find the class c's original id(w/o permutation)
                sample_idxs=torch.randperm(self.numel_per_class[label_idx])[:spc] # randomly choose 'spc' samples of class c
                batch[s]=self.indexes[label_idx][sample_idxs] # put the chosen 'spc' samples of class c to the (i*spc,(i+1)spc) postion of list 'batch'
            batch=batch[torch.randperm(len(batch))] # let samples from each class intersect with each other
            yield batch
    
    def __len__(self):
        '''
        returns the number of episodes per epoch
        '''
        return self.iterations
