import logging
import torch
import numpy as np
from collections import Counter
from operator import itemgetter
from torch.utils.data.sampler import Sampler

from pytorch_metric_learning.utils import common_functions as c_f

from common import sort_based_on_labels

def process_label_to_index(all_labels):
    label_idx, not_label_idx = {}, {}
    for label in np.unique(all_labels):
        label_idx[label] = np.where(all_labels == label)[0]
        not_label_idx[label] = np.where(all_labels != label)[0] 
    return label_idx, not_label_idx

class ProportionalClassSampler(Sampler):
    """
    At every iteration, this will return at least two samples per class.
    The additional samples for each class are proportional to how often
    they appear in the training set.
    """

    def __init__(self, labels, m, batch_size):
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        self.min_per_class = int(m)
        self.batch_size = int(batch_size) if batch_size is not None else batch_size
        self.labels_to_indices = c_f.get_labels_to_indices(labels)
        self.all_indices = np.arange(len(labels))
        self.unique_labels = list(self.labels_to_indices.keys())
        self.label_counts = np.unique(labels, return_counts=True)
        self.length_of_single_pass = self.min_per_class * len(self.unique_labels)
        self.batch_num = len(labels) // (self.batch_size - self.length_of_single_pass) + 1
        self.list_size = len(labels) + self.length_of_single_pass * self.batch_num

        assert self.list_size >= self.batch_size
        assert (
            self.length_of_single_pass <= self.batch_size
        ), "m * (number of unique labels) must be <= batch_size"

    def __len__(self):
        return self.list_size

    def __iter__(self):
        idx_list = [0] * self.list_size
        i = 0
        num_iters = self.calculate_num_iters()

        # for the reamining, we sample without replacement.
        n = len(self.all_indices)
        perm_indices = torch.randperm(n).tolist()
        k = 0
        for bcnt in range(num_iters):
            c_f.NUMPY_RANDOM.shuffle(self.unique_labels)
            for label in self.unique_labels:
                t = self.labels_to_indices[label]
                idx_list[i : i + self.min_per_class] = c_f.safe_random_choice(
                    t, size=self.min_per_class
                )
                i += self.min_per_class
            # sample remaining based on how frequent the label appears in the training set
            if bcnt < num_iters - 1:
                remain_size = self.batch_size - i % self.batch_size
            else:
                # last batch
                remain_size = self.list_size - i
            # for the reamining, we sample without replacement.
            idx_list[i : ] = perm_indices[k : k + remain_size]
            # idx_list[i : ] = c_f.safe_random_choice(self.all_indices, size=remain_size)
            i += remain_size
            k += remain_size

        return iter(idx_list)

    def calculate_num_iters(self):
        return self.batch_num

class HalfSampler(Sampler):
    """
    At every iteration, this will first sample half of the batch, and then
    fill the other half of the batch with the same label distribution.
    batch_size must be an even number
    """
    def __init__(self, labels, batch_size, upsample = None):
        assert (batch_size % 2 == 0), "batch_size must be an even number"

        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        self.batch_size = int(batch_size)
        self.index_to_label = labels
        self.labels_to_indices = c_f.get_labels_to_indices(labels)
        # upsampling the indices
        self.upsample = upsample
        if upsample is not None:
            # duplicate samples according to the counts
            self.all_indices = np.repeat(np.arange(len(labels)), upsample)
        else:
            self.all_indices = np.arange(len(labels))
        # sample half of the batch_size as self.length_of_single_pass
        self.length_of_single_pass = self.batch_size // 2
        self.batch_num = len(self.all_indices) // self.length_of_single_pass + 1
        self.list_size = len(self.all_indices) * 2

        assert self.list_size >= self.batch_size
    
    def __len__(self):
        return self.list_size
    
    def __iter__(self):
        idx_list = [0] * self.list_size
        num_iters = self.calculate_num_iters()
        
        # one pass of training data with size n
        n = len(self.all_indices)
        # self.all_indices may have repeated items after upsampling
        indices = torch.randperm(n).numpy()
        perm_indices = self.all_indices[indices]
        i = 0 # index the idx_list
        k = 0 # index the perm_indices
        for bcnt in range(num_iters):
            if bcnt < num_iters - 1:
                step =  self.length_of_single_pass
            else:
                step = len(self.index_to_label) % self.length_of_single_pass
            half_batch_indices = perm_indices[k: k + step]
            k += step
            idx_list[i : i + step] = half_batch_indices
            i += step
            # sample the other half with the same label distribution
            label_counts = Counter(self.index_to_label[half_batch_indices])
            for label, count in label_counts.items():
                t = self.labels_to_indices[label]
                idx_list[i : i + count] = c_f.safe_random_choice(
                    t, size=count
                )
                i += count
        return iter(idx_list)

    def calculate_num_iters(self):
        return self.batch_num

class TripletSampler(Sampler):
    """
    At every iteration, this will first sample half of the batch, and then
    fill the other half of the batch with the same label distribution.
    batch_size must be an even number
    """
    def __init__(self, labels, batch_size):
        assert (batch_size % 3 == 0), "batch_size must be dividable by three"

        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        self.batch_size = int(batch_size)
        self.index_to_label = labels
        self.labels_to_indices, self.not_labels_to_indices = process_label_to_index(labels)
        self.unique_labels = list(self.labels_to_indices.keys())
        self.all_indices = np.arange(len(labels))
        # sample half of the batch_size as self.length_of_single_pass
        self.length_of_single_pass = self.batch_size // 3
        self.batch_num = len(labels) // self.length_of_single_pass + 1
        self.list_size = len(labels) * 3

        assert self.list_size >= self.batch_size
    
    def __len__(self):
        return self.list_size
    
    def __iter__(self):
        idx_list = [0] * self.list_size
        num_iters = self.calculate_num_iters()
        #logging.debug(f'self.list_size {self.list_size}')
        # one pass of training data with size n
        n = len(self.all_indices)
        perm_indices = torch.randperm(n).numpy()
        i = 0 # index the idx_list
        k = 0 # index the perm_indices
        for bcnt in range(num_iters):
            if bcnt < num_iters - 1:
                step =  self.length_of_single_pass
            else:
                step = len(self.index_to_label) % self.length_of_single_pass
            start_batch_indices = perm_indices[k: k + step]
            k += step
            start_batch_labels = self.index_to_label[start_batch_indices]
            ### labels from smallest to the largest
            start_batch_indices, start_batch_labels = sort_based_on_labels(start_batch_indices, start_batch_labels)
            #logging.debug(f'start_batch_labels {start_batch_labels}')
            idx_list[i : i + step] = start_batch_indices
            i += step
            #logging.debug(f'i = {i}')
            #logging.debug(f'len(start_batch_labels) {len(start_batch_labels)}')
            
            # sample the second share with the same label distribution
            label_counts = sorted(Counter(self.index_to_label[start_batch_indices]).items(), key=itemgetter(0))
            # label counts sort labels from smallest to the largest
            for label, count in label_counts:
                t = self.labels_to_indices[label]
                idx_list[i : i + count] = c_f.safe_random_choice(t, size=count)
                i += count
            # sample the last share with different labels
            for anchor_label, count in label_counts:
                negative_list = self.not_labels_to_indices[anchor_label]
                negative = c_f.safe_random_choice(negative_list, size=count)
                idx_list[i : i + count] = negative
                i += count
            #logging.debug(f'bcnt: {bcnt}')
        return iter(idx_list)

    def calculate_num_iters(self):
        return self.batch_num
