import logging
import operator
import time
import torch
import numpy as np
from collections import Counter, defaultdict
from sklearn import neighbors
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import v_measure_score
from scipy.spatial import KDTree

from common import to_categorical
from losses import HiDistanceXentLoss
from selector_def import Selector
from util import AverageMeter
from train import pseudo_loss

class LocalPseudoLossSelector(Selector):
    def __init__(self, encoder):
        self.encoder = encoder
        self.z_train = None
        self.z_test = None
        self.y_train = None
        return
    
    def select_samples(self, args, X_train, y_train, y_train_binary, \
                    X_test, y_test_pred, \
                    total_epochs, \
                    test_offset, \
                    all_test_family, \
                    total_count, \
                    y_test = None):
        X_train_tensor = torch.from_numpy(X_train).float().cuda()
        z_train = self.encoder.encode(X_train_tensor)
        logging.info(f'Normalizing z_train to unit length...')
        z_train = torch.nn.functional.normalize(z_train)
        z_train = z_train.cpu().detach().numpy()

        X_test_tensor = torch.from_numpy(X_test).float().cuda()
        z_test = self.encoder.encode(X_test_tensor)
        logging.info(f'Normalizing z_test to unit length...')
        z_test = torch.nn.functional.normalize(z_test)
        z_test = z_test.cpu().detach().numpy()
        
        self.z_train = z_train
        self.z_test = z_test
        self.y_train = y_train

        self.sample_indices = []
        self.sample_scores = []
        
        # build the KDTree
        logging.info(f'Building KDTree...')
        tree = KDTree(z_train)
        logging.info(f'Querying KDTree...')
        # query all z_test up to a margin
        all_neighbors = tree.query(z_test, k=z_train.shape[0], workers=8)
        logging.info(f'Finished querying KDTree...')
        all_distances, all_indices = all_neighbors

        # each batch is to get one loss for one test sample
        batch_time = AverageMeter()
        data_time = AverageMeter()
        end = time.time()
        if args.plb == None:
            bsize = args.bsize
        else:
            bsize = args.plb
        # nn_loss = np.zeros([sample_num])
        sample_num = z_test.shape[0]
        for i in range(sample_num):
            test_sample = X_test_tensor[i:i+1] # on GPU
            # bsize-1 nearest neighbors of the test sample i
            batch_indices = all_indices[i][:bsize-1]
            # x_batch
            x_train_batch = X_train_tensor[batch_indices] # on GPU
            x_batch = torch.cat((test_sample, x_train_batch), 0)
            # y_batch
            y_train_batch = y_train_binary[batch_indices]
            y_batch_np = np.hstack((y_test_pred[i], y_train_batch))
            y_batch = torch.from_numpy(y_batch_np).cuda()
            # y_bin_batch
            y_bin_batch = torch.from_numpy(to_categorical(y_batch_np, num_classes=2)).float().cuda()
            # we don't need split_tensor. all samples are training samples
            # split_tensor = torch.zeros(x_batch.shape[0]).int().cuda()
            # split_tensor[test_offset:] = 1
            
            data_time.update(time.time() - end)

            # in the loss function, y_bin_batch is the categorical version
            # call the loss function once for every test sample
            if args.loss_func == 'hi-dist-xent':
                _, features, y_pred = self.encoder(x_batch)
                HiDistanceXent = HiDistanceXentLoss(reduce = args.reduce).cuda()
                loss, _, _ = HiDistanceXent(args.xent_lambda, 
                                        y_pred, y_bin_batch,
                                        features, labels=y_batch,
                                        margin = args.margin)
                loss = loss.to('cpu').detach().numpy()
            else:
                # other loss functions pending
                raise Exception(f'local pseudo loss for {args.loss_func} not implemented.')
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # update the loss values for i
            # nn_loss[i] = loss[0]
            self.sample_scores.append(loss[0])

            # only display the test samples
            if (i + 1) % (args.display_interval * 3) == 0:
                logging.debug('Train + Test: [0][{0}/{1}]\t'
                    'BT {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'DT {data_time.val:.3f} ({data_time.avg:.3f})  '
                    'i {i} loss {l}'.format(
                    i + 1, sample_num, batch_time=batch_time,
                    data_time=data_time, i=i, l=loss[0]))
        
        sorted_sample_scores = list(sorted(list(enumerate(self.sample_scores)), key=lambda item: item[1], reverse=True))
        logging.info(f'sorted_sample_scores[:100]: {sorted_sample_scores[:100]}')
        sample_cnt = 0
        for idx, score in sorted_sample_scores:
            logging.info('Sample glb idx: %d, pred: %s, true: %s, ' \
                'score: %.2f\n' % \
                (test_offset+idx, y_test_pred[idx], all_test_family[idx], \
                score))
            self.sample_indices.append(idx)
            sample_cnt += 1
            if sample_cnt == total_count:
                break
        logging.info('Added %s samples...' % (len(self.sample_indices)))
        return self.sample_indices, self.sample_scores

