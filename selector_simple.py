import logging
import torch
from common import sort_index
from selector_def import Selector

class UncertainPredScoreSelector(Selector):
    def __init__(self, classifier):
        self.classifier = classifier
        self.sample_indices = None
        return
    
    def select_samples(self, args, X_test, y_test_pred, total_count, \
                    adaptive = False, \
                    y_test = None, \
                    total_wrong = 0, \
                    max_count = None):
        # offset indicates where to start in test samples.
        # we will sort prediction scores of all test samples
        offset = 0
        self.sample_indices = []
        if args.classifier not in ['svm', 'gbdt']:
            # e.g., 'mlp' and other neural network models:
            self.classifier = self.classifier.cuda()
            X_test_tensor = torch.from_numpy(X_test).float().cuda()
            pred_scores = self.classifier.predict_proba(X_test_tensor)[:, 1].cpu().detach().numpy()
        elif args.classifier == 'svm':
            pred_scores = self.classifier.predict_proba(X_test)[:, 1]
        else:
            # 'gbdt':
            pred_scores = self.classifier.predict_proba(X_test)
        # sort abs(score-0.5) from smallest to largest
        sorted_conf = sort_index(pred_scores, offset)
        self.sample_scores = {index: 0.5-val for index, val in sorted_conf}
        if adaptive == False:
            if len(sorted_conf) >= total_count:
                self.sample_indices = [x[0] for x in sorted_conf[:total_count]]
            else:
                self.sample_indices = [x[0] for x in sorted_conf]
        else:
            sample_cnt = 0
            wrong_cnt = 0
            for idx, score in sorted_conf:
                self.sample_indices.append(idx)
                sample_cnt += 1
                if y_test_pred[idx] != y_test[idx]:
                    wrong_cnt += 1
                if wrong_cnt == total_wrong or sample_cnt == max_count:
                    break
        logging.info('Added %s uncertain samples...' % (len(self.sample_indices)))
        return self.sample_indices, self.sample_scores
    
    def cluster_and_print(self, **kwargs):
        return super().cluster_and_print(**kwargs)

class MultiUncertainPredScoreSelector(Selector):
    def __init__(self, classifier):
        self.classifier = classifier
        self.sample_indices = None
        return
    
    def select_samples(self, args, X_test, y_test_pred, total_count, \
                    adaptive = False, \
                    y_test = None, \
                    total_wrong = 0, \
                    max_count = None):
        # offset indicates where to start in test samples.
        # we will sort prediction scores of all test samples
        offset = 0
        self.sample_indices = []
        if args.classifier not in ['svm', 'gbdt']:
            # e.g., 'mlp' and other neural network models:
            self.classifier = self.classifier.cuda()
            X_test_tensor = torch.from_numpy(X_test).float().cuda()
            # shape: (n_samples, n_classes)
            pred_scores = self.classifier.predict_proba(X_test_tensor).cpu().detach().numpy()
        elif args.classifier == 'svm':
            pred_scores = self.classifier.predict_proba(X_test)
        else:
            # 'gbdt', not implemented yet
            raise Exception('Multi-class uncertainty sample selector for GBDT model not implemented yet.')
        
        # get max prediction probability for each sample
        pred_scores = pred_scores.max(axis=1)
        # uncertainty scores is 1 - max prediction probability
        # sort 1 - pred_scores from largest to smallest and get indices
        unc_scores = sorted([(1.0 - score, idx) for idx, score in enumerate(pred_scores)], reverse=True)
        self.sample_scores = {index: val for val, index in unc_scores[:total_count]}
        self.sample_indices = [x[1] for x in unc_scores[:total_count]]
        logging.info('Added %s uncertain samples...' % (len(self.sample_indices)))
        return self.sample_indices, self.sample_scores
    
    def cluster_and_print(self, **kwargs):
        return super().cluster_and_print(**kwargs)