import numpy as np
import torch
import logging
import multiprocessing as mp

from sklearn.model_selection import StratifiedKFold
from scipy import sparse
from selector_def import Selector
from transcend import calibration
from transcend import scores

def parallel_ice(folds_gen): 
    X_train, y_train = folds_gen['X_train'], folds_gen['y_train']
    X_cal, y_cal = folds_gen['X_cal'], folds_gen['y_cal']
    reg, crit = folds_gen['reg'], folds_gen['crit'] 
    idx = folds_gen['idx'] 
    
    logging.info('[{}] Starting calibration...'.format(idx))  

    cal_results_dict = calibration.train_calibration_ice( 
            X_proper_train = X_train,
            X_cal = X_cal,
            y_proper_train = y_train,
            y_cal = y_cal,
            fold_index = 'cce_{}'.format(idx),
            reg = reg
    ) 

    ncms_cal = cal_results_dict['ncms_cal']
    svm = cal_results_dict['model']
    X_test = folds_gen['X_test']
    
    pred_test = svm.predict(X_test)
    ncms_test = scores.get_svm_ncms(svm, X_test, pred_test)
    p_val_test_dict = scores.compute_p_values_cred_and_conf(
        train_ncms=ncms_cal,
        groundtruth_train=y_cal,
        test_ncms=ncms_test,
        y_test=pred_test,
        cred_only=True if crit == 'cred' else False)
    
    if crit == 'cred':
        result_arr = np.array(p_val_test_dict['cred'])
    elif crit == 'cred+conf':
        cred_arr = np.array(p_val_test_dict['cred'])
        conf_arr = np.array(p_val_test_dict['conf'])
        result_arr = np.multiply(cred_arr, conf_arr)
    else:
        raise ValueError('Unknown value for args.c')
    
    return result_arr

class TranscendSelector(Selector):
    def __init__(self, encoder, crit="cred+conf"):
        self.encoder = encoder
        self.z_train = None
        self.z_test = None
        self.y_train = None
        self.y_test = None
        self.crit = crit
        return
    
    def select_samples(self, X_train, y_train, \
                    X_test, \
                    budget, reg=1):
        self.y_train = y_train
        X_train_tensor = torch.from_numpy(X_train).float().cuda()
        z_train = self.encoder.encode(X_train_tensor)
        z_train = z_train.cpu().detach().numpy()
        self.z_train = z_train
        X_test_tensor = torch.from_numpy(X_test).float().cuda()
        z_test = self.encoder.encode(X_test_tensor)
        z_test = z_test.cpu().detach().numpy()
        self.z_test = z_test

        # convert z_train, z_test to scipy sparse matrix, y_train with only 0 and 1 label
        y_train_binary =  np.array([1 if item != 0 else 0 for item in y_train])
        z_train_sparse = [sparse.lil_matrix(z_train[i]) for i in range(z_train.shape[0])]
        z_train_sparse = sparse.vstack(z_train_sparse, format='csr')
        z_test_sparse = [sparse.lil_matrix(z_test[i]) for i in range(z_test.shape[0])]
        z_test_sparse = sparse.vstack(z_test_sparse, format='csr')
        
        logging.info('Calculating p vals for test data')
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=21) 
        folds_generator = ({ 
            'X_train': z_train_sparse[train_index],
            'y_train': y_train_binary[train_index],
            'X_cal': z_train_sparse[cal_index],
            'y_cal': y_train_binary[cal_index],
            'X_test': z_test_sparse,
            'reg': reg,
            'crit': self.crit,
            'idx': idx
        } for idx, (train_index, cal_index) in enumerate(skf.split(z_train_sparse, y_train_binary)))
        
        NCPU = mp.cpu_count() - 4 if mp.cpu_count() > 2 else 1
        pval_lst = []
        with mp.Pool(processes=NCPU) as p: 
            for res in p.imap(parallel_ice, folds_generator): 
                if res is not None:
                    pval_lst.append(res)
                else: 
                    raise RuntimeError('CCE response is empty')
        
        pval_arr = np.array(pval_lst)
        result_arr = np.median(pval_arr, axis=0)
        idx = np.argpartition(result_arr, budget)
        self.sample_indices = idx[:budget]
        sample_scores = list(result_arr)
        self.sample_indices = list(self.sample_indices)
        
        return self.sample_indices, sample_scores