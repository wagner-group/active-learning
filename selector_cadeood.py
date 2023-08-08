from selector_def import Selector
import torch
import logging
import numpy as np

def safe_division(x, y):
    if abs(y) < 1e-8:
        y = 1e-8
    return x / y

def get_latent_data_for_each_family(z_train, y_train):
    N_lst = list(np.unique(y_train))
    N = len(N_lst)
    N_family = [len(np.where(y_train == family)[0]) for family in N_lst]
    z_family = []
    for family in N_lst:
        z_tmp = z_train[np.where(y_train == family)[0]]
        z_family.append(z_tmp)

    z_len = [len(z_family[i]) for i in range(N)]

    return N, N_family, z_family, z_len


def get_latent_distance_between_sample_and_centroid(z_family, centroids, N, N_family):
    dis_family = []  # two-dimension list

    for i in range(N): # i: family index
        dis = [np.linalg.norm(z_family[i][j] - centroids[i]) for j in range(N_family[i])]
        dis_family.append(dis)

    dis_len = [len(dis_family[i]) for i in range(N)]

    return dis_family, dis_len


def get_MAD_for_each_family(dis_family, N, N_family):
    mad_family = []
    for i in range(N):
        median = np.median(dis_family[i])
        # print(f'family {i} median: {median}')
        diff_list = [np.abs(dis_family[i][j] - median) for j in range(N_family[i])]
        mad = 1.4826 * np.median(diff_list)  # 1.4826: assuming the underlying distribution is Gaussian
        mad_family.append(mad)
    # print(f'mad_family: {mad_family}')

    return mad_family

def detect_drift_samples_top(z_train, z_test, y_train):

    '''get latent data for each family in the training set'''
    N, N_family, z_family, z_len = get_latent_data_for_each_family(z_train, y_train)

    '''get centroid for each family in the latent space'''
    centroids = [np.mean(z_family[i], axis=0) for i in range(N)]
    # print(f'centroids: {centroids}')

    '''get distance between each training sample and their family's centroid in the latent space '''
    dis_family, dis_len = get_latent_distance_between_sample_and_centroid(z_family, centroids,
                                                                    N, N_family)
    assert z_len == dis_len, "training family stats size mismatches distance family stats size"
    '''get the MAD for each family'''
    mad_family = get_MAD_for_each_family(dis_family, N, N_family)

    ### return samples sorted by OOD scores
    '''detect drifting in the testing set'''
    ood_scores = []
    for k in range(len(z_test)):
        z_k = z_test[k]
        '''get distance between each testing sample and each centroid'''
        dis_k = [np.linalg.norm(z_k - centroids[i]) for i in range(N)]
        anomaly_k = [safe_division(np.abs(dis_k[i] - np.median(dis_family[i])), mad_family[i]) for i in range(N)]
        # print(f'sample-{k} - dis_k: {dis_k}')
        # print(f'sample-{k} - anomaly_k: {anomaly_k}')
        min_anomaly_score = np.min(anomaly_k)
        ood_scores.append((k, min_anomaly_score))
    return ood_scores

class OODSelector(Selector):
    def __init__(self, encoder):
        self.encoder = encoder
        self.z_train = None
        self.z_test = None
        self.y_train = None
        self.y_test = None
        return
    
    def select_samples(self, X_train, y_train, \
                    X_test, \
                    budget):
        # Is y_train already binary? No
        self.y_train = y_train
        X_train_tensor = torch.from_numpy(X_train).float().cuda()
        z_train = self.encoder.encode(X_train_tensor)
        z_train = z_train.cpu().detach().numpy()
        self.z_train = z_train
        X_test_tensor = torch.from_numpy(X_test).float().cuda()
        z_test = self.encoder.encode(X_test_tensor)
        z_test = z_test.cpu().detach().numpy()
        self.z_test = z_test

        ood_scores = detect_drift_samples_top(self.z_train, self.z_test, self.y_train)
        sample_scores = [item[1] for item in ood_scores]
        ood_scores.sort(reverse=True, key=lambda x: x[1])
        self.sample_indices = [item[0] for item in ood_scores[:budget]]
        print(ood_scores[:50])
        return self.sample_indices, sample_scores