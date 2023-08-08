import abc
from collections import Counter, defaultdict
import logging
import numpy as np
import operator
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import v_measure_score

class Selector(object):

    @abc.abstractmethod
    def __init__(self, **kwargs):
        return

    @abc.abstractmethod
    def select_samples(self, **kwargs):
        # return the list of sample indices
        return None
    
    def cluster_and_print(self, fname, cur_month_str, \
                        all_train_family, train_ben_details, \
                        all_test_family, test_ben_details, \
                        y_test, y_test_binary, y_test_pred, \
                        test_offset):
        logging.info('Running KMeans Clustering...')    
        total_fam_num = len(set(all_train_family.tolist() + all_test_family.tolist())) + 5
        logging.info(f'total_fam_num: {total_fam_num}')
        all_z = np.concatenate((self.z_train, self.z_test), axis=0)
        kmeans = KMeans(n_clusters=total_fam_num, random_state=0).fit(all_z)
    
        # train test index separation
        # idx < test_offset ? train : test

        y_train_kmeans_pred = kmeans.predict(self.z_train)
        y_test_kmeans_pred = kmeans.predict(self.z_test)
        v_score = v_measure_score(np.concatenate((self.y_train, y_test), axis=0), np.concatenate((y_train_kmeans_pred, y_test_kmeans_pred), axis=0))
        logging.info('GMM all v measure score: \t%.4f\n' % v_score)
        
        # data index and family info for each mixture
        mid_train_idx = defaultdict(list)
        mid_train_info = defaultdict(list)
        mid_test_idx = defaultdict(list)
        mid_test_info = defaultdict(list)
        mid_idx = defaultdict(list) # global index
        mid_info = defaultdict(list)
        for idx, mid in enumerate(y_train_kmeans_pred):
            mid_train_idx[mid].append(idx)
            mid_train_info[mid].append(all_train_family[idx])
            mid_idx[mid].append(idx)
            mid_info[mid].append(all_train_family[idx])
        for idx, mid in enumerate(y_test_kmeans_pred):
            mid_test_idx[mid].append(idx)
            mid_test_info[mid].append(all_test_family[idx])
            mid_idx[mid].append(idx+test_offset) # global index
            mid_info[mid].append(all_test_family[idx])
        mid_size = {mid: len(item) for mid, item in mid_idx.items()}
        # compute purity
        # some mixtures may only have test data, so mid_train_idx[mid] does not exist
        mid_purity = []
        mid_train_percent = {}
        for mid, size in mid_size.items():
            try:
                total_train_cnt = float(len(mid_train_idx[mid]))
                most_common_cnt = Counter(mid_train_info[mid]).most_common()[0][1]
                purity = most_common_cnt / total_train_cnt
                mid_train_percent[mid] = total_train_cnt / float(size)
            except IndexError:
                # unknown purity
                purity = -1.0
                mid_train_percent[mid] = 0.0
            mid_purity.append((mid, purity))
        sorted_mid_purity = sorted(mid_purity, key=operator.itemgetter(1))
        logging.info(f'sorted_mid_purity: {sorted_mid_purity}')

        cluster_out = open(fname, 'a')
        cluster_out.write('\n======= Month %s =======\n' % cur_month_str)
        # print mixtures
        for mid, purity in sorted_mid_purity:
            test_indices = mid_test_idx[mid]
            fp = 0
            fn = 0
            for index in test_indices:
                if y_test_binary[index] == 0 and y_test_pred[index] == 1:
                    fp += 1
                if y_test_binary[index] == 1 and y_test_pred[index] == 0:
                    fn += 1
            # PRINT
            cluster_out.write('\n======= Mixture %d =======\n' % mid)
            cluster_out.write('####### Train Purity: %.4f\tTrain Percent %.4f\n' % (purity, mid_train_percent[mid]))
            cluster_out.write('####### Test FP: %d Test FN: %d\n' % (fp, fn))
            ### information about the entire mixture
            # 1) Total counts per family
            cluster_out.write('####### All Family Info %s\n' % Counter(mid_info[mid]).most_common())
            # 2) Train counts per familly
            cluster_out.write('####### Train Family Info %s\n' % Counter(mid_train_info[mid]).most_common())
            # 3) Test counts per family
            cluster_out.write('####### Test Family Info %s\n' % Counter(mid_test_info[mid]).most_common())
        
            ### information about individual data points
            # sort indices from closest to furthest
            mean = kmeans.cluster_centers_[mid]
            distances = [(idx, np.linalg.norm(all_z[idx]-mean)) for idx in mid_idx[mid]]
            sorted_distances = sorted(distances, key=operator.itemgetter(1), reverse=True)
            # for each sample
            cluster_out.write('\t_idx\tclsdist'\
                    '\tfamily\tpkgname\ttest\tnew\tselect\n')
            
            for idx, distance in sorted_distances:
                # only family needs a relative idx
                if idx < test_offset:
                    family = all_train_family[idx]
                    in_test = ''
                    if family == 'benign':
                        package_name = train_ben_details[idx]
                    else:
                        package_name = ''
                    wrong = ''
                    select = ''
                else:
                    family = all_test_family[idx-test_offset]
                    test_idx = idx - test_offset
                    # NOTE: this is the customized ood score, not CADE
                    in_test = '~'
                    if family == 'benign':
                        package_name = test_ben_details[test_idx]
                    else:
                        package_name = ''
                    if y_test_binary[test_idx] == y_test_pred[test_idx]:
                        wrong = ''
                    else:
                        wrong = 'X'
                    if test_idx in self.sample_indices:
                        select = '!!!'
                    else:
                        select = ''
                is_new = family not in all_train_family
                tag = '*' if is_new else ''
                # idx is in all_z, y_proba
                #cluster_out.write('\t_idx\tdist'\
                #'\tfamily\tscore\t1nn_idx\tpseudo\t2nn_idx\tcontrast\tpkgname\ttest\tnew\n')
                cluster_out.write('\t%d\t%.2f\t%s'\
                        '\t%s\t%s\t%s\t%s\t%s\n' % \
                        (idx, distance, family, package_name, in_test, tag, wrong, select))
                cluster_out.flush()
        
        cluster_out.close()
        return {}