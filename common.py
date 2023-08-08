#! /usr/bin/env python
import numpy as np
from sklearn.metrics import confusion_matrix

def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical

def f1_score(precision, recall):
    return 2*precision*recall/float(precision+recall)

def get_model_stats(y, y_pred, multi_class = False):
    if multi_class == False:
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        acc = (tp+tn)/float(tp+tn+fp+fn)
        fpr = fp/float(fp+tn)
        tpr = tp/float(tp+fn)
        tnr = tn/float(fp+tn)
        fnr = fn/float(fn+tp)
        precision = tp/float(tp+fp)
        recall = tp/float(tp+fn)
        return tpr, tnr, fpr, fnr, acc, precision, f1_score(precision, tpr)
    else:
        # https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
        cmatrix = confusion_matrix(y, y_pred)
        FP = cmatrix.sum(axis=0) - np.diag(cmatrix)
        FN = cmatrix.sum(axis=1) - np.diag(cmatrix)
        TP = np.diag(cmatrix)
        TN = cmatrix.sum() - (FP + FN + TP)

        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP/(TP+FN)
        TPR = TPR[~np.isnan(TPR)]
        # Specificity or true negative rate
        TNR = TN/(TN+FP) 
        TNR = TNR[~np.isnan(TNR)]
        # Precision or positive predictive value
        PPV = TP/(TP+FP)
        PPV = PPV[~np.isnan(PPV)]
        # Negative predictive value
        NPV = TN/(TN+FN)
        NPV = NPV[~np.isnan(NPV)]
        # Fall out or false positive rate
        FPR = FP/(FP+TN)
        FPR = FPR[~np.isnan(FPR)]
        # False negative rate
        FNR = FN/(TP+FN)
        FNR = FNR[~np.isnan(FNR)]
        # False discovery rate
        FDR = FP/(TP+FP)
        FDR = FDR[~np.isnan(FDR)]
        # Overall accuracy
        ACC = (TP+TN)/(TP+FP+FN+TN)
        ACC = ACC[~np.isnan(ACC)]
        
        return np.average(TPR), np.average(TNR), np.average(FPR), \
                np.average(FNR), np.average(ACC), np.average(PPV), \
                f1_score(np.average(PPV), np.average(TPR))

def sort_index(pred_scores, offset, cutoff = 0.5):
    confidence = [(idx + offset, abs(score-cutoff)) for idx, score in enumerate(pred_scores)]
    sorted_conf = sorted(confidence, key=lambda x:x[1])
    return sorted_conf

def sort_based_on_labels(x_train, y_train):
    inds = y_train.argsort()
    sorted_x_train = x_train[inds]
    sorted_y_train = y_train[inds]
    return sorted_x_train, sorted_y_train
