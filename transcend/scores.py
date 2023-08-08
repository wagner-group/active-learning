# -*- coding: utf-8 -*-

"""
scores.py
~~~~~~~~~

Functions for producing the various scores used during conformal evaluation,
such as non-conformity measures, credibility and confidence p-values and
probabilities for comparison.

Note that the functions in this module currently only apply to producing
scores for a binary classification task and an SVM classifier. Different
settings and different classifiers will require their own functions for
generating non-conformity measures based on different intuitions.

"""
import numpy as np
from tqdm import tqdm


def get_svm_ncms(clf, X_in, y_in):
    """Helper functions to get NCMs across an entire pair of X,y arrays. """
    assert hasattr(clf, 'decision_function')
    return [get_single_svm_ncm(clf, x, y) for x, y in
            tqdm(zip(X_in, y_in), total=len(y_in), desc='svm ncms')]


def get_single_svm_ncm(clf, single_x, single_y):
    """Collect a non-conformity measure from the classifier for `single_x`.

    A note about SVM ncms: In binary classification with a linear SVM, the
    output score is the distance from the hyperplane with respect to the
    positive class. If the score is negative, the prediction is class 0, if
    positive, it's class 1 (in sklearn technically it will be clf.class_[0] and
    clf.class_[1] respectively). To perform thresholding with conformal
    evaluator, we need the distance from the hyperplane with respect to *both*
    classes, so we simply flip the sign to get the 'reflection' for the other
    class.

    Args:
        clf (sklearn.svm.SVC): The classifier to use for the NCMs.
        single_x (np.ndarray): An single feature vector to get the NCM for.
        single_y (int): Either the ground truth (calibration) or predicted
            label (testing) of `single_x`.

    Returns:
        float: The NCM for the given `single_x`.

    """
    assert hasattr(clf, 'decision_function')
    decision = clf.decision_function(single_x)

    # If y (ground truth in calibration, prediction in testing) is malware
    # then flip the sign to ensure the most conforming point is most minimal.
    if single_y == 1:
        return -decision
    elif single_y == 0:
        return decision
    raise Exception('Unknown class? Only binary decisions supported.')


def compute_p_values_cred_and_conf(
        train_ncms, groundtruth_train, test_ncms, y_test, cred_only=False):
    """Helper function to compute p-values across an entire array."""
    cred = [compute_single_cred_p_value(train_ncms=train_ncms,
                                        groundtruth_train=groundtruth_train,
                                        single_test_ncm=ncm,
                                        single_y_test=y)
            for ncm, y in tqdm(
            zip(test_ncms, y_test), total=len(y_test), desc='cred pvals')]
    if cred_only:
        conf = []
    else:
        conf = [compute_single_conf_p_value(train_ncms=train_ncms,
                                            groundtruth_train=groundtruth_train,
                                            single_test_ncm=ncm,
                                            single_y_test=y)
                for ncm, y in tqdm(
                zip(test_ncms, y_test), total=len(y_test), desc='conf pvals')]

    return {'cred': cred, 'conf': conf}


def compute_single_cred_p_value(
        train_ncms, groundtruth_train, single_test_ncm, single_y_test):
    """Compute a single credibility p-value.

    Credibility p-values describe how 'conformal' a point is with respect to
    the other objects of that class. They're computed as the proportion of
    points with greater NCMs (the number of points _less conforming_ than the
    reference point) over the total number of points.

    Intuitively, a point predicted as malware which is the further away from
    the decision boundary than any other point will have the highest p-value
    out of all other malware points. It will have the smallest NCM (as it is
    the least _non-conforming_) and thus no other points will have a greater
    NCM and it will have a credibility p-value of 1.

    Args:
        train_ncms (np.ndarray): An array of training NCMs to compare the
            reference point against.
        groundtruth_train (np.ndarray): An array of ground truths corresponding
            to `train_ncms`.
        single_test_ncm (float): A single reference point to compute the
            p-value of.
        single_y_test (int): Either the ground truth (calibration) or predicted
            label (testing) of `single_test_ncm`.

    See Also:
        - `compute_p_values_cred_and_conf`
        - `compute_single_conf_p_value`

    Returns:
        float: The p-value for `single_test_ncm` w.r.t. `train_ncms`.

    """
    assert len(set(groundtruth_train)) == 2  # binary classification tasks only

    how_many_are_greater_than_single_test_ncm = 0

    for ncm, groundtruth in zip(train_ncms, groundtruth_train):
        if groundtruth == single_y_test and ncm >= single_test_ncm:
            how_many_are_greater_than_single_test_ncm += 1

    single_cred_p_value = (how_many_are_greater_than_single_test_ncm /
                           sum(1 for y in groundtruth_train if
                               y == single_y_test))
    return single_cred_p_value


def compute_single_conf_p_value(
        train_ncms, groundtruth_train, single_test_ncm, single_y_test):
    """Compute a single confidence p-value.

    The confidence p-value is computed similarly to the credibility p-value,
    except it aims to capture the confidence that the classifier has that the
    point _doesn't_ belong to the opposite class.

    To achieve this we assume that point has the label of the second highest
    scoring class---in binary classification, simply the opposite class---and
    compute the credibility p-value with respect to other points of that class.
    The confidence p-value is (1 - this value).

    Note that in transductive conformal evaluation, the entire classifier
    should be retrained with the reference point given the label of the
    opposite class. Usually, this is computationally prohibitive, and so this
    approximation assumes that the decision boundary undergoes only minimal
    changes when the label of a single point is flipped.

    See Also:
        - `compute_p_values_cred_and_conf`
        - `compute_single_cred_p_value`

    Args:
        train_ncms (np.ndarray): An array of training NCMs to compare the
            reference point against.
        groundtruth_train (np.ndarray): An array of ground truths corresponding
            to `train_ncms`.
        single_test_ncm (float): A single reference point to compute the
            p-value of.
        single_y_test (int): Either the ground truth (calibration) or predicted
            label (testing) of `single_test_ncm`.

    Returns:
        float: The p-value for `single_test_ncm` w.r.t. `train_ncms`.

    """
    assert len(set(groundtruth_train)) == 2  # binary classification tasks only

    # 'Cast' NCMs to NCMs with respect to the opposite class (binary only)
    # train_ncms_opposite_class = -1 * np.array(train_ncms)
    single_y_test_opposite_class = 0 if single_y_test == 1 else 1
    single_test_ncm_opposite_class = -1 * single_test_ncm

    how_many_are_greater_than_single_test_ncm = 0

    for ncm, groundtruth in zip(train_ncms, groundtruth_train):
        if (groundtruth == single_y_test_opposite_class
                and ncm >= single_test_ncm_opposite_class):
            how_many_are_greater_than_single_test_ncm += 1

    single_cred_p_value_opposite_class = (
            how_many_are_greater_than_single_test_ncm /
            sum(1 for y in groundtruth_train if
                y == single_y_test_opposite_class))

    return 1 - single_cred_p_value_opposite_class  # confidence p value


def get_svm_probs(clf, X_in):
    """Get scores and predictions for comparison with probabilities.

    Note that this function returns the predictions _and_ probabilities given
    by the classifier and that these predictions may different from other
    outputs from the same classifier (such as `predict` or `decision_function`.
    This is due to Platt's scaling (and it's implementation in scikit-learn) in
    which a 5-fold SVM is trained and used to score the observation
    (`predict_proba()` is actually the average of these 5 classifiers).

    The takeaway is to be sure that you're always using probability scores with
    probability predictions and not with the output of other SVC functions.

    Args:
        clf (sklearn.svm.SVC): The classifier to use for the probabilities.
        X_in (np.ndarray): An array of feature vectors to classify.

    Returns:
        (list, list): (Probability scores, probability labels) for `X_in`.

    """
    assert hasattr(clf, 'predict_proba')
    probability_results = clf.predict_proba(X_in)
    probas_cal_fold = [np.max(t) for t in probability_results]
    pred_proba_cal_fold = [np.argmax(t) for t in probability_results]
    return probas_cal_fold, pred_proba_cal_fold
