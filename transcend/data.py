# -*- coding: utf-8 -*-

"""
data.py
~~~~~~~

Functions for caching and loading data during conformal evaluation.

"""
import logging
import os
import pickle
import json as json

import numpy as np
from scipy import sparse


def load_features(dataset, folder='features/transcend'):
    """Load features from datasets (e.g., Drebin and Marvin).

    The expected pickle format is a list of sparse matrices stored in
    class-separate files (denoted by the suffix '_mw_features.p' and
    '_gw_features.p').

    Each sparse matrix in the list represents the feature vector for a single
    observation (e.g., a row of the feature matrix X).

    [
        <scipy.sparse.lil.lil_matrix>,
        <scipy.sparse.lil.lil_matrix>,
        <scipy.sparse.lil.lil_matrix>,
        ...
        <scipy.sparse.lil.lil_matrix>
    ]

    A corresponding array of labels is generated based on the length of this
    list (positive class (1) == malware, negative class (0) == goodware).

    Args:
        dataset (str): The name of the dataset to load.
        folder (str): The folder to look for the datasets in.

    Returns:
        (np.ndarray, (np.ndarray): (features, ground truth) for the loaded
            dataset.

    """
    logging.info('Loading ' + dataset + '_gw features...')
    filepath = os.path.join(folder, dataset + '_gw_features.p')
    data_gw = load_csr_list(filepath)
    labels_gw = [0] * data_gw.shape[0]

    logging.info('Loading ' + dataset + '_mw features...')
    filepath = os.path.join(folder, dataset + '_mw_features.p')
    data_mw = load_csr_list(filepath)
    labels_mw = [1] * data_mw.shape[0]

    X = sparse.vstack([data_gw, data_mw], format='csr')
    y = np.array(labels_gw + labels_mw)

    return X, y


def load_csr_list(filepath):
    """Helper function to load and stack a sparse matrix.

    Args:
        filepath (str): The pickle file to load.

    Returns:
        (np.ndarray): Feature matrix X.

    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, list):
        data = sparse.vstack(data, format='csr')

    return data


def load_cached_data(data_path):
    """Load cached data (trained model, computed p-values, etc).

    Args:
        data_path: (str) To avoid mix-ups, and to allow safe caching of models
            produced during calibration, it's advised to keep this location
            'fold-specific'.

    See Also:
        - `cache_data`

    Returns:
        The previously cached data.

    """
    logging.info('Loading data from {}...'.format(data_path))
    with open(data_path, 'rb') as f:
        model = pickle.load(f)
    logging.debug('Done.')
    return model


def cache_data(model, data_path):
    """Cache data (trained model, computed p-values, etc).

    Args:
        model: The data to save.
        data_path: (str) To avoid mix-ups, and to allow safe caching of models
            produced during calibration, it's advised to keep this location
            'fold-specific'.

    See Also:
        - `load_cached_data`

    """

    model_folder_path = os.path.dirname(data_path)

    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)

    logging.info('Saving data to {}...'.format(data_path))
    with open(data_path, 'wb') as f:
        pickle.dump(model, f)
    logging.debug('Done.')


def save_results(results, args, output='txt'):
    """Helper function to save results with distinct filename based on args."""
    test_file, train_file = args.test.replace('/', ''), args.train.replace('/', '')

    filename = 'report-{}-{}-fold-{}-{}-{}'.format(
        train_file, args.folds, args.thresholds, args.criteria, test_file)
    if args.criteria == 'quartiles':
        filename += '-{}'.format(args.q_consider)
    if args.criteria == 'constrained-search':
        filename += '-{}max-{}con-{}s'.format(
            args.cs_max, args.cs_min, args.rs_samples)
    if args.criteria == 'random-search':
        filename += '-{}max-{}mwrej-{}s'.format(
            args.rs_max, args.rs_reject_limit, args.rs_samples)
    filename += '.' + output

    folder = './reports'

    if not os.path.exists(folder):
        os.makedirs(folder)

    results_path = os.path.join(folder, filename)
    logging.info('Saving results to {}...'.format(results_path))

    if output == 'txt':
        with open(results_path, 'wt') as f:
            f.write(results)
    elif output == 'json':
        with open(results_path, 'wt') as f:
            json.dump(results, f)
    elif output == 'p':
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
    else:
        msg = 'Unknown file format could not save.'
        logging.error(msg)
    logging.debug('Done.')
