# -*- coding: utf-8 -*-

"""
utils.py
~~~~~~~~

Helper functions for setting up the environment.

"""
import argparse
import logging
import multiprocessing as mp
import sys
from pprint import pformat

from termcolor import colored


def configure_logger():
    class SpecialFormatter(logging.Formatter):
        FORMATS = {
            logging.DEBUG: logging._STYLES['{'][0](
                colored("[*] {message}", 'blue')),
            logging.INFO: logging._STYLES['{'][0](
                colored("[*] {message}", 'cyan')),
            logging.WARNING: logging._STYLES['{'][0](
                colored("[!] {message}", 'yellow')),
            logging.ERROR: logging._STYLES['{'][0](
                colored("[!] {message}", 'red')),
            logging.CRITICAL: logging._STYLES['{'][0](
                colored("[!] {message}", 'white', 'on_red')),
            'DEFAULT': logging._STYLES['{'][0]("[ ] {message}")}

        def format(self, record):
            self._style = self.FORMATS.get(record.levelno,
                                           self.FORMATS['DEFAULT'])
            return logging.Formatter.format(self, record)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(SpecialFormatter())
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.DEBUG)


def parse_args():
    """Parse the command line configuration for a particular run.

    See Also:
        - `data.load_features`
        - `thresholding.find_quartile_thresholds`
        - `thresholding.find_random_search_thresholds`
        - `thresholding.sort_by_predicted_label`
        - `thresholding.get_performance_with_rejection`

    Returns:
        argparse.Namespace: A set of parsed arguments.

    """
    p = argparse.ArgumentParser()

    # Dataset options
    p.add_argument('--train', default='drebin',
                   help='The training set to use.')
    p.add_argument('--test', default='marvin_full',
                   help='The testing set to use.')
    p.add_argument('--budget', default=10, type=int, help="The number of test samples to choose for active learning.")
    p.add_argument('--reg', default=1.0, type=float, help="The regularization parameter to use for SVM.")
    # Calibration options
    p.add_argument('-k', '--folds', default=10, type=int,
                   help='The number of folds to use during calibration.')
    p.add_argument('-n', '--ncpu', default=-2, type=int,
                   help='The number of processes to use. '
                        'Negative values are interpreted as (`mpu.cpu_count()` '
                        '- abs(args.ncpu))')
    p.add_argument('--pval-consider', default='full-train',
                   choices=['full-train', 'cal-only'],
                   help='The ncms to consider when generating p values.')

    # Thresholding options
    p.add_argument('-t', '--thresholds', default='quartiles',
                   choices=['quartiles', 'random-search', 'constrained-search', 'full-search'],
                   help='The type of thresholds to use.')

    p.add_argument('-c', '--criteria', default='cred',
                   choices=['cred', 'conf', 'cred+conf'],
                   help='The p-values to threshold on.')

    # Sub-arguments for --thresholds=quartiles
    p.add_argument('--q-consider',  # default='correct',
                   choices=['correct', 'incorrect', 'all'],
                   help='Which predictions to select quartiles from.')

    # Sub-arguments for --thresholds=random-search
    p.add_argument('--rs-max',  # default='f1_k,kept_total_perc',
                   help='The performance metric(s) to maximise (comma sep).')
    p.add_argument('--rs-min',  # default='f1_r',
                   help='The performance metric(s) to minimise (comma sep).')
    p.add_argument('--rs-ceiling',  # default='0.25',
                   help='The maximum total rejections that is acceptable. '
                        'Either a float (for `total_reject_perc` or comma '
                        'separated key:value pairs '
                        '(e.g., \'total_reject_perc:0.25,f1_r:0.8\')')
    p.add_argument('--rs-samples', type=int,  # default=100,
                   help='The number of sample selections to make.')

    # Sub-arguments for --thresholds=constrained-search
    p.add_argument('--cs-max',  # default='f1_k:0.99',
                   help='The performance metric(s) to maximise. '
                        'Comma separated key:value pairs (e.g., "f1_k:0.99")')
    p.add_argument('--cs-con',  # default='kept_total_perc:0.75',
                   help='The performance metric(s) to constrain. '
                        'Comma separated key:value pairs (e.g., "kept_total_perc:0.75")')
    args = p.parse_args()

    try:
        args.rs_ceiling = float(args.rs_ceiling)
    except (TypeError, ValueError):
        pass  # Leave it as a string (e.g., 'total_reject_perc:0.25') or None

    # Resolve ncpu value (negative values are interpreted as
    # mp.cpu_count() - abs(args.ncpu)
    args.ncpu = args.ncpu if args.ncpu > 0 else mp.cpu_count() + args.ncpu
    if args.ncpu < 0:
        raise ValueError('Invalid ncpu value.')

    logging.warning('Running with configuration:\n' + pformat(vars(args)))
    return args
