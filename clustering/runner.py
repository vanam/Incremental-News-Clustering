import argparse
import logging

import os

import sys
from time import sleep

import numpy as np

from clustering.measures import Evaluation, TrainingEvaluation
from clustering.readData import read_data2

RESULT_DIR_NAME = 'results'


def kmeans(k, X, output_directory):
    # Store parameters to a file
    with open(os.path.join(output_directory, 'kmeans.par'), 'w') as f:
        f.write("k: %d\n" % k)


    sleep(1)

    clusters = []
    likelihood = [0]
    return likelihood, clusters


def em(k, X, output_directory):
    # TODO Store parameters to a file

    sleep(1)

    # TODO run clustering algorithm
    clusters = []
    likelihood = []
    return likelihood, clusters


def fbgmm(k, X, output_directory):
    # TODO Store parameters to a file

    sleep(1)

    # TODO run clustering algorithm
    clusters = []
    likelihood = []
    return likelihood, clusters


def ibgmm(k, X, output_directory):
    # TODO Store parameters to a file

    sleep(1)

    # TODO run clustering algorithm
    clusters = []
    likelihood = []
    return likelihood, clusters


def ddibgmm(k, X, output_directory):
    # TODO Store parameters to a file

    sleep(1)

    # TODO run clustering algorithm
    clusters = []
    likelihood = []
    return likelihood, clusters


def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)

    https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r\033[K%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix))

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def run(args, X, output_directory, classes=None):
    # Define algorithm functions to run
    function_lable = [
        (kmeans,  'k-means'),
        (em,      'finite bayesian GMM using EM algorithm'),
        (fbgmm,   'finite bayesian GMM using Gibbs algorithm'),
        (ibgmm,   'infinite bayesian GMM (CRP) using Gibbs algorithm'),
        (ddibgmm, 'distance dependent infinite bayesian GMM (ddCRP) using Gibbs algorithm')
    ]

    # Load algorithm flags
    flags = [args.kmeans, args.em, args.fbgmm, args.ibgmm, args.ddibgmm]

    # Get number of different clustering algorithms
    alg_count = sum(flags)

    # Number of finished clustering algorithm runs
    finished = 0

    print("Running clustering algorithms")
    evaluations = np.empty(len(flags), dtype=Evaluation)

    for i, pair in enumerate(function_lable):
        if not flags[i]:
            continue

        f, label = pair

        # Print progress bar
        print_progress(finished, alg_count, suffix='Running %s' % label, bar_length=50)

        # Run clustering algorithm
        likelihood, clusters = f(args.c, X, output_directory)

        # TODO Plot and save likelihood
        # TODO Plot and save clusters

        # Evaluate clustering method
        if classes is None:
            evaluations[i] = Evaluation(clusters, likelihood[0])
        else:
            evaluations[i] = TrainingEvaluation(clusters, classes, likelihood[0])

        # Update number of finished algorithm runs
        finished += 1

    # Print final progress bar
    print_progress(finished, alg_count, bar_length=50)

    # TODO Process evaluation results using Evaluation objects and likelihoods for all clustering methods
    for e in evaluations:
        if e is not None:
            print(e)

if __name__ == "__main__":
    # Default values
    default = {
        'c': 2,
        'kmeans': False,
        'em': False,
        'fbgmm': False,
        'ibgmm': False,
        'ddibgmm': False,
    }

    # Define arguments
    parser = argparse.ArgumentParser(description='Run clustering algorithms.')

    parser.add_argument('file', metavar='file', type=argparse.FileType('rb'),
                        help='a data file')

    parser.add_argument('-c', dest='c', type=int,
                        help='the (initial) number of clusters (default: %d)' % default['c'])
    parser.add_argument('-k, --kmeans', dest='kmeans', action='store_true',
                        help='k-means')
    parser.add_argument('-e, --em', dest='em', action='store_true',
                        help='finite bayesian GMM using EM algorithm')
    parser.add_argument('-f, --fbgmm', dest='fbgmm', action='store_true',
                        help='finite bayesian GMM using Gibbs algorithm')
    parser.add_argument('-i, --ibgmm', dest='ibgmm', action='store_true',
                        help='infinite bayesian GMM (CRP) using Gibbs algorithm')
    parser.add_argument('-d, --ddibgmm', dest='ddibgmm', action='store_true',
                        help='distance dependent infinite bayesian GMM (ddCRP) using Gibbs algorithm')
    parser.set_defaults(
        c=default['c'],
        kmeans=default['kmeans'],
        em=default['em'],
        fbgmm=default['fbgmm'],
        ibmgm=default['ibgmm'],
        ddibgmm=default['ddibgmm'],
    )

    # Parse arguments
    args = parser.parse_args()

    # Define and create output directory
    raw_filename = os.path.splitext(os.path.basename(args.file.name))[0]
    output_directory = filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..',
        RESULT_DIR_NAME,
        raw_filename
    )

    # Create result directory if does not exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Define logger
    logging.basicConfig(
        filename=os.path.join(output_directory, '%s.log' % raw_filename),
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Load data file
    X, classes = read_data2(args.file)
    # TODO read data without class assignment

    # Run selected clustering algorithms using flags
    run(args, X, output_directory, classes=None)
