import argparse
import ast
import configparser

import numpy as np
from scipy.stats import wishart

KEYS = ['mean', 'cov', 'size']

CLUSTER_DIM_RANGE = [50, 150]


def main(config_file, output_file, random=False, k=20, d=2):
    if random:
        # Mean hyper-parameters
        mean = np.zeros(d)
        cov = np.eye(d) * k * 8
        scale = np.eye(d)

        # Define config
        config = configparser.ConfigParser()

        for i in range(k):
            config['c%d' % i] = {}
            cluster_config = config['c%d' % i]
            cluster_config['mean'] = str(np.random.multivariate_normal(mean, cov, 1)[0].tolist())
            cluster_config['cov'] = str(wishart.rvs(d + 1, scale).tolist())
            cluster_config['size'] = str(np.random.randint(CLUSTER_DIM_RANGE[0], CLUSTER_DIM_RANGE[1]))

        # Write config
        config.write(config_file)

        # Move to the beginning of a file
        config_file.seek(0, 0)

    # Read config
    config = configparser.ConfigParser()
    config.read_file(config_file)

    # For each section generate points
    data = []
    for s in config.sections():
        # Read configuration
        mean = ast.literal_eval(config[s]['mean'])
        cov = ast.literal_eval(config[s]['cov'])
        size = ast.literal_eval(config[s]['size'])

        # Is multivariate?
        if isinstance(mean, list):
            data.append(np.random.multivariate_normal(mean, cov, size))
        else:
            data.append(np.transpose([np.random.normal(mean, cov, size)]))

    # Close configuration file
    config_file.close()

    # Save data
    np.save(output_file, data)

    # Close output file
    output_file.close()


def is_valid_config(string):
    # Check if file exists and is writeable
    file = argparse.FileType('w+')(string)

    # Check if file is a configuration file
    config = configparser.ConfigParser()
    try:
        config.read_file(file)
    except configparser.Error as e:
        raise argparse.ArgumentTypeError(e.message)

    # Check if file contains mandatory keys
    for s in config.sections():
        keys = list(config[s].keys())
        if not np.intersect1d(keys, KEYS).size is len(KEYS):
            message = "missing mandatory keys in section '%s'"
            raise argparse.ArgumentTypeError(message % s)

    # Move to the beginning of a file
    file.seek(0, 0)
    return file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate multivariate normal random data.')
    parser.add_argument('config', metavar='config-file', type=is_valid_config,
                        help='a data configuration file')
    parser.add_argument('output', metavar='output-file', type=argparse.FileType('wb'),
                        help='an output data file')
    parser.add_argument('-k, --clusters', dest='k', type=int,
                        help='the number of clusters (default: 10)')
    parser.add_argument('-d, --dimension', dest='d', type=int,
                        help='the data dimension (default: 2)')
    parser.add_argument('-r, --random', dest='random', action='store_true',
                        help='randomly generate data')
    parser.set_defaults(k=10, d=2, random=False)

    args = parser.parse_args()
    main(args.config, args.output, args.random, args.k)
