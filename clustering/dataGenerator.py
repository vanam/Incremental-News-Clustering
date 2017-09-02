import argparse
import ast
import configparser

import numpy as np

KEYS = ['mean', 'cov', 'size']


def main(config_file, output_file):
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
    # Check if file exists and is readable
    file = argparse.FileType('r')(string)

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

    args = parser.parse_args()
    main(args.config, args.output)
