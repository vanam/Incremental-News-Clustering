import argparse
import re

import os


def check_dir(dir_path, value):
    directory = os.path.join(dir_path, value)

    if not os.path.isdir(directory):
        raise argparse.ArgumentTypeError("%s directory not found" % value)

    return directory


def check_lang(value):
    lang_patter = re.compile("[a-z]{2}")

    if len(value) != 2 or lang_patter.match(value) is None:
        raise argparse.ArgumentTypeError("%s is an invalid language" % value)

    return value


def check_uint(value):
    value = int(value)

    if value < 1:
        raise argparse.ArgumentTypeError("%d must be positive number" % value)

    return value

