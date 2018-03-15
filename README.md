# Clustering

[![Build Status](https://travis-ci.org/vanam/clustering.svg?branch=master)](https://travis-ci.org/vanam/clustering)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

A source code for my thesis

## Installation

Requirements

* Python 3.5
* Pip
* Pipenv

### Ubuntu

```bash
$ sudo apt-get install python3 python3-tk python3-pip
$ pip3 install pipenv
```

### Project dependencies

```bash
$ pipenv install --dev
```

### ~/.bashrc

```bash
export PYTHONPATH='.'
```

## Development

### Configure PyCharm

* [Configure PyCharm to use virtualenv](http://exponential.io/blog/2015/02/10/configure-pycharm-to-use-virtualenv/)

### Activate project's virtualenv

```bash
$ pipenv shell
```

### Run script

```bash
$ pipenv run python <script_name>.py
```

### Run tests

```bash
$ pipenv run pytest tests
```