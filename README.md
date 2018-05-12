# Incremental News Clustering

[![Build Status](https://travis-ci.org/vanam/clustering.svg?branch=master)](https://travis-ci.org/vanam/clustering)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

The goal was to research model-based clustering methods, notably the
Distance Dependent Chinese Restaurant Process (ddCRP), and propose an incremental
clustering system which would be capable of maintaining the growing number
of topic clusters of news articles coming online from a crawler.
LDA, LSA, and doc2vec methods were used to represent a document as a fixed-length numeric vector.
Cluster assignments given by a proof-of-concept implementation of such a system were
evaluated using various metrics, notably purity, F-measure and V-measure.
A modification of V-measure -- NV-measure -- was introduced
in order to penalize an excessive or insufficient number of clusters.
The best results were achieved with doc2vec and ddCRP.

> Due to copyright, news articles used for experiments are only available at the university library.

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
