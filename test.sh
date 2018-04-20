#!/usr/bin/env bash

PYTHONPATH='.' pipenv run pytest "$@" --ignore=data --ignore=clustering_system