#!/usr/bin/env bash

pipenv run pytest "$@" --ignore=data --ignore=clustering_system