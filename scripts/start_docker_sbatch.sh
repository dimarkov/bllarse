#!/bin/bash

docker run --rm --gpus="device=$SLURM_JOB_GPUS"\
  -e HOME=$HOME \
  -v $HOME:$HOME \
  -v $(pwd):$(pwd) \
  -w $(pwd) \
  bllarse-dev "$@"
