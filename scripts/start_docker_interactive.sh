#!/bin/bash

docker run -it --rm --gpus=\"device=$SLURM_STEP_GPUS\"\
  -e HOME=$HOME \
  -v $HOME:$HOME \
  -v $(pwd):$(pwd) \
  -w $(pwd) \
  -u $(id -u):$(id -g) \
  bllarse-dev "${@:-bash}"
