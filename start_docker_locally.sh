#!/bin/bash

docker run -it --gpus all \
  -e HOME=$HOME \
  -v $HOME:$HOME \
  -v $(pwd):$(pwd) \
  -w $(pwd) \
  bllarse-dev "${@:-bash}"
