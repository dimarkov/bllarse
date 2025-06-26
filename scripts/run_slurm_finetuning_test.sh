#!/usr/bin/env bash

#SBATCH --job-name=finetuning_test   # Appears in queue listings
#SBATCH -G 1                         # partition
#SBATCH --mem=40GB                   # memory limit
#SBATCH --time=01:00:00              # HH:MM:SS – wall-clock time limit

./start_docker_sbatch.sh ./run_finetuning.sh
