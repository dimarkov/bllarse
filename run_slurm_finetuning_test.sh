#!/usr/bin/env bash

#SBATCH --job-name=finetuning_test           # Appears in queue listings
#SBATCH --mem=8G                     # Memory per node
#SBATCH -G=1                         #  
#SBATCH --time=01:00:00              # HH:MM:SS – wall-clock time limit

./start_docker.sh run_python_script.sh