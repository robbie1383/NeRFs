#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=100gb
#SBATCH --time=4:00:00
#SBATCH --output=output.log


python main.py
