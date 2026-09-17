#!/bin/bash
#SBATCH --job-name=butterflyAI_multiclass_detectron
#SBATCH --output=pytorch_output_2.txt
#SBATCH --error=pytorch_error_2.txt
#SBATCH --partition=gpu_partition
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1

python3 -u label_monarchs.py

