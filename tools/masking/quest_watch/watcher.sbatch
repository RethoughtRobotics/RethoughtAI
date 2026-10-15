#!/bin/bash
# Bimanual HF dataset watcher as a CPU Slurm job (login-node tmux sessions did not survive). Log: cache/watch/watcher.log
#SBATCH -J bimanual_watcher
#SBATCH -A e52938
#SBATCH -p normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -t 2-00:00:00
#SBATCH -o /projects/e52938/kmy2091/logs/%x_%j.out
exec bash /projects/e52938/kmy2091/cache/watch/run_watcher.sh
