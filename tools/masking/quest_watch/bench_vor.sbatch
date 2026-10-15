#!/bin/bash
#SBATCH -J vor_bench
#SBATCH -A e52938
#SBATCH -p gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=qgpu2014
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -t 01:30:00
#SBATCH -o /projects/e52938/kmy2091/logs/%x_%j.out
set -uo pipefail
V=/projects/e52938/kmy2091/cache/vor
WS=/projects/e52938/kmy2091
export PATH="$WS/cache/ffmpeg7/bin:$PATH" LD_LIBRARY_PATH="$WS/cache/ffmpeg7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME="$WS/cache/huggingface" HF_TOKEN_PATH="$HOME/.cache/huggingface/token" PYTHONUNBUFFERED=1
mkdir -p $V/bench; cd $V
nvidia-smi -L
[[ -f bench/clip.npz ]] || $WS/cache/sam3-venv/bin/python bench_vor.py masks 2>&1 | grep -v -i "warn\|rope\|Loading weights"
venv-minimax/bin/python bench_vor.py run minimax 12 2>&1 | grep -E "^\{|Error|Traceback|error"
venv-effecterase/bin/python bench_vor.py run effecterase 20 2>&1 | grep -E "^\{|Error|Traceback|error"
venv-effecterase/bin/python bench_vor.py run effecterase 50 2>&1 | grep -E "^\{|Error|Traceback|error"
$WS/cache/sam3-venv/bin/python bench_vor.py sheet
