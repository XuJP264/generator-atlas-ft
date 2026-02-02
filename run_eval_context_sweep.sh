#!/bin/bash
#PBS -q normal
#PBS -j oe
#PBS -l select=1:ncpus=16:ngpus=1:mem=128GB
#PBS -l walltime=12:00:00
#PBS -P personal-xu0029ng
#PBS -N generator_ctx_sweep
#PBS -o generator_ctx_sweep.log

cd ${PBS_O_WORKDIR}

# =========================
# Modules (minimal)
# =========================

module purge
module load cuda/12.2.2
# ❌ 不再 load gcc

export CUDA_HOME=/app/apps/cuda/12.2.2
export PATH=$CUDA_HOME/bin:$PATH

# 单卡 eval
export CUDA_VISIBLE_DEVICES=0

# =========================
# Conda
# =========================

source /home/users/ntu/xu0029ng/miniconda3/etc/profile.d/conda.sh
conda activate Generator

# ⚠️ 关键：让 conda 的 lib 永远优先
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CUDA_HOME/lib64

# Cache & temp dirs
export HF_HOME=/home/users/ntu/xu0029ng/scratch/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export TRITON_CACHE_DIR=/home/users/ntu/xu0029ng/scratch/triton_cache
export TMPDIR=/home/users/ntu/xu0029ng/scratch/tmp
export XDG_CACHE_HOME=/home/users/ntu/xu0029ng/scratch/.cache

# CUDA allocator
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# =========================
# ENV CHECK
# =========================

echo "==== ENV CHECK ===="
which python
python -V
python - << 'EOF'
import torch, sys
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU count:", torch.cuda.device_count())
import ctypes
print("libstdc++:", ctypes.util.find_library("stdc++"))
EOF
echo "==================="

# =========================
# Run evaluation
# =========================

cd /home/users/ntu/xu0029ng/scratch/GENERator

echo "Starting context sweep evaluation..."
date

python eval_next_kmer_context_sweep.py

date
echo "Evaluation finished."
