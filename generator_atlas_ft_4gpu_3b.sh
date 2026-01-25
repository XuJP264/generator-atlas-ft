#!/bin/bash
#PBS -q normal
#PBS -j oe
#PBS -l select=1:ncpus=20:ngpus=4:mem=440GB
#PBS -l walltime=16:00:00
#PBS -P personal-xu0029ng
#PBS -N generator_atlas_ft_3b_4gpu
#PBS -o generator_atlas_ft_3b_4gpu_1epoch.log

cd ${PBS_O_WORKDIR}

module load cuda/12.2.2
module load gcc/11.2.0

export CUDA_HOME=/app/apps/cuda/12.2.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=0,1,2,3

export HF_HOME=/home/users/ntu/xu0029ng/scratch/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export TRITON_CACHE_DIR=/home/users/ntu/xu0029ng/scratch/triton_cache
export TMPDIR=/home/users/ntu/xu0029ng/scratch/tmp
export XDG_CACHE_HOME=/home/users/ntu/xu0029ng/scratch/.cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export WANDB_PROJECT=generator-crispr
export WANDB_NAME=atlas_ft_3b_4gpu
export WANDB_MODE=offline

# ===== 正确进入 conda =====
source /home/users/ntu/xu0029ng/miniconda3/etc/profile.d/conda.sh
conda activate Generator

echo "==== ENV CHECK ===="
which python
python -V
python - << 'EOF'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU count:", torch.cuda.device_count())
EOF
echo "==================="

cd /home/users/ntu/xu0029ng/scratch/GENERator

torchrun \
  --nproc_per_node=4 \
  src/tasks/downstream/fine_tuning.py \
    --model_name GenerTeam/GENERator-v2-prokaryote-3b-base \
    --dataset_name metaXu264/crispr-cas-atlas-generator \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_length 8192 \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --save_steps 2000 \
    --output_dir results/atlas_ft_3b \
    --pad_to_multiple_of_six \
    --distributed_type deepspeed

